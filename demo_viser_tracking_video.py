# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Demo with Instance Tracking and 3D Bounding Boxes - Video Input Version

This version works directly with video files instead of pre-extracted frames:
- Extracts frames on-the-fly at specified FPS
- Supports SAM3 segmentation output format (binary PNG masks)
- Automatically cleans up temporary frames after processing

Usage:
    python demo_viser_tracking_video.py \
        --video ./data/my_video.mp4 \
        --sam3_masks ./data/sam3_output/ \
        --extract_fps 5 \
        --output_dir ./outputs/
"""

import os
import sys
import time
import json
import argparse
from typing import List, Optional
from datetime import datetime

import numpy as np
import torch
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# viser is optional - only needed for visualization
try:
    import viser
    HAS_VISER = True
except ImportError:
    HAS_VISER = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.video_utils import VideoFrameContext, get_video_info
from vggt.utils.sam3_mask_loader import load_sam3_masks, load_sam3_masks_from_image_paths

# Import tracking components from original demo
from demo_viser_tracking import (
    ImprovedTracker,
    compute_instance_bboxes,
    save_tracking_summary,
    save_trajectory_plots,
    save_kitti_labels,
    project_bboxes_to_2d,
    parse_dji_logs,
    parse_dji_logs_with_gps,
    viser_wrapper_with_tracking,
    load_grounded_sam_masks,
    extract_frame_number,
    select_frames,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT demo with video input and SAM3 mask support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with SAM3 masks
    python demo_viser_tracking_video.py \\
        --video ./data/my_video.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # With specific frame range (match SAM3 mask frame numbers)
    python demo_viser_tracking_video.py \\
        --video ./data/my_video.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --start_frame 100 --end_frame 500 \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # With DJI telemetry
    python demo_viser_tracking_video.py \\
        --video ./data/drone.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --dji_log ./data/metadata.srt \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # Backward compatible: use pre-extracted frames
    python demo_viser_tracking_video.py \\
        --image_folder ./data/frames/ \\
        --mask_dir ./data/grounded_sam_masks/ \\
        --output_dir ./outputs/
        """
    )

    # Video input (new primary interface)
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--extract_fps", type=float, default=None,
                        help="Frame rate for extraction (default: match SAM3 fps or video native)")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="First video frame number to process (0-indexed, matches mask filenames)")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="Last video frame number to process (0-indexed, inclusive, matches mask filenames)")
    parser.add_argument("--sam3_masks", type=str, default=None,
                        help="Path to SAM3 output directory (containing masks/ subfolder)")
    parser.add_argument("--sam3_fps", type=float, default=None,
                        help="FPS used when generating SAM3 masks (default: read from metadata)")
    parser.add_argument("--object_class", type=str, default="object",
                        help="Default class name for SAM3 objects (SAM3 doesn't provide classes)")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Don't delete extracted frames after processing (for debugging)")

    # Backward compatible interface (original arguments)
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Path to folder containing pre-extracted images (backward compatibility)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Path to pre-computed Grounded SAM masks (JSON) (backward compatibility)")

    # Common arguments
    parser.add_argument("--dji_log", type=str, default=None,
                        help="Path to DJI SRT file for gimbal/GPS data")
    parser.add_argument("--use_gps_refinement", action="store_true",
                        help="Enable GPS-based pose refinement")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port number for the viser server")
    parser.add_argument("--num_images", type=int, default=20,
                        help="Max number of images to process (VGGT limit ~20)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Skip factor for frame subsampling (only for image_folder mode)")
    parser.add_argument("--conf_threshold", type=float, default=25.0,
                        help="Initial confidence threshold (%%)")
    parser.add_argument("--max_distance", type=float, default=8.0,
                        help="Max 3D distance for tracking")
    parser.add_argument("--max_missing_frames", type=int, default=20,
                        help="Frames before track goes dormant")
    parser.add_argument("--dormant_timeout", type=int, default=100,
                        help="Frames before dormant track removed")
    parser.add_argument("--use_point_map", action="store_true",
                        help="Use point map instead of depth-based points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--background_mode", action="store_true",
                        help="Run the viser server in background mode")
    parser.add_argument("--no_visualization", action="store_true",
                        help="Skip viser visualization (headless mode)")

    args = parser.parse_args()

    # Validate inputs
    if args.video is None and args.image_folder is None:
        parser.error("Either --video or --image_folder must be provided")

    if args.video and args.image_folder:
        print("Warning: Both --video and --image_folder provided. Using --video.")
        args.image_folder = None

    return args


def process_video_mode(args, device):
    """Process video input with SAM3 masks."""
    print(f"\n=== Video Mode ===")
    print(f"Video: {args.video}")

    # Get video info
    video_info = get_video_info(args.video)
    print(f"Video info: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.2f} fps, {video_info['total_frames']} frames")

    # Determine extraction FPS
    if args.extract_fps:
        target_fps = args.extract_fps
    elif args.sam3_fps:
        target_fps = args.sam3_fps
    else:
        # Try to read from SAM3 metadata
        if args.sam3_masks:
            metadata_path = os.path.join(args.sam3_masks, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    sam3_metadata = json.load(f)
                target_fps = sam3_metadata.get('fps', sam3_metadata.get('effective_fps', video_info['fps']))
                print(f"Using FPS from SAM3 metadata: {target_fps}")
            else:
                target_fps = video_info['fps']
        else:
            target_fps = video_info['fps']

    print(f"Extraction FPS: {target_fps}")

    # Show frame range if specified
    if args.start_frame is not None or args.end_frame is not None:
        print(f"Frame range: {args.start_frame or 0} to {args.end_frame or 'end'}")

    # Extract frames using context manager
    with VideoFrameContext(
        args.video,
        target_fps=target_fps,
        max_frames=args.num_images,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        keep_frames=args.keep_frames
    ) as ctx:
        image_paths = ctx.frame_paths
        frame_indices = ctx.frame_indices  # Original video frame indices (for DJI telemetry matching)

        print(f"Extracted {len(image_paths)} frames")
        print(f"Video frame indices: {frame_indices[:5]}..." if len(frame_indices) > 5 else f"Video frame indices: {frame_indices}")

        # Run the main processing pipeline
        results = run_processing_pipeline(
            args=args,
            device=device,
            image_paths=image_paths,
            frame_indices=frame_indices,
            video_fps=video_info['fps'],
            target_fps=target_fps,
            is_video_mode=True
        )

    return results


def process_image_folder_mode(args, device):
    """Process pre-extracted image folder (backward compatibility)."""
    print(f"\n=== Image Folder Mode (Backward Compatible) ===")
    print(f"Image folder: {args.image_folder}")

    # Select frames using original logic
    image_paths, frame_numbers = select_frames(args.image_folder, args.num_images, args.skip)

    # Run the main processing pipeline
    results = run_processing_pipeline(
        args=args,
        device=device,
        image_paths=image_paths,
        frame_indices=frame_numbers,
        video_fps=None,  # Not applicable for image folder mode
        target_fps=None,
        is_video_mode=False
    )

    return results


def run_processing_pipeline(
    args,
    device,
    image_paths: List[str],
    frame_indices: List[int],
    video_fps: Optional[float],
    target_fps: Optional[float],
    is_video_mode: bool
):
    """Run the main VGGT processing pipeline."""
    t0 = time.time()

    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    # Load VGGT model
    print(f"\n=== Loading VGGT Model ===")
    t1 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"Model loading: {time.time() - t1:.2f}s")

    # Load and preprocess images
    t2 = time.time()
    print(f"\n=== Loading Images ===")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    print(f"Image loading: {time.time() - t2:.2f}s")

    # Run inference
    t3 = time.time()
    print(f"\n=== Running VGGT Inference ===")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    torch.cuda.synchronize()
    print(f"Inference: {time.time() - t3:.2f}s")

    # Convert pose encoding to extrinsic and intrinsic matrices
    t4 = time.time()
    print(f"\n=== Processing Outputs ===")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    print(f"Post-processing: {time.time() - t4:.2f}s")

    # Parse DJI logs if provided
    gimbal_data = None
    if args.dji_log:
        print(f"\n=== Parsing DJI Logs ===")
        if args.use_gps_refinement:
            gimbal_data = parse_dji_logs_with_gps(args.dji_log, frame_indices)
        else:
            gimbal_data = parse_dji_logs(args.dji_log, frame_indices)

    # Load original images for 2D projection
    original_images = [cv2.imread(p) for p in image_paths]

    # Initialize tracking and compute bounding boxes
    bounding_boxes = []
    all_track_ids = []
    masks_data = None

    # Load masks based on format
    if args.sam3_masks and is_video_mode:
        print(f"\n=== Loading SAM3 Masks ===")
        # Use direct frame matching when user specifies start_frame/end_frame
        # (they've explicitly chosen frame numbers that match mask filenames)
        use_direct_match = (
            getattr(args, 'start_frame', None) is not None or
            getattr(args, 'end_frame', None) is not None
        )
        masks_data = load_sam3_masks(
            sam3_output_dir=args.sam3_masks,
            extracted_frame_indices=frame_indices,
            video_fps=video_fps,
            sam3_fps=args.sam3_fps or target_fps,
            class_name=args.object_class,
            direct_frame_match=use_direct_match
        )
    elif args.sam3_masks and not is_video_mode:
        # For image folder mode with SAM3 masks, use filename-based matching
        print(f"\n=== Loading SAM3 Masks (filename matching) ===")
        masks_data = load_sam3_masks_from_image_paths(
            sam3_output_dir=args.sam3_masks,
            image_paths=image_paths,
            class_name=args.object_class
        )
    elif args.mask_dir:
        # Backward compatibility: use Grounded SAM JSON format
        print(f"\n=== Loading Grounded SAM Masks ===")
        masks_data = load_grounded_sam_masks(args.mask_dir, image_paths)

    if masks_data:
        print(f"\n=== Computing 3D BBoxes ===")
        # Get world points
        if args.use_point_map:
            world_points = predictions["world_points"]
        else:
            world_points = unproject_depth_map_to_point_map(
                predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
            )

        # Initialize tracker
        tracker = ImprovedTracker(
            max_distance=args.max_distance,
            mask_iou_threshold=0.15,
            max_missing_frames=args.max_missing_frames,
            dormant_timeout=args.dormant_timeout,
        )

        # Compute 3D bounding boxes with tracking
        model_size = (predictions["depth"].shape[1], predictions["depth"].shape[2])
        bounding_boxes = compute_instance_bboxes(
            world_points, masks_data, original_images, model_size, tracker,
            gimbal_data=gimbal_data
        )

        # Collect all track IDs
        for frame_bboxes in bounding_boxes:
            for bbox in frame_bboxes:
                if bbox.track_id is not None and bbox.track_id >= 0:
                    if bbox.track_id not in all_track_ids:
                        all_track_ids.append(bbox.track_id)
        all_track_ids = sorted(all_track_ids)

        print(f"\nTotal unique tracks: {len(all_track_ids)}")

    # Save outputs if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        if bounding_boxes and all_track_ids:
            model_size = (predictions["depth"].shape[1], predictions["depth"].shape[2])

            # Save tracking summary
            save_tracking_summary(args.output_dir, bounding_boxes, all_track_ids)

            # Save trajectory plots
            save_trajectory_plots(args.output_dir, bounding_boxes, all_track_ids)

            # Save KITTI labels
            save_kitti_labels(
                args.output_dir, bounding_boxes,
                predictions["extrinsic"], predictions["intrinsic"],
                model_size, frame_names
            )

            # Project bboxes to 2D
            project_bboxes_to_2d(
                bounding_boxes, original_images,
                predictions["extrinsic"], predictions["intrinsic"],
                args.output_dir, frame_names, all_track_ids,
                model_size=model_size
            )

    total_time = time.time() - t0

    # Save metadata
    if args.output_dir:
        metadata = {
            "mode": "video" if is_video_mode else "image_folder",
            "input_path": os.path.abspath(args.video if is_video_mode else args.image_folder),
            "output_path": os.path.abspath(args.output_dir),
            "num_frames": len(image_paths),
            "image_files": [os.path.basename(f) for f in image_paths],
            "frame_indices": frame_indices,
            "video_fps": video_fps,
            "extraction_fps": target_fps,
            "start_frame": getattr(args, 'start_frame', None),
            "end_frame": getattr(args, 'end_frame', None),
            "sam3_masks": args.sam3_masks,
            "mask_dir": args.mask_dir,
            "object_class": args.object_class,
            "settings": {
                "num_images": args.num_images,
                "skip": args.skip,
                "conf_threshold": args.conf_threshold,
                "use_point_map": args.use_point_map,
                "mask_sky": args.mask_sky,
                "max_distance": args.max_distance,
            },
            "dji_log": args.dji_log,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": {
                "model_loading": round(t2 - t1, 2),
                "image_loading": round(t3 - t2, 2),
                "inference": round(t4 - t3, 2),
                "post_processing": round(time.time() - t4, 2),
                "total": round(total_time, 2)
            }
        }
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {os.path.join(args.output_dir, 'metadata.json')}")

    print(f"\n=== Total Processing Time: {total_time:.2f}s ===")

    return {
        "predictions": predictions,
        "bounding_boxes": bounding_boxes,
        "all_track_ids": all_track_ids,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Process based on input mode
    if args.video:
        results = process_video_mode(args, device)
    else:
        results = process_image_folder_mode(args, device)

    # Start visualization
    if not args.no_visualization:
        if not HAS_VISER:
            print("\nWarning: viser not installed. Skipping visualization.")
            print("Install with: pip install viser")
        else:
            print(f"\n=== Starting Viser Visualization ===")
            viser_wrapper_with_tracking(
                results["predictions"],
                results["bounding_boxes"],
                results["all_track_ids"],
                port=args.port,
                init_conf_threshold=args.conf_threshold,
                use_point_map=args.use_point_map,
                background_mode=args.background_mode,
            )
    else:
        print("\nVisualization skipped (--no_visualization flag)")


if __name__ == "__main__":
    main()
