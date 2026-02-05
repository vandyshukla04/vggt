# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch inference script for headless cluster processing - Video Input Version

This version works directly with video files instead of pre-extracted frames:
- Extracts frames on-the-fly at specified FPS
- Supports SAM3 segmentation output format (binary PNG masks)
- Automatically cleans up temporary frames after processing

Usage:
    python batch_inference_video.py \
        --video ./data/my_video.mp4 \
        --sam3_masks ./data/sam3_output/ \
        --extract_fps 5 \
        --output_dir ./outputs/

Outputs saved:
    - predictions.pt: Full model predictions (torch tensors)
    - point_cloud.ply: Colored point cloud (PLY format)
    - cameras.json: Camera extrinsics and intrinsics
    - depth_maps.npz: Compressed depth maps
    - tracking_summary.json: Track data (if masks provided)
    - annotated_2d/: Frames with projected bboxes (if masks provided)
    - trajectory_plots/: 2D and 3D trajectory plots (if masks provided)
    - metadata.json: Run configuration and timing
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.export import save_point_cloud_ply, save_cameras_json, save_depth_maps
from vggt.utils.video_utils import VideoFrameContext, get_video_info
from vggt.utils.sam3_mask_loader import load_sam3_masks, load_sam3_masks_from_image_paths

# Import from batch_inference for backward compatibility
from batch_inference import select_frames


def run_inference(
    image_paths: List[str],
    device: str = "cuda",
) -> dict:
    """Run VGGT inference on images.

    Returns:
        Dictionary with all model predictions
    """
    print("\n=== Loading VGGT Model ===")
    t1 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"Model loading: {time.time() - t1:.2f}s")

    print("\n=== Loading Images ===")
    t2 = time.time()
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    print(f"Image loading: {time.time() - t2:.2f}s")

    print("\n=== Running VGGT Inference ===")
    t3 = time.time()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    torch.cuda.synchronize()
    print(f"Inference: {time.time() - t3:.2f}s")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("\n=== Processing Outputs ===")
    t4 = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images.cpu()

    print(f"Post-processing: {time.time() - t4:.2f}s")

    return predictions, {
        "model_loading": round(t2 - t1, 2),
        "image_loading": round(t3 - t2, 2),
        "inference": round(t4 - t3, 2),
        "post_processing": round(time.time() - t4, 2),
    }


def save_predictions(
    output_dir: str,
    predictions: dict,
    image_paths: List[str],
    conf_threshold: float = 50.0,
    use_point_map: bool = False,
) -> None:
    """Save all predictions to disk."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Saving Predictions ===")

    # Convert tensors to numpy/cpu for saving
    predictions_cpu = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions_cpu[key] = value.cpu()
        else:
            predictions_cpu[key] = value

    # Save full predictions as torch file
    predictions_path = os.path.join(output_dir, "predictions.pt")
    torch.save(predictions_cpu, predictions_path)
    print(f"Saved predictions to {predictions_path}")

    # Helper to safely remove batch dimension
    def maybe_squeeze_batch(arr):
        """Remove batch dimension if it exists and equals 1."""
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr

    # Extract numpy arrays (handle both batched and unbatched tensors)
    images = maybe_squeeze_batch(predictions_cpu["images"].numpy())  # (S, 3, H, W)
    extrinsics = maybe_squeeze_batch(predictions_cpu["extrinsic"].numpy())  # (S, 3, 4)
    intrinsics = maybe_squeeze_batch(predictions_cpu["intrinsic"].numpy())  # (S, 3, 3)
    depth = maybe_squeeze_batch(predictions_cpu["depth"].numpy())  # (S, H, W, 1)
    depth_conf = maybe_squeeze_batch(predictions_cpu["depth_conf"].numpy())  # (S, H, W)

    if use_point_map:
        world_points = maybe_squeeze_batch(predictions_cpu["world_points"].numpy())
        conf = maybe_squeeze_batch(predictions_cpu["world_points_conf"].numpy())
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = depth_conf

    # Save point cloud as PLY
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    num_points = save_point_cloud_ply(
        ply_path,
        world_points,
        colors,
        conf,
        conf_threshold_percentile=conf_threshold,
    )
    print(f"Saved point cloud ({num_points:,} points) to {ply_path}")

    # Save cameras as JSON
    image_names = [os.path.basename(p) for p in image_paths]
    cameras_path = os.path.join(output_dir, "cameras.json")
    save_cameras_json(
        cameras_path,
        extrinsics,
        intrinsics,
        image_names=image_names,
        image_size=(depth.shape[1], depth.shape[2]),
    )
    print(f"Saved cameras to {cameras_path}")

    # Save depth maps
    depth_path = os.path.join(output_dir, "depth_maps.npz")
    save_depth_maps(depth_path, depth, depth_conf)
    print(f"Saved depth maps to {depth_path}")


def run_tracking(
    output_dir: str,
    predictions: dict,
    image_paths: List[str],
    masks_data: dict,
    dji_log: Optional[str] = None,
    use_gps_refinement: bool = False,
    max_distance: float = 8.0,
    max_missing_frames: int = 20,
    dormant_timeout: int = 100,
    use_point_map: bool = False,
    frame_indices: Optional[List[int]] = None,
) -> None:
    """Run tracking pipeline and save results."""
    # Import tracking components from demo_viser_tracking
    from demo_viser_tracking import (
        compute_instance_bboxes,
        ImprovedTracker,
        save_tracking_summary,
        save_trajectory_plots,
        save_kitti_labels,
        project_bboxes_to_2d,
        parse_dji_logs,
        parse_dji_logs_with_gps,
        extract_frame_number,
    )
    import cv2

    print("\n=== Running Tracking Pipeline ===")

    def maybe_squeeze_batch(arr):
        """Remove batch dimension if it exists and equals 1."""
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr.squeeze(0)
        return arr

    # Convert predictions to numpy
    if isinstance(predictions["extrinsic"], torch.Tensor):
        extrinsics = maybe_squeeze_batch(predictions["extrinsic"].cpu().numpy())
        intrinsics = maybe_squeeze_batch(predictions["intrinsic"].cpu().numpy())
        depth = maybe_squeeze_batch(predictions["depth"].cpu().numpy())
        depth_conf = maybe_squeeze_batch(predictions["depth_conf"].cpu().numpy())
        images = maybe_squeeze_batch(predictions["images"].cpu().numpy())
    else:
        extrinsics = maybe_squeeze_batch(predictions["extrinsic"])
        intrinsics = maybe_squeeze_batch(predictions["intrinsic"])
        depth = maybe_squeeze_batch(predictions["depth"])
        depth_conf = maybe_squeeze_batch(predictions["depth_conf"])
        images = maybe_squeeze_batch(predictions["images"])

    # Get world points
    if use_point_map and "world_points" in predictions:
        if isinstance(predictions["world_points"], torch.Tensor):
            world_points = maybe_squeeze_batch(predictions["world_points"].cpu().numpy())
        else:
            world_points = maybe_squeeze_batch(predictions["world_points"])
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)

    # Parse DJI logs if provided
    if frame_indices is None:
        frame_indices = [extract_frame_number(p) for p in image_paths]

    gimbal_data = None
    if dji_log:
        print(f"\n=== Parsing DJI Logs ===")
        if use_gps_refinement:
            gimbal_data = parse_dji_logs_with_gps(dji_log, frame_indices)
        else:
            gimbal_data = parse_dji_logs(dji_log, frame_indices)

    # Load original images for 2D projection
    original_images = [cv2.imread(p) for p in image_paths]

    # Initialize tracker
    tracker = ImprovedTracker(
        max_distance=max_distance,
        mask_iou_threshold=0.15,
        max_missing_frames=max_missing_frames,
        dormant_timeout=dormant_timeout,
    )

    # Compute 3D bounding boxes with tracking
    model_size = (depth.shape[1], depth.shape[2])
    bounding_boxes = compute_instance_bboxes(
        world_points, masks_data, original_images, model_size, tracker,
        gimbal_data=gimbal_data
    )

    # Collect all track IDs
    all_track_ids = []
    for frame_bboxes in bounding_boxes:
        for bbox in frame_bboxes:
            if bbox.track_id is not None and bbox.track_id >= 0:
                if bbox.track_id not in all_track_ids:
                    all_track_ids.append(bbox.track_id)
    all_track_ids = sorted(all_track_ids)

    print(f"\nTotal unique tracks: {len(all_track_ids)}")

    # Save tracking outputs
    if bounding_boxes and all_track_ids:
        frame_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

        # Save extended tracking summary (with dimensions, rotations, velocities, 2D boxes)
        save_tracking_summary(
            output_dir, bounding_boxes, all_track_ids,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_size=model_size
        )

        # Save KITTI format labels
        save_kitti_labels(
            output_dir, bounding_boxes,
            extrinsics, intrinsics,
            model_size, frame_names
        )

        # Save trajectory plots
        save_trajectory_plots(output_dir, bounding_boxes, all_track_ids)

        # Project bboxes to 2D
        project_bboxes_to_2d(
            bounding_boxes, original_images,
            extrinsics, intrinsics,
            output_dir, frame_names, all_track_ids,
            model_size=model_size
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for headless cluster processing with video input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference from video (no tracking)
    python batch_inference_video.py --video ./data/video.mp4 --output_dir ./outputs/

    # With SAM3 masks for tracking
    python batch_inference_video.py \\
        --video ./data/video.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # With specific frame range (match SAM3 mask frame numbers)
    python batch_inference_video.py \\
        --video ./data/video.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --start_frame 100 --end_frame 500 \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # With DJI telemetry
    python batch_inference_video.py \\
        --video ./data/drone.mp4 \\
        --sam3_masks ./data/sam3_output/ \\
        --dji_log ./data/metadata.srt \\
        --extract_fps 5 \\
        --output_dir ./outputs/

    # Backward compatible: use pre-extracted frames
    python batch_inference_video.py \\
        --scene_dir ./data/scene \\
        --mask_dir ./data/scene/masks \\
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
                        help="Default class name for SAM3 objects")
    parser.add_argument("--sam3_class", type=str, default=None,
                        help="Class name(s) to load from multi-class SAM3 output. "
                             "Comma-separated for multiple classes (e.g., 'zebra,lion'). "
                             "If not specified, loads all available classes.")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Don't delete extracted frames after processing")

    # Backward compatible interface
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Path to scene directory containing 'images/' subfolder (backward compatibility)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Path to pre-computed Grounded SAM masks (JSON) (backward compatibility)")

    # Common arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--dji_log", type=str, default=None,
                        help="Path to DJI SRT file for gimbal/GPS data")
    parser.add_argument("--use_gps_refinement", action="store_true",
                        help="Enable GPS-based pose refinement")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Max number of images to process (default: all images)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Skip factor for frame subsampling (only for scene_dir mode)")
    parser.add_argument("--conf_threshold", type=float, default=50.0,
                        help="Confidence threshold percentile for point cloud filtering")
    parser.add_argument("--max_distance", type=float, default=8.0,
                        help="Max 3D distance for tracking")
    parser.add_argument("--max_missing_frames", type=int, default=20,
                        help="Frames before track goes dormant")
    parser.add_argument("--dormant_timeout", type=int, default=100,
                        help="Frames before dormant track removed")
    parser.add_argument("--use_point_map", action="store_true",
                        help="Use point map instead of depth-based points")

    args = parser.parse_args()

    # Validate inputs
    if args.video is None and args.scene_dir is None:
        parser.error("Either --video or --scene_dir must be provided")

    if args.video and args.scene_dir:
        print("Warning: Both --video and --scene_dir provided. Using --video.")
        args.scene_dir = None

    return args


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t0 = time.time()

    # Process based on input mode
    if args.video:
        # Video mode
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
            frame_indices = ctx.frame_indices  # Original video frame numbers (for DJI telemetry matching)

            print(f"Extracted {len(image_paths)} frames")
            print(f"Video frame indices: {frame_indices[:5]}..." if len(frame_indices) > 5 else f"Video frame indices: {frame_indices}")

            # Run inference
            predictions, timing = run_inference(image_paths, device)

            # Save predictions
            save_predictions(
                args.output_dir,
                predictions,
                image_paths,
                conf_threshold=args.conf_threshold,
                use_point_map=args.use_point_map,
            )

            # Run tracking if masks provided
            masks_data = None
            if args.sam3_masks:
                print(f"\n=== Loading SAM3 Masks ===")

                # Parse class names if provided
                class_names = None
                if args.sam3_class:
                    class_names = [c.strip() for c in args.sam3_class.split(',')]

                # Use direct frame matching when user specifies start_frame/end_frame
                use_direct_match = (args.start_frame is not None or args.end_frame is not None)

                # Check format and load appropriately
                from vggt.utils.sam3_mask_loader import detect_sam3_format, load_sam3_masks_multi_class
                format_info = detect_sam3_format(args.sam3_masks)

                if format_info['format'] == 'multi_class':
                    # Multi-class format
                    if class_names is None:
                        print(f"Auto-loading all {len(format_info['classes'])} class(es)")

                    masks_data = load_sam3_masks_multi_class(
                        sam3_output_dir=args.sam3_masks,
                        extracted_frame_indices=frame_indices,
                        video_fps=video_info['fps'],
                        sam3_fps=args.sam3_fps or target_fps,
                        class_names=class_names,
                        direct_frame_match=use_direct_match
                    )
                else:
                    # Single-class format (backward compatibility)
                    masks_data = load_sam3_masks(
                        sam3_output_dir=args.sam3_masks,
                        extracted_frame_indices=frame_indices,
                        video_fps=video_info['fps'],
                        sam3_fps=args.sam3_fps or target_fps,
                        class_name=args.object_class,
                        direct_frame_match=use_direct_match,
                        auto_detect_format=False  # Already detected
                    )

            if masks_data:
                run_tracking(
                    args.output_dir,
                    predictions,
                    image_paths,
                    masks_data,
                    dji_log=args.dji_log,
                    use_gps_refinement=args.use_gps_refinement,
                    max_distance=args.max_distance,
                    max_missing_frames=args.max_missing_frames,
                    dormant_timeout=args.dormant_timeout,
                    use_point_map=args.use_point_map,
                    frame_indices=frame_indices,
                )

            is_video_mode = True
            video_fps = video_info['fps']

    else:
        # Scene directory mode (backward compatible)
        print(f"\n=== Scene Directory Mode (Backward Compatible) ===")

        # Determine image folder
        image_folder = os.path.join(args.scene_dir, "images")
        if not os.path.exists(image_folder):
            # Try scene_dir directly
            image_folder = args.scene_dir

        # Select frames
        print(f"\n=== Selecting Frames ===")
        image_paths, frame_indices = select_frames(image_folder, args.num_images, args.skip)

        # Run inference
        predictions, timing = run_inference(image_paths, device)

        # Save predictions
        save_predictions(
            args.output_dir,
            predictions,
            image_paths,
            conf_threshold=args.conf_threshold,
            use_point_map=args.use_point_map,
        )

        # Load masks
        masks_data = None
        if args.sam3_masks:
            # SAM3 masks with image folder
            print(f"\n=== Loading SAM3 Masks (filename matching) ===")
            masks_data = load_sam3_masks_from_image_paths(
                sam3_output_dir=args.sam3_masks,
                image_paths=image_paths,
                class_name=args.object_class
            )
        elif args.mask_dir:
            # Backward compatibility: Grounded SAM JSON format
            from demo_viser_tracking import load_grounded_sam_masks
            print(f"\n=== Loading Grounded SAM Masks ===")
            masks_data = load_grounded_sam_masks(args.mask_dir, image_paths)

        # Run tracking if masks provided
        if masks_data:
            run_tracking(
                args.output_dir,
                predictions,
                image_paths,
                masks_data,
                dji_log=args.dji_log,
                use_gps_refinement=args.use_gps_refinement,
                max_distance=args.max_distance,
                max_missing_frames=args.max_missing_frames,
                dormant_timeout=args.dormant_timeout,
                use_point_map=args.use_point_map,
                frame_indices=frame_indices,
            )

        is_video_mode = False
        video_fps = None
        target_fps = None

    total_time = time.time() - t0
    timing["total"] = round(total_time, 2)

    # Save metadata
    metadata = {
        "mode": "video" if is_video_mode else "scene_dir",
        "input_path": os.path.abspath(args.video if is_video_mode else args.scene_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "num_frames": len(image_paths),
        "image_files": [os.path.basename(f) for f in image_paths],
        "frame_indices": frame_indices,
        "video_fps": video_fps if is_video_mode else None,
        "extraction_fps": target_fps if is_video_mode else None,
        "start_frame": getattr(args, 'start_frame', None),
        "end_frame": getattr(args, 'end_frame', None),
        "settings": {
            "num_images": args.num_images,
            "skip": args.skip,
            "conf_threshold": args.conf_threshold,
            "use_point_map": args.use_point_map,
            "sam3_masks": args.sam3_masks,
            "mask_dir": args.mask_dir,
            "object_class": args.object_class,
            "dji_log": args.dji_log,
            "max_distance": args.max_distance,
            "max_missing_frames": args.max_missing_frames,
            "dormant_timeout": args.dormant_timeout,
        },
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": timing,
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    print(f"\n=== Total Processing Time: {total_time:.2f}s ===")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("  - predictions.pt")
    print("  - point_cloud.ply")
    print("  - cameras.json")
    print("  - depth_maps.npz")
    if masks_data:
        print("  - tracking_summary.json")
        print("  - kitti_labels/")
        print("  - annotated_2d/")
        print("  - track_trajectories_*.png")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
