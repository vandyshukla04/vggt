# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch inference script for headless cluster processing.

Runs VGGT inference and optional tracking without visualization,
saving all outputs for later local visualization.

Usage:
    python batch_inference.py \
        --scene_dir ./data/scene \
        --output_dir ./outputs/scene \
        --conf_threshold 50

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
import glob
import time
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.export import save_point_cloud_ply, save_cameras_json, save_depth_maps


def select_frames(image_folder: str, num_images: int, skip: int) -> Tuple[List[str], List[int]]:
    """Select frames based on skip factor and limit."""
    import re

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    image_files = sorted(image_files)

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    print(f"Found {len(image_files)} total images in folder")

    # Apply skip factor
    selected_files = image_files[::skip]

    # Limit to num_images
    if num_images is not None and len(selected_files) > num_images:
        selected_files = selected_files[:num_images]

    # Extract frame numbers for telemetry matching
    def extract_frame_number(filename: str) -> int:
        basename = os.path.splitext(os.path.basename(filename))[0]
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[-1])
        return 0

    frame_numbers = [extract_frame_number(f) for f in selected_files]

    print(f"Selected {len(selected_files)} images (skip={skip})")

    return selected_files, frame_numbers


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
    mask_dir: str,
    dji_log: Optional[str] = None,
    use_gps_refinement: bool = False,
    max_distance: float = 8.0,
    max_missing_frames: int = 20,
    dormant_timeout: int = 100,
    use_point_map: bool = False,
) -> None:
    """Run tracking pipeline and save results."""
    # Import tracking components from demo_viser_tracking
    from demo_viser_tracking import (
        load_grounded_sam_masks,
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
    frame_numbers = [extract_frame_number(p) for p in image_paths]
    gimbal_data = None
    if dji_log:
        print(f"\n=== Parsing DJI Logs ===")
        if use_gps_refinement:
            gimbal_data = parse_dji_logs_with_gps(dji_log, frame_numbers)
        else:
            gimbal_data = parse_dji_logs(dji_log, frame_numbers)

    # Load masks
    print(f"\n=== Loading Masks ===")
    masks_data = load_grounded_sam_masks(mask_dir, image_paths)

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
        description="Batch inference for headless cluster processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference (no tracking)
    python batch_inference.py --scene_dir ./data/scene --output_dir ./outputs/scene

    # With tracking (requires pre-computed masks)
    python batch_inference.py --scene_dir ./data/scene --output_dir ./outputs/scene --mask_dir ./data/scene/masks

    # With DJI gimbal data
    python batch_inference.py --scene_dir ./data/scene --output_dir ./outputs/scene --mask_dir ./data/scene/masks --dji_log ./data/scene/metadata.srt
        """
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene directory containing 'images/' subfolder")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Path to pre-computed Grounded SAM masks (JSON). If not provided, tracking is skipped.")
    parser.add_argument("--dji_log", type=str, default=None,
                        help="Path to DJI SRT file for gimbal/GPS data")
    parser.add_argument("--use_gps_refinement", action="store_true",
                        help="Enable GPS-based pose refinement")
    parser.add_argument("--num_images", type=int, default=20,
                        help="Max number of images to process (VGGT limit ~20)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Skip factor for frame subsampling")
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine image folder
    image_folder = os.path.join(args.scene_dir, "images")
    if not os.path.exists(image_folder):
        # Try scene_dir directly
        image_folder = args.scene_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t0 = time.time()

    # Select frames
    print(f"\n=== Selecting Frames ===")
    image_paths, frame_numbers = select_frames(image_folder, args.num_images, args.skip)

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

    # Run tracking if mask_dir provided
    if args.mask_dir:
        run_tracking(
            args.output_dir,
            predictions,
            image_paths,
            args.mask_dir,
            dji_log=args.dji_log,
            use_gps_refinement=args.use_gps_refinement,
            max_distance=args.max_distance,
            max_missing_frames=args.max_missing_frames,
            dormant_timeout=args.dormant_timeout,
            use_point_map=args.use_point_map,
        )

    total_time = time.time() - t0
    timing["total"] = round(total_time, 2)

    # Save metadata
    metadata = {
        "scene_dir": os.path.abspath(args.scene_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "num_frames": len(image_paths),
        "image_files": [os.path.basename(f) for f in image_paths],
        "frame_numbers": frame_numbers,
        "settings": {
            "num_images": args.num_images,
            "skip": args.skip,
            "conf_threshold": args.conf_threshold,
            "use_point_map": args.use_point_map,
            "mask_dir": args.mask_dir,
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
    if args.mask_dir:
        print("  - tracking_summary.json")
        print("  - annotated_2d/")
        print("  - track_trajectories_*.png")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
