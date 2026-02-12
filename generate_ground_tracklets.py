#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate ground-only 3D reconstruction with smoothed tracklets.

Post-processing script that takes batch_inference_video.py outputs and:
1. Masks out animal/object point clouds to isolate the ground surface
2. Smooths tracklet trajectories using Savitzky-Golay filtering
3. Saves a combined PLY with ground points + colored tracklet points

Designed to run headlessly on a cluster (no GPU or display needed).
View results later with visualize_ground_tracklets.py.

Usage:
    # With SAM3 masks (video mode)
    python generate_ground_tracklets.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/sam3_output/ \
        --video ./data/video.mp4

    # With Grounded SAM masks (image folder mode)
    python generate_ground_tracklets.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/grounded_sam_masks/ \
        --mask_format grounded_sam

    # With custom smoothing parameters
    python generate_ground_tracklets.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/sam3_output/ \
        --savgol_window 15 --savgol_polyorder 3
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.export import save_point_cloud_ply
from demo_viser_tracking import (
    transform_mask_to_model_coordinates,
    load_grounded_sam_masks,
    COLOR_PALETTE,
)

# =============================================================================
# Mask Loading Helpers
# =============================================================================


def load_masks(
    mask_source: str,
    mask_format: str,
    metadata: dict,
    video: Optional[str] = None,
    sam3_class: Optional[str] = None,
    sam3_fps: Optional[float] = None,
) -> Dict[int, List[Dict]]:
    """Load segmentation masks from SAM3 or Grounded SAM format."""

    if mask_format == "grounded_sam":
        # Build image paths from metadata
        image_files = metadata.get("image_files", [])
        input_path = metadata.get("input_path", "")

        # Try to reconstruct image paths
        image_paths = []
        for fname in image_files:
            # Try common locations
            candidates = [
                os.path.join(input_path, fname),
                os.path.join(input_path, "images", fname),
                os.path.join(os.path.dirname(input_path), "images", fname),
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    image_paths.append(c)
                    found = True
                    break
            if not found:
                # Use a dummy path - mask loading uses basename matching
                image_paths.append(fname)

        return load_grounded_sam_masks(mask_source, image_paths)

    # SAM3 format
    from vggt.utils.sam3_mask_loader import (
        detect_sam3_format,
        load_sam3_masks,
        load_sam3_masks_multi_class,
    )

    frame_indices = metadata.get("frame_indices", list(range(metadata.get("num_frames", 0))))

    # Determine FPS values
    video_fps = metadata.get("video_fps", 30.0)
    if sam3_fps is None:
        sam3_fps = metadata.get("extraction_fps", video_fps)

    # If video is provided, get its native FPS
    if video:
        from vggt.utils.video_utils import get_video_info
        video_info = get_video_info(video)
        video_fps = video_info["fps"]

    # Check if start_frame/end_frame were used (direct frame matching)
    direct_frame_match = (
        metadata.get("start_frame") is not None
        or metadata.get("end_frame") is not None
    )

    format_info = detect_sam3_format(mask_source)

    if format_info["format"] == "multi_class":
        class_names = None
        if sam3_class:
            class_names = [c.strip() for c in sam3_class.split(",")]

        return load_sam3_masks_multi_class(
            sam3_output_dir=mask_source,
            extracted_frame_indices=frame_indices,
            video_fps=video_fps,
            sam3_fps=sam3_fps,
            class_names=class_names,
            direct_frame_match=direct_frame_match,
        )
    else:
        return load_sam3_masks(
            sam3_output_dir=mask_source,
            extracted_frame_indices=frame_indices,
            video_fps=video_fps,
            sam3_fps=sam3_fps,
            class_name=sam3_class or "object",
            direct_frame_match=direct_frame_match,
            auto_detect_format=False,
        )


# =============================================================================
# Ground Point Cloud Extraction
# =============================================================================


def build_ground_point_cloud(
    world_points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    masks_data: Dict[int, List[Dict]],
    original_images: Optional[List[np.ndarray]],
    model_size: Tuple[int, int],
    conf_threshold_percentile: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ground-only point cloud by masking out detected objects.

    Args:
        world_points: 3D points (S, H, W, 3)
        colors: RGB colors (S, H, W, 3) in [0, 1]
        conf: Confidence maps (S, H, W)
        masks_data: Dict mapping frame index to list of mask dicts
        original_images: List of original images for shape reference
        model_size: (H, W) model output size
        conf_threshold_percentile: Confidence threshold for filtering

    Returns:
        Tuple of (ground_points (N, 3), ground_colors (N, 3) in [0, 255] uint8)
    """
    S, H, W, _ = world_points.shape
    conf_flat = conf.reshape(-1)
    conf_threshold_value = np.percentile(conf_flat[conf_flat > 1e-5], conf_threshold_percentile)

    ground_points_list = []
    ground_colors_list = []

    for frame_idx in range(S):
        # Union all object masks for this frame
        animal_mask = np.zeros((H, W), dtype=bool)

        if frame_idx in masks_data:
            for mask_info in masks_data[frame_idx]:
                mask = mask_info["mask"]
                class_name = mask_info.get("class_name", "object").lower()

                # Skip background-like classes (keep them as ground)
                if class_name in {"ground", "sky", "background", "terrain", "floor"}:
                    continue

                # Determine original image shape for coordinate transform
                if original_images and frame_idx < len(original_images):
                    orig_img = original_images[frame_idx]
                    if orig_img.ndim == 3 and orig_img.shape[0] == 3:
                        orig_h, orig_w = orig_img.shape[1], orig_img.shape[2]
                    else:
                        orig_h, orig_w = orig_img.shape[:2]
                else:
                    orig_h, orig_w = mask.shape[:2]

                mask_transformed = transform_mask_to_model_coordinates(
                    mask, (orig_h, orig_w), (H, W)
                )
                animal_mask |= mask_transformed

        # Ground = everything NOT an animal, filtered by confidence
        ground_mask = ~animal_mask
        conf_mask = conf[frame_idx] >= conf_threshold_value
        valid_mask = ~np.any(
            np.isnan(world_points[frame_idx]) | np.isinf(world_points[frame_idx]),
            axis=-1,
        )
        combined_mask = ground_mask & conf_mask & valid_mask

        ground_pts = world_points[frame_idx][combined_mask]
        ground_clrs = colors[frame_idx][combined_mask]

        ground_points_list.append(ground_pts)
        ground_colors_list.append(ground_clrs)

        n_animal = int(animal_mask.sum())
        n_ground = int(combined_mask.sum())
        print(f"  Frame {frame_idx}: {n_animal:,} animal pixels masked, {n_ground:,} ground points kept")

    all_ground_points = np.concatenate(ground_points_list, axis=0)
    all_ground_colors = np.concatenate(ground_colors_list, axis=0)

    # Convert colors to uint8
    if all_ground_colors.max() <= 1.0:
        all_ground_colors = (all_ground_colors * 255).astype(np.uint8)
    else:
        all_ground_colors = all_ground_colors.astype(np.uint8)

    return all_ground_points, all_ground_colors


# =============================================================================
# Tracklet Smoothing
# =============================================================================


def smooth_tracklets(
    tracking_data: Dict,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
) -> Dict:
    """Smooth tracklet trajectories using Savitzky-Golay filtering.

    Args:
        tracking_data: Loaded tracking_summary.json dict
        savgol_window: Window length for Savitzky-Golay filter (must be odd)
        savgol_polyorder: Polynomial order (must be < window)

    Returns:
        Dict mapping track_id to smoothed track info
    """
    smoothed_tracks = {}

    for track_id, track_data in tracking_data.get("tracks", {}).items():
        centers = np.array(track_data["centers"])  # (N, 3)
        frames = track_data.get("frames", list(range(len(centers))))
        class_name = track_data.get("class_name", "object")

        if len(centers) < 2:
            print(f"  Track {track_id} ({class_name}): {len(centers)} points - too short, keeping raw")
            smoothed_tracks[track_id] = {
                "class_name": class_name,
                "frames": frames,
                "raw_centers": centers.tolist(),
                "smoothed_centers": centers.tolist(),
                "savgol_window": 0,
                "savgol_polyorder": 0,
            }
            continue

        # Determine effective window size
        effective_window = savgol_window
        if len(centers) < savgol_window:
            effective_window = len(centers)
            if effective_window % 2 == 0:
                effective_window -= 1

        if effective_window < savgol_polyorder + 2:
            # Too short for Savitzky-Golay, keep raw
            print(f"  Track {track_id} ({class_name}): {len(centers)} points - too short for filtering, keeping raw")
            smoothed_tracks[track_id] = {
                "class_name": class_name,
                "frames": frames,
                "raw_centers": centers.tolist(),
                "smoothed_centers": centers.tolist(),
                "savgol_window": 0,
                "savgol_polyorder": 0,
            }
            continue

        effective_polyorder = min(savgol_polyorder, effective_window - 1)

        # Apply Savitzky-Golay independently to x, y, z
        smoothed_x = savgol_filter(centers[:, 0], effective_window, effective_polyorder)
        smoothed_y = savgol_filter(centers[:, 1], effective_window, effective_polyorder)
        smoothed_z = savgol_filter(centers[:, 2], effective_window, effective_polyorder)
        smoothed_centers = np.column_stack([smoothed_x, smoothed_y, smoothed_z])

        # Compute smoothing error
        error = np.mean(np.linalg.norm(centers - smoothed_centers, axis=1))

        print(
            f"  Track {track_id} ({class_name}): {len(centers)} points, "
            f"window={effective_window}, polyorder={effective_polyorder}, "
            f"avg_displacement={error:.4f}"
        )

        smoothed_tracks[track_id] = {
            "class_name": class_name,
            "frames": frames,
            "raw_centers": centers.tolist(),
            "smoothed_centers": smoothed_centers.tolist(),
            "savgol_window": effective_window,
            "savgol_polyorder": effective_polyorder,
        }

    return smoothed_tracks


# =============================================================================
# Tracklet Point Sampling
# =============================================================================


def sample_tracklet_points(
    smoothed_tracks: Dict,
    tracklet_density: int = 100,
    tracklet_radius: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample dense colored points along smoothed tracklets.

    For each tracklet, interpolates a smooth curve and samples dense points
    along it with the track's assigned color.

    Args:
        smoothed_tracks: Dict from smooth_tracklets()
        tracklet_density: Number of points per unit length of trajectory
        tracklet_radius: Radius for tube-like point sampling around centerline

    Returns:
        Tuple of (tracklet_points (N, 3), tracklet_colors (N, 3) uint8)
    """
    all_points = []
    all_colors = []

    sorted_track_ids = sorted(smoothed_tracks.keys(), key=lambda x: int(x))

    for idx, track_id in enumerate(sorted_track_ids):
        track = smoothed_tracks[track_id]
        centers = np.array(track["smoothed_centers"])

        if len(centers) < 2:
            continue

        # Compute arc-length parameterization
        diffs = np.diff(centers, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)

        if total_length < 1e-6:
            continue

        # Cumulative arc length
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        t = arc_lengths / total_length  # Normalized [0, 1]

        # Create cubic spline interpolation
        try:
            cs = CubicSpline(t, centers)
        except ValueError:
            # Fallback for degenerate cases
            continue

        # Sample dense points along the spline
        num_samples = max(int(total_length * tracklet_density), 50)
        t_dense = np.linspace(0, 1, num_samples)
        dense_points = cs(t_dense)

        # Add tube-like points around the centerline for visibility
        # Sample a few rings of points perpendicular to the tangent
        tangents = cs(t_dense, 1)  # First derivative
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms[tangent_norms < 1e-8] = 1.0
        tangents = tangents / tangent_norms

        # Create perpendicular vectors using cross product with an arbitrary vector
        up = np.array([0, 1, 0])
        tube_points = [dense_points]  # centerline

        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            # For each tangent, compute a perpendicular offset
            perp1 = np.cross(tangents, up)
            perp1_norms = np.linalg.norm(perp1, axis=1, keepdims=True)
            small_norm = perp1_norms.squeeze() < 1e-6
            if np.any(small_norm):
                alt_up = np.array([1, 0, 0])
                perp1[small_norm] = np.cross(tangents[small_norm], alt_up)
                perp1_norms[small_norm] = np.linalg.norm(
                    perp1[small_norm], axis=1, keepdims=True
                )
            perp1_norms[perp1_norms < 1e-8] = 1.0
            perp1 = perp1 / perp1_norms

            perp2 = np.cross(tangents, perp1)
            perp2_norms = np.linalg.norm(perp2, axis=1, keepdims=True)
            perp2_norms[perp2_norms < 1e-8] = 1.0
            perp2 = perp2 / perp2_norms

            offset = tracklet_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            tube_points.append(dense_points + offset)

        track_points = np.concatenate(tube_points, axis=0)

        # Assign track color
        color_rgb = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        color_uint8 = np.array([int(c * 255) for c in color_rgb], dtype=np.uint8)
        track_colors = np.tile(color_uint8, (len(track_points), 1))

        all_points.append(track_points)
        all_colors.append(track_colors)

        print(
            f"  Track {track_id} ({track['class_name']}): "
            f"{len(track_points):,} points sampled along {total_length:.2f}m trajectory"
        )

    if not all_points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)


# =============================================================================
# Top-Down Image Export
# =============================================================================


def save_topdown_image(
    ground_points: np.ndarray,
    ground_colors: np.ndarray,
    smoothed_tracks: Dict,
    output_dir: str,
    dpi: int = 300,
    subsample_ratio: float = 0.3,
) -> str:
    """Render a top-down (bird's eye) image of the ground with tracklet overlays.

    Projects onto the X-Z plane (Y is height/up). Produces a rasterized PNG
    that doesn't suffer from the point-size artifacts seen in viser.

    Args:
        ground_points: (N, 3) ground point coordinates
        ground_colors: (N, 3) uint8 RGB colors
        smoothed_tracks: Dict from smooth_tracklets()
        output_dir: Where to save the image
        dpi: Output resolution (default: 300)
        subsample_ratio: Fraction of ground points to render (default: 0.3)

    Returns:
        Path to the saved image
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    # Subsample ground points for rendering speed
    n = len(ground_points)
    if subsample_ratio < 1.0 and n > 0:
        k = max(1, int(n * subsample_ratio))
        idx = np.random.choice(n, k, replace=False)
        pts = ground_points[idx]
        clrs = ground_colors[idx] / 255.0
    else:
        pts = ground_points
        clrs = ground_colors / 255.0

    # Scatter ground points on X-Z plane
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0], pts[:, 2],
            c=clrs, s=0.1, alpha=0.4, marker=".", rasterized=True,
        )

    # Overlay smoothed tracklets
    sorted_ids = sorted(smoothed_tracks.keys(), key=lambda x: int(x))
    for i, tid in enumerate(sorted_ids):
        track = smoothed_tracks[tid]
        centers = np.array(track["smoothed_centers"])
        if len(centers) < 2:
            continue
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        class_name = track.get("class_name", "object")
        ax.plot(
            centers[:, 0], centers[:, 2],
            color=color, linewidth=3, label=f"T{tid}: {class_name}",
        )
        # Start marker
        ax.scatter(
            centers[0, 0], centers[0, 2],
            color=color, s=80, marker="o", zorder=5,
            edgecolors="white", linewidths=0.5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Ground Reconstruction - Top-Down View")
    if sorted_ids:
        ax.legend(loc="upper right", fontsize=8)

    out_path = os.path.join(output_dir, "ground_topdown.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-only 3D reconstruction with smoothed tracklets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With SAM3 masks
    python generate_ground_tracklets.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/sam3_output/

    # With Grounded SAM masks
    python generate_ground_tracklets.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/masks/ \\
        --mask_format grounded_sam

    # With video for frame index matching
    python generate_ground_tracklets.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/sam3_output/ \\
        --video ./data/video.mp4

    # Custom smoothing
    python generate_ground_tracklets.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/sam3_output/ \\
        --savgol_window 15 --savgol_polyorder 3
        """,
    )

    parser.add_argument(
        "--result_dir", type=str, required=True,
        help="Directory containing batch inference outputs (predictions.pt, tracking_summary.json)",
    )
    parser.add_argument(
        "--mask_source", type=str, required=True,
        help="Path to segmentation masks (SAM3 directory or Grounded SAM JSON directory)",
    )
    parser.add_argument(
        "--mask_format", type=str, default="auto",
        choices=["auto", "sam3", "grounded_sam"],
        help="Mask format (default: auto-detect)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video (for SAM3 frame-index matching)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: result_dir/ground_tracklets/)",
    )
    parser.add_argument(
        "--savgol_window", type=int, default=11,
        help="Savitzky-Golay window length, must be odd (default: 11)",
    )
    parser.add_argument(
        "--savgol_polyorder", type=int, default=3,
        help="Savitzky-Golay polynomial order, must be < window (default: 3)",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=50.0,
        help="Confidence percentile for point filtering (default: 50.0)",
    )
    parser.add_argument(
        "--tracklet_density", type=int, default=100,
        help="Points per unit length for tracklet sampling (default: 100)",
    )
    parser.add_argument(
        "--tracklet_radius", type=float, default=0.02,
        help="Radius of tracklet tube in 3D units (default: 0.02)",
    )
    parser.add_argument(
        "--sam3_class", type=str, default=None,
        help="Class name(s) for multi-class SAM3 masks (comma-separated)",
    )
    parser.add_argument(
        "--sam3_fps", type=float, default=None,
        help="FPS used when generating SAM3 masks",
    )
    parser.add_argument(
        "--use_point_map", action="store_true",
        help="Use world_points from model instead of depth-based unprojection",
    )
    parser.add_argument(
        "--no_topdown", action="store_true",
        help="Skip generating the top-down PNG image",
    )
    parser.add_argument(
        "--topdown_dpi", type=int, default=300,
        help="DPI for the top-down image (default: 300)",
    )
    parser.add_argument(
        "--topdown_subsample", type=float, default=0.3,
        help="Fraction of ground points to render in the top-down image (default: 0.3)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    predictions_path = os.path.join(args.result_dir, "predictions.pt")
    if not os.path.exists(predictions_path):
        print(f"Error: predictions.pt not found in {args.result_dir}")
        sys.exit(1)

    tracking_path = os.path.join(args.result_dir, "tracking_summary.json")
    if not os.path.exists(tracking_path):
        print(f"Error: tracking_summary.json not found in {args.result_dir}")
        sys.exit(1)

    if not os.path.exists(args.mask_source):
        print(f"Error: Mask source not found: {args.mask_source}")
        sys.exit(1)

    # Ensure savgol_window is odd
    if args.savgol_window % 2 == 0:
        args.savgol_window += 1
        print(f"Adjusted savgol_window to {args.savgol_window} (must be odd)")

    if args.savgol_polyorder >= args.savgol_window:
        print(f"Error: savgol_polyorder ({args.savgol_polyorder}) must be < savgol_window ({args.savgol_window})")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir or os.path.join(args.result_dir, "ground_tracklets")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Ground Reconstruction with Smoothed Tracklets")
    print(f"{'='*60}")
    print(f"Result dir:  {args.result_dir}")
    print(f"Mask source: {args.mask_source}")
    print(f"Output dir:  {output_dir}")
    print(f"Smoothing:   window={args.savgol_window}, polyorder={args.savgol_polyorder}")

    # =========================================================================
    # Step 1: Load predictions
    # =========================================================================
    print(f"\n=== Step 1: Loading Predictions ===")
    predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)

    def maybe_squeeze_batch(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    images = maybe_squeeze_batch(predictions["images"])  # (S, 3, H, W)
    depth = maybe_squeeze_batch(predictions["depth"])  # (S, H, W, 1)
    depth_conf = maybe_squeeze_batch(predictions["depth_conf"])  # (S, H, W)

    # Get or compute extrinsics/intrinsics
    if "extrinsic" in predictions:
        extrinsics = maybe_squeeze_batch(predictions["extrinsic"])
        intrinsics = maybe_squeeze_batch(predictions["intrinsic"])
    else:
        pose_enc = maybe_squeeze_batch(predictions["pose_enc"])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            torch.from_numpy(pose_enc).unsqueeze(0),
            (images.shape[-2], images.shape[-1]),
        )
        extrinsics = extrinsics.numpy().squeeze(0)
        intrinsics = intrinsics.numpy().squeeze(0)

    # Compute world points
    if args.use_point_map and "world_points" in predictions:
        world_points = maybe_squeeze_batch(predictions["world_points"])
        conf = maybe_squeeze_batch(predictions["world_points_conf"])
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = depth_conf

    # Colors: (S, 3, H, W) -> (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)

    S, H, W, _ = world_points.shape
    model_size = (H, W)
    print(f"Loaded: {S} frames, {H}x{W} resolution")
    print(f"World points shape: {world_points.shape}")

    # =========================================================================
    # Step 2: Load masks
    # =========================================================================
    print(f"\n=== Step 2: Loading Masks ===")

    # Load metadata for frame index info
    metadata_path = os.path.join(args.result_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Auto-detect mask format
    mask_format = args.mask_format
    if mask_format == "auto":
        from vggt.utils.sam3_mask_loader import detect_sam3_format
        format_info = detect_sam3_format(args.mask_source)
        if format_info["format"] in ("single_class", "multi_class"):
            mask_format = "sam3"
            print(f"Auto-detected mask format: SAM3 ({format_info['format']})")
        else:
            # Check for JSON files (Grounded SAM)
            json_files = [f for f in os.listdir(args.mask_source) if f.endswith("_results.json")]
            if json_files:
                mask_format = "grounded_sam"
                print(f"Auto-detected mask format: Grounded SAM ({len(json_files)} JSON files)")
            else:
                print("Warning: Could not auto-detect mask format. Trying SAM3.")
                mask_format = "sam3"

    # Load original images for shape reference
    original_images = None
    image_files = metadata.get("image_files", [])
    input_path = metadata.get("input_path", "")
    if image_files and input_path:
        original_images = []
        for fname in image_files:
            candidates = [
                os.path.join(input_path, fname),
                os.path.join(input_path, "images", fname),
            ]
            loaded = False
            for c in candidates:
                if os.path.exists(c):
                    img = cv2.imread(c)
                    if img is not None:
                        original_images.append(img)
                        loaded = True
                        break
            if not loaded:
                # Use a placeholder with model size
                original_images.append(np.zeros((H, W, 3), dtype=np.uint8))

    masks_data = load_masks(
        mask_source=args.mask_source,
        mask_format=mask_format,
        metadata=metadata,
        video=args.video,
        sam3_class=args.sam3_class,
        sam3_fps=args.sam3_fps,
    )

    total_masks = sum(len(v) for v in masks_data.values())
    print(f"Loaded masks for {len(masks_data)} frames ({total_masks} total instances)")

    # =========================================================================
    # Step 3: Build ground point cloud
    # =========================================================================
    print(f"\n=== Step 3: Building Ground Point Cloud ===")

    ground_points, ground_colors = build_ground_point_cloud(
        world_points=world_points,
        colors=colors,
        conf=conf,
        masks_data=masks_data,
        original_images=original_images,
        model_size=model_size,
        conf_threshold_percentile=args.conf_threshold,
    )

    print(f"\nGround point cloud: {len(ground_points):,} points")

    # Save ground-only PLY
    ground_ply_path = os.path.join(output_dir, "ground_point_cloud.ply")
    n_saved = save_point_cloud_ply(ground_ply_path, ground_points, ground_colors)
    print(f"Saved ground point cloud ({n_saved:,} points) to {ground_ply_path}")

    # =========================================================================
    # Step 4: Smooth tracklets
    # =========================================================================
    print(f"\n=== Step 4: Smoothing Tracklets ===")

    with open(tracking_path, "r") as f:
        tracking_data = json.load(f)

    smoothed_tracks = smooth_tracklets(
        tracking_data,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
    )

    print(f"\nSmoothed {len(smoothed_tracks)} tracklets")

    # Save smoothed tracklets JSON
    smoothed_json = {
        "tracks": smoothed_tracks,
        "total_tracks": len(smoothed_tracks),
        "total_frames": tracking_data.get("total_frames", S),
        "settings": {
            "savgol_window": args.savgol_window,
            "savgol_polyorder": args.savgol_polyorder,
            "conf_threshold": args.conf_threshold,
            "tracklet_density": args.tracklet_density,
            "tracklet_radius": args.tracklet_radius,
        },
    }

    tracklets_json_path = os.path.join(output_dir, "smoothed_tracklets.json")
    with open(tracklets_json_path, "w") as f:
        json.dump(smoothed_json, f, indent=2)
    print(f"Saved smoothed tracklets to {tracklets_json_path}")

    # =========================================================================
    # Step 5: Sample tracklet points and combine
    # =========================================================================
    print(f"\n=== Step 5: Sampling Tracklet Points ===")

    tracklet_points, tracklet_colors = sample_tracklet_points(
        smoothed_tracks,
        tracklet_density=args.tracklet_density,
        tracklet_radius=args.tracklet_radius,
    )

    print(f"\nTracklet points: {len(tracklet_points):,}")

    # =========================================================================
    # Step 6: Combine and save final output
    # =========================================================================
    print(f"\n=== Step 6: Saving Combined Output ===")

    if len(tracklet_points) > 0:
        combined_points = np.concatenate([ground_points, tracklet_points], axis=0)
        combined_colors = np.concatenate([ground_colors, tracklet_colors], axis=0)
    else:
        combined_points = ground_points
        combined_colors = ground_colors

    combined_ply_path = os.path.join(output_dir, "ground_with_tracklets.ply")
    n_combined = save_point_cloud_ply(combined_ply_path, combined_points, combined_colors)

    print(f"\nSaved combined PLY ({n_combined:,} points) to {combined_ply_path}")
    print(f"  Ground points: {len(ground_points):,}")
    print(f"  Tracklet points: {len(tracklet_points):,}")

    # =========================================================================
    # Step 7: Save top-down image
    # =========================================================================
    topdown_path = None
    if not args.no_topdown:
        print(f"\n=== Step 7: Rendering Top-Down Image ===")
        topdown_path = save_topdown_image(
            ground_points=ground_points,
            ground_colors=ground_colors,
            smoothed_tracks=smoothed_tracks,
            output_dir=output_dir,
            dpi=args.topdown_dpi,
            subsample_ratio=args.topdown_subsample,
        )
        print(f"Saved top-down image to {topdown_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("Output Files:")
    print(f"{'='*60}")
    print(f"  {ground_ply_path}")
    print(f"  {combined_ply_path}")
    print(f"  {tracklets_json_path}")
    if topdown_path:
        print(f"  {topdown_path}")
    print(f"\nTo visualize:")
    print(f"  python visualize_ground_tracklets.py --result_dir {output_dir}")


if __name__ == "__main__":
    main()
