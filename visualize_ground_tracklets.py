#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive 3D Visualization of Ground Point Cloud with Smoothed Tracklets

Loads outputs from generate_ground_tracklets.py and displays them
in an interactive viser 3D viewer.

Supports two modes:
  1. PLY mode (default): loads pre-baked ground_point_cloud.ply
  2. Frame mode (--frame N): loads per-frame 3D data from predictions.pt,
     shows cumulative point cloud (frames 0..N), trimmed tracklet splines,
     and animal instances at frame N highlighted with tracklet colours.

Usage:
    # PLY mode (original)
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/

    # Frame mode with instance highlighting
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ \\
        --frame 40 --mask_source ./data/sam3_output/zebra/
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import viser

# Open3D for PLY loading
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. Install with: pip install open3d")

# Color palette matching generate_ground_tracklets.py
COLOR_PALETTE = [
    (1.0, 0.2, 0.2),  # Red
    (0.2, 1.0, 0.2),  # Green
    (0.2, 0.2, 1.0),  # Blue
    (1.0, 1.0, 0.2),  # Yellow
    (1.0, 0.2, 1.0),  # Magenta
    (0.2, 1.0, 1.0),  # Cyan
    (1.0, 0.6, 0.2),  # Orange
    (0.6, 0.2, 1.0),  # Purple
    (0.2, 0.8, 0.2),  # Forest Green
    (0.8, 0.2, 0.6),  # Pink
]


def load_point_cloud(filepath: str) -> Optional[dict]:
    """Load a PLY point cloud file."""
    if not HAS_OPEN3D:
        print("Error: open3d is required for PLY loading")
        return None

    if not os.path.exists(filepath):
        print(f"Warning: PLY file not found: {filepath}")
        return None

    print(f"Loading point cloud from {filepath}...")
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    print(f"Loaded {len(points):,} points")
    return {"points": points, "colors": colors}


def load_smoothed_tracklets(filepath: str) -> Optional[Dict]:
    """Load smoothed tracklets JSON."""
    if not os.path.exists(filepath):
        print(f"Warning: Tracklets file not found: {filepath}")
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {data.get('total_tracks', 0)} tracklets")
    return data


# =========================================================================
# Frame-mode helpers
# =========================================================================


def load_predictions_for_frames(
    predictions_dir: str,
    use_point_map: bool = False,
    conf_threshold_percentile: float = 50.0,
) -> Dict:
    """Load predictions.pt and prepare per-frame point cloud data.

    Returns dict with world_points (S,H,W,3), colors (S,H,W,3),
    conf (S,H,W), conf_threshold, model_size, images, num_frames.
    """
    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    predictions_path = os.path.join(predictions_dir, "predictions.pt")
    print(f"Loading predictions from {predictions_path}...")
    predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)

    def maybe_squeeze_batch(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    images = maybe_squeeze_batch(predictions["images"])        # (S, 3, H, W)
    depth = maybe_squeeze_batch(predictions["depth"])          # (S, H, W, 1)
    depth_conf = maybe_squeeze_batch(predictions["depth_conf"])  # (S, H, W)

    # Get or compute extrinsics/intrinsics
    if "extrinsic" in predictions:
        extrinsics = maybe_squeeze_batch(predictions["extrinsic"])
        intrinsics = maybe_squeeze_batch(predictions["intrinsic"])
    else:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pose_enc = maybe_squeeze_batch(predictions["pose_enc"])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            torch.from_numpy(pose_enc).unsqueeze(0),
            (images.shape[-2], images.shape[-1]),
        )
        extrinsics = extrinsics.numpy().squeeze(0)
        intrinsics = intrinsics.numpy().squeeze(0)

    # Compute world points
    if use_point_map and "world_points" in predictions:
        world_points = maybe_squeeze_batch(predictions["world_points"])
        conf = maybe_squeeze_batch(predictions.get("world_points_conf", predictions["depth_conf"]))
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = depth_conf

    # Colors: (S, 3, H, W) -> (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)

    S, H, W, _ = world_points.shape
    print(f"Loaded: {S} frames, {H}x{W} resolution")

    # Compute confidence threshold
    conf_flat = conf.reshape(-1)
    valid_conf = conf_flat[conf_flat > 1e-5]
    conf_threshold = np.percentile(valid_conf, conf_threshold_percentile) if len(valid_conf) > 0 else 0.0

    return {
        "world_points": world_points,
        "colors": colors,
        "conf": conf,
        "conf_threshold": conf_threshold,
        "model_size": (H, W),
        "images_raw": images,  # (S, 3, H, W) for shape reference
        "num_frames": S,
    }


def precompute_per_frame_data(
    world_points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
) -> tuple:
    """Pre-compute per-frame valid point arrays for fast cumulative building.

    Returns (per_frame_points, per_frame_colors) — lists of S arrays.
    """
    S = world_points.shape[0]
    per_frame_points = []
    per_frame_colors = []

    for f in range(S):
        valid = (
            (conf[f] >= conf_threshold)
            & ~np.any(np.isnan(world_points[f]) | np.isinf(world_points[f]), axis=-1)
        )
        pts = world_points[f][valid].astype(np.float32)
        clrs = (np.clip(colors[f][valid], 0, 1) * 255).astype(np.uint8)
        per_frame_points.append(pts)
        per_frame_colors.append(clrs)

    total = sum(len(p) for p in per_frame_points)
    print(f"Pre-computed {S} frames, {total:,} valid points total")
    return per_frame_points, per_frame_colors


def save_topdown_image(
    points: np.ndarray,
    colors: np.ndarray,
    tracklets_data: Optional[Dict],
    output_path: str,
    dpi: int = 300,
    subsample_ratio: float = 0.3,
) -> str:
    """Render a top-down (bird's eye) image of the ground with tracklet overlays.

    Projects onto the X-Z plane (Y is height/up). Produces a rasterized PNG
    that doesn't suffer from the point-size artifacts seen in viser.
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    # Subsample ground points for rendering speed
    n = len(points)
    if subsample_ratio < 1.0 and n > 0:
        k = max(1, int(n * subsample_ratio))
        idx = np.random.choice(n, k, replace=False)
        pts = points[idx]
        clrs = colors[idx] / 255.0
    else:
        pts = points
        clrs = colors / 255.0

    if len(pts) > 0:
        ax.scatter(
            pts[:, 0], pts[:, 2],
            c=clrs, s=0.1, alpha=0.4, marker=".", rasterized=True,
        )

    # Overlay smoothed tracklets
    if tracklets_data:
        tracks = tracklets_data.get("tracks", {})
        sorted_ids = sorted(tracks.keys(), key=lambda x: int(x))
        for i, tid in enumerate(sorted_ids):
            track = tracks[tid]
            centers = np.array(track.get("smoothed_centers", []))
            if len(centers) < 2:
                continue
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            class_name = track.get("class_name", "object")
            ax.plot(
                centers[:, 0], centers[:, 2],
                color=color, linewidth=3, label=f"T{tid}: {class_name}",
            )
            ax.scatter(
                centers[0, 0], centers[0, 2],
                color=color, s=80, marker="o", zorder=5,
                edgecolors="white", linewidths=0.5,
            )
        if sorted_ids:
            ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Ground Reconstruction - Top-Down View")

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-down image to {output_path}")
    return output_path


def visualize(
    result_dir: str,
    port: int = 8080,
    point_size: float = 0.01,
    # Frame-mode parameters
    frame_n: Optional[int] = None,
    predictions_dir: Optional[str] = None,
    mask_source: Optional[str] = None,
    mask_format: str = "auto",
    video: Optional[str] = None,
    conf_threshold: float = 50.0,
    sam3_class: Optional[str] = None,
    sam3_fps: Optional[float] = None,
    use_point_map: bool = False,
):
    """Launch interactive viser visualization.

    Args:
        result_dir: Directory containing ground_point_cloud.ply and smoothed_tracklets.json
        port: Viser server port
        point_size: Point size for rendering
        frame_n: Frame number for frame-based mode (None = PLY mode)
        predictions_dir: Directory containing predictions.pt
        mask_source: Path to SAM3 mask directory for instance highlighting
        mask_format: Mask format (auto/sam3/grounded_sam)
        video: Optional video path for SAM3 frame matching
        conf_threshold: Confidence percentile for point filtering
        sam3_class: Class name(s) for multi-class SAM3 masks
        sam3_fps: FPS used when generating SAM3 masks
        use_point_map: Use world_points directly instead of depth unprojection
    """
    print(f"\n{'='*60}")
    print("Ground Tracklets Visualization")
    print(f"{'='*60}")
    print(f"Loading from: {result_dir}\n")

    # Load tracklets (used in both modes)
    tracklets_data = load_smoothed_tracklets(
        os.path.join(result_dir, "smoothed_tracklets.json")
    )

    # Detect frame mode
    frame_mode = False
    if frame_n is not None and predictions_dir is not None:
        pred_path = os.path.join(predictions_dir, "predictions.pt")
        if os.path.exists(pred_path):
            frame_mode = True
        else:
            print(f"Warning: predictions.pt not found at {pred_path}")
            print("Falling back to PLY mode.")

    if frame_mode:
        _visualize_frame_mode(
            result_dir=result_dir,
            port=port,
            point_size=point_size,
            frame_n=frame_n,
            predictions_dir=predictions_dir,
            tracklets_data=tracklets_data,
            mask_source=mask_source,
            mask_format=mask_format,
            video=video,
            conf_threshold=conf_threshold,
            sam3_class=sam3_class,
            sam3_fps=sam3_fps,
            use_point_map=use_point_map,
        )
    else:
        _visualize_ply_mode(
            result_dir=result_dir,
            port=port,
            point_size=point_size,
            tracklets_data=tracklets_data,
        )


def _visualize_ply_mode(
    result_dir: str,
    port: int,
    point_size: float,
    tracklets_data: Optional[Dict],
):
    """Original PLY-based visualization (backward compatible)."""
    # Load ground PLY
    ground_ply = os.path.join(result_dir, "ground_point_cloud.ply")
    combined_ply = os.path.join(result_dir, "ground_with_tracklets.ply")
    if os.path.exists(ground_ply):
        ply_path = ground_ply
    else:
        ply_path = combined_ply

    pcd_data = load_point_cloud(ply_path)
    if pcd_data is None:
        print("Error: Could not load point cloud. Exiting.")
        sys.exit(1)

    points = pcd_data["points"]
    colors = pcd_data["colors"]

    # Center the scene
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center

    # Start viser server
    print(f"\nStarting viser server on port {port}...")
    server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # =========================================================================
    # GUI Controls
    # =========================================================================
    with server.gui.add_folder("Display Options"):
        gui_show_ground = server.gui.add_checkbox("Show Ground Points", initial_value=True)
        gui_show_tracklets = server.gui.add_checkbox("Show Tracklet Splines", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.001, initial_value=point_size
        )
        gui_tracklet_width = server.gui.add_slider(
            "Tracklet Line Width", min=1.0, max=10.0, step=0.5, initial_value=4.0
        )
        gui_subsample = server.gui.add_slider(
            "Point Density %", min=10, max=100, step=5, initial_value=100
        )
        gui_marker_size = server.gui.add_slider(
            "Marker Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )
        gui_show_labels = server.gui.add_checkbox("Show Track Labels", initial_value=True)
        gui_show_markers = server.gui.add_checkbox("Show Start/End Markers", initial_value=True)

    # Per-track toggle checkboxes
    track_checkboxes = {}
    if tracklets_data:
        tracks = tracklets_data.get("tracks", {})
        if tracks:
            with server.gui.add_folder("Track Filters"):
                for track_id, track_info in tracks.items():
                    class_name = track_info.get("class_name", "object")
                    n_points = len(track_info.get("smoothed_centers", []))
                    track_checkboxes[track_id] = server.gui.add_checkbox(
                        f"Track {track_id} ({class_name}, {n_points}pts)",
                        initial_value=True,
                    )

    # =========================================================================
    # Visualization State
    # =========================================================================
    viz_handles = {"point_cloud": None, "tracklets": []}

    def update_point_cloud():
        if viz_handles["point_cloud"] is not None:
            viz_handles["point_cloud"].remove()
            viz_handles["point_cloud"] = None

        if not gui_show_ground.value:
            return

        density = gui_subsample.value / 100.0
        if density < 1.0:
            n = len(points_centered)
            indices = np.random.choice(n, int(n * density), replace=False)
            display_points = points_centered[indices]
            display_colors = colors[indices]
        else:
            display_points = points_centered
            display_colors = colors

        viz_handles["point_cloud"] = server.scene.add_point_cloud(
            name="/ground_cloud",
            points=display_points,
            colors=display_colors,
            point_size=gui_point_size.value,
            point_shape="circle",
        )

    def update_tracklets():
        for handle in viz_handles["tracklets"]:
            handle.remove()
        viz_handles["tracklets"] = []

        if not gui_show_tracklets.value or tracklets_data is None:
            return

        tracks = tracklets_data.get("tracks", {})
        sorted_ids = sorted(tracks.keys(), key=lambda x: int(x))
        line_width = gui_tracklet_width.value

        for idx, track_id in enumerate(sorted_ids):
            if track_id in track_checkboxes and not track_checkboxes[track_id].value:
                continue

            track = tracks[track_id]
            centers = np.array(track.get("smoothed_centers", []))

            if len(centers) < 2:
                continue

            centers_centered = centers - scene_center
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            handle = server.scene.add_spline_catmull_rom(
                name=f"/tracklet_{track_id}",
                positions=centers_centered.astype(np.float32),
                color=color,
                line_width=line_width,
                segments=max(1, len(centers_centered) * 4),
            )
            viz_handles["tracklets"].append(handle)

            if gui_show_markers.value:
                marker_r = gui_marker_size.value
                start_handle = server.scene.add_icosphere(
                    name=f"/tracklet_{track_id}_start",
                    radius=marker_r,
                    position=centers_centered[0],
                    color=color,
                )
                viz_handles["tracklets"].append(start_handle)

                end_handle = server.scene.add_icosphere(
                    name=f"/tracklet_{track_id}_end",
                    radius=marker_r * 0.6,
                    position=centers_centered[-1],
                    color=color,
                )
                viz_handles["tracklets"].append(end_handle)

            if gui_show_labels.value:
                mid_idx = len(centers_centered) // 2
                class_name = track.get("class_name", "object")
                label_handle = server.scene.add_label(
                    name=f"/tracklet_label_{track_id}",
                    text=f"T{track_id}: {class_name}",
                    position=centers_centered[mid_idx],
                )
                viz_handles["tracklets"].append(label_handle)

    def update_all():
        update_point_cloud()
        update_tracklets()

    # =========================================================================
    # Callbacks
    # =========================================================================
    @gui_show_ground.on_update
    def _(_):
        update_point_cloud()

    @gui_show_tracklets.on_update
    def _(_):
        update_tracklets()

    @gui_point_size.on_update
    def _(_):
        if viz_handles["point_cloud"] is not None:
            viz_handles["point_cloud"].point_size = gui_point_size.value

    @gui_tracklet_width.on_update
    def _(_):
        update_tracklets()

    @gui_subsample.on_update
    def _(_):
        update_point_cloud()

    @gui_marker_size.on_update
    def _(_):
        update_tracklets()

    @gui_show_labels.on_update
    def _(_):
        update_tracklets()

    @gui_show_markers.on_update
    def _(_):
        update_tracklets()

    def _make_track_cb(cb):
        @cb.on_update
        def _(_):
            update_tracklets()

    for tid, checkbox in track_checkboxes.items():
        _make_track_cb(checkbox)

    # Initial render
    update_all()

    print(f"\n{'='*60}")
    print(f"Visualization ready! Open your browser to:")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def _visualize_frame_mode(
    result_dir: str,
    port: int,
    point_size: float,
    frame_n: int,
    predictions_dir: str,
    tracklets_data: Optional[Dict],
    mask_source: Optional[str],
    mask_format: str,
    video: Optional[str],
    conf_threshold: float,
    sam3_class: Optional[str],
    sam3_fps: Optional[float],
    use_point_map: bool,
):
    """Frame-based visualization: ground PLY + trimmed tracklets + instance highlights."""
    import cv2

    # =========================================================================
    # Step 1: Load ground PLY (already excludes animal pixels)
    # =========================================================================
    ground_ply = os.path.join(result_dir, "ground_point_cloud.ply")
    combined_ply = os.path.join(result_dir, "ground_with_tracklets.ply")
    if os.path.exists(ground_ply):
        ply_path = ground_ply
    else:
        ply_path = combined_ply

    pcd_data = load_point_cloud(ply_path)
    if pcd_data is None:
        print("Error: Could not load ground point cloud. Exiting.")
        sys.exit(1)

    points = pcd_data["points"]
    colors = pcd_data["colors"]
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center

    # =========================================================================
    # Step 2: Load predictions (only needed for instance 3D extraction)
    # =========================================================================
    pred_data = load_predictions_for_frames(
        predictions_dir=predictions_dir,
        use_point_map=use_point_map,
        conf_threshold_percentile=conf_threshold,
    )

    world_points = pred_data["world_points"]
    frame_colors = pred_data["colors"]
    model_size = pred_data["model_size"]
    num_frames = pred_data["num_frames"]

    # Load metadata for frame number mapping (original video frame IDs)
    metadata_path = os.path.join(predictions_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    frame_numbers = metadata.get("frame_numbers", list(range(num_frames)))

    # Map --frame (original frame ID) to internal extracted index
    if frame_n in frame_numbers:
        frame_idx = frame_numbers.index(frame_n)
        print(f"\nFrame {frame_n} -> extracted index {frame_idx}")
    elif 0 <= frame_n < num_frames:
        frame_idx = frame_n
        print(f"\nFrame {frame_n} treated as extracted index (no matching original ID)")
    else:
        print(f"Error: --frame {frame_n} not found in original frame IDs {frame_numbers[:3]}...{frame_numbers[-3:]}")
        print(f"  Valid original IDs: {frame_numbers[0]}..{frame_numbers[-1]}")
        print(f"  Valid extracted indices: 0..{num_frames - 1}")
        sys.exit(1)

    # Use frame_idx internally from here on
    frame_n = frame_idx
    print(f"Frame mode: tracklets up to index {frame_n}, instances at index {frame_n} (of {num_frames})")

    # =========================================================================
    # Step 3: Load SAM3 masks for instance highlighting
    # =========================================================================
    masks_data = None
    original_images = None

    if mask_source is not None:
        from save_masked_frames import load_masks

        # Auto-detect mask format (same logic as save_masked_frames.py main())
        resolved_mask_format = mask_format
        if resolved_mask_format == "auto":
            from vggt.utils.sam3_mask_loader import detect_sam3_format
            format_info = detect_sam3_format(mask_source)
            if format_info["format"] in ("single_class", "multi_class"):
                resolved_mask_format = "sam3"
                print(f"Auto-detected mask format: SAM3 ({format_info['format']})")
            else:
                resolved_mask_format = "grounded_sam"
                print("Auto-detected mask format: Grounded SAM")

        # metadata already loaded above for frame number mapping
        masks_data = load_masks(
            mask_source=mask_source,
            mask_format=resolved_mask_format,
            metadata=metadata,
            video=video,
            sam3_class=sam3_class,
            sam3_fps=sam3_fps,
        )
        total_masks = sum(len(v) for v in masks_data.values())
        print(f"Loaded masks for {len(masks_data)} frames ({total_masks} total instances)")

        # Load original images for mask coordinate transform
        image_files = metadata.get("image_files", [])
        input_path = metadata.get("input_path", "")
        H, W = model_size
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
                    original_images.append(np.zeros((H, W, 3), dtype=np.uint8))
    else:
        print("No --mask_source provided. Instance highlighting disabled.")

    # =========================================================================
    # Step 4: Build instance highlight point clouds at frame_n
    # =========================================================================
    instance_clouds = {}  # track_id -> (points, colors_uint8)

    if masks_data is not None and frame_n in masks_data:
        from save_masked_frames import transform_mask_to_model_coordinates

        tracks = tracklets_data.get("tracks", {}) if tracklets_data else {}
        sorted_ids = sorted(tracks.keys(), key=lambda x: int(x))
        H, W = model_size

        # Determine whether masks carry object_id (SAM3) or need IoU matching
        # (Grounded SAM). Check the first mask at this frame.
        frame_masks = masks_data[frame_n]
        has_object_id = any(
            "object_id" in m and m["object_id"] >= 0 for m in frame_masks
        )

        if has_object_id:
            # SAM3: direct object_id -> track_id mapping
            mask_to_track = {}
            for mi, m in enumerate(frame_masks):
                oid = m.get("object_id", -1)
                tid_str = str(oid)
                if tid_str in sorted_ids:
                    mask_to_track[mi] = (tid_str, sorted_ids.index(tid_str))
        else:
            # Grounded SAM: match via IoU with tracking_summary bbox_2d
            # (same approach as save_masked_frames.py)
            tracking_path = os.path.join(predictions_dir, "tracking_summary.json")
            if os.path.exists(tracking_path):
                with open(tracking_path, "r") as f:
                    tracking_summary = json.load(f)
                orig_shapes = {}
                for m in frame_masks:
                    orig_shapes[frame_n] = m["mask"].shape[:2]
                    break
                from save_masked_frames import match_tracks_to_masks
                mask_to_track = match_tracks_to_masks(
                    frame_idx=frame_n,
                    tracking=tracking_summary,
                    masks_data=masks_data,
                    model_size=model_size,
                    orig_shapes=orig_shapes,
                )
            else:
                print(f"Warning: tracking_summary.json not found for Grounded SAM matching")
                mask_to_track = {}

        for mask_idx, mask_info in enumerate(frame_masks):
            if mask_idx not in mask_to_track:
                continue

            track_id_str, sort_idx = mask_to_track[mask_idx]
            mask = mask_info["mask"]

            color = COLOR_PALETTE[sort_idx % len(COLOR_PALETTE)]
            color_uint8 = np.array([int(c * 255) for c in color], dtype=np.uint8)

            # Transform mask to model coordinates
            if original_images is not None and frame_n < len(original_images):
                orig_img = original_images[frame_n]
                orig_h, orig_w = orig_img.shape[:2]
            else:
                orig_h, orig_w = mask.shape[:2]

            mask_model = transform_mask_to_model_coordinates(
                mask, (orig_h, orig_w), (H, W)
            )

            # Extract 3D points at mask pixels
            # Use relaxed filtering for instances: mask already constrains
            # spatially, and animals typically have lower depth confidence
            # than the static ground.
            instance_pts = world_points[frame_n][mask_model]

            valid = ~np.any(
                np.isnan(instance_pts) | np.isinf(instance_pts), axis=-1
            )
            instance_pts = instance_pts[valid].astype(np.float32)

            if len(instance_pts) == 0:
                continue

            # Blend original image colors with track color for subtle tint
            instance_img_colors = frame_colors[frame_n][mask_model][valid]
            instance_img_colors = (np.clip(instance_img_colors, 0, 1) * 255).astype(np.uint8)
            alpha = 0.5
            blended_clrs = (alpha * color_uint8 + (1 - alpha) * instance_img_colors).astype(np.uint8)
            instance_clouds[track_id_str] = (instance_pts, blended_clrs)
            class_name = tracks.get(track_id_str, {}).get("class_name", "object")
            print(f"  Instance T{track_id_str} ({class_name}): {len(instance_pts):,} points")

    if instance_clouds:
        print(f"Highlighting {len(instance_clouds)} instances at frame {frame_n}")
    elif mask_source is not None:
        print(f"No instance masks found at frame {frame_n}")

    # =========================================================================
    # Step 5: Start viser server and render
    # =========================================================================
    print(f"\nStarting viser server on port {port}...")
    server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # --- Frame info ---
    orig_frame_id = frame_numbers[frame_n] if frame_n < len(frame_numbers) else frame_n
    with server.gui.add_folder("Frame Info"):
        server.gui.add_markdown(
            f"**Original frame {orig_frame_id}** (index {frame_n} / {num_frames - 1})\n\n"
            f"Ground points: {len(points):,}\n\n"
            f"Instances at frame {orig_frame_id}: {len(instance_clouds)}"
        )

    # --- Display Options ---
    with server.gui.add_folder("Display Options"):
        gui_show_ground = server.gui.add_checkbox("Show Ground Points", initial_value=True)
        gui_show_tracklets = server.gui.add_checkbox("Show Tracklet Splines", initial_value=True)
        gui_show_instances = server.gui.add_checkbox("Show Instance Highlights", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.001, initial_value=point_size
        )
        gui_tracklet_width = server.gui.add_slider(
            "Tracklet Line Width", min=1.0, max=10.0, step=0.5, initial_value=4.0
        )
        gui_subsample = server.gui.add_slider(
            "Point Density %", min=10, max=100, step=5, initial_value=100
        )
        gui_marker_size = server.gui.add_slider(
            "Marker Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )
        gui_show_labels = server.gui.add_checkbox("Show Track Labels", initial_value=True)
        gui_show_markers = server.gui.add_checkbox("Show Start/End Markers", initial_value=True)

    # Per-track toggle checkboxes
    track_checkboxes = {}
    if tracklets_data:
        tracks = tracklets_data.get("tracks", {})
        if tracks:
            with server.gui.add_folder("Track Filters"):
                for track_id, track_info in tracks.items():
                    class_name = track_info.get("class_name", "object")
                    n_points = len(track_info.get("smoothed_centers", []))
                    track_checkboxes[track_id] = server.gui.add_checkbox(
                        f"Track {track_id} ({class_name}, {n_points}pts)",
                        initial_value=True,
                    )

    # =========================================================================
    # Visualization State
    # =========================================================================
    viz_handles = {"point_cloud": None, "tracklets": [], "instances": [], "instance_labels": []}

    def update_point_cloud():
        if viz_handles["point_cloud"] is not None:
            viz_handles["point_cloud"].remove()
            viz_handles["point_cloud"] = None

        if not gui_show_ground.value:
            return

        density = gui_subsample.value / 100.0
        if density < 1.0:
            n = len(points_centered)
            indices = np.random.choice(n, int(n * density), replace=False)
            display_points = points_centered[indices]
            display_colors = colors[indices]
        else:
            display_points = points_centered
            display_colors = colors

        viz_handles["point_cloud"] = server.scene.add_point_cloud(
            name="/ground_cloud",
            points=display_points,
            colors=display_colors,
            point_size=gui_point_size.value,
            point_shape="circle",
        )

    def update_tracklets():
        """Render tracklet splines trimmed to frames 0..frame_n."""
        for handle in viz_handles["tracklets"]:
            handle.remove()
        viz_handles["tracklets"] = []

        if not gui_show_tracklets.value or tracklets_data is None:
            return

        tracks = tracklets_data.get("tracks", {})
        sorted_ids = sorted(tracks.keys(), key=lambda x: int(x))
        line_width = gui_tracklet_width.value

        for idx, track_id in enumerate(sorted_ids):
            if track_id in track_checkboxes and not track_checkboxes[track_id].value:
                continue

            track = tracks[track_id]
            all_frames = track.get("frames", [])
            all_centers = np.array(track.get("smoothed_centers", []))

            if len(all_centers) == 0:
                continue

            # Trim to frames <= frame_n
            trim_mask = np.array(all_frames) <= frame_n
            trimmed_centers = all_centers[trim_mask]

            if len(trimmed_centers) == 0:
                continue

            centers_centered = trimmed_centers - scene_center
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            # Single point — just a marker
            if len(trimmed_centers) == 1:
                if gui_show_markers.value:
                    marker_r = gui_marker_size.value
                    handle = server.scene.add_icosphere(
                        name=f"/tracklet_{track_id}_start",
                        radius=marker_r,
                        position=centers_centered[0],
                        color=color,
                    )
                    viz_handles["tracklets"].append(handle)
                continue

            # Render trimmed spline
            handle = server.scene.add_spline_catmull_rom(
                name=f"/tracklet_{track_id}",
                positions=centers_centered.astype(np.float32),
                color=color,
                line_width=line_width,
                segments=max(1, len(centers_centered) * 4),
            )
            viz_handles["tracklets"].append(handle)

            # Start marker
            if gui_show_markers.value:
                marker_r = gui_marker_size.value
                start_handle = server.scene.add_icosphere(
                    name=f"/tracklet_{track_id}_start",
                    radius=marker_r,
                    position=centers_centered[0],
                    color=color,
                )
                viz_handles["tracklets"].append(start_handle)

                # Current-position marker (end of trimmed spline)
                current_handle = server.scene.add_icosphere(
                    name=f"/tracklet_{track_id}_current",
                    radius=marker_r * 0.8,
                    position=centers_centered[-1],
                    color=color,
                )
                viz_handles["tracklets"].append(current_handle)

            # Label at current position
            if gui_show_labels.value:
                class_name = track.get("class_name", "object")
                label_handle = server.scene.add_label(
                    name=f"/tracklet_label_{track_id}",
                    text=f"T{track_id}: {class_name}",
                    position=centers_centered[-1],
                )
                viz_handles["tracklets"].append(label_handle)

    def update_instances():
        """Render instance highlight point clouds at frame_n."""
        for handle in viz_handles["instances"]:
            handle.remove()
        viz_handles["instances"] = []
        for handle in viz_handles["instance_labels"]:
            handle.remove()
        viz_handles["instance_labels"] = []

        if not gui_show_instances.value or not instance_clouds:
            return

        tracks = tracklets_data.get("tracks", {}) if tracklets_data else {}

        for track_id_str, (pts, clrs) in instance_clouds.items():
            # Respect per-track checkbox
            if track_id_str in track_checkboxes and not track_checkboxes[track_id_str].value:
                continue

            pts_centered = pts - scene_center
            handle = server.scene.add_point_cloud(
                name=f"/instance_{track_id_str}",
                points=pts_centered,
                colors=clrs,
                point_size=gui_point_size.value,
                point_shape="circle",
            )
            viz_handles["instances"].append(handle)

            # Instance ID label
            if gui_show_labels.value and len(pts_centered) > 0:
                centroid = np.mean(pts_centered, axis=0).astype(np.float32)
                class_name = tracks.get(track_id_str, {}).get("class_name", "object")
                label_handle = server.scene.add_label(
                    name=f"/instance_label_{track_id_str}",
                    text=f"T{track_id_str}: {class_name}",
                    position=centroid,
                )
                viz_handles["instance_labels"].append(label_handle)

    def update_all():
        update_point_cloud()
        update_tracklets()
        update_instances()

    # =========================================================================
    # Callbacks
    # =========================================================================
    @gui_show_ground.on_update
    def _(_):
        update_point_cloud()

    @gui_show_tracklets.on_update
    def _(_):
        update_tracklets()

    @gui_show_instances.on_update
    def _(_):
        update_instances()

    @gui_point_size.on_update
    def _(_):
        if viz_handles["point_cloud"] is not None:
            viz_handles["point_cloud"].point_size = gui_point_size.value
        update_instances()

    @gui_tracklet_width.on_update
    def _(_):
        update_tracklets()

    @gui_subsample.on_update
    def _(_):
        update_point_cloud()

    @gui_marker_size.on_update
    def _(_):
        update_tracklets()

    @gui_show_labels.on_update
    def _(_):
        update_tracklets()
        update_instances()

    @gui_show_markers.on_update
    def _(_):
        update_tracklets()

    def _make_track_cb(cb):
        @cb.on_update
        def _(_):
            update_tracklets()
            update_instances()

    for tid, checkbox in track_checkboxes.items():
        _make_track_cb(checkbox)

    # Initial render
    update_all()

    print(f"\n{'='*60}")
    print(f"Frame-mode visualization ready! Open your browser to:")
    print(f"  http://localhost:{port}")
    print(f"  Ground PLY + tracklets up to index {frame_n} (frame {orig_frame_id}) + instances at that frame")
    print(f"{'='*60}")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground point cloud with smoothed tracklets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PLY mode (original)
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/

    # Frame mode with instance highlighting
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ \\
        --frame 40 --mask_source ./data/sam3_output/zebra/

    # Frame mode without masks
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ \\
        --frame 40
        """,
    )

    parser.add_argument(
        "--result_dir", type=str, required=True,
        help="Directory containing ground_point_cloud.ply and smoothed_tracklets.json",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for viser server (default: 8080)",
    )
    parser.add_argument(
        "--point_size", type=float, default=0.01,
        help="Initial point size (default: 0.01)",
    )
    parser.add_argument(
        "--save_topdown", type=str, default=None,
        help="Save a top-down PNG image to this path and exit (no viser). "
             "Example: --save_topdown ./topdown.png",
    )
    parser.add_argument(
        "--topdown_dpi", type=int, default=300,
        help="DPI for the top-down image (default: 300)",
    )
    parser.add_argument(
        "--topdown_subsample", type=float, default=0.3,
        help="Fraction of ground points to render in top-down image (default: 0.3)",
    )

    # Frame-mode arguments
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Original video frame ID to visualize up to (enables frame mode). "
             "Uses frame_numbers from metadata.json. Shows cumulative cloud "
             "and instances at that frame.",
    )
    parser.add_argument(
        "--predictions_dir", type=str, default=None,
        help="Directory containing predictions.pt (default: parent of result_dir)",
    )
    parser.add_argument(
        "--mask_source", type=str, default=None,
        help="Path to SAM3 mask directory for instance highlighting",
    )
    parser.add_argument(
        "--mask_format", type=str, default="auto",
        choices=["auto", "sam3", "grounded_sam"],
        help="Mask format for instance highlighting (default: auto)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video (for SAM3 frame-index matching)",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=50.0,
        help="Confidence percentile for point filtering in frame mode (default: 50.0)",
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

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Directory not found: {args.result_dir}")
        sys.exit(1)

    # Top-down image mode: render and exit without launching viser
    if args.save_topdown:
        ground_ply = os.path.join(args.result_dir, "ground_point_cloud.ply")
        combined_ply = os.path.join(args.result_dir, "ground_with_tracklets.ply")
        ply_path = ground_ply if os.path.exists(ground_ply) else combined_ply

        pcd_data = load_point_cloud(ply_path)
        if pcd_data is None:
            print("Error: Could not load point cloud.")
            sys.exit(1)

        tracklets_data = load_smoothed_tracklets(
            os.path.join(args.result_dir, "smoothed_tracklets.json")
        )

        save_topdown_image(
            points=pcd_data["points"],
            colors=pcd_data["colors"],
            tracklets_data=tracklets_data,
            output_path=args.save_topdown,
            dpi=args.topdown_dpi,
            subsample_ratio=args.topdown_subsample,
        )
        sys.exit(0)

    # Resolve predictions_dir: default to parent of result_dir
    predictions_dir = args.predictions_dir
    if predictions_dir is None and args.frame is not None:
        predictions_dir = os.path.dirname(args.result_dir.rstrip("/"))

    visualize(
        result_dir=args.result_dir,
        port=args.port,
        point_size=args.point_size,
        frame_n=args.frame,
        predictions_dir=predictions_dir,
        mask_source=args.mask_source,
        mask_format=args.mask_format,
        video=args.video,
        conf_threshold=args.conf_threshold,
        sam3_class=args.sam3_class,
        sam3_fps=args.sam3_fps,
        use_point_map=args.use_point_map,
    )


if __name__ == "__main__":
    main()
