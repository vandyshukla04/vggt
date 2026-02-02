# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize saved VGGT outputs with viser.

Loads predictions saved by batch_inference.py and renders them locally.

Usage:
    python visualize_saved.py --output_dir ./outputs/scene

Requires:
    - predictions.pt or (point_cloud.ply + cameras.json)
    - Optional: tracking_summary.json for bounding box visualization
"""

import os
import sys
import json
import time
import argparse
import threading
from typing import List, Optional, Dict

import numpy as np
import torch
import viser
import viser.transforms as viser_tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.export import load_cameras_json, load_depth_maps


# Color palette for track visualization
COLOR_PALETTE = [
    (1.0, 0.2, 0.2),   # Red
    (0.2, 1.0, 0.2),   # Green
    (0.2, 0.2, 1.0),   # Blue
    (1.0, 1.0, 0.2),   # Yellow
    (1.0, 0.2, 1.0),   # Magenta
    (0.2, 1.0, 1.0),   # Cyan
    (1.0, 0.6, 0.2),   # Orange
    (0.6, 0.2, 1.0),   # Purple
    (0.2, 0.8, 0.2),   # Forest Green
    (0.8, 0.2, 0.6),   # Pink
]


def get_track_color(track_id: int, all_track_ids: List[int]) -> tuple:
    """Get consistent color for a track ID."""
    sorted_ids = sorted(all_track_ids)
    idx = sorted_ids.index(track_id) if track_id in sorted_ids else track_id
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def load_predictions(output_dir: str) -> dict:
    """Load predictions from saved files."""
    predictions_path = os.path.join(output_dir, "predictions.pt")

    if os.path.exists(predictions_path):
        print(f"Loading predictions from {predictions_path}")
        predictions = torch.load(predictions_path, map_location="cpu")

        # Convert to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].numpy().squeeze(0)

        return predictions

    # Fallback: load from individual files
    print("predictions.pt not found, loading from individual files...")

    cameras_path = os.path.join(output_dir, "cameras.json")
    depth_path = os.path.join(output_dir, "depth_maps.npz")

    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"cameras.json not found in {output_dir}")

    extrinsics, intrinsics, image_names = load_cameras_json(cameras_path)

    predictions = {
        "extrinsic": extrinsics,
        "intrinsic": intrinsics,
    }

    if os.path.exists(depth_path):
        depth, depth_conf = load_depth_maps(depth_path)
        predictions["depth"] = depth
        predictions["depth_conf"] = depth_conf

        # Compute world points from depth
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        predictions["world_points"] = world_points
        predictions["world_points_conf"] = depth_conf

    return predictions


def load_tracking_data(output_dir: str) -> Optional[Dict]:
    """Load tracking summary if available."""
    tracking_path = os.path.join(output_dir, "tracking_summary.json")
    if os.path.exists(tracking_path):
        print(f"Loading tracking data from {tracking_path}")
        with open(tracking_path, 'r') as f:
            return json.load(f)
    return None


def visualize_with_viser(
    predictions: dict,
    tracking_data: Optional[Dict] = None,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = True,
):
    """Visualize predictions with viser."""
    print(f"\nStarting viser server on port {port}")
    print(f"Open http://localhost:{port} in your browser")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Get data from predictions
    extrinsics = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics = predictions["intrinsic"]  # (S, 3, 3)

    if "world_points" in predictions:
        world_points = predictions["world_points"]  # (S, H, W, 3)
        conf = predictions.get("world_points_conf", predictions.get("depth_conf"))
    elif "depth" in predictions:
        depth = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = predictions.get("depth_conf")
    else:
        raise ValueError("No point cloud data found in predictions")

    # Get colors from images if available
    if "images" in predictions:
        images = predictions["images"]  # (S, 3, H, W)
        colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    else:
        # Default gray colors
        colors = np.ones_like(world_points) * 0.5

    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1) if conf is not None else np.ones(len(points))

    # Compute camera-to-world transforms
    cam_to_world_mat = closed_form_inverse_se3(extrinsics)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Center the scene
    valid_mask = conf_flat > np.percentile(conf_flat, 10)
    scene_center = np.mean(points[valid_mask], axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    frame_indices = np.repeat(np.arange(S), H * W)

    # Build GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_show_tracks = server.gui.add_checkbox("Show Track Trajectories", initial_value=True)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_slider = server.gui.add_slider(
        "Frame", min=-1, max=S-1, step=1, initial_value=-1
    )
    server.gui.add_markdown("*Frame -1 = All frames*")

    # Create initial point cloud
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    track_handles = []

    def visualize_frames():
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        if not gui_show_frames.value:
            return

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle):
            @frustum.on_click
            def _(_):
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in range(S):
            cam2world_3x4 = cam_to_world[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Add frustum with image if available
            if "images" in predictions:
                img = predictions["images"][img_id]
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                h, w = img.shape[:2]
                fy = 1.1 * h
                fov = 2 * np.arctan2(h / 2, fy)

                frustum_cam = server.scene.add_camera_frustum(
                    f"frame_{img_id}/frustum",
                    fov=fov,
                    aspect=w / h,
                    scale=0.05,
                    image=img,
                    line_width=1.0
                )
            else:
                frustum_cam = server.scene.add_camera_frustum(
                    f"frame_{img_id}/frustum",
                    fov=1.0,
                    aspect=1.0,
                    scale=0.05,
                    line_width=1.0
                )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def visualize_tracks():
        for th in track_handles:
            try:
                th.remove()
            except:
                pass
        track_handles.clear()

        if not gui_show_tracks.value or tracking_data is None:
            return

        tracks = tracking_data.get("tracks", {})
        all_track_ids = [int(tid) for tid in tracks.keys()]

        for tid_str, track_info in tracks.items():
            tid = int(tid_str)
            centers = np.array(track_info["centers"]) - scene_center

            if len(centers) < 2:
                continue

            color = get_track_color(tid, all_track_ids)

            # Draw trajectory as spline
            try:
                spline = server.scene.add_spline_catmull_rom(
                    f"track_{tid}",
                    positions=centers,
                    color=color,
                )
                track_handles.append(spline)

                # Add start marker
                marker = server.scene.add_icosphere(
                    f"track_{tid}_start",
                    radius=0.02,
                    color=color,
                    position=centers[0],
                )
                track_handles.append(marker)
            except:
                pass

    def update_point_cloud():
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        selected_frame = int(gui_frame_slider.value)
        if selected_frame < 0:
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            frame_mask = frame_indices == selected_frame

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_):
        update_point_cloud()

    @gui_frame_slider.on_update
    def _(_):
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_):
        visualize_frames()

    @gui_show_tracks.on_update
    def _(_):
        visualize_tracks()

    # Initial visualization
    visualize_frames()
    visualize_tracks()

    print("\nViser server started. Press Ctrl+C to stop.")

    # Keep server running
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved VGGT outputs with viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization
    python visualize_saved.py --output_dir ./outputs/scene

    # Custom port and confidence threshold
    python visualize_saved.py --output_dir ./outputs/scene --port 8081 --conf_threshold 60
        """
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing saved outputs from batch_inference.py")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for viser server")
    parser.add_argument("--conf_threshold", type=float, default=50.0,
                        help="Initial confidence threshold percentile")
    parser.add_argument("--use_point_map", action="store_true",
                        help="Use point map instead of depth-based points")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        sys.exit(1)

    # Load predictions
    predictions = load_predictions(args.output_dir)

    # Load tracking data if available
    tracking_data = load_tracking_data(args.output_dir)

    # Visualize
    visualize_with_viser(
        predictions,
        tracking_data=tracking_data,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
    )


if __name__ == "__main__":
    main()
