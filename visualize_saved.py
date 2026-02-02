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

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


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


def speed_to_color(speed: float, min_speed: float, max_speed: float) -> tuple:
    """Map speed to color: blue (slow) -> cyan -> green -> yellow -> red (fast).

    Args:
        speed: Current speed value
        min_speed: Minimum speed in the dataset
        max_speed: Maximum speed in the dataset

    Returns:
        RGB tuple with values in [0, 1]
    """
    if max_speed <= min_speed:
        return (0.5, 0.5, 0.5)  # Gray if no variation

    # Normalize speed to [0, 1]
    t = (speed - min_speed) / (max_speed - min_speed)
    t = np.clip(t, 0, 1)

    # Color gradient: blue -> cyan -> green -> yellow -> red
    if t < 0.25:
        # Blue to Cyan
        r, g, b = 0, t * 4, 1
    elif t < 0.5:
        # Cyan to Green
        r, g, b = 0, 1, 1 - (t - 0.25) * 4
    elif t < 0.75:
        # Green to Yellow
        r, g, b = (t - 0.5) * 4, 1, 0
    else:
        # Yellow to Red
        r, g, b = 1, 1 - (t - 0.75) * 4, 0

    return (r, g, b)


def compute_bbox_corners(center: np.ndarray, dimensions: np.ndarray,
                         rotation_matrix: np.ndarray) -> np.ndarray:
    """Compute 8 corner points of a 3D bounding box.

    Args:
        center: (3,) center position
        dimensions: (3,) [length, width, height]
        rotation_matrix: (3, 3) rotation matrix

    Returns:
        (8, 3) array of corner positions
    """
    l, w, h = dimensions
    corners_local = np.array([
        [-l/2, -w/2, -h/2],
        [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2],
        [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2],
        [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2],
        [-l/2, +w/2, +h/2],
    ])
    corners_world = (rotation_matrix @ corners_local.T).T + center
    return corners_world


# Bounding box edge indices for wireframe
BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
]


def load_predictions(output_dir: str) -> dict:
    """Load predictions from saved files."""
    predictions_path = os.path.join(output_dir, "predictions.pt")

    def maybe_squeeze_batch(arr):
        """Remove batch dimension if it exists and equals 1."""
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr

    if os.path.exists(predictions_path):
        print(f"Loading predictions from {predictions_path}")
        predictions = torch.load(predictions_path, map_location="cpu")

        # Convert to numpy (safely handle batch dimension)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = maybe_squeeze_batch(predictions[key].numpy())

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
    gui_show_bboxes = server.gui.add_checkbox("Show 3D Bounding Boxes", initial_value=False)
    gui_speed_color = server.gui.add_checkbox("Speed Color Coding", initial_value=False)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_slider = server.gui.add_slider(
        "Frame", min=-1, max=S-1, step=1, initial_value=-1
    )
    server.gui.add_markdown("*Frame -1 = All frames*")
    server.gui.add_markdown("**Speed Legend**: Blue=Slow, Red=Fast")

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
    bbox_handles = []

    # Precompute speed statistics for color mapping
    min_speed, max_speed = 0.0, 1.0
    if tracking_data is not None:
        all_speeds = []
        for track_info in tracking_data.get("tracks", {}).values():
            velocities = track_info.get("velocities", [])
            for vel in velocities:
                speed = np.linalg.norm(vel)
                all_speeds.append(speed)
        if all_speeds:
            min_speed = np.min(all_speeds)
            max_speed = np.max(all_speeds)
            if max_speed <= min_speed:
                max_speed = min_speed + 1e-6

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
        use_speed_color = gui_speed_color.value

        for tid_str, track_info in tracks.items():
            tid = int(tid_str)
            centers = np.array(track_info["centers"]) - scene_center
            velocities = track_info.get("velocities", [[0, 0, 0]] * len(centers))

            if len(centers) < 2:
                continue

            if use_speed_color and len(velocities) >= len(centers) - 1:
                # Draw segments with speed-based colors
                for i in range(len(centers) - 1):
                    vel = velocities[i] if i < len(velocities) else [0, 0, 0]
                    speed = np.linalg.norm(vel)
                    color = speed_to_color(speed, min_speed, max_speed)

                    try:
                        segment = server.scene.add_spline_catmull_rom(
                            f"track_{tid}_seg_{i}",
                            positions=np.array([centers[i], centers[i + 1]]),
                            color=color,
                        )
                        track_handles.append(segment)
                    except:
                        pass

                # Add start marker with speed color
                start_speed = np.linalg.norm(velocities[0]) if velocities else 0
                start_color = speed_to_color(start_speed, min_speed, max_speed)
                try:
                    marker = server.scene.add_icosphere(
                        f"track_{tid}_start",
                        radius=0.02,
                        color=start_color,
                        position=centers[0],
                    )
                    track_handles.append(marker)
                except:
                    pass
            else:
                # Use track ID-based color (original behavior)
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

    def visualize_bboxes():
        """Visualize 3D bounding boxes as wireframes."""
        for bh in bbox_handles:
            try:
                bh.remove()
            except:
                pass
        bbox_handles.clear()

        if not gui_show_bboxes.value or tracking_data is None:
            return

        tracks = tracking_data.get("tracks", {})
        all_track_ids = [int(tid) for tid in tracks.keys()]
        selected_frame = int(gui_frame_slider.value)

        for tid_str, track_info in tracks.items():
            tid = int(tid_str)
            frames_list = track_info.get("frames", [])
            centers = track_info.get("centers", [])
            dimensions = track_info.get("dimensions", [])
            rotations = track_info.get("rotation_matrices", [])

            if not dimensions or not rotations:
                continue

            color = get_track_color(tid, all_track_ids)

            for i, f_idx in enumerate(frames_list):
                # Filter by frame if specified
                if selected_frame >= 0 and f_idx != selected_frame:
                    continue

                if i >= len(dimensions) or i >= len(rotations) or i >= len(centers):
                    continue

                # Reconstruct bounding box
                center = np.array(centers[i]) - scene_center
                dims = np.array(dimensions[i])
                rotation = np.array(rotations[i])

                # Get corners
                corners = compute_bbox_corners(center, dims, rotation)

                # Draw wireframe edges
                for edge_idx, (a, b) in enumerate(BBOX_EDGES):
                    try:
                        line = server.scene.add_spline_catmull_rom(
                            f"bbox_{tid}_{f_idx}_{edge_idx}",
                            positions=np.array([corners[a], corners[b]]),
                            color=color,
                        )
                        bbox_handles.append(line)
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
        visualize_bboxes()  # Update bboxes when frame changes

    @gui_show_frames.on_update
    def _(_):
        visualize_frames()

    @gui_show_tracks.on_update
    def _(_):
        visualize_tracks()

    @gui_show_bboxes.on_update
    def _(_):
        visualize_bboxes()

    @gui_speed_color.on_update
    def _(_):
        visualize_tracks()  # Re-render tracks with new color mode

    # Initial visualization
    visualize_frames()
    visualize_tracks()
    visualize_bboxes()

    print("\nViser server started. Press Ctrl+C to stop.")

    # Keep server running
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


def draw_bbox_wireframe(img: np.ndarray, corners_2d: np.ndarray, color: tuple, thickness: int = 2):
    """Draw 3D bounding box wireframe on image.

    Args:
        img: Image to draw on (modified in place)
        corners_2d: (8, 2) array of projected corner coordinates
        color: BGR color tuple
        thickness: Line thickness
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    for a, b in edges:
        pt1 = tuple(corners_2d[a].astype(int))
        pt2 = tuple(corners_2d[b].astype(int))
        cv2.line(img, pt1, pt2, color, thickness)


def generate_annotated_2d(output_dir: str, predictions: dict, tracking_data: Dict):
    """Generate annotated 2D images with projected bounding boxes.

    Args:
        output_dir: Output directory (will create annotated_2d subfolder)
        predictions: Predictions dict with extrinsics, intrinsics
        tracking_data: Tracking summary dict with tracks
    """
    if not HAS_CV2:
        print("Error: OpenCV (cv2) not available. Cannot generate annotated images.")
        return

    if tracking_data is None:
        print("Error: No tracking data available. Cannot generate annotated images.")
        return

    # Load metadata to get image paths
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {output_dir}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    image_files = metadata.get("image_files", [])
    if not image_files:
        print("Error: No image files found in metadata")
        return

    extrinsics = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics = predictions["intrinsic"]  # (S, 3, 3)

    annotated_dir = os.path.join(output_dir, "annotated_2d")
    os.makedirs(annotated_dir, exist_ok=True)

    tracks = tracking_data.get("tracks", {})
    all_track_ids = [int(tid) for tid in tracks.keys()]

    print(f"\n=== Generating Annotated 2D Images ===")

    # Build per-frame bounding box data
    num_frames = len(image_files)
    frame_bboxes = {i: [] for i in range(num_frames)}

    for tid_str, track_info in tracks.items():
        tid = int(tid_str)
        frames_list = track_info.get("frames", [])
        centers = track_info.get("centers", [])
        dimensions = track_info.get("dimensions", [])
        rotations = track_info.get("rotation_matrices", [])
        class_name = track_info.get("class_name", "object")

        if not dimensions or not rotations:
            continue

        for i, f_idx in enumerate(frames_list):
            if i >= len(dimensions) or i >= len(rotations) or i >= len(centers):
                continue
            if f_idx >= num_frames:
                continue

            frame_bboxes[f_idx].append({
                'track_id': tid,
                'center': np.array(centers[i]),
                'dimensions': np.array(dimensions[i]),
                'rotation': np.array(rotations[i]),
                'class_name': class_name,
            })

    # Process each frame
    for frame_idx, img_path in enumerate(image_files):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        K = intrinsics[frame_idx]  # (3, 3)
        ext = extrinsics[frame_idx]  # (3, 4)

        # Scale intrinsics if needed (model size vs original size)
        # Assume intrinsics are calibrated for the model output size
        if "depth" in predictions:
            model_h, model_w = predictions["depth"].shape[1:3]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h
            K_scaled = K.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
        else:
            K_scaled = K

        for bbox_data in frame_bboxes[frame_idx]:
            tid = bbox_data['track_id']
            center = bbox_data['center']
            dims = bbox_data['dimensions']
            rotation = bbox_data['rotation']
            class_name = bbox_data['class_name']

            # Compute 3D corners
            corners_3d = compute_bbox_corners(center, dims, rotation)

            # Transform to camera coordinates
            corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
            corners_cam = (ext @ corners_h.T).T  # (8, 3)

            # Check if in front of camera
            if np.any(corners_cam[:, 2] <= 0):
                continue

            # Project to image
            corners_2d_h = (K_scaled @ corners_cam.T).T
            corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

            # Check bounds
            if np.any(corners_2d < -orig_w) or np.any(corners_2d > 2 * orig_w):
                continue

            # Get color for this track
            color_rgb = get_track_color(tid, all_track_ids)
            color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])

            # Draw wireframe
            draw_bbox_wireframe(img, corners_2d, color_bgr)

            # Add label
            center_2d = np.mean(corners_2d, axis=0).astype(int)
            center_2d[0] = max(10, min(center_2d[0], orig_w - 100))
            center_2d[1] = max(20, min(center_2d[1], orig_h - 10))
            label = f"T{tid}: {class_name}"
            cv2.putText(img, label, tuple(center_2d),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

        # Save annotated image
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(annotated_dir, f"{frame_name}_tracked.png")
        cv2.imwrite(output_path, img)

    print(f"Saved annotated frames to {annotated_dir}")


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

    # Generate annotated 2D images only (no viser)
    python visualize_saved.py --output_dir ./outputs/scene --generate_annotated_2d
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
    parser.add_argument("--generate_annotated_2d", action="store_true",
                        help="Generate annotated 2D images with projected bboxes (requires tracking data)")
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

    # Generate annotated 2D images if requested
    if args.generate_annotated_2d:
        generate_annotated_2d(args.output_dir, predictions, tracking_data)
        return  # Exit after generating images

    # Visualize with viser
    visualize_with_viser(
        predictions,
        tracking_data=tracking_data,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
    )


if __name__ == "__main__":
    main()
