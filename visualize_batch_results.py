#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize Batch Inference Results with Trajectory Tracking

Loads predictions.pt from batch_inference_video.py output and visualizes using viser.
Enhanced with trajectory visualization to show tracking results up to each frame and
per-frame bounding box information.

Features:
- Interactive frame slider to see tracking progression
- Trajectory lines showing object paths up to current frame
- Per-frame bounding box visualization
- Track statistics and active detections per frame

Usage:
    python visualize_batch_results.py --result_dir ./output/vggt/ --port 8080
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import visualization function
from demo_viser_tracking import viser_wrapper_with_tracking, BoundingBox3D, get_track_color, closed_form_inverse_se3
import viser.transforms as viser_tf

try:
    import viser
    HAS_VISER = True
except ImportError:
    HAS_VISER = False
    print("Warning: viser not installed. Install with: pip install viser")


def load_predictions(result_dir: str):
    """Load predictions.pt file and convert to numpy"""
    pred_path = os.path.join(result_dir, "predictions.pt")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.pt not found in {result_dir}")

    print(f"Loading predictions from {pred_path}...")
    predictions = torch.load(pred_path, map_location='cpu')

    # Convert all tensors to numpy arrays and squeeze batch dimension
    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            arr = value.cpu().numpy()
            # Squeeze out batch dimension if present (first dim = 1)
            if arr.ndim > 0 and arr.shape[0] == 1 and key != 'images':
                arr = arr.squeeze(0)
            predictions_np[key] = arr
        else:
            predictions_np[key] = value

    print(f"Loaded predictions with {predictions_np['images'].shape[0]} frames")
    return predictions_np


def viser_wrapper_with_trajectories(
    pred_dict: dict,
    bounding_boxes,
    all_track_ids,
    trajectories: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
):
    """Enhanced visualization with trajectory tracking up to current frame."""
    if not HAS_VISER:
        raise ImportError("viser is required. Install with: pip install viser")

    import time
    import threading
    from demo_viser_tracking import unproject_depth_map_to_point_map

    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]
    world_points_map = pred_dict["world_points"]
    conf_map = pred_dict["world_points_conf"]
    depth_map = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsics_cam = pred_dict["extrinsic"]
    intrinsics_cam = pred_dict["intrinsic"]

    # Compute world points
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Convert images
    colors = images.transpose(0, 2, 3, 1)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    frame_indices = np.repeat(np.arange(S), H * W)

    # Build GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_show_bboxes = server.gui.add_checkbox("Show Bounding Boxes", initial_value=True)
    gui_show_trajectories = server.gui.add_checkbox("Show Trajectories", initial_value=True)
    gui_show_all_bboxes = server.gui.add_checkbox("Show All Frame Bboxes", initial_value=False)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_bbox_thickness = server.gui.add_slider(
        "Bbox Line Thickness", min=0.5, max=5.0, step=0.5, initial_value=2.0
    )
    gui_trajectory_thickness = server.gui.add_slider(
        "Trajectory Line Thickness", min=0.5, max=5.0, step=0.5, initial_value=2.0
    )
    gui_frame_slider = server.gui.add_slider(
        "Frame", min=0, max=S-1, step=1, initial_value=S-1
    )
    server.gui.add_markdown("*Slide to see tracking progression*")

    # Add frame info display
    gui_frame_info = server.gui.add_markdown("")

    # Create point cloud
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames = []
    frustums = []
    bbox_handles = []
    trajectory_handles = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray):
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum, frame):
            @frustum.on_click
            def _(_):
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
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

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def visualize_trajectories():
        """Visualize tracking trajectories up to selected frame."""
        # Remove existing trajectories
        for th in trajectory_handles:
            try:
                th.remove()
            except:
                pass
        trajectory_handles.clear()

        if not gui_show_trajectories.value:
            return

        selected_frame = int(gui_frame_slider.value)
        line_thickness = gui_trajectory_thickness.value

        # For each track, draw trajectory from first frame to selected frame
        for track_id, traj_data in trajectories.items():
            track_frames = traj_data['frames']
            track_centers = traj_data['centers']

            # Filter to frames up to selected_frame
            valid_indices = [i for i, f in enumerate(track_frames) if f <= selected_frame]
            if len(valid_indices) < 2:
                continue

            # Get centers for these frames
            traj_points = np.array([track_centers[i] for i in valid_indices])
            traj_points_centered = traj_points - scene_center

            # Get track color
            color = get_track_color(track_id, all_track_ids)

            # Draw trajectory line
            try:
                line = server.scene.add_spline_catmull_rom(
                    f"trajectory_{track_id}",
                    positions=traj_points_centered,
                    color=color,
                    line_width=line_thickness,
                )
                trajectory_handles.append(line)

                # Add markers at each detection point
                for i, point in enumerate(traj_points_centered):
                    marker = server.scene.add_icosphere(
                        f"trajectory_marker_{track_id}_{i}",
                        radius=0.05,
                        color=color,
                        position=point,
                    )
                    trajectory_handles.append(marker)
            except Exception as e:
                pass

    def visualize_bboxes():
        """Visualize bounding boxes for selected frame."""
        for bh in bbox_handles:
            try:
                bh.remove()
            except:
                pass
        bbox_handles.clear()

        if not gui_show_bboxes.value:
            return

        selected_frame = int(gui_frame_slider.value)
        show_all = gui_show_all_bboxes.value
        line_thickness = gui_bbox_thickness.value

        for frame_idx, frame_bboxes in enumerate(bounding_boxes):
            if not show_all and frame_idx != selected_frame:
                continue

            for bbox in frame_bboxes:
                if bbox.track_id is None or bbox.track_id < 0:
                    continue

                color = get_track_color(bbox.track_id, all_track_ids)
                corners, edges = bbox.get_wireframe_edges()
                corners_centered = corners - scene_center

                for edge_idx, (a, b) in enumerate(edges):
                    try:
                        line = server.scene.add_spline_catmull_rom(
                            f"bbox_{frame_idx}_{bbox.track_id}_{edge_idx}",
                            positions=np.array([corners_centered[a], corners_centered[b]]),
                            color=color,
                            line_width=line_thickness,
                        )
                        bbox_handles.append(line)
                    except:
                        pass

    def update_frame_info():
        """Update frame information display."""
        selected_frame = int(gui_frame_slider.value)

        # Count active tracks at this frame
        active_tracks = set()
        tracks_at_frame = []
        if selected_frame < len(bounding_boxes):
            for bbox in bounding_boxes[selected_frame]:
                if bbox.track_id is not None and bbox.track_id >= 0:
                    active_tracks.add(bbox.track_id)
                    tracks_at_frame.append(f"Track {bbox.track_id} ({bbox.class_name})")

        # Count total tracks seen up to this frame
        tracks_up_to_frame = set()
        for track_id, traj_data in trajectories.items():
            if traj_data['first_frame'] <= selected_frame:
                tracks_up_to_frame.add(track_id)

        info_text = f"**Frame {selected_frame}/{S-1}**\n"
        info_text += f"- Active tracks: {len(active_tracks)}\n"
        info_text += f"- Total tracks seen: {len(tracks_up_to_frame)}\n"
        if tracks_at_frame:
            info_text += f"- Detections: {', '.join(tracks_at_frame[:5])}"
            if len(tracks_at_frame) > 5:
                info_text += f" (+{len(tracks_at_frame)-5} more)"

        gui_frame_info.content = info_text

    def update_point_cloud():
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        selected_frame = int(gui_frame_slider.value)
        frame_mask = frame_indices == selected_frame

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    def update_all():
        update_point_cloud()
        visualize_bboxes()
        visualize_trajectories()
        update_frame_info()

    @gui_points_conf.on_update
    def _(_):
        update_point_cloud()

    @gui_frame_slider.on_update
    def _(_):
        update_all()

    @gui_show_frames.on_update
    def _(_):
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @gui_show_bboxes.on_update
    def _(_):
        visualize_bboxes()

    @gui_show_trajectories.on_update
    def _(_):
        visualize_trajectories()

    @gui_show_all_bboxes.on_update
    def _(_):
        visualize_bboxes()

    @gui_bbox_thickness.on_update
    def _(_):
        visualize_bboxes()

    @gui_trajectory_thickness.on_update
    def _(_):
        visualize_trajectories()

    visualize_frames(cam_to_world, images)
    update_all()

    print("Viser server started with trajectory visualization")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


def load_tracking_summary(result_dir: str):
    """Load tracking_summary.json and reconstruct bounding boxes and trajectories"""
    tracking_path = os.path.join(result_dir, "tracking_summary.json")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"tracking_summary.json not found in {result_dir}")

    with open(tracking_path, 'r') as f:
        tracking = json.load(f)

    print(f"Loading tracking data: {tracking['total_tracks']} tracks, {tracking['total_frames']} frames")

    # Reconstruct bounding boxes
    num_frames = tracking['total_frames']
    bounding_boxes = [[] for _ in range(num_frames)]
    all_track_ids = []

    # Store trajectories: {track_id: {'frames': [...], 'centers': [...], 'class_name': str}}
    trajectories = {}

    for track_id_str, track_data in tracking['tracks'].items():
        track_id = int(track_id_str)
        all_track_ids.append(track_id)

        frames = track_data['frames']
        centers = track_data['centers']
        dimensions = track_data['dimensions']
        rotation_matrices = track_data['rotation_matrices']
        velocities = track_data.get('velocities', [[0, 0, 0]] * len(frames))
        confidences = track_data.get('confidences', [1.0] * len(frames))

        # Store trajectory data
        trajectories[track_id] = {
            'frames': frames,
            'centers': centers,
            'class_name': track_data['class_name'],
            'first_frame': min(frames),
            'last_frame': max(frames),
        }

        for i, frame_idx in enumerate(frames):
            bbox = BoundingBox3D(
                center=np.array(centers[i]),
                dimensions=np.array(dimensions[i]),
                rotation_matrix=np.array(rotation_matrices[i]),
                class_name=track_data['class_name'],
                confidence=confidences[i],
                instance_id=track_id  # Use track_id as instance_id
            )
            # Set track_id and velocity as attributes
            bbox.track_id = track_id
            bbox.velocity = np.array(velocities[i]) if velocities[i] else None
            bounding_boxes[frame_idx].append(bbox)

    all_track_ids = sorted(set(all_track_ids))
    print(f"Reconstructed {len(all_track_ids)} tracks")

    return bounding_boxes, all_track_ids, trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Batch Inference Results with Trajectory Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize results from batch_inference_video.py
    python visualize_batch_results.py --result_dir ./output/vggt/

    # Use custom port
    python visualize_batch_results.py --result_dir ./output/vggt/ --port 8888

    # Use point map instead of depth
    python visualize_batch_results.py --result_dir ./output/vggt/ --use_point_map

Features:
    - Use the "Frame" slider to see tracking results up to each frame
    - Toggle "Show Trajectories" to see object paths over time
    - Toggle "Show Bounding Boxes" to see detections at current frame
    - Frame info shows active tracks and total tracks seen up to current frame
        """
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to directory containing batch_inference_video.py outputs "
             "(predictions.pt, tracking_summary.json, etc.)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server (default: 8080)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=50.0,
        help="Initial confidence threshold percentile (default: 50.0)"
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        default=True,
        help="Use predicted point map instead of unprojected depth (default: True)"
    )
    parser.add_argument(
        "--background_mode",
        action="store_true",
        help="Run in background mode (no blocking)"
    )

    args = parser.parse_args()

    # Validate result directory
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    # Check for required files
    required_files = ["predictions.pt", "tracking_summary.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.result_dir, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required files in {args.result_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nThese files are generated by batch_inference_video.py")
        sys.exit(1)

    if not HAS_VISER:
        print("Error: viser is required for visualization")
        print("Install with: pip install viser")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"VGGT Batch Results Visualization")
    print(f"{'='*60}")
    print(f"Loading results from: {args.result_dir}\n")

    # Load predictions
    predictions = load_predictions(args.result_dir)

    # Load tracking data with trajectories
    bounding_boxes, all_track_ids, trajectories = load_tracking_summary(args.result_dir)

    print(f"\n{'='*60}")
    print(f"Starting interactive visualization with trajectory tracking...")
    print(f"Open your browser to: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    # Launch enhanced visualization with trajectory support
    viser_wrapper_with_trajectories(
        predictions,
        bounding_boxes,
        all_track_ids,
        trajectories,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
    )


if __name__ == "__main__":
    main()
