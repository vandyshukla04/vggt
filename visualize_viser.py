#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive 3D Visualization of VGGT Batch Inference Results

Loads pre-computed results from batch_inference_video.py and displays them
in an interactive viser 3D viewer with controls for:
- Toggle tracks on/off
- Toggle bounding boxes
- Toggle point cloud
- Frame-by-frame or cumulative point cloud
- Camera trajectory visualization

Usage:
    python visualize_viser.py --result_dir ./output/vggt/ --port 8080
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import viser

# Open3D is optional - only needed for PLY loading
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. Point cloud visualization will be disabled.")
    print("Install with: pip install open3d")

if TYPE_CHECKING:
    from open3d.geometry import PointCloud


def load_tracking_summary(result_dir: str) -> Dict:
    """Load tracking_summary.json"""
    tracking_path = os.path.join(result_dir, "tracking_summary.json")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"tracking_summary.json not found in {result_dir}")

    with open(tracking_path, 'r') as f:
        return json.load(f)


def load_cameras(result_dir: str) -> Dict:
    """Load cameras.json"""
    cameras_path = os.path.join(result_dir, "cameras.json")
    if not os.path.exists(cameras_path):
        print(f"Warning: cameras.json not found in {result_dir}")
        return {}

    with open(cameras_path, 'r') as f:
        return json.load(f)


def load_point_cloud(result_dir: str) -> Optional['PointCloud']:
    """Load point_cloud.ply"""
    if not HAS_OPEN3D:
        return None

    ply_path = os.path.join(result_dir, "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Warning: point_cloud.ply not found in {result_dir}")
        return None

    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded point cloud with {len(pcd.points)} points")
    return pcd


def load_depth_maps(result_dir: str) -> Optional[np.ndarray]:
    """Load depth_maps.npz for per-frame point clouds"""
    depth_path = os.path.join(result_dir, "depth_maps.npz")
    if not os.path.exists(depth_path):
        print(f"Warning: depth_maps.npz not found in {result_dir}")
        return None

    print(f"Loading depth maps from {depth_path}...")
    depth_data = np.load(depth_path)
    return depth_data


def get_bbox_corners(center: List[float], dimensions: List[float], rotation: Optional[List[float]] = None) -> np.ndarray:
    """
    Get 8 corners of a 3D bounding box.

    Args:
        center: [x, y, z]
        dimensions: [length, width, height]
        rotation: Optional rotation (not used for now, assumes axis-aligned)

    Returns:
        corners: (8, 3) array of corner positions
    """
    cx, cy, cz = center
    l, w, h = dimensions

    # 8 corners of the box (axis-aligned)
    corners = np.array([
        [cx - l/2, cy - w/2, cz - h/2],
        [cx + l/2, cy - w/2, cz - h/2],
        [cx + l/2, cy + w/2, cz - h/2],
        [cx - l/2, cy + w/2, cz - h/2],
        [cx - l/2, cy - w/2, cz + h/2],
        [cx + l/2, cy - w/2, cz + h/2],
        [cx + l/2, cy + w/2, cz + h/2],
        [cx - l/2, cy + w/2, cz + h/2],
    ])

    return corners


def get_bbox_edges() -> List[tuple]:
    """Get edges connecting bbox corners"""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    return edges


def create_color_map(num_tracks: int) -> Dict[int, tuple]:
    """Create distinct colors for different tracks"""
    np.random.seed(42)
    colors = {}
    for i in range(num_tracks):
        colors[i] = tuple(np.random.randint(50, 255, 3).astype(float) / 255.0)
    return colors


def visualize_results(
    result_dir: str,
    port: int = 8080,
    share: bool = False
):
    """
    Launch interactive viser visualization of batch inference results.

    Args:
        result_dir: Path to directory containing VGGT output files
        port: Port for viser server
        share: Whether to create a shareable link
    """
    print(f"\n{'='*60}")
    print(f"VGGT Results Visualization")
    print(f"{'='*60}")
    print(f"Loading results from: {result_dir}\n")

    # Load all data
    tracking = load_tracking_summary(result_dir)
    cameras = load_cameras(result_dir)
    point_cloud = load_point_cloud(result_dir)
    depth_maps = load_depth_maps(result_dir)

    # Extract tracking info
    tracks = tracking.get('tracks', {})
    total_frames = tracking.get('total_frames', 0)

    print(f"\nLoaded:")
    print(f"  - {len(tracks)} tracks")
    print(f"  - {total_frames} frames")
    if point_cloud:
        print(f"  - Point cloud with {len(point_cloud.points)} points")
    if cameras:
        print(f"  - {len(cameras.get('frames', {}))} camera poses")

    # Create color map for tracks
    color_map = create_color_map(len(tracks))

    # Initialize viser server
    print(f"\nStarting viser server on port {port}...")
    server = viser.ViserServer(port=port, verbose=False)

    if share:
        print(f"Shareable link: {server.get_share_url()}")

    print(f"\n{'='*60}")
    print(f"Visualization ready! Open your browser to:")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}\n")

    # Add GUI controls
    with server.gui.add_folder("Display Options"):
        show_point_cloud = server.gui.add_checkbox("Show Point Cloud", initial_value=True)
        show_bboxes = server.gui.add_checkbox("Show Bounding Boxes", initial_value=True)
        show_tracks = server.gui.add_checkbox("Show Track Trajectories", initial_value=True)
        show_cameras = server.gui.add_checkbox("Show Camera Trajectory", initial_value=True)

        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, total_frames - 1),
            step=1,
            initial_value=0
        )

        point_cloud_opacity = server.gui.add_slider(
            "Point Cloud Opacity",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=0.5
        )

    with server.gui.add_folder("Track Filters"):
        track_checkboxes = {}
        for track_id, track_data in tracks.items():
            class_name = track_data.get('class_name', 'object')
            track_checkboxes[track_id] = server.gui.add_checkbox(
                f"Track {track_id} ({class_name})",
                initial_value=True
            )

    # Visualization state
    current_frame = [0]
    viz_handles = {'bboxes': [], 'trajectories': [], 'point_cloud': None, 'cameras': []}

    def clear_visualization():
        """Clear all visualization handles"""
        for handle in viz_handles['bboxes']:
            handle.remove()
        for handle in viz_handles['trajectories']:
            handle.remove()
        for handle in viz_handles['cameras']:
            handle.remove()
        if viz_handles['point_cloud'] is not None:
            viz_handles['point_cloud'].remove()

        viz_handles['bboxes'] = []
        viz_handles['trajectories'] = []
        viz_handles['cameras'] = []
        viz_handles['point_cloud'] = None

    def update_visualization():
        """Update the visualization based on current settings"""
        clear_visualization()

        frame_idx = current_frame[0]

        # Show point cloud
        if show_point_cloud.value and point_cloud is not None:
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)

            # Subsample if too large (for performance)
            if len(points) > 100000:
                indices = np.random.choice(len(points), 100000, replace=False)
                points = points[indices]
                colors = colors[indices]

            handle = server.scene.add_point_cloud(
                name="/point_cloud",
                points=points,
                colors=colors,
                point_size=0.02,
                # Set opacity through colors alpha channel
            )
            viz_handles['point_cloud'] = handle

        # Show bounding boxes for current frame
        if show_bboxes.value:
            for track_id, track_data in tracks.items():
                if not track_checkboxes[track_id].value:
                    continue

                frame_data_dict = track_data.get('frame_data', {})

                # Check if track exists at this frame
                if str(frame_idx) in frame_data_dict:
                    frame_data = frame_data_dict[str(frame_idx)]
                    center = frame_data['center']
                    dimensions = frame_data['dimensions']

                    # Get corners
                    corners = get_bbox_corners(center, dimensions)

                    # Draw bbox edges
                    color = color_map[int(track_id)]
                    edges = get_bbox_edges()

                    for i, (start_idx, end_idx) in enumerate(edges):
                        handle = server.scene.add_spline_catmull_rom(
                            name=f"/bbox_{track_id}_{i}",
                            positions=np.array([corners[start_idx], corners[end_idx]]),
                            color=color,
                            line_width=3.0,
                            segments=1
                        )
                        viz_handles['bboxes'].append(handle)

                    # Add track ID label
                    class_name = track_data.get('class_name', 'object')
                    label_handle = server.scene.add_label(
                        name=f"/label_{track_id}",
                        text=f"Track {track_id}\n{class_name}",
                        position=center,

                    )
                    viz_handles['bboxes'].append(label_handle)

        # Show track trajectories
        if show_tracks.value:
            for track_id, track_data in tracks.items():
                if not track_checkboxes[track_id].value:
                    continue

                frame_data_dict = track_data.get('frame_data', {})

                # Collect all centers up to current frame
                centers = []
                for f in range(frame_idx + 1):
                    if str(f) in frame_data_dict:
                        centers.append(frame_data_dict[str(f)]['center'])

                if len(centers) > 1:
                    color = color_map[int(track_id)]
                    handle = server.scene.add_spline_catmull_rom(
                        name=f"/trajectory_{track_id}",
                        positions=np.array(centers),
                        color=color,
                        line_width=2.0,
                        segments=max(1, len(centers) - 1)
                    )
                    viz_handles['trajectories'].append(handle)

        # Show camera trajectory
        if show_cameras.value and cameras:
            camera_frames = cameras.get('frames', {})
            camera_positions = []

            for f in range(frame_idx + 1):
                frame_key = f"frame_{f:06d}"
                if frame_key in camera_frames:
                    cam_data = camera_frames[frame_key]
                    # Camera position is typically stored in extrinsics or pose
                    if 'position' in cam_data:
                        camera_positions.append(cam_data['position'])
                    elif 'extrinsic' in cam_data:
                        # Extract position from 4x4 extrinsic matrix
                        extrinsic = np.array(cam_data['extrinsic'])
                        position = extrinsic[:3, 3]
                        camera_positions.append(position.tolist())

            if len(camera_positions) > 1:
                handle = server.scene.add_spline_catmull_rom(
                    name="/camera_trajectory",
                    positions=np.array(camera_positions),
                    color=(0.2, 0.8, 0.2),  # Green
                    line_width=3.0,
                    segments=max(1, len(camera_positions) - 1)
                )
                viz_handles['cameras'].append(handle)

            # Add current camera pose indicator
            if len(camera_positions) > 0:
                current_pos = camera_positions[-1]
                handle = server.scene.add_icosphere(
                    name="/camera_current",
                    radius=0.3,
                    position=current_pos,
                    color=(0.2, 1.0, 0.2)
                )
                viz_handles['cameras'].append(handle)

    # Set up callbacks
    @frame_slider.on_update
    def _(_) -> None:
        current_frame[0] = frame_slider.value
        update_visualization()

    @show_point_cloud.on_update
    def _(_) -> None:
        update_visualization()

    @show_bboxes.on_update
    def _(_) -> None:
        update_visualization()

    @show_tracks.on_update
    def _(_) -> None:
        update_visualization()

    @show_cameras.on_update
    def _(_) -> None:
        update_visualization()

    # Set up track checkbox callbacks
    for track_id, checkbox in track_checkboxes.items():
        @checkbox.on_update
        def _(_) -> None:
            update_visualization()

    # Initial visualization
    update_visualization()

    # Keep server running
    print("Visualization is running. Press Ctrl+C to exit.")
    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Visualization of VGGT Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize results from batch_inference_video.py
    python visualize_viser.py --result_dir ./output/vggt/

    # Use custom port
    python visualize_viser.py --result_dir ./output/vggt/ --port 8888

    # Create shareable link
    python visualize_viser.py --result_dir ./output/vggt/ --share
        """
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to directory containing VGGT output files "
             "(tracking_summary.json, point_cloud.ply, etc.)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server (default: 8080)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link for remote viewing"
    )

    args = parser.parse_args()

    # Validate result directory
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    # Check for required files
    required_files = ["tracking_summary.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.result_dir, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required files in {args.result_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)

    # Launch visualization
    visualize_results(
        result_dir=args.result_dir,
        port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
