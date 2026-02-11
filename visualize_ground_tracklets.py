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

Usage:
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ --port 8888
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
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


def visualize(
    result_dir: str,
    port: int = 8080,
    point_size: float = 0.01,
    use_combined: bool = True,
):
    """Launch interactive viser visualization.

    Args:
        result_dir: Directory containing ground_with_tracklets.ply and smoothed_tracklets.json
        port: Viser server port
        point_size: Point size for rendering
        use_combined: If True, load combined PLY; if False, load ground-only PLY
    """
    print(f"\n{'='*60}")
    print("Ground Tracklets Visualization")
    print(f"{'='*60}")
    print(f"Loading from: {result_dir}\n")

    # Load data
    if use_combined:
        ply_path = os.path.join(result_dir, "ground_with_tracklets.ply")
    else:
        ply_path = os.path.join(result_dir, "ground_point_cloud.ply")

    pcd_data = load_point_cloud(ply_path)
    tracklets_data = load_smoothed_tracklets(
        os.path.join(result_dir, "smoothed_tracklets.json")
    )

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
        """Update point cloud display."""
        if viz_handles["point_cloud"] is not None:
            viz_handles["point_cloud"].remove()
            viz_handles["point_cloud"] = None

        if not gui_show_ground.value:
            return

        # Subsample for performance
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
        """Update tracklet spline display."""
        for handle in viz_handles["tracklets"]:
            handle.remove()
        viz_handles["tracklets"] = []

        if not gui_show_tracklets.value or tracklets_data is None:
            return

        tracks = tracklets_data.get("tracks", {})
        sorted_ids = sorted(tracks.keys(), key=lambda x: int(x))
        line_width = gui_tracklet_width.value

        for idx, track_id in enumerate(sorted_ids):
            # Check per-track toggle
            if track_id in track_checkboxes and not track_checkboxes[track_id].value:
                continue

            track = tracks[track_id]
            centers = np.array(track.get("smoothed_centers", []))

            if len(centers) < 2:
                continue

            # Center the tracklet
            centers_centered = centers - scene_center

            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            # Render as Catmull-Rom spline
            handle = server.scene.add_spline_catmull_rom(
                name=f"/tracklet_{track_id}",
                positions=centers_centered.astype(np.float32),
                color=color,
                line_width=line_width,
                segments=max(1, len(centers_centered) * 4),
            )
            viz_handles["tracklets"].append(handle)

            # Add start marker
            start_handle = server.scene.add_icosphere(
                name=f"/tracklet_{track_id}_start",
                radius=0.05,
                position=centers_centered[0],
                color=color,
            )
            viz_handles["tracklets"].append(start_handle)

            # Add end marker (different size)
            end_handle = server.scene.add_icosphere(
                name=f"/tracklet_{track_id}_end",
                radius=0.03,
                position=centers_centered[-1],
                color=color,
            )
            viz_handles["tracklets"].append(end_handle)

            # Add label at midpoint
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

    for tid, checkbox in track_checkboxes.items():
        @checkbox.on_update
        def _(_):
            update_tracklets()

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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground point cloud with smoothed tracklets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ --port 8888
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ --ground_only
        """,
    )

    parser.add_argument(
        "--result_dir", type=str, required=True,
        help="Directory containing ground_with_tracklets.ply and smoothed_tracklets.json",
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
        "--ground_only", action="store_true",
        help="Load ground-only PLY instead of combined PLY",
    )

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Directory not found: {args.result_dir}")
        sys.exit(1)

    visualize(
        result_dir=args.result_dir,
        port=args.port,
        point_size=args.point_size,
        use_combined=not args.ground_only,
    )


if __name__ == "__main__":
    main()
