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
):
    """Launch interactive viser visualization.

    Args:
        result_dir: Directory containing ground_point_cloud.ply and smoothed_tracklets.json
        port: Viser server port
        point_size: Point size for rendering
    """
    print(f"\n{'='*60}")
    print("Ground Tracklets Visualization")
    print(f"{'='*60}")
    print(f"Loading from: {result_dir}\n")

    # Always load ground-only PLY for the point cloud so track filters
    # can properly hide tracklets (the combined PLY has tracklet colors baked in).
    # Fall back to combined PLY if ground-only doesn't exist.
    ground_ply = os.path.join(result_dir, "ground_point_cloud.ply")
    combined_ply = os.path.join(result_dir, "ground_with_tracklets.ply")
    if os.path.exists(ground_ply):
        ply_path = ground_ply
    else:
        ply_path = combined_ply

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

            # Add start/end markers
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

            # Add label at midpoint
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground point cloud with smoothed tracklets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/
    python visualize_ground_tracklets.py --result_dir ./output/vggt/ground_tracklets/ --port 8888
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

    visualize(
        result_dir=args.result_dir,
        port=args.port,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
