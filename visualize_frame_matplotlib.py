#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Matplotlib-based Frame Visualization

Visualizes a specific frame from batch inference results with:
- Point cloud (with configurable confidence threshold)
- 3D bounding boxes
- Track trajectories up to that frame
- Interactive 3D rotation
- Save to image file

Usage:
    # Visualize frame 50
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50

    # With custom confidence threshold
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --conf_threshold 0.5

    # Save to file without showing
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --save frame_50.png --no_show

    # Show all frames point cloud
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --all_frames
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class BoundingBox3D:
    """Minimal BoundingBox3D class for visualization."""

    def __init__(self, center, dimensions, rotation_matrix, class_name, confidence, instance_id):
        self.center = np.array(center)
        self.dimensions = np.array(dimensions)
        self.rotation_matrix = np.array(rotation_matrix)
        self.class_name = class_name
        self.confidence = confidence
        self.instance_id = instance_id
        self.track_id = None
        self.velocity = None

    def get_wireframe_edges(self):
        """Get corners and edges for wireframe visualization."""
        # Get 8 corners of the bounding box
        l, w, h = self.dimensions

        # Local corners (before rotation)
        local_corners = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])

        # Apply rotation and translation
        rotated_corners = (self.rotation_matrix @ local_corners.T).T
        corners = rotated_corners + self.center

        # Define edges connecting corners
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]

        return corners, edges


def load_predictions(result_dir: str) -> Dict:
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

    # Check images shape - could be (S, C, H, W) or (1, S, C, H, W)
    images_shape = predictions_np['images'].shape
    if len(images_shape) == 5:
        num_frames = images_shape[1]  # Has batch dimension
    else:
        num_frames = images_shape[0]  # No batch dimension
    print(f"Loaded predictions with {num_frames} frames")
    return predictions_np


def load_tracking_summary(result_dir: str) -> Tuple[List[List], List[int]]:
    """Load tracking_summary.json and reconstruct bounding boxes"""
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

    for track_id_str, track_data in tracking['tracks'].items():
        track_id = int(track_id_str)
        all_track_ids.append(track_id)

        frames = track_data['frames']
        centers = track_data['centers']
        dimensions = track_data['dimensions']
        rotation_matrices = track_data['rotation_matrices']
        velocities = track_data.get('velocities', [[0, 0, 0]] * len(frames))
        confidences = track_data.get('confidences', [1.0] * len(frames))

        for i, frame_idx in enumerate(frames):
            bbox = BoundingBox3D(
                center=np.array(centers[i]),
                dimensions=np.array(dimensions[i]),
                rotation_matrix=np.array(rotation_matrices[i]),
                class_name=track_data['class_name'],
                confidence=confidences[i],
                instance_id=track_id
            )
            bbox.track_id = track_id
            bbox.velocity = np.array(velocities[i]) if velocities[i] else None
            bounding_boxes[frame_idx].append(bbox)

    all_track_ids = sorted(set(all_track_ids))
    print(f"Reconstructed {len(all_track_ids)} tracks")

    return bounding_boxes, all_track_ids, tracking


def get_track_color(track_id: int, all_track_ids: List[int]) -> Tuple[float, float, float]:
    """Get consistent color for a track ID with good contrast"""
    np.random.seed(42)
    # Use gist_rainbow for more distinctive colors
    num_colors = max(20, len(all_track_ids))
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_colors))
    idx = all_track_ids.index(track_id) % len(colors)
    color = np.array(colors[idx][:3])

    # Ensure minimum brightness for visibility
    if np.mean(color) < 0.3:
        color = color + (0.3 - np.mean(color))

    return tuple(color)


def extract_point_cloud(predictions: Dict, frame_idx: int, conf_threshold: float = 0.0,
                       all_frames: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract point cloud for visualization.

    Args:
        predictions: Predictions dict with world_points, world_points_conf, images
        frame_idx: Frame index to visualize
        conf_threshold: Confidence threshold (0.0 = all points)
        all_frames: If True, show all frames; if False, only selected frame

    Returns:
        Tuple of (points, colors) as numpy arrays
    """
    world_points = predictions['world_points']  # (S, H, W, 3)
    confidence = predictions['world_points_conf']  # (S, H, W) - note: no trailing dimension
    images = predictions['images']  # Could be (S, C, H, W) or (1, S, C, H, W)

    S, H, W, _ = world_points.shape

    if frame_idx < 0 or frame_idx >= S:
        raise ValueError(f"Frame index {frame_idx} out of range [0, {S-1}]")

    # Determine which frames to include
    if all_frames:
        frame_mask = np.ones((S, H, W), dtype=bool)
    else:
        frame_mask = np.zeros((S, H, W), dtype=bool)
        frame_mask[frame_idx] = True

    # Flatten everything
    points_flat = world_points.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)
    frame_mask_flat = frame_mask.reshape(-1)

    # Extract colors from images
    # Handle both (S, C, H, W) and (1, S, C, H, W) formats
    if len(images.shape) == 5:
        images_no_batch = images[0]  # (S, C, H, W)
    else:
        images_no_batch = images  # Already (S, C, H, W)

    colors_list = []
    for s in range(S):
        img = images_no_batch[s]  # (C, H, W)
        img = img.transpose(1, 2, 0)  # (H, W, C)
        colors_list.append(img)
    colors = np.stack(colors_list, axis=0)  # (S, H, W, C)
    colors_flat = colors.reshape(-1, 3)

    # Apply confidence threshold and frame mask
    conf_mask = conf_flat >= conf_threshold
    combined_mask = conf_mask & frame_mask_flat

    points = points_flat[combined_mask]
    colors = colors_flat[combined_mask]

    print(f"Point cloud: {len(points)} points (conf >= {conf_threshold})")

    return points, colors


def get_track_trajectory(bounding_boxes: List[List], track_id: int, up_to_frame: int) -> np.ndarray:
    """Get trajectory points for a track up to specified frame"""
    centers = []
    for frame_idx in range(up_to_frame + 1):
        for bbox in bounding_boxes[frame_idx]:
            if bbox.track_id == track_id:
                centers.append(bbox.center)
                break
    return np.array(centers) if centers else np.zeros((0, 3))


def plot_frame_3d(
    predictions: Dict,
    bounding_boxes: List[List],
    all_track_ids: List[int],
    tracking_data: Dict,
    frame_idx: int,
    conf_threshold: float = 0.0,
    all_frames: bool = False,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 20,
    azim: float = 45,
    point_size: float = 0.5,
    point_alpha: float = 0.6,
    show_trajectories: bool = True,
    show_bboxes: bool = True,
    save_path: str = None,
    show: bool = True
):
    """
    Create 3D visualization of frame using matplotlib.

    Args:
        predictions: Predictions dict
        bounding_boxes: List of bounding boxes per frame
        all_track_ids: List of all track IDs
        tracking_data: Tracking summary dict
        frame_idx: Frame index to visualize
        conf_threshold: Confidence threshold for points
        all_frames: Show all frames point cloud
        figsize: Figure size (width, height)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        point_size: Point size for scatter plot
        show_trajectories: Show track trajectories
        show_bboxes: Show bounding boxes
        save_path: Path to save figure (None = don't save)
        show: Show interactive plot
    """
    # Extract point cloud
    points, colors = extract_point_cloud(predictions, frame_idx, conf_threshold, all_frames)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud with adaptive alpha
    if len(points) > 0:
        # Use adaptive alpha: lower for very dense clouds, higher for sparse ones
        adaptive_alpha = point_alpha if len(points) < 100000 else max(0.3, point_alpha * 0.7)
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            s=point_size,
            alpha=adaptive_alpha,
            marker='.'
        )
        print(f"Plotted {len(points)} points (alpha={adaptive_alpha:.2f})")

    # Plot bounding boxes for current frame
    if show_bboxes:
        frame_bboxes = bounding_boxes[frame_idx]
        print(f"Plotting {len(frame_bboxes)} bounding boxes")

        for bbox in frame_bboxes:
            if bbox.track_id is None or bbox.track_id < 0:
                continue

            color = get_track_color(bbox.track_id, all_track_ids)
            corners, edges = bbox.get_wireframe_edges()

            # Draw bbox edges with outline for better visibility
            for a, b in edges:
                # Draw black outline first for contrast
                ax.plot(
                    [corners[a, 0], corners[b, 0]],
                    [corners[a, 1], corners[b, 1]],
                    [corners[a, 2], corners[b, 2]],
                    color='black',
                    linewidth=3.5,
                    alpha=1.0,
                    zorder=1
                )
                # Draw colored bbox on top
                ax.plot(
                    [corners[a, 0], corners[b, 0]],
                    [corners[a, 1], corners[b, 1]],
                    [corners[a, 2], corners[b, 2]],
                    color=color,
                    linewidth=2.5,
                    alpha=1.0,
                    zorder=2
                )

            # Add label above bbox center with background box
            # ax.text(
            #     bbox.center[0],
            #     bbox.center[1],
            #     bbox.center[2] + bbox.dimensions[2] * 0.3,  # Position slightly above
            #     f"ID {bbox.track_id}\n{bbox.class_name}",
            #     color=color,
            #     fontsize=9,
            #     fontweight='bold',
            #     ha='center',
            #     bbox=dict(
            #         boxstyle='round,pad=0.3',
            #         facecolor='white',
            #         alpha=0.8,
            #         edgecolor=color,
            #         linewidth=1.5
            #     ),
            #     zorder=3
            # )

    # Plot track trajectories up to this frame
    if show_trajectories:
        # Get active tracks in current frame for legend filtering
        active_track_ids = set(bbox.track_id for bbox in frame_bboxes if bbox.track_id is not None and bbox.track_id >= 0)

        print(f"Plotting trajectories for {len(all_track_ids)} tracks")
        for track_id in all_track_ids:
            trajectory = get_track_trajectory(bounding_boxes, track_id, frame_idx)
            if len(trajectory) > 1:
                color = get_track_color(track_id, all_track_ids)
                # Only add legend label for tracks visible in current frame
                label = f"Track {track_id}" if track_id in active_track_ids else None
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color=color,
                    linewidth=2.0,
                    alpha=0.8,
                    marker='o',
                    markersize=3,
                    markevery=max(1, len(trajectory) // 10),  # Show markers at intervals
                    label=label
                )

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    frame_mode = "All Frames" if all_frames else f"Frame {frame_idx}"
    ax.set_title(f'VGGT 3D Visualization - {frame_mode}\n'
                 f'Conf >= {conf_threshold}, {len(frame_bboxes)} objects, '
                 f'{tracking_data["total_tracks"]} tracks')

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    if len(points) > 0:
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add legend if showing trajectories (limit to avoid clutter)
    if show_trajectories and len(all_track_ids) > 0:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            # Limit legend to 10 entries to avoid clutter
            if len(handles) > 10:
                extra_count = len(handles) - 10
                handles = handles[:10]
                labels = labels[:10]
                labels.append(f"...and {extra_count} more tracks")
                # Add a dummy handle for the extra text
                handles.append(plt.Line2D([0], [0], color='none'))
            ax.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save if requested
    if save_path:
        print(f"Saving figure to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    # Show if requested
    if show:
        print("\nShowing interactive 3D plot. Rotate with mouse, close window to exit.")
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Matplotlib-based Frame Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize frame 50 with interactive rotation
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50

    # With custom confidence threshold (0 = all points)
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --conf_threshold 0

    # Show all frames point cloud
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --all_frames

    # Save to file without showing
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --save output.png --no_show

    # Custom view angles
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --elev 30 --azim 60

    # Hide trajectories
    python visualize_frame_matplotlib.py --result_dir ./output/vggt/ --frame 50 --no_trajectories
        """
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to directory containing batch_inference_video.py outputs"
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Frame index to visualize (0-indexed)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for point cloud (default: 0.0 = all points)"
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Show point cloud from all frames instead of just selected frame"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 10],
        help="Figure size in inches (width height) (default: 12 10)"
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20,
        help="Elevation angle for 3D view (default: 20)"
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=45,
        help="Azimuth angle for 3D view (default: 45)"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.5,
        help="Point size for scatter plot (default: 0.5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Point cloud alpha/transparency (default: 0.6)"
    )
    parser.add_argument(
        "--no_trajectories",
        action="store_true",
        help="Don't show track trajectories"
    )
    parser.add_argument(
        "--no_bboxes",
        action="store_true",
        help="Don't show bounding boxes"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save figure (e.g., frame_50.png)"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't show interactive plot (useful with --save)"
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
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"VGGT Frame Visualization")
    print(f"{'='*60}")
    print(f"Loading results from: {args.result_dir}\n")

    # Load data
    predictions = load_predictions(args.result_dir)
    bounding_boxes, all_track_ids, tracking_data = load_tracking_summary(args.result_dir)

    # Validate frame index
    images_shape = predictions['images'].shape
    if len(images_shape) == 5:
        num_frames = images_shape[1]  # (1, S, C, H, W)
    else:
        num_frames = images_shape[0]  # (S, C, H, W)

    if args.frame < 0 or args.frame >= num_frames:
        print(f"Error: Frame index {args.frame} out of range [0, {num_frames-1}]")
        print(f"\nNote: predictions.pt only contains {num_frames} frames,")
        print(f"but tracking_summary.json has {tracking_data['total_frames']} frames.")
        print(f"This can happen if batch_inference_video.py was run with --num_images={num_frames}")
        print(f"or a limited frame range.")
        print(f"\nAvailable frames in predictions.pt: 0 to {num_frames-1}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Creating visualization...")
    print(f"{'='*60}\n")

    # Create visualization
    plot_frame_3d(
        predictions=predictions,
        bounding_boxes=bounding_boxes,
        all_track_ids=all_track_ids,
        tracking_data=tracking_data,
        frame_idx=args.frame,
        conf_threshold=args.conf_threshold,
        all_frames=args.all_frames,
        figsize=tuple(args.figsize),
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
        point_alpha=args.alpha,
        show_trajectories=not args.no_trajectories,
        show_bboxes=not args.no_bboxes,
        save_path=args.save,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()
