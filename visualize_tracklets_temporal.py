#!/usr/bin/env python3
"""
Temporal 3D Tracklet Visualization

Elegant matplotlib visualization of 3D tracklets with temporal color encoding.
Tracklets are rendered with glow effects on a dark background, colored by time
using perceptually uniform colormaps (plasma/inferno/magma).

Modes:
  multipanel  - 2x3 grid showing tracklet growth at 6 time snapshots
  single      - Single large plot with all tracklets, linewidth/alpha gradient

Usage:
    # Multi-panel growth montage (default)
    python visualize_tracklets_temporal.py \\
        --tracklets ./output/.../ground_tracklets/smoothed_tracklets.json \\
        --mode multipanel --output tracklets_growth.png

    # Single elegant panel
    python visualize_tracklets_temporal.py \\
        --tracklets ./output/.../ground_tracklets/smoothed_tracklets.json \\
        --mode single --output tracklets_single.png

    # Custom colormap and view angle
    python visualize_tracklets_temporal.py \\
        --tracklets ./output/.../ground_tracklets/smoothed_tracklets.json \\
        --cmap inferno --elev 30 --azim -45
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tracklets(filepath: str) -> dict:
    """Load smoothed_tracklets.json and return the parsed data."""
    with open(filepath, "r") as f:
        data = json.load(f)
    n_tracks = data["total_tracks"]
    n_frames = data["total_frames"]
    print(f"Loaded {n_tracks} tracklets over {n_frames} frames from {filepath}")
    return data


def center_tracks(tracks: dict):
    """Shift all tracklet coordinates so the centroid is at the origin.

    Returns the centroid used for centering (for reference).
    """
    all_pts = []
    for track in tracks.values():
        all_pts.extend(track["smoothed_centers"])
    centroid = np.mean(all_pts, axis=0)

    for track in tracks.values():
        shifted = (np.array(track["smoothed_centers"]) - centroid).tolist()
        track["smoothed_centers"] = shifted
    return centroid


def compute_global_bounds(tracks: dict):
    """Compute per-axis tight limits from all tracklet points.

    Returns (mins, maxs) arrays with 5% padding per axis.
    """
    all_pts = []
    for track in tracks.values():
        all_pts.extend(track["smoothed_centers"])
    all_pts = np.array(all_pts)

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ranges = maxs - mins
    pad = ranges * 0.12  # 12% padding each side
    return mins - pad, maxs + pad


# ---------------------------------------------------------------------------
# Axes Styling
# ---------------------------------------------------------------------------

def style_axes_dark(ax, show_labels=True):
    """Apply elegant dark styling to a 3D axes."""
    # Transparent panes with nearly invisible edges
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.line.set_color((1, 1, 1, 0.04))
        axis.line.set_linewidth(0.3)

    # Subtle ticks
    for axis_name in ("x", "y", "z"):
        ax.tick_params(
            axis=axis_name, colors=(1, 1, 1, 0.25), labelsize=7, pad=0,
            length=2, width=0.4,
        )

    if show_labels:
        ax.set_xlabel("X (m)", color=(1, 1, 1, 0.4), fontsize=8, labelpad=1)
        ax.set_ylabel("Y (m)", color=(1, 1, 1, 0.4), fontsize=8, labelpad=1)
        ax.set_zlabel("Z (m)", color=(1, 1, 1, 0.4), fontsize=8, labelpad=1)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Fewer ticks for cleanliness
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax.grid(True, alpha=0.04, color="white", linewidth=0.3)


# ---------------------------------------------------------------------------
# Core Drawing
# ---------------------------------------------------------------------------

def draw_tracklets_on_axes(
    ax,
    tracks: dict,
    up_to_frame: int,
    total_frames: int,
    cmap,
    norm,
    show_glow: bool = True,
    show_markers: bool = True,
    linewidth_base: float = 1.8,
    grow_width: bool = False,
):
    """
    Draw all tracklets up to *up_to_frame* on the given 3D axes.

    Parameters
    ----------
    grow_width : bool
        If True, linewidth and alpha ramp up with frame (for single-panel mode).
    """
    for _tid, track in sorted(tracks.items(), key=lambda x: int(x[0])):
        frames = np.array(track["frames"])
        centers = np.array(track["smoothed_centers"])

        # Trim to frames <= up_to_frame
        mask = frames <= up_to_frame
        t_frames = frames[mask]
        t_centers = centers[mask]

        if len(t_centers) < 2:
            # Single point — just a marker
            if len(t_centers) == 1 and show_markers:
                c = cmap(norm(t_frames[0]))
                ax.scatter(
                    t_centers[0, 0], t_centers[0, 1], t_centers[0, 2],
                    color=c, s=18, zorder=5, alpha=0.8,
                )
            continue

        # Build segments (N-1, 2, 3)
        segments = np.stack([t_centers[:-1], t_centers[1:]], axis=1)
        seg_t = (t_frames[:-1] + t_frames[1:]) / 2.0  # midpoint frame
        seg_colors = cmap(norm(seg_t))

        if grow_width:
            # Linewidth ramps from 0.6 → linewidth_base over time
            frac = seg_t / max(total_frames - 1, 1)
            widths = 0.6 + (linewidth_base - 0.6) * frac
            # Alpha ramps 0.35 → 0.95
            seg_colors[:, 3] = 0.35 + 0.60 * frac
        else:
            widths = linewidth_base
            seg_colors[:, 3] = 0.88

        # --- Glow layers (two passes for richer bloom) ---
        if show_glow:
            # Outer glow — very wide, very faint
            glow_outer = seg_colors.copy()
            glow_outer[:, 3] = np.clip(seg_colors[:, 3] * 0.08, 0.01, 0.10)
            glow_w_outer = (widths * 7) if isinstance(widths, np.ndarray) else widths * 7
            ax.add_collection3d(Line3DCollection(segments, colors=glow_outer, linewidths=glow_w_outer))
            # Inner glow — moderately wide
            glow_inner = seg_colors.copy()
            glow_inner[:, 3] = np.clip(seg_colors[:, 3] * 0.22, 0.03, 0.25)
            glow_w_inner = (widths * 3.5) if isinstance(widths, np.ndarray) else widths * 3.5
            ax.add_collection3d(Line3DCollection(segments, colors=glow_inner, linewidths=glow_w_inner))

        # --- Core layer ---
        ax.add_collection3d(Line3DCollection(segments, colors=seg_colors, linewidths=widths))

        # --- Markers ---
        if show_markers:
            # Start marker (small, faded)
            sc = cmap(norm(t_frames[0]))
            ax.scatter(
                t_centers[0, 0], t_centers[0, 1], t_centers[0, 2],
                color=sc, s=14, marker="o",
                edgecolors="white", linewidths=0.3, zorder=5, alpha=0.55,
            )
            # End marker (larger, bright diamond)
            ec = cmap(norm(t_frames[-1]))
            ax.scatter(
                t_centers[-1, 0], t_centers[-1, 1], t_centers[-1, 2],
                color=ec, s=30, marker="D",
                edgecolors="white", linewidths=0.5, zorder=6, alpha=1.0,
            )


# ---------------------------------------------------------------------------
# Multi-Panel Mode
# ---------------------------------------------------------------------------

def create_multipanel(
    tracks: dict,
    total_frames: int,
    output_path: str,
    cmap_name: str = "plasma",
    snapshot_frames=None,
    elev: float = 25,
    azim: float = -60,
    dpi: int = 300,
    show: bool = False,
):
    """2x3 grid showing tracklet growth at 6 time snapshots."""
    if snapshot_frames is None:
        snapshot_frames = [0, 40, 80, 120, 160, total_frames - 1]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 13))

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=total_frames - 1)

    lims_lo, lims_hi = compute_global_bounds(tracks)

    for idx, snap in enumerate(snapshot_frames):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")

        draw_tracklets_on_axes(
            ax, tracks, snap, total_frames, cmap, norm,
            show_glow=True, show_markers=True, linewidth_base=2.0,
        )

        # Consistent limits (tight per-axis)
        ax.set_xlim(lims_lo[0], lims_hi[0])
        ax.set_ylim(lims_lo[1], lims_hi[1])
        ax.set_zlim(lims_lo[2], lims_hi[2])
        ax.view_init(elev=elev, azim=azim)

        style_axes_dark(ax, show_labels=False)

        # Panel title
        pct = int(round(100 * snap / max(total_frames - 1, 1)))
        ax.set_title(
            f"t = {snap}  ({pct}%)",
            color="white", fontsize=13, fontweight="bold", pad=6,
            fontfamily="monospace",
        )

    # --- Shared colorbar ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.28, 0.04, 0.44, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Frame", color="white", fontsize=11, labelpad=4)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.outline.set_edgecolor((1, 1, 1, 0.2))

    # --- Suptitle ---
    fig.suptitle(
        "3D Tracklet Evolution",
        color="white", fontsize=20, fontweight="bold",
        y=0.97, fontfamily="monospace",
    )

    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.92, bottom=0.08,
        wspace=0.05, hspace=0.12,
    )

    print(f"Saving multi-panel figure to {output_path} (dpi={dpi})...")
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single-Panel Mode
# ---------------------------------------------------------------------------

def create_single_panel(
    tracks: dict,
    total_frames: int,
    output_path: str,
    cmap_name: str = "plasma",
    elev: float = 25,
    azim: float = -60,
    dpi: int = 300,
    show: bool = False,
):
    """Single large 3D plot with linewidth/alpha gradient for temporal depth."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=total_frames - 1)

    draw_tracklets_on_axes(
        ax, tracks, total_frames - 1, total_frames, cmap, norm,
        show_glow=True, show_markers=True,
        linewidth_base=2.5, grow_width=True,
    )

    lims_lo, lims_hi = compute_global_bounds(tracks)
    ax.set_xlim(lims_lo[0], lims_hi[0])
    ax.set_ylim(lims_lo[1], lims_hi[1])
    ax.set_zlim(lims_lo[2], lims_hi[2])
    ax.view_init(elev=elev, azim=azim)

    style_axes_dark(ax, show_labels=True)

    # --- Colorbar ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=25)
    cbar.set_label("Frame", color="white", fontsize=11, labelpad=6)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.outline.set_edgecolor((1, 1, 1, 0.2))

    # --- Title ---
    ax.set_title(
        "3D Tracklets  —  Temporal Encoding",
        color="white", fontsize=16, fontweight="bold", pad=14,
        fontfamily="monospace",
    )

    fig.tight_layout()
    print(f"Saving single-panel figure to {output_path} (dpi={dpi})...")
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Animation (GIF) Mode
# ---------------------------------------------------------------------------

def create_animation(
    tracks: dict,
    total_frames: int,
    output_path: str,
    cmap_name: str = "plasma",
    elev: float = 25,
    azim: float = -60,
    dpi: int = 150,
    fps: int = 15,
    frame_step: int = 2,
):
    """Animated GIF showing tracklets growing over time."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=total_frames - 1)

    lims_lo, lims_hi = compute_global_bounds(tracks)

    # Colorbar (static, drawn once)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08, aspect=25)
    cbar.set_label("Frame", color="white", fontsize=10, labelpad=4)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.outline.set_edgecolor((1, 1, 1, 0.15))

    anim_frames = list(range(0, total_frames, frame_step))
    if anim_frames[-1] != total_frames - 1:
        anim_frames.append(total_frames - 1)

    print(f"Rendering {len(anim_frames)} animation frames (step={frame_step}, fps={fps})...")

    def update(frame_idx):
        ax.clear()
        style_axes_dark(ax, show_labels=True)

        draw_tracklets_on_axes(
            ax, tracks, frame_idx, total_frames, cmap, norm,
            show_glow=True, show_markers=True,
            linewidth_base=2.2, grow_width=True,
        )

        ax.set_xlim(lims_lo[0], lims_hi[0])
        ax.set_ylim(lims_lo[1], lims_hi[1])
        ax.set_zlim(lims_lo[2], lims_hi[2])
        ax.view_init(elev=elev, azim=azim)

        pct = int(round(100 * frame_idx / max(total_frames - 1, 1)))
        ax.set_title(
            f"t = {frame_idx}  ({pct}%)",
            color="white", fontsize=14, fontweight="bold", pad=10,
            fontfamily="monospace",
        )
        return []

    anim = FuncAnimation(fig, update, frames=anim_frames, blit=False)

    print(f"Saving animation to {output_path} (dpi={dpi}, fps={fps})...")
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Temporal 3D Tracklet Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tracklets", type=str, required=True,
        help="Path to smoothed_tracklets.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path (default: auto-named next to input)",
    )
    parser.add_argument(
        "--mode", type=str, default="multipanel",
        choices=["multipanel", "single", "animation"],
        help="Visualization mode (default: multipanel)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    parser.add_argument("--fps", type=int, default=15, help="Animation FPS (default: 15)")
    parser.add_argument("--frame_step", type=int, default=2, help="Frame step for animation (default: 2)")
    parser.add_argument(
        "--cmap", type=str, default="plasma",
        help="Colormap name (default: plasma). Try: inferno, magma, viridis",
    )
    parser.add_argument("--elev", type=float, default=25, help="View elevation (default: 25)")
    parser.add_argument("--azim", type=float, default=-60, help="View azimuth (default: -60)")
    parser.add_argument(
        "--snapshot_frames", type=int, nargs="+", default=None,
        help="Custom snapshot frames for multipanel mode (default: 0 40 80 120 160 199)",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive plot")

    args = parser.parse_args()

    if not os.path.exists(args.tracklets):
        print(f"Error: file not found: {args.tracklets}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        base_dir = os.path.dirname(args.tracklets)
        ext = ".gif" if args.mode == "animation" else ".png"
        args.output = os.path.join(base_dir, f"tracklets_{args.mode}{ext}")

    data = load_tracklets(args.tracklets)
    tracks = data["tracks"]
    total_frames = data["total_frames"]

    # Center data at origin for cleaner axis labels
    centroid = center_tracks(tracks)
    print(f"Centered data (shifted by {centroid})")

    if args.mode == "multipanel":
        create_multipanel(
            tracks, total_frames, args.output,
            cmap_name=args.cmap,
            snapshot_frames=args.snapshot_frames,
            elev=args.elev, azim=args.azim,
            dpi=args.dpi, show=args.show,
        )
    elif args.mode == "single":
        create_single_panel(
            tracks, total_frames, args.output,
            cmap_name=args.cmap,
            elev=args.elev, azim=args.azim,
            dpi=args.dpi, show=args.show,
        )
    elif args.mode == "animation":
        create_animation(
            tracks, total_frames, args.output,
            cmap_name=args.cmap,
            elev=args.elev, azim=args.azim,
            dpi=min(args.dpi, 150),  # cap for GIF file size
            fps=args.fps, frame_step=args.frame_step,
        )


if __name__ == "__main__":
    main()
