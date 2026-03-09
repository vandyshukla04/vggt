#!/usr/bin/env python3
"""
Interactive 3D visualization of masked animal point clouds from VGGT output.

Loads per-frame 3D world points from predictions.pt, applies SAM3 masks
to extract per-instance (animal) point clouds, and displays them in a
viser viewer with play/pause animation controls and tracklet splines.

Usage:
    python visualize_masked_pointcloud.py \
        --vggt_dir ./output/wd_data/lions/DJI_20250116125746_0001_V_1k/vggt \
        --mask_dir ./output/wd_data/lions/DJI_20250116125746_0001_V_1k/masks

Controls:
    - Play / Pause: animate through frames at low FPS
    - Frame slider: jump to specific frame
    - Per-track toggles: show/hide individual animals
    - Point size, tracklet width, confidence threshold sliders
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

import viser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(vggt_dir: str, use_point_map: bool = True) -> Dict:
    """Load predictions.pt and return world points, colors, confidence."""
    predictions_path = os.path.join(vggt_dir, "predictions.pt")
    print(f"Loading predictions from {predictions_path} ...")
    predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)

    def squeeze(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    images = squeeze(predictions["images"])              # (S, 3, H, W)
    depth_conf = squeeze(predictions["depth_conf"])      # (S, H, W)

    if use_point_map and "world_points" in predictions:
        world_points = squeeze(predictions["world_points"])          # (S, H, W, 3)
        conf = squeeze(predictions.get("world_points_conf", predictions["depth_conf"]))
    else:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        depth = squeeze(predictions["depth"])
        extrinsics = squeeze(predictions["extrinsic"])
        intrinsics = squeeze(predictions["intrinsic"])
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = depth_conf

    # (S, 3, H, W) -> (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)

    S, H, W, _ = world_points.shape
    print(f"  {S} frames, {H}x{W} resolution")
    return {
        "world_points": world_points,
        "colors": colors,
        "conf": conf,
        "model_size": (H, W),
        "num_frames": S,
    }


def _rle_decode(rle: Dict, height: int, width: int) -> np.ndarray:
    """Decode COCO-style RLE segmentation to a binary mask."""
    counts = rle["counts"]
    if isinstance(counts, str):
        # Compressed RLE string – decode using pycocotools if available,
        # otherwise fall back to a pure-Python implementation.
        try:
            from pycocotools import mask as mask_utils
            binary = mask_utils.decode(rle).astype(bool)
            return binary
        except ImportError:
            pass
        # Pure-python compressed RLE decoder (COCO format)
        m = 0
        p = 0
        cnts = []
        while p < len(counts):
            x = 0
            k = 0
            more = True
            while more:
                c = ord(counts[p]) - 48
                p += 1
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                k += 1
            if m > 2:
                x += cnts[m - 2]
            cnts.append(x)
            m += 1
        flat = np.zeros(height * width, dtype=bool)
        pos = 0
        for i, c in enumerate(cnts):
            if i % 2 == 1:  # odd runs are foreground
                flat[pos : pos + c] = True
            pos += c
        return flat.reshape((height, width), order="F")
    else:
        # Uncompressed RLE (list of ints)
        flat = np.zeros(height * width, dtype=bool)
        pos = 0
        for i, c in enumerate(counts):
            if i % 2 == 1:
                flat[pos : pos + c] = True
            pos += c
        return flat.reshape((height, width), order="F")


def load_grounded_sam_masks(
    mask_dir: str, image_files: List[str],
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, str]]:
    """Load Grounded-SAM per-frame *_results.json masks.

    Since Grounded-SAM has no cross-frame tracking, we use a simple
    spatial-overlap heuristic to assign consistent object IDs across frames.

    Returns:
        masks: dict mapping frame_index -> {obj_id: binary_mask}
        obj_classes: dict mapping obj_id -> class_name
    """
    mask_dir_p = Path(mask_dir)
    result_files = sorted(mask_dir_p.glob("*_results.json"))
    if not result_files:
        print(f"Warning: No *_results.json files found in {mask_dir}")
        return {}, {}

    print(f"Found {len(result_files)} Grounded-SAM result files")

    # Build lookup: frame stem (e.g. "5400") -> VGGT frame index
    stem_to_idx = {}
    for i, fname in enumerate(image_files):
        stem = Path(fname).stem  # "5400.jpg" -> "5400"
        stem_to_idx[stem] = i

    # First pass: load all per-frame detections (untracked)
    # frame_detections[frame_idx] = [(class_name, binary_mask), ...]
    frame_detections: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    for rf in result_files:
        stem = rf.stem.replace("_results", "")  # "5400_results" -> "5400"
        if stem not in stem_to_idx:
            continue
        frame_idx = stem_to_idx[stem]
        with open(str(rf), "r") as f:
            data = json.load(f)
        dets = []
        for ann in data.get("annotations", []):
            seg = ann.get("segmentation")
            if seg is None:
                continue
            h, w = seg["size"]
            binary = _rle_decode(seg, h, w)
            if not binary.any():
                continue
            dets.append((ann.get("class_name", "object"), binary))
        if dets:
            frame_detections[frame_idx] = dets

    # Second pass: assign consistent IDs via greedy IoU matching
    next_obj_id = 0
    # prev_masks tracks {obj_id: binary_mask} from the previous frame
    prev_masks: Dict[int, np.ndarray] = {}
    obj_classes: Dict[int, str] = {}
    masks: Dict[int, Dict[int, np.ndarray]] = {}

    for frame_idx in sorted(frame_detections.keys()):
        dets = frame_detections[frame_idx]
        assigned: Dict[int, np.ndarray] = {}
        used_prev_ids = set()
        unmatched_dets = list(range(len(dets)))

        if prev_masks:
            # Compute IoU between each detection and each previous object
            iou_pairs = []
            for di, (cls, det_mask) in enumerate(dets):
                for pid, prev_mask in prev_masks.items():
                    # Resize if shapes differ
                    if det_mask.shape != prev_mask.shape:
                        import cv2
                        prev_resized = cv2.resize(
                            prev_mask.astype(np.uint8) * 255,
                            (det_mask.shape[1], det_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ) > 127
                    else:
                        prev_resized = prev_mask
                    inter = np.sum(det_mask & prev_resized)
                    union = np.sum(det_mask | prev_resized)
                    iou = inter / union if union > 0 else 0.0
                    if iou > 0.1:
                        iou_pairs.append((iou, di, pid))
            # Greedy assignment: highest IoU first
            iou_pairs.sort(key=lambda x: -x[0])
            for iou_val, di, pid in iou_pairs:
                if di not in unmatched_dets or pid in used_prev_ids:
                    continue
                assigned[pid] = dets[di][1]
                obj_classes[pid] = dets[di][0]
                used_prev_ids.add(pid)
                unmatched_dets.remove(di)

        # Create new IDs for unmatched detections
        for di in unmatched_dets:
            assigned[next_obj_id] = dets[di][1]
            obj_classes[next_obj_id] = dets[di][0]
            next_obj_id += 1

        masks[frame_idx] = assigned
        prev_masks = assigned

    total_objs = len(set(oid for fm in masks.values() for oid in fm.keys()))
    total_frames_with_masks = len(masks)
    print(f"  Tracked {total_objs} objects across {total_frames_with_masks} frames (IoU matching)")
    for oid in sorted(obj_classes.keys()):
        count = sum(1 for fm in masks.values() if oid in fm)
        print(f"    obj_{oid} ({obj_classes[oid]}): {count} frames")
    return masks, obj_classes


def load_sam3_masks(mask_dir: str, image_files: List[str]) -> Dict[int, Dict[int, np.ndarray]]:
    """Load SAM3 masks from per-object directories.

    Returns:
        dict mapping frame_index -> {obj_id: binary_mask}
    """
    from PIL import Image

    obj_dirs = sorted(
        [d for d in Path(mask_dir).iterdir() if d.is_dir() and d.name.startswith("obj_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not obj_dirs:
        print(f"Warning: No obj_* directories found in {mask_dir}")
        return {}

    print(f"Found {len(obj_dirs)} object mask directories")

    # Build lookup: frame filename -> extraction index
    fname_to_idx = {fname: i for i, fname in enumerate(image_files)}

    masks: Dict[int, Dict[int, np.ndarray]] = {}
    for obj_dir in obj_dirs:
        obj_id = int(obj_dir.name.split("_")[1])
        mask_files = sorted(obj_dir.glob("frame_*.png"))
        loaded = 0
        for mf in mask_files:
            fname = mf.name
            if fname not in fname_to_idx:
                continue
            frame_idx = fname_to_idx[fname]
            mask_arr = np.array(Image.open(str(mf)))
            binary = mask_arr > 127
            if not binary.any():
                continue
            if frame_idx not in masks:
                masks[frame_idx] = {}
            masks[frame_idx][obj_id] = binary
            loaded += 1
        print(f"  obj_{obj_id}: {loaded} frames with valid masks")

    return masks


def resize_mask_to_model(mask: np.ndarray, orig_size: Tuple[int, int],
                         model_size: Tuple[int, int]) -> np.ndarray:
    """Resize a binary mask from original image resolution to model resolution
    using letterbox-aware mapping (same as VGGT preprocessing)."""
    import cv2
    orig_h, orig_w = orig_size
    model_h, model_w = model_size

    # VGGT uses a square-then-resize approach: determine the letterbox
    # scaling. If the original image was resized to fit into (model_h, model_w)
    # preserving aspect ratio with padding, we need that mapping.
    # For simplicity, just resize the mask directly.
    mask_uint8 = mask.astype(np.uint8) * 255
    resized = cv2.resize(mask_uint8, (model_w, model_h), interpolation=cv2.INTER_NEAREST)
    return resized > 127


def load_tracking_summary(vggt_dir: str) -> Optional[Dict]:
    """Load tracking_summary.json for tracklet trajectory data."""
    path = os.path.join(vggt_dir, "tracking_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded tracking: {data['total_tracks']} tracks, {data['total_frames']} frames")
    return data


# ---------------------------------------------------------------------------
# Per-frame instance extraction
# ---------------------------------------------------------------------------

def extract_instance_points(
    world_points_frame: np.ndarray,  # (H, W, 3)
    colors_frame: np.ndarray,        # (H, W, 3) float [0,1]
    conf_frame: np.ndarray,          # (H, W)
    mask: np.ndarray,                # (H, W) bool at model resolution
    conf_threshold: float,
    track_color: Tuple[float, float, float],
    color_blend: float = 0.4,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract 3D points for a masked instance at one frame.

    Returns (points_Nx3, colors_Nx3_uint8) or None if no valid points.
    """
    # Apply mask + confidence + validity filtering
    valid = (
        mask
        & (conf_frame >= conf_threshold)
        & ~np.any(np.isnan(world_points_frame) | np.isinf(world_points_frame), axis=-1)
    )

    pts = world_points_frame[valid].astype(np.float32)
    if len(pts) == 0:
        # Try without confidence filter for masked pixels (animals often
        # have lower confidence than static ground)
        valid_relaxed = (
            mask
            & ~np.any(np.isnan(world_points_frame) | np.isinf(world_points_frame), axis=-1)
        )
        pts = world_points_frame[valid_relaxed].astype(np.float32)
        clrs_raw = colors_frame[valid_relaxed]
    else:
        clrs_raw = colors_frame[valid]

    if len(pts) == 0:
        return None

    # Blend image colors with track color
    img_colors = (np.clip(clrs_raw, 0, 1) * 255).astype(np.uint8)
    tc = np.array([int(c * 255) for c in track_color], dtype=np.uint8)
    blended = (color_blend * tc + (1 - color_blend) * img_colors).astype(np.uint8)
    return pts, blended


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize masked animal point clouds with viser")
    parser.add_argument("--vggt_dir", type=str, required=True,
                        help="Path to VGGT output directory (contains predictions.pt)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Path to SAM3 mask directory (contains obj_0/, obj_1/, ...)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.008)
    parser.add_argument("--conf_percentile", type=float, default=30.0,
                        help="Confidence percentile threshold for background points")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Playback frames per second")
    parser.add_argument("--color_blend", type=float, default=0.35,
                        help="Blend factor for track color onto instance (0=image, 1=track color)")
    parser.add_argument("--no_background", action="store_true",
                        help="Hide background (non-masked) points entirely")
    parser.add_argument("--use_depth", action="store_true",
                        help="Use depth unprojection instead of world_points")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Masked Point Cloud Visualization")
    print(f"{'='*60}\n")

    # Metadata
    metadata_path = os.path.join(args.vggt_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    image_files = metadata.get("image_files", [])

    # Predictions
    pred = load_predictions(args.vggt_dir, use_point_map=not args.use_depth)
    world_points = pred["world_points"]   # (S, H, W, 3)
    colors = pred["colors"]               # (S, H, W, 3) float
    conf = pred["conf"]                   # (S, H, W)
    model_size = pred["model_size"]       # (H, W)
    num_frames = pred["num_frames"]

    # Fill image_files if missing
    if not image_files:
        image_files = [f"frame_{i:06d}.png" for i in range(num_frames)]

    # Confidence threshold for background
    conf_flat = conf.reshape(-1)
    valid_conf = conf_flat[conf_flat > 1e-5]
    bg_conf_threshold = float(np.percentile(valid_conf, args.conf_percentile)) if len(valid_conf) > 0 else 0.0
    print(f"Background confidence threshold (p{args.conf_percentile:.0f}): {bg_conf_threshold:.4f}")

    # Masks: auto-detect format (SAM3 obj_* dirs vs Grounded-SAM *_results.json)
    gsam_obj_classes: Dict[int, str] = {}
    mask_dir_p = Path(args.mask_dir)
    has_obj_dirs = any(
        d.is_dir() and d.name.startswith("obj_") for d in mask_dir_p.iterdir()
    )
    has_gsam_json = any(mask_dir_p.glob("*_results.json"))

    if has_obj_dirs:
        print("Detected SAM3 mask format (obj_* directories)")
        masks = load_sam3_masks(args.mask_dir, image_files)
    elif has_gsam_json:
        print("Detected Grounded-SAM mask format (*_results.json)")
        masks, gsam_obj_classes = load_grounded_sam_masks(args.mask_dir, image_files)
    else:
        print(f"Warning: No recognized mask format in {args.mask_dir}")
        masks = {}

    if not masks:
        print("Warning: No masks loaded. Only background points will be shown.")

    # Tracking summary (for trajectory splines)
    tracking = load_tracking_summary(args.vggt_dir)

    # Determine mask original resolution (from first mask)
    mask_orig_size = None
    for frame_idx, frame_masks in masks.items():
        for obj_id, m in frame_masks.items():
            mask_orig_size = m.shape[:2]
            break
        if mask_orig_size is not None:
            break
    print(f"Mask original size: {mask_orig_size}, model size: {model_size}")

    # Map object IDs to sorted track indices for consistent coloring
    all_obj_ids = sorted(set(oid for fm in masks.values() for oid in fm.keys()))
    obj_id_to_sort_idx = {oid: i for i, oid in enumerate(all_obj_ids)}
    print(f"Object IDs: {all_obj_ids}")

    # ------------------------------------------------------------------
    # Pre-compute per-frame background points (masked out animals)
    # ------------------------------------------------------------------
    print("\nPre-computing per-frame data ...")

    per_frame_bg_points = []
    per_frame_bg_colors = []
    per_frame_instances = []  # list of dicts: {obj_id: (pts, clrs)}

    for f in range(num_frames):
        # Aggregate all object masks for this frame
        combined_animal_mask = np.zeros(model_size, dtype=bool)
        frame_instances = {}

        if f in masks:
            for obj_id, raw_mask in masks[f].items():
                # Resize mask to model coordinates
                if raw_mask.shape[:2] != model_size:
                    model_mask = resize_mask_to_model(raw_mask, mask_orig_size, model_size)
                else:
                    model_mask = raw_mask
                combined_animal_mask |= model_mask

                sort_idx = obj_id_to_sort_idx[obj_id]
                track_color = COLOR_PALETTE[sort_idx % len(COLOR_PALETTE)]

                result = extract_instance_points(
                    world_points[f], colors[f], conf[f],
                    model_mask, bg_conf_threshold * 0.3,  # relaxed threshold for animals
                    track_color, args.color_blend,
                )
                if result is not None:
                    frame_instances[obj_id] = result

        per_frame_instances.append(frame_instances)

        # Background: everything NOT masked, above confidence
        if not args.no_background:
            bg_valid = (
                ~combined_animal_mask
                & (conf[f] >= bg_conf_threshold)
                & ~np.any(np.isnan(world_points[f]) | np.isinf(world_points[f]), axis=-1)
            )
            bg_pts = world_points[f][bg_valid].astype(np.float32)
            bg_clrs = (np.clip(colors[f][bg_valid], 0, 1) * 255).astype(np.uint8)
        else:
            bg_pts = np.empty((0, 3), dtype=np.float32)
            bg_clrs = np.empty((0, 3), dtype=np.uint8)

        per_frame_bg_points.append(bg_pts)
        per_frame_bg_colors.append(bg_clrs)

        if (f + 1) % 50 == 0 or f == num_frames - 1:
            n_inst = len(frame_instances)
            n_bg = len(bg_pts)
            print(f"  Frame {f+1}/{num_frames}: {n_inst} instances, {n_bg:,} bg points")

    # Compute scene center from all points for centering
    all_bg = [p for p in per_frame_bg_points if len(p) > 0]
    all_inst = [pts for fi in per_frame_instances for pts, _ in fi.values()]
    all_combined = all_bg + all_inst
    if all_combined:
        sampled = np.concatenate([p[::max(1, len(p)//1000)] for p in all_combined if len(p) > 0])
        scene_center = np.mean(sampled, axis=0)
    else:
        scene_center = np.zeros(3, dtype=np.float32)
    print(f"Scene center: {scene_center}")

    # ------------------------------------------------------------------
    # Build tracklet data for splines
    # ------------------------------------------------------------------
    track_splines = {}  # track_id_str -> {frames: [], centers: []}
    if tracking:
        for tid_str, tdata in tracking["tracks"].items():
            frames_list = tdata.get("frames", [])
            centers_list = tdata.get("centers", [])
            if frames_list and centers_list:
                track_splines[tid_str] = {
                    "frames": np.array(frames_list),
                    "centers": np.array(centers_list, dtype=np.float32),
                    "class_name": tdata.get("class_name", "animal"),
                }

    # ------------------------------------------------------------------
    # Viser server
    # ------------------------------------------------------------------
    print(f"\nStarting viser server on port {args.port} ...")
    server = viser.ViserServer(host="0.0.0.0", port=args.port, verbose=False)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # --- Playback controls ---
    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=num_frames - 1, step=1, initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("Playing", initial_value=False)
        gui_fps = server.gui.add_slider(
            "FPS", min=0.5, max=10.0, step=0.5, initial_value=args.fps,
        )
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)

    # --- Display options ---
    with server.gui.add_folder("Display"):
        gui_show_bg = server.gui.add_checkbox("Show Background", initial_value=not args.no_background)
        gui_show_instances = server.gui.add_checkbox("Show Instances", initial_value=True)
        gui_show_tracklets = server.gui.add_checkbox("Show Tracklets", initial_value=True)
        gui_show_labels = server.gui.add_checkbox("Show Labels", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size,
        )
        gui_instance_point_size = server.gui.add_slider(
            "Instance Point Size", min=0.002, max=0.05, step=0.001,
            initial_value=args.point_size * 1.5,
        )
        gui_tracklet_width = server.gui.add_slider(
            "Tracklet Width", min=1.0, max=10.0, step=0.5, initial_value=4.0,
        )
        gui_bg_density = server.gui.add_slider(
            "BG Density %", min=5, max=100, step=5, initial_value=50,
        )
        gui_cumulative_bg = server.gui.add_checkbox("Cumulative Background", initial_value=False)

    # --- Per-track toggles ---
    track_checkboxes: Dict[int, object] = {}
    with server.gui.add_folder("Tracks"):
        for obj_id in all_obj_ids:
            sort_idx = obj_id_to_sort_idx[obj_id]
            color = COLOR_PALETTE[sort_idx % len(COLOR_PALETTE)]
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            # Find class name from tracking summary or Grounded-SAM classes
            class_name = gsam_obj_classes.get(obj_id, "animal")
            if tracking:
                tdata = tracking["tracks"].get(str(obj_id), {})
                class_name = tdata.get("class_name", class_name)
            track_checkboxes[obj_id] = server.gui.add_checkbox(
                f"[{color_hex}] Obj {obj_id} ({class_name})",
                initial_value=True,
            )

    # ------------------------------------------------------------------
    # Scene handles
    # ------------------------------------------------------------------
    handles = {
        "bg": None,
        "instances": {},     # obj_id -> handle
        "tracklets": [],     # list of handles
        "labels": [],        # list of handles
    }

    def _safe_remove(h):
        """Remove a viser handle, ignoring errors if already removed."""
        try:
            h.remove()
        except (KeyError, Exception):
            pass

    def clear_handles(key):
        if key == "bg":
            if handles["bg"] is not None:
                _safe_remove(handles["bg"])
                handles["bg"] = None
        elif key == "instances":
            for h in handles["instances"].values():
                _safe_remove(h)
            handles["instances"] = {}
        elif key in ("tracklets", "labels"):
            for h in handles[key]:
                _safe_remove(h)
            handles[key] = []

    def render_frame(frame_idx: int):
        """Render a single frame: background + instances + tracklets."""
        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))

        # --- Background ---
        clear_handles("bg")
        if gui_show_bg.value:
            if gui_cumulative_bg.value:
                # Accumulate background from frame 0 to frame_idx
                bg_pts_list = []
                bg_clrs_list = []
                density = gui_bg_density.value / 100.0
                for i in range(frame_idx + 1):
                    pts_i = per_frame_bg_points[i]
                    clrs_i = per_frame_bg_colors[i]
                    if len(pts_i) == 0:
                        continue
                    if density < 1.0:
                        k = max(1, int(len(pts_i) * density))
                        idx = np.random.choice(len(pts_i), k, replace=False)
                        bg_pts_list.append(pts_i[idx])
                        bg_clrs_list.append(clrs_i[idx])
                    else:
                        bg_pts_list.append(pts_i)
                        bg_clrs_list.append(clrs_i)
                if bg_pts_list:
                    bg_pts = np.concatenate(bg_pts_list) - scene_center
                    bg_clrs = np.concatenate(bg_clrs_list)
                    handles["bg"] = server.scene.add_point_cloud(
                        name="/background",
                        points=bg_pts,
                        colors=bg_clrs,
                        point_size=gui_point_size.value,
                        point_shape="circle",
                    )
            else:
                bg_pts = per_frame_bg_points[frame_idx]
                bg_clrs = per_frame_bg_colors[frame_idx]
                if len(bg_pts) > 0:
                    density = gui_bg_density.value / 100.0
                    if density < 1.0:
                        k = max(1, int(len(bg_pts) * density))
                        idx = np.random.choice(len(bg_pts), k, replace=False)
                        bg_pts = bg_pts[idx]
                        bg_clrs = bg_clrs[idx]
                    handles["bg"] = server.scene.add_point_cloud(
                        name="/background",
                        points=bg_pts - scene_center,
                        colors=bg_clrs,
                        point_size=gui_point_size.value,
                        point_shape="circle",
                    )

        # --- Instances ---
        clear_handles("instances")
        if gui_show_instances.value:
            frame_inst = per_frame_instances[frame_idx]
            for obj_id, (pts, clrs) in frame_inst.items():
                if obj_id in track_checkboxes and not track_checkboxes[obj_id].value:
                    continue
                handles["instances"][obj_id] = server.scene.add_point_cloud(
                    name=f"/instance/obj_{obj_id}",
                    points=pts - scene_center,
                    colors=clrs,
                    point_size=gui_instance_point_size.value,
                    point_shape="circle",
                )

        # --- Tracklets (splines up to current frame) ---
        clear_handles("tracklets")
        clear_handles("labels")
        if gui_show_tracklets.value and track_splines:
            for tid_str, sp_data in track_splines.items():
                obj_id = int(tid_str)
                if obj_id in track_checkboxes and not track_checkboxes[obj_id].value:
                    continue

                sort_idx = obj_id_to_sort_idx.get(obj_id, int(tid_str))
                color = COLOR_PALETTE[sort_idx % len(COLOR_PALETTE)]

                sp_frames = sp_data["frames"]
                sp_centers = sp_data["centers"]

                # Trim to frames <= frame_idx
                trim_mask = sp_frames <= frame_idx
                trimmed = sp_centers[trim_mask]
                if len(trimmed) < 2:
                    # Still show a marker for a single point
                    if len(trimmed) == 1:
                        h = server.scene.add_icosphere(
                            name=f"/tracklet/t{tid_str}_marker",
                            radius=0.02,
                            position=(trimmed[0] - scene_center).astype(np.float32),
                            color=color,
                        )
                        handles["tracklets"].append(h)
                    continue

                centered = (trimmed - scene_center).astype(np.float32)
                h = server.scene.add_spline_catmull_rom(
                    name=f"/tracklet/t{tid_str}",
                    positions=centered,
                    color=color,
                    line_width=gui_tracklet_width.value,
                    segments=max(1, len(centered) * 4),
                )
                handles["tracklets"].append(h)

                # Current position marker
                h2 = server.scene.add_icosphere(
                    name=f"/tracklet/t{tid_str}_head",
                    radius=0.015,
                    position=centered[-1],
                    color=color,
                )
                handles["tracklets"].append(h2)

                # Label
                if gui_show_labels.value:
                    class_name = sp_data.get("class_name", "animal")
                    lh = server.scene.add_label(
                        name=f"/track_labels/label_{tid_str}",
                        text=f"T{tid_str}: {class_name}",
                        position=centered[-1],
                    )
                    handles["labels"].append(lh)

    # ------------------------------------------------------------------
    # Playback thread
    # ------------------------------------------------------------------
    render_lock = threading.Lock()

    # Wrap render_frame with lock
    _render_frame_inner = render_frame

    def render_frame_safe(frame_idx):
        with render_lock:
            _render_frame_inner(frame_idx)

    render_frame = render_frame_safe

    def playback_loop():
        while True:
            if gui_playing.value:
                current = gui_frame.value
                next_frame = current + 1
                if next_frame >= num_frames:
                    if gui_loop.value:
                        next_frame = 0
                    else:
                        gui_playing.value = False
                        time.sleep(0.05)
                        continue
                gui_frame.value = next_frame
                time.sleep(1.0 / gui_fps.value)
            else:
                time.sleep(0.05)

    playback_thread = threading.Thread(target=playback_loop, daemon=True)
    playback_thread.start()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    @gui_frame.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_show_bg.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_show_instances.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_show_tracklets.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_show_labels.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_point_size.on_update
    def _(_):
        if handles["bg"] is not None:
            handles["bg"].point_size = gui_point_size.value

    @gui_instance_point_size.on_update
    def _(_):
        for h in handles["instances"].values():
            h.point_size = gui_instance_point_size.value

    @gui_bg_density.on_update
    def _(_):
        render_frame(gui_frame.value)

    @gui_cumulative_bg.on_update
    def _(_):
        render_frame(gui_frame.value)

    def _make_track_cb(cb):
        @cb.on_update
        def _(_):
            render_frame(gui_frame.value)

    for _, checkbox in track_checkboxes.items():
        _make_track_cb(checkbox)

    # Initial render
    render_frame(0)

    print(f"\n{'='*60}")
    print(f"Visualization ready! Open your browser to:")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*60}")
    print(f"  {num_frames} frames | {len(all_obj_ids)} tracked objects")
    print(f"  Press Play or use the frame slider to animate")
    print(f"  Press Ctrl+C to exit\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
