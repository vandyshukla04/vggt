#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Save per-frame images with colored mask overlays matching viser track colors.

Standalone script that loads predictions.pt (for images), segmentation masks,
and tracking_summary.json (for track-to-mask matching). Produces per-frame PNGs
with semi-transparent colored mask overlays + contours + track labels.

The mask colors use the same COLOR_PALETTE and sorted track ID order as
visualize_ground_tracklets.py's viser splines, so the colors are consistent
between the 2D frame images and the 3D visualization.

Usage:
    python save_masked_frames.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/grounded-sam/

    python save_masked_frames.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/sam3_output/ \
        --mask_format sam3 --video ./data/video.mp4

    python save_masked_frames.py \
        --result_dir ./output/vggt/ \
        --mask_source ./data/grounded-sam/ \
        --frames 0,5,10,15 --alpha 0.5
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import pycocotools for RLE mask decoding (Grounded SAM format)
try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False

# Color palette matching visualize_ground_tracklets.py and demo_viser_tracking.py
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


# =============================================================================
# Self-contained mask utilities (from demo_viser_tracking.py)
# =============================================================================


def decode_rle_mask(rle_data: Dict) -> Optional[np.ndarray]:
    """Decode RLE mask using pycocotools."""
    if not HAS_PYCOCOTOOLS:
        return None
    try:
        if isinstance(rle_data, dict) and "size" in rle_data and "counts" in rle_data:
            decoded = mask_utils.decode(rle_data)
            return decoded.astype(bool)
    except Exception as e:
        print(f"RLE decode failed: {e}")
    return None


def load_grounded_sam_masks(
    mask_dir: str, img_paths: List[str]
) -> Dict[int, List[Dict]]:
    """Load Grounded SAM masks from JSON files."""
    masks_data = {}

    for i, img_path in enumerate(img_paths):
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_file = os.path.join(mask_dir, f"{frame_name}_results.json")

        if os.path.exists(mask_file):
            try:
                with open(mask_file, "r") as f:
                    data = json.load(f)

                print(
                    f"Loading masks for {frame_name}: "
                    f"{len(data['annotations'])} annotations"
                )

                frame_masks = []
                for ann_idx, ann in enumerate(data["annotations"]):
                    try:
                        mask = decode_rle_mask(ann["segmentation"])
                        if mask is not None and np.sum(mask) > 0:
                            score = (
                                ann["score"][0]
                                if isinstance(ann["score"], list)
                                else ann["score"]
                            )
                            frame_masks.append(
                                {
                                    "mask": mask,
                                    "class_name": ann["class_name"],
                                    "score": score,
                                    "bbox": ann["bbox"],
                                }
                            )
                    except Exception as e:
                        print(f"  Error processing annotation {ann_idx}: {e}")

                masks_data[i] = frame_masks

            except Exception as e:
                print(f"Error loading {mask_file}: {e}")
                masks_data[i] = []
        else:
            print(f"Mask file not found: {mask_file}")
            masks_data[i] = []

    return masks_data


def transform_mask_to_model_coordinates(
    mask: np.ndarray,
    original_shape: Tuple[int, int],
    model_shape: Tuple[int, int],
) -> np.ndarray:
    """Transform mask from original image coordinates to model coordinates."""
    orig_h, orig_w = original_shape
    model_h, model_w = model_shape

    scale = min(model_w / orig_w, model_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    pad_x = (model_w - new_w) // 2
    pad_y = (model_h - new_h) // 2

    mask_resized = cv2.resize(
        mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    mask_padded = np.zeros((model_h, model_w), dtype=bool)
    mask_padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = mask_resized

    return mask_padded


# =============================================================================
# Mask Loading
# =============================================================================


def load_masks(
    mask_source: str,
    mask_format: str,
    metadata: dict,
    video: Optional[str] = None,
    sam3_class: Optional[str] = None,
    sam3_fps: Optional[float] = None,
) -> Dict[int, List[Dict]]:
    """Load segmentation masks from SAM3 or Grounded SAM format."""

    if mask_format == "grounded_sam":
        image_files = metadata.get("image_files", [])
        input_path = metadata.get("input_path", "")

        image_paths = []
        for fname in image_files:
            candidates = [
                os.path.join(input_path, fname),
                os.path.join(input_path, "images", fname),
                os.path.join(os.path.dirname(input_path), "images", fname),
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    image_paths.append(c)
                    found = True
                    break
            if not found:
                image_paths.append(fname)

        return load_grounded_sam_masks(mask_source, image_paths)

    # SAM3 format
    from vggt.utils.sam3_mask_loader import (
        detect_sam3_format,
        load_sam3_masks,
        load_sam3_masks_multi_class,
    )

    frame_indices = metadata.get(
        "frame_indices", list(range(metadata.get("num_frames", 0)))
    )

    video_fps = metadata.get("video_fps", 30.0)
    if sam3_fps is None:
        sam3_fps = metadata.get("extraction_fps", video_fps)

    if video:
        from vggt.utils.video_utils import get_video_info

        video_info = get_video_info(video)
        video_fps = video_info["fps"]

    direct_frame_match = (
        metadata.get("start_frame") is not None
        or metadata.get("end_frame") is not None
    )

    format_info = detect_sam3_format(mask_source)

    if format_info["format"] == "multi_class":
        class_names = None
        if sam3_class:
            class_names = [c.strip() for c in sam3_class.split(",")]

        return load_sam3_masks_multi_class(
            sam3_output_dir=mask_source,
            extracted_frame_indices=frame_indices,
            video_fps=video_fps,
            sam3_fps=sam3_fps,
            class_names=class_names,
            direct_frame_match=direct_frame_match,
        )
    else:
        return load_sam3_masks(
            sam3_output_dir=mask_source,
            extracted_frame_indices=frame_indices,
            video_fps=video_fps,
            sam3_fps=sam3_fps,
            class_name=sam3_class or "object",
            direct_frame_match=direct_frame_match,
            auto_detect_format=False,
        )


# =============================================================================
# Track-to-Mask Matching
# =============================================================================


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [left, top, right, bottom] bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> List[float]:
    """Get [left, top, right, bottom] bounding box from a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def match_tracks_to_masks(
    frame_idx: int,
    tracking: Dict,
    masks_data: Dict[int, List[Dict]],
    model_size: Tuple[int, int],
    orig_shapes: Dict[int, Tuple[int, int]],
) -> Dict[int, Tuple[str, int]]:
    """Match tracking_summary tracks to mask detections at a given frame.

    Uses IoU between track bbox_2d and mask bounding box (after transforming
    mask to model coordinates) with Hungarian assignment.

    Returns:
        Dict mapping mask_index -> (track_id_str, sorted_track_index)
        where sorted_track_index determines the COLOR_PALETTE color.
    """
    sorted_track_ids = sorted(tracking["tracks"].keys(), key=lambda x: int(x))

    # Gather active tracks at this frame with their bbox_2d
    active_tracks = []
    for sort_idx, tid in enumerate(sorted_track_ids):
        track = tracking["tracks"][tid]
        frames = track.get("frames", [])
        if frame_idx not in frames:
            continue
        frame_pos = frames.index(frame_idx)
        bbox_2d = track.get("bbox_2d", [])
        if frame_pos < len(bbox_2d):
            b = bbox_2d[frame_pos]
            # Skip invalid projections
            if b[0] >= 0:
                active_tracks.append((tid, sort_idx, b))

    frame_masks = masks_data.get(frame_idx, [])
    if not active_tracks or not frame_masks:
        return {}

    # Compute mask bboxes in model coordinates
    mask_bboxes = []
    for mask_info in frame_masks:
        raw_mask = mask_info["mask"]
        orig_shape = orig_shapes.get(frame_idx, raw_mask.shape[:2])
        mask_transformed = transform_mask_to_model_coordinates(
            raw_mask, orig_shape, model_size
        )
        mask_bboxes.append(mask_bbox(mask_transformed))

    # Build IoU cost matrix: (num_tracks, num_masks)
    n_tracks = len(active_tracks)
    n_masks = len(mask_bboxes)
    cost_matrix = np.zeros((n_tracks, n_masks))

    for ti, (_, _, track_box) in enumerate(active_tracks):
        for mi, m_box in enumerate(mask_bboxes):
            cost_matrix[ti, mi] = 1.0 - compute_iou(track_box, m_box)

    # Hungarian assignment
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    result = {}
    for ri, ci in zip(row_ind, col_ind):
        iou_val = 1.0 - cost_matrix[ri, ci]
        if iou_val > 0.1:  # Minimum IoU threshold
            tid, sort_idx, _ = active_tracks[ri]
            result[ci] = (tid, sort_idx)

    return result


# =============================================================================
# Frame Rendering
# =============================================================================


def render_masked_frame(
    frame_idx: int,
    images: np.ndarray,
    masks_data: Dict[int, List[Dict]],
    tracking: Dict,
    model_size: Tuple[int, int],
    orig_shapes: Dict[int, Tuple[int, int]],
    alpha: float = 0.45,
    original_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a single frame with colored mask overlays.

    Args:
        frame_idx: Frame index
        images: Preprocessed images (S, 3, H, W) in [0, 1]
        masks_data: Per-frame mask detections
        tracking: Tracking summary dict
        model_size: (H, W) of model output
        orig_shapes: Per-frame original image shapes {frame_idx: (h, w)}
        alpha: Mask overlay transparency
        original_image: Optional high-res original image (H, W, 3) uint8

    Returns:
        Rendered image as (H, W, 3) uint8
    """
    # Get base image
    if original_image is not None:
        image = original_image.copy()
    else:
        image = (images[frame_idx].transpose(1, 2, 0) * 255).astype(np.uint8)

    overlay = image.copy().astype(np.float32)

    # Match tracks to masks
    mask_to_track = match_tracks_to_masks(
        frame_idx, tracking, masks_data, model_size, orig_shapes
    )

    frame_masks = masks_data.get(frame_idx, [])

    for mask_idx, mask_info in enumerate(frame_masks):
        if mask_idx not in mask_to_track:
            continue

        track_id_str, sort_idx = mask_to_track[mask_idx]
        color = COLOR_PALETTE[sort_idx % len(COLOR_PALETTE)]
        color_rgb = np.array(color) * 255

        # Transform mask to model coordinates
        raw_mask = mask_info["mask"]
        orig_shape = orig_shapes.get(frame_idx, raw_mask.shape[:2])
        mask_transformed = transform_mask_to_model_coordinates(
            raw_mask, orig_shape, model_size
        )

        # If using original high-res image, resize mask to match
        if original_image is not None:
            img_h, img_w = original_image.shape[:2]
            model_h, model_w = model_size
            if (img_h, img_w) != (model_h, model_w):
                mask_transformed = cv2.resize(
                    mask_transformed.astype(np.uint8),
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        mask_bool = mask_transformed > 0

        # Apply semi-transparent colored overlay
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color_rgb[c] * alpha,
                overlay[:, :, c],
            )

        # Draw contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.drawContours(
            overlay_uint8,
            contours,
            -1,
            (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])),
            thickness=2,
        )
        overlay = overlay_uint8.astype(np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Save per-frame images with colored mask overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python save_masked_frames.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/grounded-sam/

    python save_masked_frames.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/sam3_output/ \\
        --mask_format sam3 --video ./data/video.mp4

    python save_masked_frames.py \\
        --result_dir ./output/vggt/ \\
        --mask_source ./data/grounded-sam/ \\
        --frames 0,5,10,15 --alpha 0.5
        """,
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="VGGT output directory (has predictions.pt, tracking_summary.json)",
    )
    parser.add_argument(
        "--mask_source",
        type=str,
        required=True,
        help="Path to masks directory (Grounded SAM JSON or SAM3)",
    )
    parser.add_argument(
        "--mask_format",
        type=str,
        default="auto",
        choices=["auto", "grounded_sam", "sam3"],
        help="Mask format (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save frame PNGs (default: result_dir/masked_frames/)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file (for SAM3 frame-index matching)",
    )
    parser.add_argument(
        "--sam3_class",
        type=str,
        default=None,
        help="Class name filter for SAM3 multi-class masks",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to original high-res images (instead of predictions.pt)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame indices to export (e.g., 0,5,10). Default: all",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Mask overlay transparency (default: 0.45)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.result_dir, "masked_frames")
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Load predictions.pt
    # =========================================================================
    print("=== Loading Predictions ===")
    predictions_path = os.path.join(args.result_dir, "predictions.pt")
    if not os.path.exists(predictions_path):
        print(f"Error: predictions.pt not found in {args.result_dir}")
        sys.exit(1)

    predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)

    def maybe_squeeze_batch(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    images = maybe_squeeze_batch(predictions["images"])  # (S, 3, H, W)
    S = images.shape[0]
    model_h, model_w = images.shape[2], images.shape[3]
    model_size = (model_h, model_w)
    print(f"Loaded {S} frames, model size: {model_h}x{model_w}")

    # =========================================================================
    # Step 2: Load tracking summary
    # =========================================================================
    print("\n=== Loading Tracking Summary ===")
    tracking_path = os.path.join(args.result_dir, "tracking_summary.json")
    if not os.path.exists(tracking_path):
        print(f"Error: tracking_summary.json not found in {args.result_dir}")
        sys.exit(1)

    with open(tracking_path, "r") as f:
        tracking = json.load(f)
    print(f"Loaded {tracking.get('total_tracks', 0)} tracks")

    # =========================================================================
    # Step 3: Load metadata
    # =========================================================================
    metadata_path = os.path.join(args.result_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # =========================================================================
    # Step 4: Load masks
    # =========================================================================
    print("\n=== Loading Masks ===")
    mask_format = args.mask_format
    if mask_format == "auto":
        from vggt.utils.sam3_mask_loader import detect_sam3_format

        format_info = detect_sam3_format(args.mask_source)
        if format_info["format"] in ("single_class", "multi_class"):
            mask_format = "sam3"
            print(f"Auto-detected: SAM3 ({format_info['format']})")
        else:
            mask_format = "grounded_sam"
            print("Auto-detected: Grounded SAM")

    masks_data = load_masks(
        mask_source=args.mask_source,
        mask_format=mask_format,
        metadata=metadata,
        video=args.video,
        sam3_class=args.sam3_class,
    )
    print(f"Loaded masks for {len(masks_data)} frames")

    # Determine original image shapes from mask data
    orig_shapes: Dict[int, Tuple[int, int]] = {}
    for frame_idx, frame_masks in masks_data.items():
        if frame_masks:
            mask = frame_masks[0]["mask"]
            orig_shapes[frame_idx] = mask.shape[:2]

    # =========================================================================
    # Step 5: Load original images if requested
    # =========================================================================
    original_images: Dict[int, np.ndarray] = {}
    if args.image_dir:
        print(f"\n=== Loading Original Images from {args.image_dir} ===")
        image_files = sorted(
            [
                f
                for f in os.listdir(args.image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        # If metadata has image_files, use that ordering
        if metadata.get("image_files"):
            image_files = metadata["image_files"]

        for i, fname in enumerate(image_files):
            if i >= S:
                break
            img_path = os.path.join(args.image_dir, fname)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    original_images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Loaded {len(original_images)} original images")

    # =========================================================================
    # Step 6: Determine which frames to export
    # =========================================================================
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(",")]
        frame_indices = [i for i in frame_indices if 0 <= i < S]
    else:
        frame_indices = list(range(S))

    # =========================================================================
    # Step 7: Render and save
    # =========================================================================
    print(f"\n=== Rendering {len(frame_indices)} frames ===")
    for i, frame_idx in enumerate(frame_indices):
        orig_img = original_images.get(frame_idx)

        rendered = render_masked_frame(
            frame_idx=frame_idx,
            images=images,
            masks_data=masks_data,
            tracking=tracking,
            model_size=model_size,
            orig_shapes=orig_shapes,
            alpha=args.alpha,
            original_image=orig_img,
        )

        out_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

        if (i + 1) % 5 == 0 or (i + 1) == len(frame_indices):
            print(f"  Saved {i + 1}/{len(frame_indices)} frames")

    print(f"\nDone! Saved {len(frame_indices)} masked frames to: {output_dir}")


if __name__ == "__main__":
    main()
