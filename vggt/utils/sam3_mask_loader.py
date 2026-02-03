# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for loading SAM3 segmentation masks.

SAM3 output format:
- masks/obj_{id}/frame_{idx:06d}.png - Binary PNG masks (0/255)
- metadata.json - Contains fps, resolution, object_ids, etc.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def load_sam3_metadata(sam3_output_dir: str) -> Dict[str, Any]:
    """
    Load SAM3 metadata.json file.

    Args:
        sam3_output_dir: Root directory of SAM3 output

    Returns:
        Dict with metadata (fps, resolution, object_ids, etc.)
        Returns empty dict with defaults if metadata.json not found
    """
    metadata_path = os.path.join(sam3_output_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.json not found at {metadata_path}, using defaults")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load metadata.json: {e}")
        return {}


def get_sam3_object_ids(sam3_output_dir: str) -> List[int]:
    """
    Discover object IDs from directory structure.

    Scans for directories matching pattern: masks/obj_{id}/

    Args:
        sam3_output_dir: Root directory of SAM3 output

    Returns:
        Sorted list of object IDs found
    """
    masks_dir = os.path.join(sam3_output_dir, "masks")

    if not os.path.exists(masks_dir):
        print(f"Warning: masks directory not found at {masks_dir}")
        return []

    object_ids = []
    for entry in os.listdir(masks_dir):
        if entry.startswith("obj_") and os.path.isdir(os.path.join(masks_dir, entry)):
            try:
                obj_id = int(entry.split("_")[1])
                object_ids.append(obj_id)
            except (ValueError, IndexError):
                continue

    return sorted(object_ids)


def load_sam3_mask_for_frame(
    sam3_output_dir: str,
    object_id: int,
    frame_idx: int,
    frame_pattern: str = "frame_{:06d}.png"
) -> Optional[np.ndarray]:
    """
    Load a single SAM3 binary mask for a specific object and frame.

    Args:
        sam3_output_dir: Root directory of SAM3 output
        object_id: Object ID (0, 1, 2, ...)
        frame_idx: Frame index (0-based)
        frame_pattern: Frame filename pattern

    Returns:
        Boolean numpy array (H, W) or None if mask doesn't exist
    """
    mask_filename = frame_pattern.format(frame_idx)
    mask_path = os.path.join(
        sam3_output_dir,
        "masks",
        f"obj_{object_id}",
        mask_filename
    )

    if not os.path.exists(mask_path):
        return None

    # Load grayscale PNG
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Convert to boolean (SAM3 uses 0/255)
    return mask > 127


def map_video_frame_to_sam3_frame(
    video_frame_idx: int,
    video_fps: float,
    sam3_fps: float,
    rounding: str = "nearest"
) -> int:
    """
    Map a video frame index to corresponding SAM3 frame index.

    Args:
        video_frame_idx: Original video frame index
        video_fps: Original video FPS
        sam3_fps: FPS used for SAM3 processing
        rounding: "nearest", "floor", or "ceil"

    Returns:
        Corresponding SAM3 frame index
    """
    if video_fps <= 0 or sam3_fps <= 0:
        return video_frame_idx

    # Convert video frame to time, then to SAM3 frame
    time_seconds = video_frame_idx / video_fps
    sam3_frame_float = time_seconds * sam3_fps

    if rounding == "floor":
        return int(sam3_frame_float)
    elif rounding == "ceil":
        return int(np.ceil(sam3_frame_float))
    else:  # nearest
        return int(round(sam3_frame_float))


def compute_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """
    Compute [x, y, width, height] bounding box from binary mask.

    Args:
        mask: Boolean numpy array (H, W)

    Returns:
        [x, y, w, h] in pixel coordinates, or [0, 0, 0, 0] if mask is empty
    """
    if mask is None or not mask.any():
        return [0, 0, 0, 0]

    # Find rows and columns with True values
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def load_sam3_masks(
    sam3_output_dir: str,
    extracted_frame_indices: List[int],
    video_fps: float,
    sam3_fps: Optional[float] = None,
    class_name: str = "object",
    direct_frame_match: bool = False
) -> Dict[int, List[Dict]]:
    """
    Load SAM3 masks for all extracted frames.

    This is the main function that replaces load_grounded_sam_masks().

    Args:
        sam3_output_dir: Root directory of SAM3 output
        extracted_frame_indices: List of original video frame indices for each extracted frame.
            These are the ACTUAL video frame numbers (0-indexed) which are used for:
            - DJI telemetry matching (SRT FrameCnt)
            - SAM3 mask file matching when direct_frame_match=True
        video_fps: Original video FPS
        sam3_fps: FPS used for SAM3 processing (from metadata or argument)
        class_name: Default class name for all objects (SAM3 doesn't provide classes)
        direct_frame_match: If True, use video frame indices directly as SAM3 frame indices
            (useful when extraction fps matches sam3 fps, or when --start_frame/--end_frame
            are used to match SAM3 mask frame numbers)

    Returns:
        Dict mapping extracted frame index (0, 1, 2, ...) to list of mask dicts:
        {
            frame_idx: [
                {
                    'mask': np.ndarray (H, W, bool),
                    'class_name': str,
                    'score': float (default 1.0 for SAM3),
                    'bbox': [x, y, w, h],
                    'object_id': int  # SAM3 object ID for tracking continuity
                },
                ...
            ]
        }
    """
    # Load metadata to get sam3_fps if not provided
    metadata = load_sam3_metadata(sam3_output_dir)

    if sam3_fps is None:
        sam3_fps = metadata.get('fps', metadata.get('effective_fps', video_fps))
        print(f"Using SAM3 FPS from metadata: {sam3_fps}")

    # Determine if we should use direct frame matching
    # If extraction fps equals sam3 fps (within tolerance), use direct matching
    if not direct_frame_match and sam3_fps is not None and video_fps is not None:
        if abs(sam3_fps - video_fps) < 0.1:
            direct_frame_match = True
            print(f"Extraction FPS ({video_fps}) matches SAM3 FPS ({sam3_fps}), using direct frame matching")

    # Discover all object IDs
    object_ids = get_sam3_object_ids(sam3_output_dir)
    if not object_ids:
        print(f"Warning: No object directories found in {sam3_output_dir}/masks/")
        return {}

    print(f"Found {len(object_ids)} objects: {object_ids}")
    if direct_frame_match:
        print(f"Using direct frame matching (video frame idx = SAM3 frame idx)")
    else:
        print(f"Using FPS-based frame mapping (video_fps={video_fps}, sam3_fps={sam3_fps})")

    masks_data = {}
    total_masks_loaded = 0
    missing_frames = []

    for i, video_frame_idx in enumerate(extracted_frame_indices):
        # Map to SAM3 frame index
        if direct_frame_match:
            sam3_frame_idx = video_frame_idx
        else:
            sam3_frame_idx = map_video_frame_to_sam3_frame(
                video_frame_idx, video_fps, sam3_fps
            )

        frame_masks = []
        frame_has_mask = False
        for obj_id in object_ids:
            mask = load_sam3_mask_for_frame(
                sam3_output_dir, obj_id, sam3_frame_idx
            )

            if mask is not None and mask.sum() > 0:
                bbox = compute_bbox_from_mask(mask)
                frame_masks.append({
                    'mask': mask,
                    'class_name': class_name,
                    'score': 1.0,  # SAM3 doesn't provide confidence scores
                    'bbox': bbox,
                    'object_id': obj_id
                })
                total_masks_loaded += 1
                frame_has_mask = True

        if not frame_has_mask:
            missing_frames.append(sam3_frame_idx)

        masks_data[i] = frame_masks

    print(f"Loaded {total_masks_loaded} masks across {len(extracted_frame_indices)} frames")
    if missing_frames:
        print(f"Warning: No masks found for SAM3 frame indices: {missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}")

    return masks_data


def load_sam3_masks_from_image_paths(
    sam3_output_dir: str,
    image_paths: List[str],
    sam3_fps: Optional[float] = None,
    class_name: str = "object"
) -> Dict[int, List[Dict]]:
    """
    Load SAM3 masks matching image file names.

    This version extracts frame indices from image filenames instead of
    requiring explicit video frame indices. Useful when working with
    pre-extracted frames that follow SAM3 naming convention.

    Args:
        sam3_output_dir: Root directory of SAM3 output
        image_paths: List of image file paths
        sam3_fps: Not used in this version (direct name matching)
        class_name: Default class name for all objects

    Returns:
        Dict mapping frame index to list of mask dicts
    """
    import re

    # Discover all object IDs
    object_ids = get_sam3_object_ids(sam3_output_dir)
    if not object_ids:
        print(f"Warning: No object directories found in {sam3_output_dir}/masks/")
        return {}

    print(f"Found {len(object_ids)} objects: {object_ids}")

    masks_data = {}
    total_masks_loaded = 0

    for i, image_path in enumerate(image_paths):
        # Extract frame number from filename
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]

        # Try to extract frame index from filename
        # Supports patterns like: frame_000001, 000001, 1, etc.
        match = re.search(r'(\d+)', name_without_ext)
        if match:
            sam3_frame_idx = int(match.group(1))
        else:
            # Fallback to sequential index
            sam3_frame_idx = i

        frame_masks = []
        for obj_id in object_ids:
            mask = load_sam3_mask_for_frame(
                sam3_output_dir, obj_id, sam3_frame_idx
            )

            if mask is not None and mask.sum() > 0:
                bbox = compute_bbox_from_mask(mask)
                frame_masks.append({
                    'mask': mask,
                    'class_name': class_name,
                    'score': 1.0,
                    'bbox': bbox,
                    'object_id': obj_id
                })
                total_masks_loaded += 1

        masks_data[i] = frame_masks

    print(f"Loaded {total_masks_loaded} masks across {len(image_paths)} frames")
    return masks_data
