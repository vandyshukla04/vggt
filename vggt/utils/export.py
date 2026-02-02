# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Export utilities for saving VGGT outputs in portable formats.
"""

import json
import numpy as np
from typing import Dict, Optional, Tuple


def save_point_cloud_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    conf_threshold_percentile: float = 50.0,
) -> int:
    """Save point cloud to PLY format.

    Args:
        filepath: Output path for PLY file
        points: Point coordinates (N, 3) or (S, H, W, 3)
        colors: RGB colors (N, 3) or (S, H, W, 3), values in [0, 1] or [0, 255]
        confidence: Confidence values (N,) or (S, H, W)
        conf_threshold_percentile: Filter points below this confidence percentile

    Returns:
        Number of points saved
    """
    # Flatten if needed
    if points.ndim == 4:
        points = points.reshape(-1, 3)
    if colors is not None and colors.ndim == 4:
        colors = colors.reshape(-1, 3)
    if confidence is not None and confidence.ndim > 1:
        confidence = confidence.reshape(-1)

    # Apply confidence filtering
    if confidence is not None:
        threshold = np.percentile(confidence, conf_threshold_percentile)
        mask = (confidence >= threshold) & (confidence > 1e-5)
        points = points[mask]
        if colors is not None:
            colors = colors[mask]

    # Convert colors to uint8 if needed
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Filter invalid points
    valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]

    num_points = len(points)

    # Write PLY file
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(num_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    return num_points


def save_cameras_json(
    filepath: str,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_names: Optional[list] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Save camera parameters to JSON format.

    Args:
        filepath: Output path for JSON file
        extrinsics: Camera extrinsics (S, 3, 4) - world-to-camera transform
        intrinsics: Camera intrinsics (S, 3, 3)
        image_names: Optional list of image filenames
        image_size: Optional (height, width) tuple
    """
    S = extrinsics.shape[0]

    cameras = []
    for i in range(S):
        cam = {
            "frame_index": i,
            "extrinsic": extrinsics[i].tolist(),  # 3x4 [R|t]
            "intrinsic": intrinsics[i].tolist(),  # 3x3 K matrix
        }
        if image_names is not None and i < len(image_names):
            cam["image_name"] = image_names[i]
        if image_size is not None:
            cam["image_height"] = image_size[0]
            cam["image_width"] = image_size[1]
        cameras.append(cam)

    output = {
        "num_cameras": S,
        "cameras": cameras,
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def save_depth_maps(
    filepath: str,
    depth: np.ndarray,
    depth_conf: Optional[np.ndarray] = None,
) -> None:
    """Save depth maps to compressed numpy format.

    Args:
        filepath: Output path for .npz file
        depth: Depth maps (S, H, W) or (S, H, W, 1)
        depth_conf: Optional confidence maps (S, H, W)
    """
    # Squeeze if needed
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    data = {"depth": depth}
    if depth_conf is not None:
        data["depth_conf"] = depth_conf

    np.savez_compressed(filepath, **data)


def load_depth_maps(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load depth maps from compressed numpy format.

    Returns:
        Tuple of (depth, depth_conf) where depth_conf may be None
    """
    data = np.load(filepath)
    depth = data["depth"]
    depth_conf = data.get("depth_conf", None)
    return depth, depth_conf


def load_cameras_json(filepath: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load camera parameters from JSON format.

    Returns:
        Tuple of (extrinsics, intrinsics, image_names)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    S = data["num_cameras"]
    extrinsics = np.zeros((S, 3, 4))
    intrinsics = np.zeros((S, 3, 3))
    image_names = []

    for cam in data["cameras"]:
        i = cam["frame_index"]
        extrinsics[i] = np.array(cam["extrinsic"])
        intrinsics[i] = np.array(cam["intrinsic"])
        if "image_name" in cam:
            image_names.append(cam["image_name"])

    return extrinsics, intrinsics, image_names
