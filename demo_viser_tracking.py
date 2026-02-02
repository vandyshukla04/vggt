# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Demo with Instance Tracking and 3D Bounding Boxes

This demo combines:
- VGGT's 3D reconstruction pipeline
- Instance segmentation via pre-computed Grounded SAM masks
- Kalman-based tracking with dormant re-identification (from CUT3R's retrack_3d.py)
- 3D bounding box visualization using PCA-based oriented bounding boxes
"""

import os
import re
import glob
import time
import json
import threading
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from scipy.optimize import linear_sum_assignment

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not found. Install with: pip install pycocotools")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# =============================================================================
# Color Palette for Track Visualization
# =============================================================================

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


def get_track_color(track_id: int, all_track_ids: List[int]) -> Tuple[float, float, float]:
    """Get consistent color for a track ID."""
    sorted_ids = sorted(all_track_ids)
    idx = sorted_ids.index(track_id) if track_id in sorted_ids else track_id
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


# =============================================================================
# BoundingBox3D Class
# =============================================================================

class BoundingBox3D:
    """3D Bounding Box representation (adapted from CUT3R demo_masks_fixed.py)"""

    def __init__(self, center: np.ndarray, dimensions: np.ndarray,
                 rotation_matrix: np.ndarray, class_name: str,
                 confidence: float, instance_id: int):
        self.center = np.array(center)  # [x, y, z]
        self.dimensions = np.array(dimensions)  # [length, width, height]
        self.rotation_matrix = np.array(rotation_matrix)  # 3x3 rotation matrix
        self.class_name = class_name
        self.confidence = confidence
        self.instance_id = instance_id
        self.track_id = None
        self.persistent_instance_id = None
        self.mask = None  # 2D mask for tracking

    def get_corners(self) -> np.ndarray:
        """Get 8 corner points of the bounding box"""
        l, w, h = self.dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2],
            [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2],
            [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2],
            [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2],
            [-l/2, +w/2, +h/2],
        ])
        corners_world = (self.rotation_matrix @ corners_local.T).T + self.center
        return corners_world

    def get_wireframe_edges(self) -> Tuple[np.ndarray, List[List[int]]]:
        """Get edges for wireframe visualization"""
        corners = self.get_corners()
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        return corners, edges


# =============================================================================
# Kalman Track (from CUT3R retrack_3d.py)
# =============================================================================

class KalmanTrack:
    """Per-track 3D Kalman filter with constant-velocity model.
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    def __init__(self, center: np.ndarray, track_id: int, class_name: str,
                 confidence: float, dt: float = 1.0):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.dt = dt

        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.x[:3] = center

        # State covariance
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0  # High initial velocity uncertainty

        # Transition matrix (constant velocity)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        # Process noise
        q = 0.05
        self.Q = np.eye(6) * q
        self.Q[3:, 3:] *= 2.0

        # Measurement noise
        self.R = np.eye(3) * 0.1

        # Track metadata
        self.frames_missing = 0
        self.detection_count = 1
        self.first_frame = -1
        self.last_frame = -1
        self.last_mask = None
        self.last_confidence = confidence

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    @property
    def predicted_center(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def update(self, measurement: np.ndarray):
        """Update state with measurement [x, y, z]."""
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.frames_missing = 0
        self.detection_count += 1


# =============================================================================
# Improved Tracker (from CUT3R retrack_3d.py)
# =============================================================================

class ImprovedTracker:
    """3D tracker with Kalman prediction and dormant track re-identification."""

    def __init__(self, max_distance: float = 8.0, mask_iou_threshold: float = 0.15,
                 distance_weight: float = 0.7, iou_weight: float = 0.3,
                 max_missing_frames: int = 20, dormant_timeout: int = 100):
        self.max_distance = max_distance
        self.mask_iou_threshold = mask_iou_threshold
        self.distance_weight = distance_weight
        self.iou_weight = iou_weight
        self.max_missing_frames = max_missing_frames
        self.dormant_timeout = dormant_timeout

        self.active_tracks: Dict[int, KalmanTrack] = {}
        self.dormant_tracks: Dict[int, KalmanTrack] = {}
        self.next_track_id = 0

    def update(self, detections: List[Dict], frame_idx: int) -> List[int]:
        """Process detections for one frame.

        detections: list of dicts with keys: center, class_name, confidence, mask, dimensions, rotation_matrix
        Returns: list of assigned track_ids (same order as detections)
        """
        # First frame — initialize all tracks
        if not self.active_tracks and not self.dormant_tracks:
            return self._initialize_tracks(detections, frame_idx)

        # Predict all active tracks
        for track in self.active_tracks.values():
            track.predict()
        for track in self.dormant_tracks.values():
            track.predict()

        if len(detections) == 0:
            self._increment_missing(frame_idx)
            return []

        # Build cost matrix for active tracks
        track_ids = list(self.active_tracks.keys())
        n_det = len(detections)
        n_trk = len(track_ids)

        assignments: Dict[int, int] = {}
        unmatched_dets = set(range(n_det))
        matched_tracks = set()

        if n_trk > 0:
            cost_matrix = np.full((n_det, n_trk), 1e6)

            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    track = self.active_tracks[tid]

                    # Class filter
                    if det['class_name'].lower() != track.class_name.lower():
                        continue

                    # 3D distance (predicted vs measured)
                    dist = np.linalg.norm(det['center'] - track.predicted_center)
                    if dist > self.max_distance:
                        continue

                    # Mask IoU
                    iou = self._compute_mask_iou(det.get('mask'), track.last_mask)
                    if iou < self.mask_iou_threshold and track.last_mask is not None and det.get('mask') is not None:
                        if dist > 1.0:
                            continue

                    dist_cost = dist / self.max_distance
                    iou_cost = 1.0 - iou
                    cost_matrix[i, j] = self.distance_weight * dist_cost + self.iou_weight * iou_cost

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e5:
                    tid = track_ids[c]
                    assignments[r] = tid
                    unmatched_dets.discard(r)
                    matched_tracks.add(tid)

                    # Update track
                    track = self.active_tracks[tid]
                    track.update(detections[r]['center'])
                    track.last_mask = detections[r].get('mask')
                    track.last_confidence = detections[r]['confidence']
                    track.last_frame = frame_idx

        # Try to re-identify unmatched detections from dormant tracks
        still_unmatched = set()
        for det_idx in unmatched_dets:
            reid_tid = self._try_reidentify(detections[det_idx], frame_idx)
            if reid_tid is not None:
                assignments[det_idx] = reid_tid
            else:
                still_unmatched.add(det_idx)

        # Create new tracks for remaining unmatched detections
        for det_idx in still_unmatched:
            det = detections[det_idx]
            tid = self.next_track_id
            self.next_track_id += 1
            track = KalmanTrack(det['center'], tid, det['class_name'], det['confidence'])
            track.first_frame = frame_idx
            track.last_frame = frame_idx
            track.last_mask = det.get('mask')
            self.active_tracks[tid] = track
            assignments[det_idx] = tid

        # Handle unmatched active tracks
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_tracks:
                self.active_tracks[tid].frames_missing += 1
                if self.active_tracks[tid].frames_missing > self.max_missing_frames:
                    self._move_to_dormant(tid)

        # Expire old dormant tracks
        for tid in list(self.dormant_tracks.keys()):
            if self.dormant_tracks[tid].frames_missing > self.dormant_timeout:
                del self.dormant_tracks[tid]

        return [assignments.get(i, -1) for i in range(n_det)]

    def _initialize_tracks(self, detections: List[Dict], frame_idx: int) -> List[int]:
        track_ids = []
        for det in detections:
            tid = self.next_track_id
            self.next_track_id += 1
            track = KalmanTrack(det['center'], tid, det['class_name'], det['confidence'])
            track.first_frame = frame_idx
            track.last_frame = frame_idx
            track.last_mask = det.get('mask')
            self.active_tracks[tid] = track
            track_ids.append(tid)
        return track_ids

    def _move_to_dormant(self, track_id: int):
        track = self.active_tracks.pop(track_id)
        self.dormant_tracks[track_id] = track

    def _try_reidentify(self, detection: Dict, frame_idx: int) -> Optional[int]:
        """Try to match a detection against dormant tracks."""
        best_tid = None
        best_cost = 1e6

        for tid, track in self.dormant_tracks.items():
            if detection['class_name'].lower() != track.class_name.lower():
                continue

            dist = np.linalg.norm(detection['center'] - track.predicted_center)
            if dist > self.max_distance * 2.0:
                continue

            cost = dist / self.max_distance
            if cost < best_cost:
                best_cost = cost
                best_tid = tid

        if best_tid is not None and best_cost < 1.5:
            track = self.dormant_tracks.pop(best_tid)
            track.update(detection['center'])
            track.last_mask = detection.get('mask')
            track.last_confidence = detection['confidence']
            track.last_frame = frame_idx
            self.active_tracks[best_tid] = track
            return best_tid

        return None

    def _increment_missing(self, frame_idx: int):
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid].frames_missing += 1
            if self.active_tracks[tid].frames_missing > self.max_missing_frames:
                self._move_to_dormant(tid)
        for tid in list(self.dormant_tracks.keys()):
            if self.dormant_tracks[tid].frames_missing > self.dormant_timeout:
                del self.dormant_tracks[tid]

    @staticmethod
    def _compute_mask_iou(mask1: Optional[np.ndarray], mask2: Optional[np.ndarray]) -> float:
        if mask1 is None or mask2 is None:
            return 0.0
        if mask1.shape != mask2.shape:
            return 0.0
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 0.0
        return intersection / union


# =============================================================================
# DJI SRT Log Parsing
# =============================================================================

def parse_dji_logs(log_file_path: str, frame_indices: List[int]) -> Optional[Dict]:
    """Parse DJI SRT log file to extract gimbal data for specific frames."""
    gimbal_data = {}

    if not os.path.exists(log_file_path):
        print(f"DJI log file not found: {log_file_path}")
        return None

    print(f"Parsing DJI log: {log_file_path}")
    with open(log_file_path, 'r') as f:
        content = f.read()

    print(f"SRT file size: {len(content):,} characters")

    # Pattern for gimbal data
    pattern = r'FrameCnt: (\d+).*?\[rel_alt: ([\d.]+).*?\[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+) gb_roll: ([-\d.]+)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(matches):,} gimbal entries in SRT file")

    if not matches:
        print("No gimbal data found in SRT file")
        return None

    srt_frame_indices = [int(match[0]) for match in matches]
    srt_min, srt_max = min(srt_frame_indices), max(srt_frame_indices)
    print(f"SRT FrameCnt range: {srt_min:,} -> {srt_max:,}")

    if frame_indices:
        img_min, img_max = min(frame_indices), max(frame_indices)
        print(f"Image frame range: {img_min:,} -> {img_max:,}")

        direct_matches = set(frame_indices) & set(srt_frame_indices)
        print(f"Direct frame matches: {len(direct_matches):,} out of {len(frame_indices):,}")

        if len(direct_matches) == 0:
            print("Warning: No direct matches! Using ALL SRT frames instead")
            target_frames = set(srt_frame_indices)
        else:
            target_frames = set(frame_indices)
    else:
        target_frames = set(srt_frame_indices)

    for frame_cnt_str, altitude, yaw, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            gimbal_data[frame_cnt] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'altitude': float(altitude)
            }

    print(f"Parsed gimbal data for {len(gimbal_data):,} frames")
    return gimbal_data if gimbal_data else None


def parse_dji_logs_with_gps(log_file_path: str, frame_indices: List[int]) -> Optional[Dict]:
    """Parse DJI SRT log file to extract gimbal AND GPS data."""
    data = {}

    if not os.path.exists(log_file_path):
        print(f"DJI log file not found: {log_file_path}")
        return None

    with open(log_file_path, 'r') as f:
        content = f.read()

    # Extended pattern with GPS
    pattern = r'FrameCnt: (\d+).*?\[focal_len: ([\d.]+)\].*?\[latitude: ([-\d.]+)\] \[longitude: ([-\d.]+)\] \[rel_alt: ([\d.]+) abs_alt: ([\d.]+)\] \[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+) gb_roll: ([-\d.]+)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        print("GPS pattern not matched, falling back to gimbal-only parsing")
        return parse_dji_logs(log_file_path, frame_indices)

    print(f"Found {len(matches):,} entries with GPS data")

    srt_frame_indices = [int(match[0]) for match in matches]
    target_frames = set(frame_indices) if frame_indices else set(srt_frame_indices)

    for frame_cnt_str, focal_len, latitude, longitude, rel_alt, abs_alt, yaw, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'altitude': float(rel_alt),
                'abs_altitude': float(abs_alt),
                'latitude': float(latitude),
                'longitude': float(longitude),
                'focal_len': float(focal_len),
            }

    print(f"Parsed GPS data for {len(data):,} frames")
    return data if data else None


# =============================================================================
# Mask Loading
# =============================================================================

def decode_rle_mask(rle_data: Dict) -> Optional[np.ndarray]:
    """Decode RLE mask using pycocotools."""
    if not HAS_PYCOCOTOOLS:
        return None
    try:
        if isinstance(rle_data, dict) and 'size' in rle_data and 'counts' in rle_data:
            decoded = mask_utils.decode(rle_data)
            return decoded.astype(bool)
    except Exception as e:
        print(f"RLE decode failed: {e}")
    return None


def load_grounded_sam_masks(mask_dir: str, img_paths: List[str]) -> Dict[int, List[Dict]]:
    """Load Grounded SAM masks from JSON files."""
    masks_data = {}

    for i, img_path in enumerate(img_paths):
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_file = os.path.join(mask_dir, f"{frame_name}_results.json")

        if os.path.exists(mask_file):
            try:
                with open(mask_file, 'r') as f:
                    data = json.load(f)

                print(f"Loading masks for {frame_name}: {len(data['annotations'])} annotations")

                frame_masks = []
                for ann_idx, ann in enumerate(data['annotations']):
                    try:
                        mask = decode_rle_mask(ann['segmentation'])

                        if mask is not None and np.sum(mask) > 0:
                            score = ann['score'][0] if isinstance(ann['score'], list) else ann['score']
                            frame_masks.append({
                                'mask': mask,
                                'class_name': ann['class_name'],
                                'score': score,
                                'bbox': ann['bbox']
                            })
                            print(f"  - {ann['class_name']}: {mask.shape}, {np.sum(mask)} pixels")
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


# =============================================================================
# 3D BBox Computation
# =============================================================================

def compute_oriented_bbox_pca(points_3d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute oriented bounding box using PCA (numpy SVD implementation)."""
    if len(points_3d) < 3:
        return None

    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid

    # PCA using SVD
    _, _, Vt = np.linalg.svd(centered_points, full_matrices=False)
    components = Vt  # Principal components (rows are eigenvectors)

    # Transform points to PCA space
    transformed_points = centered_points @ components.T
    min_vals = np.min(transformed_points, axis=0)
    max_vals = np.max(transformed_points, axis=0)
    dimensions = max_vals - min_vals

    # Center in PCA space, then transform back to world
    center_pca = (min_vals + max_vals) / 2
    center_world = (center_pca @ components) + centroid
    rotation_matrix = components.T  # Columns are principal axes

    return center_world, dimensions, rotation_matrix


def filter_outliers(points_3d: np.ndarray, outlier_factor: float = 1.5) -> np.ndarray:
    """Remove outlier points using IQR filtering."""
    if len(points_3d) < 10:
        return points_3d

    points_3d = np.array(points_3d)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        if points_3d.size % 3 == 0:
            points_3d = points_3d.reshape(-1, 3)
        else:
            return points_3d

    centroid = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - centroid, axis=1)

    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - outlier_factor * iqr)
    upper_bound = q3 + outlier_factor * iqr
    inliers = (distances >= lower_bound) & (distances <= upper_bound)

    return points_3d[inliers]


# =============================================================================
# Ground Plane Alignment (from CUT3R demo_masks_fixed.py)
# =============================================================================

def get_ground_plane_from_gimbal(gimbal_data: Dict) -> Optional[Tuple[np.ndarray, float]]:
    """Extract ground plane normal from gimbal data.

    Args:
        gimbal_data: Dict mapping frame indices to gimbal info (pitch, roll, yaw, altitude)

    Returns:
        Tuple of (ground_normal, avg_altitude) or None if no gimbal data
    """
    if not gimbal_data:
        return None

    # Average gimbal angles across all frames for stability
    pitches = [data['pitch'] for data in gimbal_data.values()]
    rolls = [data['roll'] for data in gimbal_data.values()]
    altitudes = [data['altitude'] for data in gimbal_data.values()]

    avg_pitch = np.mean(pitches)
    avg_roll = np.mean(rolls)
    avg_altitude = np.mean(altitudes)

    print(f"Gimbal summary: pitch={avg_pitch:.1f} deg, roll={avg_roll:.1f} deg, alt={avg_altitude:.1f}m")

    # Convert to radians
    pitch_rad = np.radians(avg_pitch)
    roll_rad = np.radians(avg_roll)

    # Compute ground normal from gimbal orientation
    # For DJI: pitch negative means camera pointing down
    # Ground normal points UP from the ground plane
    ground_normal = np.array([
        np.sin(roll_rad),                          # X component from roll
        np.cos(pitch_rad) * np.cos(roll_rad),      # Y component (up direction)
        -np.sin(pitch_rad) * np.cos(roll_rad)      # Z component from pitch
    ])

    # Normalize to unit vector
    ground_normal = ground_normal / np.linalg.norm(ground_normal)

    print(f"Computed ground normal: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")

    return ground_normal, avg_altitude


def align_bbox_to_ground_plane(bbox: BoundingBox3D, ground_normal: np.ndarray) -> BoundingBox3D:
    """Align bounding box orientation to ground plane.

    This ensures the bounding box's "up" axis aligns with the ground normal,
    which provides more consistent box orientations across frames.

    Args:
        bbox: Input bounding box with PCA-derived rotation
        ground_normal: Unit vector pointing "up" from ground plane

    Returns:
        New BoundingBox3D with ground-aligned rotation
    """
    # Current bbox has rotation matrix from PCA
    original_rotation = bbox.rotation_matrix.copy()

    # Ground normal is our "up" direction (Y-axis)
    up_axis = ground_normal / np.linalg.norm(ground_normal)

    # Find the PCA axis most aligned with ground normal (this becomes our height)
    pca_axes = original_rotation  # PCA components are columns

    # Find which PCA axis is most aligned with up direction
    dots = [abs(np.dot(up_axis, axis)) for axis in pca_axes.T]
    height_axis_idx = np.argmax(dots)

    # Reorder dimensions: height axis becomes the Y (up) axis
    old_dims = bbox.dimensions.copy()
    old_axes = pca_axes.T.copy()

    # Create new aligned rotation matrix
    # Y-axis: align with ground normal (up)
    new_y = up_axis

    # X-axis: project one of the remaining PCA axes onto ground plane
    remaining_axes = [i for i in range(3) if i != height_axis_idx]
    candidate_x = old_axes[remaining_axes[0]]

    # Project onto ground plane (remove component along ground normal)
    new_x = candidate_x - np.dot(candidate_x, up_axis) * up_axis
    if np.linalg.norm(new_x) < 1e-6:
        # Fallback if candidate is parallel to up
        candidate_x = old_axes[remaining_axes[1]] if len(remaining_axes) > 1 else np.array([1, 0, 0])
        new_x = candidate_x - np.dot(candidate_x, up_axis) * up_axis
    new_x = new_x / np.linalg.norm(new_x)

    # Z-axis: cross product to complete right-handed system
    new_z = np.cross(new_x, new_y)
    new_z = new_z / np.linalg.norm(new_z)

    # Construct aligned rotation matrix
    aligned_rotation = np.column_stack([new_x, new_y, new_z])

    # Reorder dimensions to match new axes
    # Height dimension goes to Y, others distributed to X and Z
    aligned_dims = bbox.dimensions.copy()
    aligned_dims[1] = old_dims[height_axis_idx]  # Height -> Y

    remaining_dims = [old_dims[i] for i in remaining_axes]
    aligned_dims[0] = remaining_dims[0]  # -> X
    aligned_dims[2] = remaining_dims[1] if len(remaining_dims) > 1 else remaining_dims[0]  # -> Z

    # Create aligned bounding box
    aligned_bbox = BoundingBox3D(
        center=bbox.center.copy(),
        dimensions=aligned_dims,
        rotation_matrix=aligned_rotation,
        class_name=bbox.class_name,
        confidence=bbox.confidence,
        instance_id=bbox.instance_id
    )
    aligned_bbox.track_id = bbox.track_id
    aligned_bbox.mask = bbox.mask

    return aligned_bbox


def transform_mask_to_model_coordinates(mask: np.ndarray, original_shape: Tuple[int, int],
                                         model_shape: Tuple[int, int]) -> np.ndarray:
    """Transform mask from original image coordinates to model coordinates."""
    orig_h, orig_w = original_shape
    model_h, model_w = model_shape

    # Letterbox scaling
    scale = min(model_w / orig_w, model_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    pad_x = (model_w - new_w) // 2
    pad_y = (model_h - new_h) // 2

    mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    mask_padded = np.zeros((model_h, model_w), dtype=bool)
    end_y = pad_y + new_h
    end_x = pad_x + new_w
    mask_padded[pad_y:end_y, pad_x:end_x] = mask_resized

    return mask_padded


def compute_instance_bboxes(world_points: np.ndarray, masks_data: Dict[int, List[Dict]],
                            original_images: List[np.ndarray], model_size: Tuple[int, int],
                            tracker: ImprovedTracker,
                            gimbal_data: Optional[Dict] = None) -> List[List[BoundingBox3D]]:
    """Compute 3D bounding boxes for all instances across frames and apply tracking.

    Args:
        world_points: 3D point cloud (S, H, W, 3)
        masks_data: Dict mapping frame index to list of mask dicts
        original_images: List of original images
        model_size: (H, W) tuple of model output size
        tracker: ImprovedTracker instance
        gimbal_data: Optional dict of gimbal data for ground plane alignment
    """
    all_bboxes = []
    S, H, W, _ = world_points.shape

    print("\n=== Computing 3D Bounding Boxes ===")

    # Compute ground plane from gimbal data if available
    ground_info = get_ground_plane_from_gimbal(gimbal_data) if gimbal_data else None
    ground_normal = ground_info[0] if ground_info else None

    if ground_normal is not None:
        print(f"Using gimbal-based ground plane alignment")
    else:
        print(f"No gimbal data - using PCA-only bbox orientation")

    for frame_idx in range(S):
        frame_bboxes = []

        if frame_idx not in masks_data or len(masks_data[frame_idx]) == 0:
            all_bboxes.append(frame_bboxes)
            continue

        # Get original image shape
        orig_img = original_images[frame_idx] if frame_idx < len(original_images) else None
        if orig_img is not None:
            if orig_img.ndim == 3 and orig_img.shape[0] == 3:  # (C, H, W)
                orig_h, orig_w = orig_img.shape[1], orig_img.shape[2]
            else:
                orig_h, orig_w = orig_img.shape[:2]
        else:
            orig_h, orig_w = H, W

        pts3d = world_points[frame_idx]  # (H, W, 3)

        detections_for_tracker = []

        for mask_idx, mask_info in enumerate(masks_data[frame_idx]):
            mask = mask_info['mask']
            class_name = mask_info['class_name'].lower()
            confidence = mask_info['score']

            # Skip background classes
            if class_name in {'ground', 'sky', 'background'}:
                continue

            # Transform mask to model coordinates
            mask_transformed = transform_mask_to_model_coordinates(
                mask, (orig_h, orig_w), (H, W)
            )

            # Get 3D points for this instance
            instance_points = pts3d[mask_transformed]

            if len(instance_points) < 10:
                continue

            # Filter outliers
            filtered_points = filter_outliers(instance_points)

            if len(filtered_points) < 5:
                continue

            # Compute oriented bounding box
            bbox_result = compute_oriented_bbox_pca(filtered_points)
            if bbox_result is None:
                continue

            center, dimensions, rotation = bbox_result

            # Create detection dict for tracker
            det = {
                'center': center,
                'dimensions': dimensions,
                'rotation_matrix': rotation,
                'class_name': class_name,
                'confidence': confidence,
                'mask': mask_transformed,
                'instance_id': mask_idx + 1,
            }
            detections_for_tracker.append(det)

        # Apply tracking
        track_ids = tracker.update(detections_for_tracker, frame_idx)

        # Create BoundingBox3D objects with track IDs
        for det, tid in zip(detections_for_tracker, track_ids):
            bbox = BoundingBox3D(
                center=det['center'],
                dimensions=det['dimensions'],
                rotation_matrix=det['rotation_matrix'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                instance_id=det['instance_id']
            )
            bbox.track_id = tid
            bbox.mask = det['mask']

            # Apply ground plane alignment if gimbal data is available
            if ground_normal is not None:
                bbox = align_bbox_to_ground_plane(bbox, ground_normal)

            frame_bboxes.append(bbox)

        print(f"Frame {frame_idx}: {len(frame_bboxes)} bboxes, {len(tracker.active_tracks)} active tracks")
        all_bboxes.append(frame_bboxes)

    return all_bboxes


# =============================================================================
# Frame Selection
# =============================================================================

def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like '2004.jpg' -> 2004."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Try to extract number from filename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    return 0


def select_frames(image_folder: str, num_images: int, skip: int) -> Tuple[List[str], List[int]]:
    """Select frames based on skip factor and limit."""
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    image_files = sorted(image_files)

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    print(f"Found {len(image_files)} total images in folder")

    # Apply skip factor
    selected_files = image_files[::skip]

    # Limit to num_images
    if num_images is not None and len(selected_files) > num_images:
        selected_files = selected_files[:num_images]

    # Extract frame numbers for telemetry matching
    frame_numbers = [extract_frame_number(f) for f in selected_files]

    print(f"Selected {len(selected_files)} images (skip={skip})")
    print(f"Frame numbers: {frame_numbers[:5]}..." if len(frame_numbers) > 5 else f"Frame numbers: {frame_numbers}")

    return selected_files, frame_numbers


# =============================================================================
# 2D Projection
# =============================================================================

def draw_3d_bbox_wireframe(img: np.ndarray, corners_2d: np.ndarray,
                           color_bgr: Tuple[int, int, int], thickness: int = 2):
    """Draw 3D bounding box wireframe."""
    try:
        c = corners_2d.astype(int)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        for a, b in edges:
            cv2.line(img, tuple(c[a]), tuple(c[b]), color_bgr, thickness)
    except Exception:
        pass


def project_bboxes_to_2d(bounding_boxes: List[List[BoundingBox3D]],
                          original_images: List[np.ndarray],
                          extrinsics: np.ndarray, intrinsics: np.ndarray,
                          output_dir: str, frame_names: List[str],
                          all_track_ids: List[int],
                          model_size: Tuple[int, int] = None):
    """Project 3D bounding boxes to 2D images.

    Args:
        model_size: (H, W) tuple of model output size. If provided, images are resized
                   to match intrinsics which are calibrated for model size.
    """
    annotated_dir = os.path.join(output_dir, "annotated_2d")
    os.makedirs(annotated_dir, exist_ok=True)

    print("\n=== Projecting 3D BBoxes to 2D ===")

    for frame_idx, (frame_bboxes, orig_img) in enumerate(zip(bounding_boxes, original_images)):
        if len(frame_bboxes) == 0:
            continue

        # Convert image format
        if torch.is_tensor(orig_img):
            if orig_img.dim() == 4:
                img_np = orig_img[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_np = orig_img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_display = (img_np * 255).astype(np.uint8)
            else:
                img_display = img_np.astype(np.uint8)
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        else:
            img_np = orig_img
            if img_np.dtype != np.uint8:
                img_display = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            else:
                img_display = img_np.copy()

        # Resize image to model size if specified (intrinsics are calibrated for model size)
        if model_size is not None:
            model_h, model_w = model_size
            img_display = cv2.resize(img_display, (model_w, model_h))

        # Get camera parameters
        K = intrinsics[frame_idx]  # (3, 3)
        ext = extrinsics[frame_idx]  # (3, 4)

        # Build 4x4 pose matrix
        camera_pose = np.eye(4)
        camera_pose[:3, :] = ext

        annotated_img = img_display.copy()

        for bbox in frame_bboxes:
            if bbox.track_id is None or bbox.track_id < 0:
                continue

            color_rgb = get_track_color(bbox.track_id, all_track_ids)
            color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])

            corners_3d = bbox.get_corners()

            try:
                # Transform to camera coordinates
                corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
                corners_cam = (camera_pose @ corners_h.T)[:3].T

                # Check if points are in front of camera
                if np.any(corners_cam[:, 2] <= 0):
                    continue

                # Project to image coordinates
                corners_2d_h = (K @ corners_cam.T).T
                corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

                # Check bounds
                img_h, img_w = img_display.shape[:2]
                if np.any(corners_2d < -img_w) or np.any(corners_2d > 2*img_w):
                    continue

                draw_3d_bbox_wireframe(annotated_img, corners_2d, color_bgr)

                # Add label
                center_2d = np.mean(corners_2d, axis=0).astype(int)
                center_2d[0] = max(10, min(center_2d[0], img_w - 100))
                center_2d[1] = max(20, min(center_2d[1], img_h - 10))
                label = f"T{bbox.track_id}: {bbox.class_name}"
                cv2.putText(annotated_img, label, tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

            except Exception as e:
                continue

        # Save annotated image
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx:06d}"
        output_path = os.path.join(annotated_dir, f"{frame_name}_tracked.png")
        cv2.imwrite(output_path, annotated_img)

    print(f"Saved annotated frames to {annotated_dir}")


# =============================================================================
# Output Functions
# =============================================================================

def save_tracking_summary(output_dir: str, bounding_boxes: List[List[BoundingBox3D]],
                          all_track_ids: List[int]):
    """Save tracking summary to JSON."""
    track_info = {}

    for frame_idx, frame_bboxes in enumerate(bounding_boxes):
        for bbox in frame_bboxes:
            if bbox.track_id is None or bbox.track_id < 0:
                continue

            tid = bbox.track_id
            if tid not in track_info:
                track_info[tid] = {
                    'class_name': bbox.class_name,
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'frames': [],
                    'centers': [],
                    'confidences': [],
                }

            track_info[tid]['last_frame'] = frame_idx
            track_info[tid]['frames'].append(frame_idx)
            track_info[tid]['centers'].append(bbox.center.tolist())
            track_info[tid]['confidences'].append(bbox.confidence)

    # Compute statistics
    for tid, info in track_info.items():
        info['length'] = len(info['frames'])
        info['avg_confidence'] = float(np.mean(info['confidences']))

    summary = {
        'total_tracks': len(track_info),
        'total_frames': len(bounding_boxes),
        'total_detections': sum(len(fb) for fb in bounding_boxes),
        'avg_tracklet_length': float(np.mean([t['length'] for t in track_info.values()])) if track_info else 0,
        'max_tracklet_length': max([t['length'] for t in track_info.values()]) if track_info else 0,
        'tracks': {str(k): v for k, v in track_info.items()},
    }

    summary_path = os.path.join(output_dir, 'tracking_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved tracking summary to {summary_path}")
    print(f"  Total tracks: {summary['total_tracks']}")
    print(f"  Avg tracklet length: {summary['avg_tracklet_length']:.1f}")

    return track_info


def save_trajectory_plots(output_dir: str, bounding_boxes: List[List[BoundingBox3D]],
                          all_track_ids: List[int]):
    """Save 2D and 3D trajectory plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for trajectory plots")
        return

    # Build trajectories
    trajectories = {}
    for frame_idx, frame_bboxes in enumerate(bounding_boxes):
        for bbox in frame_bboxes:
            if bbox.track_id is None or bbox.track_id < 0:
                continue
            tid = bbox.track_id
            if tid not in trajectories:
                trajectories[tid] = []
            trajectories[tid].append({
                'frame': frame_idx,
                'center': bbox.center,
            })

    if not trajectories:
        return

    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for tid, traj in trajectories.items():
        color = get_track_color(tid, all_track_ids)
        centers = np.array([t['center'] for t in traj])
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color=color, linewidth=2, label=f"T{tid}")
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                  color=color, s=50, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Track Trajectories')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_trajectories_3d.png'), dpi=150)
    plt.close()

    # 2D plot (top-down XZ)
    fig, ax = plt.subplots(figsize=(12, 8))
    for tid, traj in trajectories.items():
        color = get_track_color(tid, all_track_ids)
        centers = np.array([t['center'] for t in traj])
        ax.plot(centers[:, 0], centers[:, 2], color=color, linewidth=2, label=f"T{tid}")
        ax.scatter(centers[0, 0], centers[0, 2], color=color, s=50, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('2D Track Trajectories - Top Down')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_trajectories_2d.png'), dpi=150)
    plt.close()

    print(f"Saved trajectory plots to {output_dir}")


# =============================================================================
# Viser Visualization with Bounding Boxes
# =============================================================================

def viser_wrapper_with_tracking(
    pred_dict: dict,
    bounding_boxes: List[List[BoundingBox3D]],
    all_track_ids: List[int],
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
):
    """Visualize predicted 3D points, camera poses, and tracked bounding boxes with viser."""
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
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
    gui_show_all_bboxes = server.gui.add_checkbox("Show All Frame Bboxes", initial_value=False)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    # Use slider instead of dropdown for frame selection
    gui_frame_slider = server.gui.add_slider(
        "Frame", min=-1, max=S-1, step=1, initial_value=-1
    )
    server.gui.add_markdown("*Frame -1 = All frames*")

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

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    bbox_handles = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray):
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle):
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

    def visualize_bboxes():
        """Visualize bounding boxes for selected frame or all frames."""
        for bh in bbox_handles:
            try:
                bh.remove()
            except:
                pass
        bbox_handles.clear()

        if not gui_show_bboxes.value:
            return

        selected_frame = int(gui_frame_slider.value)
        show_all = gui_show_all_bboxes.value or selected_frame < 0

        for frame_idx, frame_bboxes in enumerate(bounding_boxes):
            # Only show bboxes for selected frame, unless "show all" is enabled
            if not show_all and frame_idx != selected_frame:
                continue

            for bbox in frame_bboxes:
                if bbox.track_id is None or bbox.track_id < 0:
                    continue

                color = get_track_color(bbox.track_id, all_track_ids)
                corners, edges = bbox.get_wireframe_edges()
                corners_centered = corners - scene_center

                # Add as line segments
                for edge_idx, (a, b) in enumerate(edges):
                    try:
                        line = server.scene.add_spline_catmull_rom(
                            f"bbox_{frame_idx}_{bbox.track_id}_{edge_idx}",
                            positions=np.array([corners_centered[a], corners_centered[b]]),
                            color=color,
                        )
                        bbox_handles.append(line)
                    except:
                        pass

    def update_point_cloud():
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        selected_frame = int(gui_frame_slider.value)
        if selected_frame < 0:
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            frame_mask = frame_indices == selected_frame

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_):
        update_point_cloud()

    @gui_frame_slider.on_update
    def _(_):
        update_point_cloud()
        visualize_bboxes()

    @gui_show_frames.on_update
    def _(_):
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @gui_show_bboxes.on_update
    def _(_):
        visualize_bboxes()

    @gui_show_all_bboxes.on_update
    def _(_):
        visualize_bboxes()

    visualize_frames(cam_to_world, images)
    visualize_bboxes()

    print("Viser server started")
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


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT demo with instance tracking and 3D bounding boxes")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Path to pre-computed Grounded SAM masks (JSON)")
    parser.add_argument("--dji_log", type=str, default=None, help="Path to DJI SRT file for gimbal/GPS data")
    parser.add_argument("--use_gps_refinement", action="store_true", help="Enable GPS-based pose refinement")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument("--num_images", type=int, default=20, help="Max number of images to process (VGGT limit ~20)")
    parser.add_argument("--skip", type=int, default=1, help="Skip factor for frame subsampling")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial confidence threshold (%)")
    parser.add_argument("--max_distance", type=float, default=8.0, help="Max 3D distance for tracking")
    parser.add_argument("--max_missing_frames", type=int, default=20, help="Frames before track goes dormant")
    parser.add_argument("--dormant_timeout", type=int, default=100, help="Frames before dormant track removed")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Select frames
    t0 = time.time()
    print(f"\n=== Selecting Frames ===")
    image_paths, frame_numbers = select_frames(args.image_folder, args.num_images, args.skip)
    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    # Load VGGT model
    print(f"\n=== Loading VGGT Model ===")
    t1 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"Model loading: {time.time() - t1:.2f}s")

    # Load and preprocess images
    t2 = time.time()
    print(f"\n=== Loading Images ===")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    print(f"Image loading: {time.time() - t2:.2f}s")

    # Run inference
    t3 = time.time()
    print(f"\n=== Running VGGT Inference ===")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    torch.cuda.synchronize()
    print(f"Inference: {time.time() - t3:.2f}s")

    # Convert pose encoding to extrinsic and intrinsic matrices
    t4 = time.time()
    print(f"\n=== Processing Outputs ===")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    print(f"Post-processing: {time.time() - t4:.2f}s")

    # Parse DJI logs if provided
    gimbal_data = None
    if args.dji_log:
        print(f"\n=== Parsing DJI Logs ===")
        if args.use_gps_refinement:
            gimbal_data = parse_dji_logs_with_gps(args.dji_log, frame_numbers)
        else:
            gimbal_data = parse_dji_logs(args.dji_log, frame_numbers)

    # Load original images for 2D projection
    original_images = [cv2.imread(p) for p in image_paths]

    # Initialize tracking and compute bounding boxes
    bounding_boxes = []
    all_track_ids = []

    if args.mask_dir:
        print(f"\n=== Loading Masks and Computing 3D BBoxes ===")
        masks_data = load_grounded_sam_masks(args.mask_dir, image_paths)

        # Get world points
        if args.use_point_map:
            world_points = predictions["world_points"]
        else:
            world_points = unproject_depth_map_to_point_map(
                predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
            )

        # Initialize tracker
        tracker = ImprovedTracker(
            max_distance=args.max_distance,
            mask_iou_threshold=0.15,
            max_missing_frames=args.max_missing_frames,
            dormant_timeout=args.dormant_timeout,
        )

        # Compute 3D bounding boxes with tracking
        model_size = (predictions["depth"].shape[1], predictions["depth"].shape[2])
        bounding_boxes = compute_instance_bboxes(
            world_points, masks_data, original_images, model_size, tracker,
            gimbal_data=gimbal_data
        )

        # Collect all track IDs
        for frame_bboxes in bounding_boxes:
            for bbox in frame_bboxes:
                if bbox.track_id is not None and bbox.track_id >= 0:
                    if bbox.track_id not in all_track_ids:
                        all_track_ids.append(bbox.track_id)
        all_track_ids = sorted(all_track_ids)

        print(f"\nTotal unique tracks: {len(all_track_ids)}")

    # Save outputs if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        if bounding_boxes and all_track_ids:
            # Save tracking summary
            save_tracking_summary(args.output_dir, bounding_boxes, all_track_ids)

            # Save trajectory plots
            save_trajectory_plots(args.output_dir, bounding_boxes, all_track_ids)

            # Project bboxes to 2D
            # Get model size from depth map shape (H, W)
            model_size = (predictions["depth"].shape[1], predictions["depth"].shape[2])
            project_bboxes_to_2d(
                bounding_boxes, original_images,
                predictions["extrinsic"], predictions["intrinsic"],
                args.output_dir, frame_names, all_track_ids,
                model_size=model_size
            )

    total_time = time.time() - t0

    # Save metadata (always save if output_dir specified)
    if args.output_dir:
        metadata = {
            "input_path": os.path.abspath(args.image_folder),
            "output_path": os.path.abspath(args.output_dir),
            "num_frames": len(image_paths),
            "image_files": [os.path.basename(f) for f in image_paths],
            "frame_numbers": frame_numbers,
            "max_images_setting": args.num_images,
            "skip": args.skip,
            "conf_threshold": args.conf_threshold,
            "use_point_map": args.use_point_map,
            "mask_sky": args.mask_sky,
            "mask_dir": args.mask_dir,
            "dji_log": args.dji_log,
            "image_shape": list(images.shape),
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": {
                "model_loading": round(t2 - t1, 2),
                "image_loading": round(t3 - t2, 2),
                "inference": round(t4 - t3, 2),
                "post_processing": round(time.time() - t4, 2),
                "total": round(total_time, 2)
            }
        }
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {os.path.join(args.output_dir, 'metadata.json')}")
    print(f"\n=== Total Processing Time: {total_time:.2f}s ===")

    # Start visualization
    print(f"\n=== Starting Viser Visualization ===")
    viser_wrapper_with_tracking(
        predictions,
        bounding_boxes,
        all_track_ids,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
    )


if __name__ == "__main__":
    main()
