#!/usr/bin/env python3
"""
VGGT 3D Bounding Box Annotator Tool

Interactive viser-based editor for refining 3D bounding boxes produced by
the VGGT batch inference pipeline (batch_inference_segment.py).

Features:
- Edit 3D bounding boxes (position, rotation, dimensions) via gizmo + sliders
- Frame-by-frame navigation with point cloud rendering
- Species-specific dimension snapping
- Ground plane detection (RANSAC, percentile, track-based) and bbox grounding
- Copy bbox from adjacent frames
- Keyframe interpolation (position + rotation + dimensions via SLERP/LERP)
- Propagate corrections to all frames
- Semantic face labeling (front, top, left) with interpolation
- SAM3 mask-based track point highlighting
- Session persistence (reload previous corrections)
- Auto-save with debounce
- Saves back to VGGT-compatible tracking_summary.json + KITTI labels

Usage:
    # Auto-discovers sam3_masks as sibling directory
    python annotator_tool.py --result_dir /path/to/seg1/vggt_results/

    # Explicit paths
    python annotator_tool.py \\
        --result_dir /path/to/seg1/vggt_results/ \\
        --sam3_masks /path/to/seg1/sam3_masks/ \\
        --output_dir /path/to/seg1/vggt_results/annotations/ \\
        --port 8080
"""

import argparse
import copy
import json
import math
import threading
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import viser

# Optional imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# BBox3D Class
# ============================================================================

class BBox3D:
    """3D oriented bounding box."""

    def __init__(self, center, dimensions, rotation_matrix, class_name, track_id, frame_idx,
                 confidence=None):
        self.center = np.array(center, dtype=np.float64)
        self.dimensions = np.array(dimensions, dtype=np.float64)
        self.rotation_matrix = np.array(rotation_matrix, dtype=np.float64)
        self.class_name = class_name
        self.track_id = track_id
        self.frame_idx = frame_idx
        self.confidence = confidence if confidence is not None else 1.0

    def get_corners(self):
        """Get 8 corners of the bbox in world coordinates."""
        l, w, h = self.dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        return (self.rotation_matrix @ corners_local.T).T + self.center

    def get_edges(self):
        """Get edge index pairs for wireframe rendering."""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]

    def get_faces(self):
        """Get 6 faces with corner indices, centers, and normals."""
        faces = {
            0: [0, 1, 5, 4],  # Front (+X)
            1: [2, 3, 7, 6],  # Back (-X)
            2: [0, 3, 7, 4],  # Left (-Y)
            3: [1, 2, 6, 5],  # Right (+Y)
            4: [4, 5, 6, 7],  # Top (+Z)
            5: [0, 1, 2, 3],  # Bottom (-Z)
        }
        corners = self.get_corners()
        face_data = {}
        for face_id, corner_ids in faces.items():
            face_corners = corners[corner_ids]
            face_center = face_corners.mean(axis=0)
            v1 = face_corners[1] - face_corners[0]
            v2 = face_corners[2] - face_corners[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            face_data[face_id] = {
                'corners': face_corners,
                'center': face_center,
                'normal': normal,
            }
        return face_data


# ============================================================================
# VGGTBBoxEditor
# ============================================================================

class VGGTBBoxEditor:
    """Interactive 3D bounding box editor for VGGT pipeline outputs."""

    SPECIES_PROPORTIONS = {
        'elephant': {'length': 1.72, 'width': 0.78},
        'rhino': {'length': 1.80, 'width': 0.85},
        'zebra': {'length': 1.65, 'width': 0.55},
        'giraffe': {'length': 1.10, 'width': 0.50},
        'default': {'length': 1.72, 'width': 0.78},
    }

    GROUND_CONFIG = {
        'search_radius': 3.0,
        'ransac_threshold': 0.05,
        'ransac_iterations': 500,
        'min_inliers': 50,
        'ground_normal_tolerance': 0.4,
        'percentile_fallback': 10,
    }

    def __init__(self, result_dir, output_dir=None, sam3_masks_dir=None,
                 reload_annotations=True, port=8080):
        self.result_dir = Path(result_dir)
        self.output_dir = Path(output_dir) if output_dir else self.result_dir / "annotations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()
        self.frame_numbers = self.metadata.get('frame_numbers', [])

        # Auto-discover SAM3 masks
        if sam3_masks_dir:
            self.sam3_masks_dir = Path(sam3_masks_dir)
        else:
            self.sam3_masks_dir = self._auto_find_sam3_masks()

        # Load SAM3 metadata for class name
        self.sam3_metadata = self._load_sam3_metadata()
        self.detected_class_name = None
        if self.sam3_metadata:
            self.detected_class_name = self.sam3_metadata.get('text_prompt')
            print(f"SAM3 class: {self.detected_class_name}")

        # Load camera parameters
        self.cam_params = self._load_camera_params()

        # Load tracking data
        self.tracking_summary = self._load_tracking_summary()
        self.auto_bboxes = self._load_bboxes()
        self.frame_indices = sorted(self.auto_bboxes.keys())

        # Corrections and semantic labels
        self.corrections = {}  # track_id -> {frame_idx -> BBox3D}
        self.semantic_faces = {}  # track_id -> {frame_idx -> {'front': id, ...}}

        # Load previous session if requested
        if reload_annotations:
            self._load_previous_session()

        # Point cloud cache
        self.point_clouds = {}
        self.depth_maps = None
        self.depth_conf = None
        self._load_depth_data()

        # Scene PLY fallback
        self.scene_pcd_points = None
        self.scene_pcd_colors = None

        # Viser server
        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction("-y")

        # State
        self.current_frame_idx = 0
        self.current_frame = self.frame_indices[0] if self.frame_indices else 0
        self.selected_track = None

        # Scene handles
        self.pc_handle = None
        self.track_pc_handle = None
        self.bbox_handles = {}
        self.original_bbox_handles = []
        self.gizmo_handle = None
        self.face_handles = []

        # Flags
        self.updating_sliders = False
        self.updating_dropdowns = False

        # Keyframe state
        self.keyframe_1 = None
        self.keyframe_2 = None
        self.pending_interpolation = None

        # Auto-save state
        self.auto_save_timer = None
        self.last_save_time = None

        # Setup UI and render
        self._setup_ui()
        self._render_frame()

        print(f"\n{'='*60}")
        print(f"VGGT BBOX EDITOR READY")
        print(f"{'='*60}")
        print(f"Open: http://localhost:{port}")
        print(f"Frames: {len(self.frame_indices)}")
        print(f"Tracks: {len(set(t for bboxes in self.auto_bboxes.values() for t in [b.track_id for b in bboxes]))}")
        print(f"Result dir: {self.result_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")

    # ======================================================================
    # Data Loading
    # ======================================================================

    def _load_metadata(self):
        """Load vggt_metadata.json."""
        # Try both naming conventions
        for name in ["vggt_metadata.json", "metadata.json"]:
            path = self.result_dir / name
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                print(f"Loaded metadata: {path}")
                return data
        print("No metadata.json found")
        return {}

    def _auto_find_sam3_masks(self):
        """Auto-discover SAM3 masks as sibling directory."""
        sam3_dir = self.result_dir.parent / "sam3_masks"
        if sam3_dir.exists():
            print(f"Auto-detected SAM3 masks: {sam3_dir}")
            return sam3_dir
        print("SAM3 masks not found (use --sam3_masks to specify)")
        return None

    def _load_sam3_metadata(self):
        """Load SAM3 metadata.json for class name."""
        if self.sam3_masks_dir is None:
            return None
        meta_path = self.sam3_masks_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def _load_tracking_summary(self):
        """Load tracking_summary.json."""
        path = self.result_dir / "tracking_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"tracking_summary.json not found in {self.result_dir}")
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded tracking: {data.get('total_tracks', 0)} tracks, {data.get('total_frames', 0)} frames")
        return data

    def _load_bboxes(self) -> Dict[int, List[BBox3D]]:
        """Parse tracking_summary.json into per-frame BBox3D dicts."""
        bboxes = {}
        for track_id_str, track_data in self.tracking_summary.get('tracks', {}).items():
            track_id = int(track_id_str)
            frames = track_data['frames']
            centers = track_data['centers']
            dimensions = track_data['dimensions']
            rotation_matrices = track_data['rotation_matrices']
            confidences = track_data.get('confidences', [1.0] * len(frames))

            # Use SAM3 class name if available, otherwise use tracking class
            class_name = self.detected_class_name or track_data.get('class_name', 'object')

            for i, frame_idx in enumerate(frames):
                bbox = BBox3D(
                    center=centers[i],
                    dimensions=dimensions[i],
                    rotation_matrix=rotation_matrices[i],
                    class_name=class_name,
                    track_id=track_id,
                    frame_idx=frame_idx,
                    confidence=confidences[i] if i < len(confidences) else 1.0,
                )
                bboxes.setdefault(frame_idx, []).append(bbox)
        return bboxes

    def _load_camera_params(self):
        """Load cameras.json."""
        path = self.result_dir / "cameras.json"
        if not path.exists():
            print("cameras.json not found")
            return None

        with open(path) as f:
            data = json.load(f)

        cameras = data.get('cameras', [])
        if not cameras:
            return None

        extrinsics = []
        intrinsics = []
        image_names = []
        image_h = cameras[0].get('image_height', 294)
        image_w = cameras[0].get('image_width', 518)

        for cam in cameras:
            extrinsics.append(np.array(cam['extrinsic']))
            intrinsics.append(np.array(cam['intrinsic']))
            image_names.append(cam.get('image_name', ''))

        print(f"Loaded {len(cameras)} cameras ({image_w}x{image_h})")
        return {
            'extrinsics': np.array(extrinsics),  # (S, 3, 4) w2c
            'intrinsics': np.array(intrinsics),   # (S, 3, 3)
            'image_names': image_names,
            'image_height': image_h,
            'image_width': image_w,
        }

    def _load_depth_data(self):
        """Load depth_maps.npz for per-frame point cloud generation."""
        path = self.result_dir / "depth_maps.npz"
        if path.exists():
            data = np.load(path)
            self.depth_maps = data.get('depth', None)
            self.depth_conf = data.get('depth_conf', None)
            if self.depth_maps is not None:
                print(f"Loaded depth maps: {self.depth_maps.shape}")
        else:
            print("depth_maps.npz not found — will use scene PLY")

    def _load_point_cloud(self, frame_idx):
        """Load/generate point cloud for a frame."""
        if frame_idx in self.point_clouds:
            return self.point_clouds[frame_idx]

        # Strategy 1: Per-frame from depth maps
        if self.depth_maps is not None and self.cam_params is not None:
            points, colors = self._generate_from_depth(frame_idx)
            if points is not None and len(points) > 0:
                self.point_clouds[frame_idx] = (points, colors)
                return points, colors

        # Strategy 2: Scene PLY (loaded once, subsampled)
        if self.scene_pcd_points is None:
            self._load_scene_ply()

        if self.scene_pcd_points is not None:
            self.point_clouds[frame_idx] = (self.scene_pcd_points, self.scene_pcd_colors)
            return self.scene_pcd_points, self.scene_pcd_colors

        return None, None

    def _generate_from_depth(self, frame_idx):
        """Generate per-frame point cloud from depth map + camera params."""
        if frame_idx >= len(self.depth_maps) or frame_idx >= len(self.cam_params['extrinsics']):
            return None, None

        try:
            from vggt.utils.geometry import depth_to_world_coords_points
        except ImportError:
            return None, None

        depth = self.depth_maps[frame_idx]
        if depth.ndim == 3:
            depth = depth.squeeze(-1)

        ext = self.cam_params['extrinsics'][frame_idx]  # (3, 4) w2c
        intr = self.cam_params['intrinsics'][frame_idx]  # (3, 3)

        world_points, point_mask, valid_mask = depth_to_world_coords_points(depth, ext, intr)

        if world_points is None:
            return None, None

        # Apply confidence filtering
        if self.depth_conf is not None and frame_idx < len(self.depth_conf):
            conf = self.depth_conf[frame_idx]
            if conf.ndim == 3:
                conf = conf.squeeze(-1)
            conf_threshold = np.percentile(conf[conf > 0], 50) if np.any(conf > 0) else 0
            combined_mask = valid_mask & (conf >= conf_threshold)
        else:
            combined_mask = valid_mask

        points = world_points[combined_mask].reshape(-1, 3)
        # Use gray colors (no image data loaded)
        colors = np.ones((len(points), 3)) * 0.6

        return points, colors

    def _load_scene_ply(self):
        """Load and subsample scene point cloud."""
        ply_path = self.result_dir / "point_cloud.ply"
        if not ply_path.exists():
            return

        if HAS_OPEN3D:
            print(f"Loading scene PLY: {ply_path}")
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.5

            # Subsample for performance
            if len(points) > 500000:
                indices = np.random.choice(len(points), 500000, replace=False)
                points = points[indices]
                colors = colors[indices]

            self.scene_pcd_points = points
            self.scene_pcd_colors = colors
            print(f"Scene PLY: {len(points)} points (subsampled)")
        else:
            print("open3d not available — cannot load scene PLY")

    # ======================================================================
    # SAM3 Mask Loading
    # ======================================================================

    def _load_sam3_mask(self, frame_idx, track_id):
        """Load SAM3 binary mask PNG for a frame and track.

        SAM3 masks are at: sam3_masks/masks/obj_{track_id}/frame_{orig_num:06d}.png
        """
        if self.sam3_masks_dir is None:
            return None

        # Map sequential frame index to original frame number
        if frame_idx < len(self.frame_numbers):
            orig_frame = self.frame_numbers[frame_idx]
        else:
            return None

        mask_path = self.sam3_masks_dir / "masks" / f"obj_{track_id}" / f"frame_{orig_frame:06d}.png"
        if not mask_path.exists():
            return None

        try:
            if HAS_CV2:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return (mask > 127).astype(np.uint8)
            else:
                from PIL import Image
                img = Image.open(mask_path).convert('L')
                mask = np.array(img)
                return (mask > 127).astype(np.uint8)
        except Exception as e:
            print(f"Failed to load mask {mask_path}: {e}")
        return None

    def _backproject_mask_to_points(self, mask, points, frame_idx):
        """Backproject 2D mask to identify which 3D points are inside."""
        if self.cam_params is None or frame_idx >= len(self.cam_params['extrinsics']):
            return None

        ext = self.cam_params['extrinsics'][frame_idx]  # (3, 4) w2c
        intr = self.cam_params['intrinsics'][frame_idx]  # (3, 3)
        mask_h, mask_w = mask.shape

        # Scale intrinsics to mask resolution
        model_h = self.cam_params['image_height']
        model_w = self.cam_params['image_width']
        scale_x = mask_w / model_w
        scale_y = mask_h / model_h

        K = intr.copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y

        # Transform points to camera space: p_cam = ext @ [X, Y, Z, 1]^T
        points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        points_cam = (ext @ points_h.T).T  # (N, 3)

        # Filter behind camera
        valid_depth = points_cam[:, 2] > 0.01

        # Project to 2D
        points_2d = np.zeros((len(points), 2))
        if np.any(valid_depth):
            proj = (K @ points_cam[valid_depth].T).T
            points_2d[valid_depth] = proj[:, :2] / proj[:, 2:3]

        u = points_2d[:, 0].astype(int)
        v = points_2d[:, 1].astype(int)

        in_bounds = (u >= 0) & (u < mask_w) & (v >= 0) & (v < mask_h) & valid_depth
        inside_mask = np.zeros(len(points), dtype=bool)
        inside_mask[in_bounds] = mask[v[in_bounds], u[in_bounds]] > 0

        return inside_mask

    # ======================================================================
    # Session Persistence
    # ======================================================================

    def _load_previous_session(self):
        """Restore corrections and semantic faces from previous session."""
        loaded = False

        # Load corrected tracking summary
        corrected_summary = self.output_dir / "tracking_summary.json"
        if corrected_summary.exists():
            print(f"Found previous session: {self.output_dir}")
            with open(corrected_summary) as f:
                prev_summary = json.load(f)
            # Rebuild bboxes from corrected summary
            self.tracking_summary = prev_summary
            self.auto_bboxes = self._load_bboxes()
            self.frame_indices = sorted(self.auto_bboxes.keys())
            print(f"  Restored corrected tracking summary")
            loaded = True

        # Load corrections dict
        corrections_path = self.output_dir / "corrections.json"
        if corrections_path.exists():
            try:
                with open(corrections_path) as f:
                    data = json.load(f)
                for track_id_str, frames in data.items():
                    track_id = int(track_id_str)
                    self.corrections[track_id] = {}
                    for frame_idx_str, bbox_dict in frames.items():
                        frame_idx = int(frame_idx_str)
                        self.corrections[track_id][frame_idx] = BBox3D(
                            center=bbox_dict['center'],
                            dimensions=bbox_dict['dimensions'],
                            rotation_matrix=bbox_dict['rotation_matrix'],
                            class_name=bbox_dict.get('class_name', 'object'),
                            track_id=track_id,
                            frame_idx=frame_idx,
                        )
                total = sum(len(f) for f in self.corrections.values())
                print(f"  Restored {total} corrections for {len(self.corrections)} tracks")
                loaded = True
            except Exception as e:
                print(f"  Failed to load corrections.json: {e}")

        # Load semantic faces
        semantic_path = self.output_dir / "semantic_faces" / "manual_labels.json"
        if semantic_path.exists():
            try:
                with open(semantic_path) as f:
                    data = json.load(f)
                for track_id_str, frames in data.items():
                    track_id = int(track_id_str)
                    self.semantic_faces[track_id] = {}
                    for frame_idx_str, labels in frames.items():
                        self.semantic_faces[track_id][int(frame_idx_str)] = labels
                total = sum(len(f) for f in self.semantic_faces.values())
                print(f"  Restored {total} semantic face labels")
                loaded = True
            except Exception as e:
                print(f"  Failed to load semantic faces: {e}")

        if loaded:
            print("  Session restored successfully!")
        else:
            print("No previous session found — starting fresh")

    # ======================================================================
    # UI Setup
    # ======================================================================

    def _setup_ui(self):
        """Setup all GUI controls."""
        gui = self.server.gui

        # --- Navigation ---
        with gui.add_folder("Navigation"):
            prev_btn = gui.add_button("Prev Frame")
            next_btn = gui.add_button("Next Frame")

            self.frame_slider = gui.add_slider(
                "Frame", min=0, max=max(0, len(self.frame_indices) - 1),
                step=1, initial_value=0
            )

            self.info_text = gui.add_text("Info", initial_value="Select a track to edit", disabled=True)

        @prev_btn.on_click
        def _(_):
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        @next_btn.on_click
        def _(_):
            if self.current_frame_idx < len(self.frame_indices) - 1:
                self.current_frame_idx += 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        @self.frame_slider.on_update
        def _(_):
            if not self.updating_sliders:
                self.current_frame_idx = int(self.frame_slider.value)
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self._render_frame()

        # --- Track Selection ---
        with gui.add_folder("Track Selection"):
            all_tracks = set()
            for bboxes in self.auto_bboxes.values():
                for bbox in bboxes:
                    all_tracks.add(bbox.track_id)

            track_options = ["(None)"] + [f"Track {tid}" for tid in sorted(all_tracks)]
            self.track_dropdown = gui.add_dropdown("Select Track", options=track_options)

        @self.track_dropdown.on_update
        def _(_):
            if self.track_dropdown.value == "(None)":
                self.selected_track = None
            else:
                self.selected_track = int(self.track_dropdown.value.split()[-1])
            self._update_selection()
            self._render_frame()

        # --- Point Cloud ---
        with gui.add_folder("Point Cloud"):
            self.point_size_slider = gui.add_slider(
                "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005
            )
            self.highlight_track_checkbox = gui.add_checkbox(
                "Highlight Track Points", initial_value=True
            )

        @self.point_size_slider.on_update
        def _(_):
            if self.pc_handle is not None:
                self.pc_handle.point_size = self.point_size_slider.value

        @self.highlight_track_checkbox.on_update
        def _(_):
            self._render_frame()

        # --- Bbox Editing ---
        with gui.add_folder("Bbox Editing"):
            self.dim_0_slider = gui.add_slider("dim[0]", min=0.001, max=2.0, step=0.001, initial_value=0.1)
            self.dim_1_slider = gui.add_slider("dim[1]", min=0.001, max=2.0, step=0.001, initial_value=0.1)
            self.dim_2_slider = gui.add_slider("dim[2]", min=0.001, max=2.0, step=0.001, initial_value=0.1)

            self.height_is = gui.add_dropdown(
                "Height is", options=["dim[0]", "dim[1]", "dim[2]"], initial_value="dim[2]"
            )
            self.snap_btn = gui.add_button("Snap to Proportions")

        @self.dim_0_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_1_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_2_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.snap_btn.on_click
        def _(_):
            self._snap_to_proportions()

        # --- Ground Snapping ---
        with gui.add_folder("Ground Snapping"):
            self.ground_method_dropdown = gui.add_dropdown(
                "Ground Method",
                options=["RANSAC (robust)", "Lowest Points", "Track Points"],
                initial_value="RANSAC (robust)"
            )
            snap_ground_btn = gui.add_button("Snap to Ground")
            snap_ground_all_btn = gui.add_button("Snap All Frames to Ground")

        @snap_ground_btn.on_click
        def _(_):
            self._snap_to_ground()

        @snap_ground_all_btn.on_click
        def _(_):
            self._snap_to_ground_all_frames()

        # --- Copy / Propagate ---
        with gui.add_folder("Copy / Propagate"):
            copy_prev_btn = gui.add_button("Copy from Previous Frame")
            copy_next_btn = gui.add_button("Copy from Next Frame")
            propagate_btn = gui.add_button("Propagate to All Frames")

        @copy_prev_btn.on_click
        def _(_):
            self._copy_from_adjacent_frame(-1)

        @copy_next_btn.on_click
        def _(_):
            self._copy_from_adjacent_frame(+1)

        @propagate_btn.on_click
        def _(_):
            self._propagate_to_all_frames()

        # --- Keyframe Interpolation ---
        with gui.add_folder("Keyframe Interpolation"):
            kf1_btn = gui.add_button("Mark as Keyframe 1")
            kf2_btn = gui.add_button("Mark as Keyframe 2")
            interp_btn = gui.add_button("Interpolate Between Keyframes")
            interp_semantic_btn = gui.add_button("Interpolate Semantic Faces Only")
            self.confirm_interpolate_btn = gui.add_button("Confirm Overwrite & Interpolate", visible=False)
            self.keyframe_1_text = gui.add_text("Keyframe 1", initial_value="Not set")
            self.keyframe_2_text = gui.add_text("Keyframe 2", initial_value="Not set")

        @kf1_btn.on_click
        def _(_):
            self._mark_keyframe(1)

        @kf2_btn.on_click
        def _(_):
            self._mark_keyframe(2)

        @interp_btn.on_click
        def _(_):
            self._interpolate_between_keyframes()

        @interp_semantic_btn.on_click
        def _(_):
            self._interpolate_semantic_faces_only()

        @self.confirm_interpolate_btn.on_click
        def _(_):
            if self.pending_interpolation:
                self._perform_interpolation(*self.pending_interpolation)
                self.pending_interpolation = None
                self.confirm_interpolate_btn.visible = False

        # --- Semantic Faces ---
        with gui.add_folder("Semantic Face Labels"):
            face_options = ["(None)"] + [f"Face {i}" for i in range(6)]
            self.front_face_dropdown = gui.add_dropdown("Front Face", options=face_options, initial_value="(None)")
            self.top_face_dropdown = gui.add_dropdown("Top Face", options=face_options, initial_value="(None)")
            self.left_face_dropdown = gui.add_dropdown("Left Face", options=face_options, initial_value="(None)")

        @self.front_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_semantic_labels()

        @self.top_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_semantic_labels()

        @self.left_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_semantic_labels()

        # --- Save / Compare ---
        with gui.add_folder("Save / Compare"):
            save_btn = gui.add_button("Save Now")
            next_unann_btn = gui.add_button("Next Unannotated Frame")
            self.show_original_checkbox = gui.add_checkbox("Show Original Bboxes", initial_value=False)

        @save_btn.on_click
        def _(_):
            self._manual_save()

        @next_unann_btn.on_click
        def _(_):
            self._next_unannotated_frame()

        @self.show_original_checkbox.on_update
        def _(_):
            self._render_frame()

    # ======================================================================
    # Rendering
    # ======================================================================

    def _clear_scene(self):
        """Remove all scene handles."""
        if self.pc_handle is not None:
            try: self.pc_handle.remove()
            except: pass
            self.pc_handle = None

        if self.track_pc_handle is not None:
            try: self.track_pc_handle.remove()
            except: pass
            self.track_pc_handle = None

        for handles in list(self.bbox_handles.values()):
            for h in handles:
                try: h.remove()
                except: pass
        self.bbox_handles.clear()

        for h in list(self.original_bbox_handles):
            try: h.remove()
            except: pass
        self.original_bbox_handles.clear()

        for h in list(self.face_handles):
            try: h.remove()
            except: pass
        self.face_handles.clear()

        if self.gizmo_handle is not None:
            try: self.gizmo_handle.remove()
            except: pass
            self.gizmo_handle = None

    def _render_frame(self):
        """Full re-render of current frame."""
        self._clear_scene()

        # Point cloud
        points, colors = self._load_point_cloud(self.current_frame)
        if points is not None:
            point_size = self.point_size_slider.value

            if (self.selected_track is not None and
                self.highlight_track_checkbox.value):
                dimmed = colors * 0.3
                self.pc_handle = self.server.scene.add_point_cloud(
                    name="/pc", points=points, colors=dimmed,
                    point_size=point_size * 0.7
                )
                self._render_track_highlighted_points(points, colors, point_size)
            else:
                self.pc_handle = self.server.scene.add_point_cloud(
                    name="/pc", points=points, colors=colors,
                    point_size=point_size
                )

        # Bboxes
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        if frame_bboxes:
            if self.selected_track is not None:
                for bbox in frame_bboxes:
                    if bbox.track_id == self.selected_track:
                        self._render_bbox(bbox)
                        break
            else:
                for bbox in frame_bboxes:
                    self._render_bbox(bbox)

        # Show original for comparison
        if (self.show_original_checkbox.value and
            self.selected_track is not None):
            for bbox in frame_bboxes:
                if bbox.track_id == self.selected_track:
                    if self._has_correction(bbox.track_id, self.current_frame):
                        self._render_bbox_original(bbox)
                    break

        self._update_selection()
        self._update_info_text()

    def _render_bbox(self, bbox):
        """Render a single bbox wireframe."""
        # Use corrected version if exists
        if self._has_correction(bbox.track_id, self.current_frame):
            bbox = self.corrections[bbox.track_id][self.current_frame]
            color = (0.0, 1.0, 0.0)  # Green = corrected
        else:
            color = (1.0, 0.0, 0.0)  # Red = auto

        if self.selected_track is not None and bbox.track_id == self.selected_track:
            color = (0.0, 1.0, 1.0)  # Cyan = selected
            line_width = 5.0
        else:
            line_width = 2.0

        bbox_id = f"bbox_{bbox.track_id}"
        if bbox_id not in self.bbox_handles:
            self.bbox_handles[bbox_id] = []

        corners = bbox.get_corners()
        for i, (si, ei) in enumerate(bbox.get_edges()):
            h = self.server.scene.add_spline_catmull_rom(
                name=f"/{bbox_id}_edge_{i}",
                positions=np.array([corners[si], corners[ei]]),
                color=color, line_width=line_width, segments=2
            )
            if bbox_id in self.bbox_handles:
                self.bbox_handles[bbox_id].append(h)

        # Center point
        h = self.server.scene.add_point_cloud(
            name=f"/{bbox_id}_center",
            points=bbox.center.reshape(1, 3),
            colors=np.array(color).reshape(1, 3),
            point_size=0.02
        )
        if bbox_id in self.bbox_handles:
            self.bbox_handles[bbox_id].append(h)

        # Semantic face labels
        if self.selected_track is not None and bbox.track_id == self.selected_track:
            self._render_face_labels(bbox)

    def _render_face_labels(self, bbox):
        """Render color-coded edges for semantic face labels."""
        selected_faces = {}
        if self.front_face_dropdown.value != "(None)":
            selected_faces['front'] = int(self.front_face_dropdown.value.split()[-1])
        if self.top_face_dropdown.value != "(None)":
            selected_faces['top'] = int(self.top_face_dropdown.value.split()[-1])
        if self.left_face_dropdown.value != "(None)":
            selected_faces['left'] = int(self.left_face_dropdown.value.split()[-1])

        semantic_colors = {
            'front': (1.0, 0.0, 0.0),
            'top': (0.0, 1.0, 0.0),
            'left': (0.0, 0.0, 1.0),
        }
        face_corner_indices = {
            0: [0, 1, 5, 4], 1: [2, 3, 7, 6], 2: [0, 3, 7, 4],
            3: [1, 2, 6, 5], 4: [4, 5, 6, 7], 5: [0, 1, 2, 3],
        }

        corners = bbox.get_corners()
        for semantic_name, face_id in selected_faces.items():
            if face_id not in face_corner_indices:
                continue
            fc = semantic_colors[semantic_name]
            cidxs = face_corner_indices[face_id]
            for i in range(4):
                si, ei = cidxs[i], cidxs[(i + 1) % 4]
                h = self.server.scene.add_spline_catmull_rom(
                    name=f"/face_{semantic_name}_edge_{i}",
                    positions=np.array([corners[si], corners[ei]]),
                    color=fc, line_width=2.0, segments=2
                )
                self.face_handles.append(h)

    def _render_bbox_original(self, bbox):
        """Render original bbox in gray for comparison."""
        color = (0.8, 0.8, 0.8)
        bbox_id = f"original_{bbox.track_id}"
        corners = bbox.get_corners()
        for i, (si, ei) in enumerate(bbox.get_edges()):
            h = self.server.scene.add_spline_catmull_rom(
                name=f"/{bbox_id}_edge_{i}",
                positions=np.array([corners[si], corners[ei]]),
                color=color, line_width=0.5, segments=2
            )
            self.original_bbox_handles.append(h)

    def _render_track_highlighted_points(self, all_points, all_colors, point_size):
        """Highlight points belonging to selected track."""
        if self.selected_track is None:
            return

        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            return

        # Try SAM3 mask first
        mask = self._load_sam3_mask(self.current_frame, self.selected_track)
        if mask is not None:
            track_mask = self._backproject_mask_to_points(mask, all_points, self.current_frame)
            if track_mask is not None and np.any(track_mask):
                track_points = all_points[track_mask]
                bright_colors = np.clip(all_colors[track_mask] * 1.5 + 0.2, 0, 1)
                if len(track_points) > 0:
                    self.track_pc_handle = self.server.scene.add_point_cloud(
                        name="/track_pc", points=track_points,
                        colors=bright_colors, point_size=point_size * 1.5
                    )
                return

        # Fallback: bbox interior
        self._render_bbox_interior_points(all_points, all_colors, current_bbox, point_size)

    def _render_bbox_interior_points(self, all_points, all_colors, bbox, point_size):
        """Highlight points inside the bbox volume."""
        points_local = (np.linalg.inv(bbox.rotation_matrix) @ (all_points - bbox.center).T).T
        half_dims = bbox.dimensions / 2
        inside = (
            (np.abs(points_local[:, 0]) <= half_dims[0]) &
            (np.abs(points_local[:, 1]) <= half_dims[1]) &
            (np.abs(points_local[:, 2]) <= half_dims[2])
        )
        if not np.any(inside):
            return

        track_points = all_points[inside]
        bright_colors = np.clip(all_colors[inside] * 1.5 + 0.2, 0, 1)
        self.track_pc_handle = self.server.scene.add_point_cloud(
            name="/track_pc", points=track_points,
            colors=bright_colors, point_size=point_size * 1.5
        )

    def _rerender_bboxes(self):
        """Re-render only bboxes (faster than full re-render)."""
        for handles in list(self.bbox_handles.values()):
            for h in handles:
                try: h.remove()
                except: pass
        self.bbox_handles.clear()

        for h in list(self.face_handles):
            try: h.remove()
            except: pass
        self.face_handles.clear()

        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        if self.selected_track is not None:
            for bbox in frame_bboxes:
                if bbox.track_id == self.selected_track:
                    self._render_bbox(bbox)
                    break
        else:
            for bbox in frame_bboxes:
                self._render_bbox(bbox)

    # ======================================================================
    # Selection & Editing
    # ======================================================================

    def _update_selection(self):
        """Update gizmo and sliders for selected bbox."""
        if self.gizmo_handle is not None:
            try: self.gizmo_handle.remove()
            except: pass
            self.gizmo_handle = None

        if self.selected_track is None:
            return

        selected_bbox = self._get_current_bbox()
        if selected_bbox is None:
            self._update_info_text()
            return

        # Update sliders
        self.updating_sliders = True
        self.dim_0_slider.value = float(selected_bbox.dimensions[0])
        self.dim_1_slider.value = float(selected_bbox.dimensions[1])
        self.dim_2_slider.value = float(selected_bbox.dimensions[2])
        self.updating_sliders = False

        # Transform gizmo
        quat_xyzw = R.from_matrix(selected_bbox.rotation_matrix).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        self.gizmo_handle = self.server.scene.add_transform_controls(
            name="/gizmo",
            position=tuple(selected_bbox.center),
            wxyz=tuple(quat_wxyz),
        )

        @self.gizmo_handle.on_update
        def _(transform):
            self._update_bbox_from_gizmo(transform)

        # Update semantic dropdowns
        self._update_semantic_dropdowns()

        # Update snap button
        if selected_bbox.class_name and selected_bbox.class_name.lower() in self.SPECIES_PROPORTIONS:
            self.snap_btn.label = f"Snap to {selected_bbox.class_name.capitalize()} Proportions"
        else:
            self.snap_btn.label = "Snap to Default Proportions"

        self._update_info_text()

    def _get_current_bbox(self):
        """Get bbox for current track/frame (corrected if exists)."""
        if self.selected_track is None:
            return None
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        bbox = next((b for b in frame_bboxes if b.track_id == self.selected_track), None)
        if bbox is None:
            return None
        if self._has_correction(self.selected_track, self.current_frame):
            return self.corrections[self.selected_track][self.current_frame]
        return bbox

    def _has_correction(self, track_id, frame_idx):
        return track_id in self.corrections and frame_idx in self.corrections[track_id]

    def _get_or_create_correction(self, track_id, frame_idx):
        """Get existing correction or create one from original bbox."""
        if not self._has_correction(track_id, frame_idx):
            frame_bboxes = self.auto_bboxes.get(frame_idx, [])
            original = next((b for b in frame_bboxes if b.track_id == track_id), None)
            if original is None:
                return None
            if track_id not in self.corrections:
                self.corrections[track_id] = {}
            self.corrections[track_id][frame_idx] = BBox3D(
                center=original.center.copy(),
                dimensions=original.dimensions.copy(),
                rotation_matrix=original.rotation_matrix.copy(),
                class_name=original.class_name,
                track_id=original.track_id,
                frame_idx=original.frame_idx,
                confidence=original.confidence,
            )
        return self.corrections[track_id][frame_idx]

    def _update_bbox_from_gizmo(self, transform):
        """Apply gizmo position/rotation to bbox."""
        if self.selected_track is None:
            return
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return
        bbox.center = np.array(transform.position)
        quat_wxyz = np.array(transform.wxyz)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        bbox.rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
        self._rerender_bboxes()
        self._schedule_auto_save()

    def _update_bbox_dimensions(self):
        """Apply slider values to bbox dimensions."""
        if self.selected_track is None:
            return
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return
        bbox.dimensions[0] = self.dim_0_slider.value
        bbox.dimensions[1] = self.dim_1_slider.value
        bbox.dimensions[2] = self.dim_2_slider.value
        self._rerender_bboxes()
        self._schedule_auto_save()

    def _snap_to_proportions(self):
        """Snap bbox dimensions to species-specific proportions."""
        if self.selected_track is None:
            return
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        height_idx = int(self.height_is.value.split('[')[1].split(']')[0])
        other_indices = [i for i in range(3) if i != height_idx]
        height = bbox.dimensions[height_idx]

        species = bbox.class_name.lower() if bbox.class_name else 'default'
        proportions = self.SPECIES_PROPORTIONS.get(species, self.SPECIES_PROPORTIONS['default'])
        target_length = height * proportions['length']
        target_width = height * proportions['width']

        idx1, idx2 = other_indices
        if bbox.dimensions[idx1] > bbox.dimensions[idx2]:
            bbox.dimensions[idx1] = target_length
            bbox.dimensions[idx2] = target_width
        else:
            bbox.dimensions[idx2] = target_length
            bbox.dimensions[idx1] = target_width

        self.updating_sliders = True
        self.dim_0_slider.value = float(bbox.dimensions[0])
        self.dim_1_slider.value = float(bbox.dimensions[1])
        self.dim_2_slider.value = float(bbox.dimensions[2])
        self.updating_sliders = False

        self._rerender_bboxes()
        print(f"Snapped to {species} proportions: {bbox.dimensions}")
        self._schedule_auto_save()

    # ======================================================================
    # Ground Snapping
    # ======================================================================

    def _detect_vertical_axis(self, points):
        """Detect vertical axis. With set_up_direction('-y'), Y is vertical, negative is up."""
        return 1, -1

    def _get_points_near_bbox(self, points, colors, bbox, radius=None):
        if radius is None:
            radius = self.GROUND_CONFIG['search_radius']
        vert_axis = 1
        horiz_axes = [0, 2]
        center_horiz = bbox.center[horiz_axes]
        points_horiz = points[:, horiz_axes]
        distances = np.linalg.norm(points_horiz - center_horiz, axis=1)
        mask = distances < radius
        return points[mask], colors[mask] if colors is not None else None, np.where(mask)[0]

    def _fit_ground_plane_ransac(self, points):
        config = self.GROUND_CONFIG
        n_points = len(points)
        if n_points < 3:
            return None, None

        best_inliers = None
        best_n_inliers = 0
        best_plane = None

        for _ in range(config['ransac_iterations']):
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len
            d = -np.dot(normal, p1)

            vert_axis = 1
            if abs(normal[vert_axis]) < np.cos(config['ground_normal_tolerance']):
                continue

            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < config['ransac_threshold']
            n_inliers = np.sum(inliers)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)

        if best_n_inliers < config['min_inliers']:
            return None, None
        return best_plane, best_inliers

    def _estimate_ground_level_percentile(self, points):
        vert_axis, up_dir = 1, -1
        vert_coords = points[:, vert_axis]
        # Up is negative, ground is at maximum Y
        ground_height = np.percentile(vert_coords, 100 - self.GROUND_CONFIG['percentile_fallback'])
        return ground_height, vert_axis

    def _compute_ground_level(self, points, colors, bbox, method='ransac'):
        if len(points) == 0:
            return None, None, False
        local_points, local_colors, _ = self._get_points_near_bbox(points, colors, bbox)
        if len(local_points) < 10:
            return None, None, False

        vert_axis = 1
        if method == 'ransac':
            plane, inliers = self._fit_ground_plane_ransac(local_points)
            if plane is not None:
                horiz_axes = [0, 2]
                if abs(plane[vert_axis]) > 1e-8:
                    horiz_contrib = sum(plane[ax] * bbox.center[ax] for ax in horiz_axes)
                    ground_height = -(horiz_contrib + plane[3]) / plane[vert_axis]
                    return ground_height, vert_axis, True
            method = 'percentile'

        if method == 'percentile':
            ground_height, va = self._estimate_ground_level_percentile(local_points)
            return ground_height, va, True

        if method == 'track':
            # Use mask/bbox points
            mask = self._load_sam3_mask(self.current_frame, bbox.track_id)
            if mask is not None:
                track_mask = self._backproject_mask_to_points(mask, points, self.current_frame)
                if track_mask is not None and np.sum(track_mask) > 10:
                    gh, va = self._estimate_ground_level_percentile(points[track_mask])
                    return gh, va, True
            gh, va = self._estimate_ground_level_percentile(local_points)
            return gh, va, True

        return None, None, False

    def _snap_bbox_to_ground(self, bbox, ground_height, vert_axis):
        corners = bbox.get_corners()
        vert_coords = corners[:, vert_axis]
        # Up is negative, bottom is max Y
        current_bottom = np.max(vert_coords)
        offset = ground_height - current_bottom
        bbox.center[vert_axis] += offset
        return offset

    def _snap_to_ground(self):
        if self.selected_track is None:
            print("No track selected!")
            return
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return
        points, colors = self._load_point_cloud(self.current_frame)
        if points is None:
            return

        method_map = {"RANSAC (robust)": "ransac", "Lowest Points": "percentile", "Track Points": "track"}
        method = method_map.get(self.ground_method_dropdown.value, "ransac")

        gh, va, ok = self._compute_ground_level(points, colors, bbox, method)
        if not ok:
            print("Ground detection failed!")
            return
        offset = self._snap_bbox_to_ground(bbox, gh, va)
        print(f"Snapped to ground: offset={offset:.4f}")
        self._update_selection()
        self._rerender_bboxes()
        self._schedule_auto_save()

    def _snap_to_ground_all_frames(self):
        if self.selected_track is None:
            return
        method_map = {"RANSAC (robust)": "ransac", "Lowest Points": "percentile", "Track Points": "track"}
        method = method_map.get(self.ground_method_dropdown.value, "ransac")

        ok_count = 0
        for frame_idx in self.frame_indices:
            if not any(b.track_id == self.selected_track for b in self.auto_bboxes.get(frame_idx, [])):
                continue
            bbox = self._get_or_create_correction(self.selected_track, frame_idx)
            if bbox is None:
                continue
            points, colors = self._load_point_cloud(frame_idx)
            if points is None:
                continue
            gh, va, ok = self._compute_ground_level(points, colors, bbox, method)
            if ok:
                self._snap_bbox_to_ground(bbox, gh, va)
                ok_count += 1

        print(f"Ground-snapped {ok_count} frames")
        self._render_frame()
        self._schedule_auto_save()

    # ======================================================================
    # Copy / Propagate
    # ======================================================================

    def _copy_from_adjacent_frame(self, direction):
        """Copy bbox from prev (-1) or next (+1) frame."""
        if self.selected_track is None:
            print("No track selected!")
            return

        target_idx = self.current_frame_idx + direction
        if target_idx < 0 or target_idx >= len(self.frame_indices):
            print("No adjacent frame!")
            return

        adj_frame = self.frame_indices[target_idx]
        adj_bbox = None
        if self._has_correction(self.selected_track, adj_frame):
            adj_bbox = self.corrections[self.selected_track][adj_frame]
        else:
            adj_bboxes = self.auto_bboxes.get(adj_frame, [])
            adj_bbox = next((b for b in adj_bboxes if b.track_id == self.selected_track), None)

        if adj_bbox is None:
            print(f"Track {self.selected_track} not in frame {adj_frame}")
            return

        if self.selected_track not in self.corrections:
            self.corrections[self.selected_track] = {}

        self.corrections[self.selected_track][self.current_frame] = BBox3D(
            center=adj_bbox.center.copy(),
            dimensions=adj_bbox.dimensions.copy(),
            rotation_matrix=adj_bbox.rotation_matrix.copy(),
            class_name=adj_bbox.class_name,
            track_id=adj_bbox.track_id,
            frame_idx=self.current_frame,
            confidence=adj_bbox.confidence,
        )

        # Copy semantic faces too
        if (self.selected_track in self.semantic_faces and
            adj_frame in self.semantic_faces[self.selected_track]):
            if self.selected_track not in self.semantic_faces:
                self.semantic_faces[self.selected_track] = {}
            self.semantic_faces[self.selected_track][self.current_frame] = \
                self.semantic_faces[self.selected_track][adj_frame].copy()

        print(f"Copied bbox from frame {adj_frame}")
        self._update_selection()
        self._rerender_bboxes()
        self._schedule_auto_save()

    def _propagate_to_all_frames(self):
        """Propagate dimensions/rotation/faces from current frame to all frames."""
        if self.selected_track is None:
            return
        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            return

        current_faces = None
        if self.selected_track in self.semantic_faces:
            current_faces = self.semantic_faces[self.selected_track].get(self.current_frame)

        count = 0
        for frame_idx in self.frame_indices:
            original = next((b for b in self.auto_bboxes.get(frame_idx, [])
                           if b.track_id == self.selected_track), None)
            if original is None:
                continue

            if self.selected_track not in self.corrections:
                self.corrections[self.selected_track] = {}

            self.corrections[self.selected_track][frame_idx] = BBox3D(
                center=original.center.copy(),  # Keep position
                dimensions=current_bbox.dimensions.copy(),
                rotation_matrix=current_bbox.rotation_matrix.copy(),
                class_name=current_bbox.class_name,
                track_id=self.selected_track,
                frame_idx=frame_idx,
                confidence=original.confidence,
            )

            if current_faces:
                if self.selected_track not in self.semantic_faces:
                    self.semantic_faces[self.selected_track] = {}
                self.semantic_faces[self.selected_track][frame_idx] = current_faces.copy()

            count += 1

        print(f"Propagated to {count} frames (dims+rotation, kept positions)")
        self._rerender_bboxes()
        self._schedule_auto_save()

    # ======================================================================
    # Keyframe Interpolation
    # ======================================================================

    def _mark_keyframe(self, kf_num):
        """Mark current frame as keyframe 1 or 2."""
        if self.selected_track is None:
            print("No track selected!")
            return
        bbox = self._get_current_bbox()
        if bbox is None:
            return

        semantic_labels = None
        if (self.selected_track in self.semantic_faces and
            self.current_frame in self.semantic_faces[self.selected_track]):
            semantic_labels = self.semantic_faces[self.selected_track][self.current_frame].copy()

        kf_data = (self.current_frame, self.selected_track, bbox, semantic_labels)

        if kf_num == 1:
            self.keyframe_1 = kf_data
            self.keyframe_1_text.value = f"Frame {self.current_frame}, Track {self.selected_track}"
        else:
            self.keyframe_2 = kf_data
            self.keyframe_2_text.value = f"Frame {self.current_frame}, Track {self.selected_track}"

        print(f"Marked Keyframe {kf_num}: Frame {self.current_frame}")

    def _interpolate_between_keyframes(self):
        """Interpolate position, rotation, dimensions between keyframes."""
        if self.keyframe_1 is None or self.keyframe_2 is None:
            print("Both keyframes must be set!")
            return

        kf1_frame, kf1_track, kf1_bbox, kf1_labels = self.keyframe_1
        kf2_frame, kf2_track, kf2_bbox, kf2_labels = self.keyframe_2

        if kf1_track != kf2_track:
            print("Keyframes must be same track!")
            return
        if kf1_frame >= kf2_frame:
            print("Keyframe 1 must come before Keyframe 2!")
            return

        track_id = kf1_track
        intermediate = [f for f in self.frame_indices if kf1_frame < f <= kf2_frame]
        if not intermediate:
            return

        # Check for existing corrections
        existing = [f for f in intermediate if self._has_correction(track_id, f)]
        if existing:
            print(f"{len(existing)} frames have corrections — click Confirm to overwrite")
            self.pending_interpolation = (
                track_id, kf1_frame, kf2_frame,
                kf1_bbox, kf2_bbox, kf1_labels, kf2_labels, intermediate
            )
            self.confirm_interpolate_btn.visible = True
            return

        self._perform_interpolation(
            track_id, kf1_frame, kf2_frame,
            kf1_bbox, kf2_bbox, kf1_labels, kf2_labels, intermediate
        )

    def _perform_interpolation(self, track_id, start_frame, end_frame,
                               kf1_bbox, kf2_bbox, kf1_labels, kf2_labels,
                               intermediate_frames):
        """Execute the interpolation."""
        pos1, pos2 = kf1_bbox.center.copy(), kf2_bbox.center.copy()
        rot1_m, rot2_m = kf1_bbox.rotation_matrix.copy(), kf2_bbox.rotation_matrix.copy()
        dims1, dims2 = kf1_bbox.dimensions.copy(), kf2_bbox.dimensions.copy()

        key_rots = R.from_matrix(np.array([rot1_m, rot2_m]))
        slerp = Slerp(np.array([0.0, 1.0]), key_rots)

        use_both = kf1_labels is not None and kf2_labels is not None

        if track_id not in self.corrections:
            self.corrections[track_id] = {}
        if track_id not in self.semantic_faces:
            self.semantic_faces[track_id] = {}

        start_pos = self.frame_indices.index(start_frame)
        end_pos = self.frame_indices.index(end_frame)

        for frame_idx in intermediate_frames:
            frame_pos = self.frame_indices.index(frame_idx)
            t = (frame_pos - start_pos) / (end_pos - start_pos)

            if frame_idx == end_frame:
                ip, ir, ids = pos2.copy(), rot2_m.copy(), dims2.copy()
                labels = kf2_labels
            else:
                ip = pos1 + t * (pos2 - pos1)
                ir = slerp([t])[0].as_matrix()
                ids = dims1 + t * (dims2 - dims1)
                labels = kf1_labels if (not use_both or t < 0.5) else kf2_labels

            self.corrections[track_id][frame_idx] = BBox3D(
                center=ip, dimensions=ids, rotation_matrix=ir,
                class_name=kf1_bbox.class_name, track_id=track_id,
                frame_idx=frame_idx, confidence=kf1_bbox.confidence,
            )
            if labels is not None:
                self.semantic_faces[track_id][frame_idx] = labels.copy()

        print(f"Interpolated {len(intermediate_frames)} frames")
        self._render_frame()
        self._schedule_auto_save()

        self.keyframe_1 = None
        self.keyframe_2 = None
        self.keyframe_1_text.value = "Not set"
        self.keyframe_2_text.value = "Not set"

    def _interpolate_semantic_faces_only(self):
        """Propagate semantic labels between keyframes without changing bboxes."""
        if self.keyframe_1 is None or self.keyframe_2 is None:
            print("Both keyframes must be set!")
            return

        kf1_frame, kf1_track, _, kf1_labels = self.keyframe_1
        kf2_frame, kf2_track, _, kf2_labels = self.keyframe_2

        if kf1_track != kf2_track:
            print("Keyframes must be same track!")
            return
        if kf1_frame >= kf2_frame:
            print("Keyframe 1 must come before Keyframe 2!")
            return
        if kf1_labels is None and kf2_labels is None:
            print("At least one keyframe must have semantic labels!")
            return

        track_id = kf1_track
        frames_to_update = [f for f in self.frame_indices if kf1_frame <= f <= kf2_frame]
        valid_frames = [f for f in frames_to_update
                       if any(b.track_id == track_id for b in self.auto_bboxes.get(f, []))]

        if not valid_frames:
            return

        if track_id not in self.semantic_faces:
            self.semantic_faces[track_id] = {}

        start_pos = self.frame_indices.index(kf1_frame)
        end_pos = self.frame_indices.index(kf2_frame)
        mid_pos = (start_pos + end_pos) // 2

        for frame_idx in valid_frames:
            fp = self.frame_indices.index(frame_idx)
            if kf1_labels is not None and kf2_labels is not None:
                labels = kf1_labels.copy() if fp <= mid_pos else kf2_labels.copy()
            elif kf1_labels is not None:
                labels = kf1_labels.copy()
            else:
                labels = kf2_labels.copy()
            self.semantic_faces[track_id][frame_idx] = labels

        print(f"Applied semantic labels to {len(valid_frames)} frames")
        self._render_frame()
        self._update_semantic_dropdowns()
        self._schedule_auto_save()

        self.keyframe_1 = None
        self.keyframe_2 = None
        self.keyframe_1_text.value = "Not set"
        self.keyframe_2_text.value = "Not set"

    # ======================================================================
    # Semantic Faces
    # ======================================================================

    def _update_semantic_dropdowns(self):
        self.updating_dropdowns = True
        self.front_face_dropdown.value = "(None)"
        self.top_face_dropdown.value = "(None)"
        self.left_face_dropdown.value = "(None)"

        if (self.selected_track in self.semantic_faces and
            self.current_frame in self.semantic_faces[self.selected_track]):
            labels = self.semantic_faces[self.selected_track][self.current_frame]
            if 'front' in labels:
                self.front_face_dropdown.value = f"Face {labels['front']}"
            if 'top' in labels:
                self.top_face_dropdown.value = f"Face {labels['top']}"
            if 'left' in labels:
                self.left_face_dropdown.value = f"Face {labels['left']}"

        self.updating_dropdowns = False

    def _auto_apply_semantic_labels(self):
        """Auto-apply and save when dropdown changes."""
        labels = {}
        if self.front_face_dropdown.value != "(None)":
            labels['front'] = int(self.front_face_dropdown.value.split()[-1])
        if self.top_face_dropdown.value != "(None)":
            labels['top'] = int(self.top_face_dropdown.value.split()[-1])
        if self.left_face_dropdown.value != "(None)":
            labels['left'] = int(self.left_face_dropdown.value.split()[-1])

        if labels and self.selected_track is not None:
            if self.selected_track not in self.semantic_faces:
                self.semantic_faces[self.selected_track] = {}
            self.semantic_faces[self.selected_track][self.current_frame] = labels
            self._rerender_bboxes()

        self._schedule_auto_save()

    # ======================================================================
    # Info
    # ======================================================================

    def _update_info_text(self):
        frame_name = ""
        if self.current_frame < len(self.frame_numbers):
            frame_name = f" (frame_{self.frame_numbers[self.current_frame]:06d})"

        if self.selected_track is None:
            base = f"Frame {self.current_frame}{frame_name}"
        else:
            bbox = self._get_current_bbox()
            if bbox is None:
                base = f"Track {self.selected_track} not in frame {self.current_frame}"
            else:
                corrected = " [corrected]" if self._has_correction(self.selected_track, self.current_frame) else ""
                base = f"T{self.selected_track} | {bbox.class_name} | Frame {self.current_frame}{frame_name}{corrected}"

        if self.last_save_time:
            elapsed = (datetime.datetime.now() - self.last_save_time).total_seconds()
            if elapsed < 60:
                base += f" | saved {int(elapsed)}s ago"
            else:
                base += f" | saved {int(elapsed/60)}m ago"

        self.info_text.value = base

    def _next_unannotated_frame(self):
        if self.selected_track is None:
            return

        # Search forward from current
        for i in list(range(self.current_frame_idx + 1, len(self.frame_indices))) + \
                 list(range(0, self.current_frame_idx)):
            fi = self.frame_indices[i]
            if not any(b.track_id == self.selected_track for b in self.auto_bboxes.get(fi, [])):
                continue
            if not self._has_correction(self.selected_track, fi):
                self.updating_sliders = True
                self.frame_slider.value = i
                self.updating_sliders = False
                self.current_frame_idx = i
                self.current_frame = fi
                self._render_frame()
                print(f"Jumped to unannotated frame {fi}")
                return

        print(f"All frames annotated for track {self.selected_track}!")

    # ======================================================================
    # Save
    # ======================================================================

    def _schedule_auto_save(self):
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()
        self.auto_save_timer = threading.Timer(2.0, self._perform_auto_save)
        self.auto_save_timer.start()

    def _perform_auto_save(self):
        self._save_all()
        self.last_save_time = datetime.datetime.now()
        self.auto_save_timer = None
        self._update_info_text()

    def _manual_save(self):
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None
        self._save_all()
        self.last_save_time = datetime.datetime.now()
        self._update_info_text()
        print("Manual save completed!")

    def _save_all(self):
        """Save everything: tracking_summary, corrections, kitti, semantic faces."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build updated tracking_summary.json
        updated = copy.deepcopy(self.tracking_summary)
        for track_id, frame_corrections in self.corrections.items():
            track_key = str(track_id)
            if track_key not in updated.get('tracks', {}):
                continue
            track = updated['tracks'][track_key]
            frame_to_idx = {f: i for i, f in enumerate(track['frames'])}
            for frame_idx, bbox in frame_corrections.items():
                if frame_idx in frame_to_idx:
                    i = frame_to_idx[frame_idx]
                    track['centers'][i] = bbox.center.tolist()
                    track['dimensions'][i] = bbox.dimensions.tolist()
                    track['rotation_matrices'][i] = bbox.rotation_matrix.tolist()

        with open(self.output_dir / "tracking_summary.json", 'w') as f:
            json.dump(updated, f, indent=2)

        # 2. Save corrections.json (for session persistence)
        corrections_data = {}
        for track_id, frames in self.corrections.items():
            corrections_data[str(track_id)] = {}
            for frame_idx, bbox in frames.items():
                corrections_data[str(track_id)][str(frame_idx)] = {
                    'center': bbox.center.tolist(),
                    'dimensions': bbox.dimensions.tolist(),
                    'rotation_matrix': bbox.rotation_matrix.tolist(),
                    'class_name': bbox.class_name,
                }
        with open(self.output_dir / "corrections.json", 'w') as f:
            json.dump(corrections_data, f, indent=2)

        # 3. Save semantic faces
        if self.semantic_faces:
            sem_dir = self.output_dir / "semantic_faces"
            sem_dir.mkdir(exist_ok=True)
            sem_data = {}
            for track_id, frames in self.semantic_faces.items():
                sem_data[str(track_id)] = {str(fi): labels for fi, labels in frames.items()}
            with open(sem_dir / "manual_labels.json", 'w') as f:
                json.dump(sem_data, f, indent=2)

        # 4. Save KITTI labels
        self._save_kitti_labels(updated)

        # 5. Generate 2D projections
        self._generate_2d_projections(updated)

    def _save_kitti_labels(self, tracking_summary):
        """Regenerate KITTI-format labels from updated tracking summary."""
        if self.cam_params is None:
            return

        kitti_dir = self.output_dir / "kitti_labels"
        kitti_dir.mkdir(exist_ok=True)

        # Build per-frame bbox data
        per_frame = {}
        for track_id_str, track_data in tracking_summary.get('tracks', {}).items():
            track_id = int(track_id_str)
            class_name = self.detected_class_name or track_data.get('class_name', 'object')
            for i, frame_idx in enumerate(track_data['frames']):
                if frame_idx >= len(self.cam_params['extrinsics']):
                    continue
                center = np.array(track_data['centers'][i])
                dims = np.array(track_data['dimensions'][i])
                rot = np.array(track_data['rotation_matrices'][i])
                conf = track_data['confidences'][i] if i < len(track_data.get('confidences', [])) else 1.0

                ext = self.cam_params['extrinsics'][frame_idx]
                intr = self.cam_params['intrinsics'][frame_idx]

                # Transform center to camera coords
                center_h = np.append(center, 1.0)
                center_cam = ext @ center_h  # (3,)

                # Rotation angle around Y axis in camera frame
                rot_y = math.atan2(rot[0, 2], rot[2, 2])

                # Alpha (observation angle)
                alpha = rot_y - math.atan2(center_cam[0], center_cam[2])
                alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

                # 2D bbox from 3D projection
                bbox3d = BBox3D(center, dims, rot, class_name, track_id, frame_idx)
                corners_3d = bbox3d.get_corners()
                corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
                corners_cam = (ext @ corners_h.T).T
                valid = corners_cam[:, 2] > 0.01
                if not np.any(valid):
                    continue

                proj = (intr @ corners_cam[valid].T).T
                proj_2d = proj[:, :2] / proj[:, 2:3]

                img_h = self.cam_params['image_height']
                img_w = self.cam_params['image_width']
                x_min = max(0, np.min(proj_2d[:, 0]))
                y_min = max(0, np.min(proj_2d[:, 1]))
                x_max = min(img_w, np.max(proj_2d[:, 0]))
                y_max = min(img_h, np.max(proj_2d[:, 1]))

                # KITTI format: h, w, l (reorder from l, w, h)
                h, w, l = dims[2], dims[1], dims[0]
                x, y, z = center_cam

                # Get original frame number for filename
                if frame_idx < len(self.frame_numbers):
                    orig_frame = self.frame_numbers[frame_idx]
                else:
                    orig_frame = frame_idx

                per_frame.setdefault(orig_frame, []).append(
                    f"{class_name} 0.00 0 {alpha:.2f} "
                    f"{x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "
                    f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} "
                    f"{rot_y:.2f} {conf:.2f}"
                )

        for orig_frame, lines in per_frame.items():
            fname = kitti_dir / f"frame_{orig_frame:06d}.txt"
            with open(fname, 'w') as f:
                f.write('\n'.join(lines) + '\n')

    def _generate_2d_projections(self, tracking_summary):
        """Generate 2D bbox projection images."""
        if not HAS_CV2 or self.cam_params is None:
            return

        # Try to find source images
        images_dir = None
        # Check for extracted frames referenced in metadata
        if self.metadata.get('image_files'):
            # Look for images in common locations
            for candidate in [self.result_dir / "images", self.result_dir / "frames",
                            self.result_dir.parent / "frames"]:
                if candidate.exists():
                    images_dir = candidate
                    break

        if images_dir is None:
            return

        vis_dir = self.output_dir / "annotated_2d"
        vis_dir.mkdir(exist_ok=True)

        for track_id_str, track_data in tracking_summary.get('tracks', {}).items():
            for i, frame_idx in enumerate(track_data['frames']):
                if frame_idx >= len(self.cam_params['extrinsics']):
                    continue

                orig_frame = self.frame_numbers[frame_idx] if frame_idx < len(self.frame_numbers) else frame_idx

                # Find image file
                img_file = None
                for ext_str in ['.jpg', '.png']:
                    for fmt in [f"frame_{orig_frame:06d}{ext_str}", f"{orig_frame:06d}{ext_str}"]:
                        candidate = images_dir / fmt
                        if candidate.exists():
                            img_file = candidate
                            break
                    if img_file:
                        break

                if img_file is None:
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                center = np.array(track_data['centers'][i])
                dims = np.array(track_data['dimensions'][i])
                rot = np.array(track_data['rotation_matrices'][i])
                class_name = self.detected_class_name or track_data.get('class_name', 'object')

                ext = self.cam_params['extrinsics'][frame_idx]
                intr = self.cam_params['intrinsics'][frame_idx]

                # Scale intrinsics to image size
                img_h, img_w = img.shape[:2]
                model_h = self.cam_params['image_height']
                model_w = self.cam_params['image_width']
                K = intr.copy()
                K[0, :] *= img_w / model_w
                K[1, :] *= img_h / model_h

                bbox3d = BBox3D(center, dims, rot, class_name, int(track_id_str), frame_idx)
                corners_3d = bbox3d.get_corners()
                corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
                corners_cam = (ext @ corners_h.T).T
                valid = corners_cam[:, 2] > 0.01
                if not np.any(valid):
                    continue

                proj = (K @ corners_cam.T).T
                corners_2d = (proj[:, :2] / proj[:, 2:3]).astype(int)

                # Corrected = green, original = blue
                is_corrected = self._has_correction(int(track_id_str), frame_idx)
                color = (0, 255, 0) if is_corrected else (255, 0, 0)

                for si, ei in bbox3d.get_edges():
                    cv2.line(img, tuple(corners_2d[si]), tuple(corners_2d[ei]), color, 2)

                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(img, f"T{track_id_str}", tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                out_file = vis_dir / f"frame_{orig_frame:06d}.png"
                cv2.imwrite(str(out_file), img)

    # ======================================================================
    # Main Loop
    # ======================================================================

    def run(self):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nSaving...")
            if self.auto_save_timer is not None:
                self.auto_save_timer.cancel()
            self._save_all()
            print("Done!")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VGGT 3D Bounding Box Annotator Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-discovers sam3_masks as sibling directory
    python annotator_tool.py --result_dir /path/to/seg1/vggt_results/

    # Explicit paths
    python annotator_tool.py \\
        --result_dir /path/to/seg1/vggt_results/ \\
        --sam3_masks /path/to/seg1/sam3_masks/ \\
        --output_dir /path/to/seg1/vggt_results/annotations/ \\
        --port 8080
        """
    )

    parser.add_argument("--result_dir", required=True,
                        help="Directory containing VGGT outputs (tracking_summary.json, cameras.json, etc.)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for annotations (default: result_dir/annotations/)")
    parser.add_argument("--sam3_masks", default=None,
                        help="Path to SAM3 masks directory (auto-detected as sibling if not given)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port (default: 8080)")
    parser.add_argument("--no_reload", action="store_true",
                        help="Don't reload previous session annotations")

    args = parser.parse_args()

    editor = VGGTBBoxEditor(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        sam3_masks_dir=args.sam3_masks,
        reload_annotations=not args.no_reload,
        port=args.port,
    )

    editor.run()


if __name__ == "__main__":
    main()
