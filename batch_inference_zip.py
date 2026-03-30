#!/usr/bin/env python3
"""
Run VGGT inference and 3D tracking on video segments with SAM3 masks.

Takes a directory (or zip) containing pre-extracted frame segments with SAM3
masks (from sam3/scripts/generate_sam3_masks.py), runs VGGT to get
depth/cameras/3D points, then runs tracking with DJI SRT gimbal grounding
to produce KITTI-format 3D bounding box labels.

Input structure (directory or zip):
    vid1/
        vid1.SRT
        seg1/
            frame_000300.jpg ...
            metadata.json
            sam3_masks/
                masks/obj_0/frame_000300.png ...
                metadata.json
        seg2/ ...
    vid2/ ...

Output adds per segment:
    seg1/vggt_results/
        cameras.json
        depth_maps.npz
        point_cloud.ply
        kitti_labels/frame_000300.txt ...
        tracking_summary.json
        annotated_2d/frame_000300.jpg ...
        vggt_metadata.json

Usage:
    # Directory input/output (recommended for inspecting intermediate results)
    python batch_inference_zip.py \
        --input-dir /path/to/segments_with_masks/ \
        --output-dir /path/to/output/ \
        --conf-threshold 50.0

    # Zip input (extracts to output-dir, results saved in place)
    python batch_inference_zip.py \
        --input-zip /path/to/segments_with_masks.zip \
        --output-dir /path/to/output/ \
        --conf-threshold 50.0
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add parent directory to path for imports (same pattern as other batch_inference scripts)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.export import save_point_cloud_ply, save_cameras_json, save_depth_maps
from vggt.utils.sam3_mask_loader import (
    load_sam3_masks, detect_sam3_format, load_sam3_masks_multi_class,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT inference + tracking on video segments"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-zip", type=str, default=None,
                             help="Path to input zip (extracted to --output-dir)")
    input_group.add_argument("--input-dir", type=str, default=None,
                             help="Path to input directory (used directly, or copied to --output-dir)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results (segments + vggt_results/)")
    parser.add_argument("--conf-threshold", type=float, default=50.0,
                        help="Confidence percentile for point cloud filtering (default: 50.0)")
    parser.add_argument("--max-distance", type=float, default=8.0,
                        help="Max 3D distance for tracking (default: 8.0)")
    parser.add_argument("--max-missing-frames", type=int, default=20,
                        help="Frames before track goes dormant (default: 20)")
    parser.add_argument("--dormant-timeout", type=int, default=100,
                        help="Frames before dormant track removed (default: 100)")
    parser.add_argument("--object-class", type=str, default="object",
                        help="Default class name for SAM3 objects (default: 'object')")
    parser.add_argument("--sam3-class", type=str, default=None,
                        help="Class name(s) to load (comma-separated)")
    parser.add_argument("--use-point-map", action="store_true",
                        help="Use point map instead of depth-based points")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip tracking (geometry only: cameras, depth, point cloud)")
    return parser.parse_args()


def discover_segments(work_dir: str):
    """
    Walk the work directory and find all segment folders (those with metadata.json
    and frame_numbers).
    Returns list of (video_dir, seg_dir, metadata) tuples sorted by path.
    """
    segments = []
    for root, dirs, files in os.walk(work_dir):
        if "metadata.json" in files:
            # Skip SAM3/VGGT subdirectory metadata
            if "sam3_masks" in root or "vggt_results" in root:
                continue
            meta_path = os.path.join(root, "metadata.json")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "frame_numbers" in meta:
                    video_dir = os.path.dirname(root)
                    segments.append((video_dir, root, meta))
            except (json.JSONDecodeError, KeyError):
                continue

    segments.sort(key=lambda x: x[1])
    return segments


def load_progress(work_dir: str, params: dict) -> dict:
    """Load or create vggt_progress.json."""
    progress_path = os.path.join(work_dir, "vggt_progress.json")
    fresh = {"params": params, "completed_segments": [], "timestamp": None}

    if not os.path.isfile(progress_path):
        return fresh

    with open(progress_path) as f:
        progress = json.load(f)

    if progress.get("params") != params:
        print("  VGGT parameters changed, restarting processing")
        return fresh

    return progress


def save_progress(work_dir: str, progress: dict):
    """Save vggt_progress.json atomically."""
    progress_path = os.path.join(work_dir, "vggt_progress.json")
    progress["timestamp"] = datetime.now().isoformat()
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)


def is_segment_vggt_complete(seg_dir: str):
    """Check if VGGT results already exist and look complete."""
    results_dir = os.path.join(seg_dir, "vggt_results")
    if not os.path.isdir(results_dir):
        return False
    required = ["cameras.json", "depth_maps.npz", "vggt_metadata.json"]
    return all(os.path.isfile(os.path.join(results_dir, f)) for f in required)


def resolve_srt_path(seg_dir: str, metadata: dict) -> Optional[str]:
    """
    Resolve the SRT file path from segment metadata.
    metadata has srt_path as relative from segment folder (e.g., "../vid1.SRT").
    """
    srt_rel = metadata.get("srt_path")
    if not srt_rel:
        return None
    srt_path = os.path.normpath(os.path.join(seg_dir, srt_rel))
    if os.path.isfile(srt_path):
        return srt_path
    # Also check for common SRT extensions
    for ext in [".SRT", ".srt"]:
        candidate = os.path.splitext(srt_path)[0] + ext
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# VGGT Processing
# ---------------------------------------------------------------------------

def load_vggt_model(device: str = "cuda"):
    """Load VGGT model once."""
    print("Loading VGGT model...")
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s")
    return model


def run_segment_inference(model, image_paths: List[str], device: str = "cuda"):
    """
    Run VGGT inference on a list of image paths.
    Returns predictions dict (on CPU) and timing dict.
    """
    t0 = time.time()
    images = load_and_preprocess_images(image_paths).to(device)
    t_load = time.time() - t0

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    t1 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    torch.cuda.synchronize()
    t_infer = time.time() - t1

    # Post-process
    t2 = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images.cpu()
    t_post = time.time() - t2

    # Move everything to CPU
    predictions_cpu = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions_cpu[key] = value.cpu()
        else:
            predictions_cpu[key] = value

    # Free GPU memory
    del predictions, images
    torch.cuda.empty_cache()

    timing = {
        "image_loading": round(t_load, 2),
        "inference": round(t_infer, 2),
        "post_processing": round(t_post, 2),
    }

    return predictions_cpu, timing


def save_vggt_outputs(output_dir: str, predictions_cpu: dict,
                      image_paths: List[str], conf_threshold: float,
                      use_point_map: bool):
    """Save VGGT geometry outputs (cameras, depth, point cloud). No predictions.pt."""
    os.makedirs(output_dir, exist_ok=True)

    def maybe_squeeze_batch(arr):
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr

    images = maybe_squeeze_batch(predictions_cpu["images"].numpy())
    extrinsics = maybe_squeeze_batch(predictions_cpu["extrinsic"].numpy())
    intrinsics = maybe_squeeze_batch(predictions_cpu["intrinsic"].numpy())
    depth = maybe_squeeze_batch(predictions_cpu["depth"].numpy())
    depth_conf = maybe_squeeze_batch(predictions_cpu["depth_conf"].numpy())

    if use_point_map:
        world_points = maybe_squeeze_batch(predictions_cpu["world_points"].numpy())
        conf = maybe_squeeze_batch(predictions_cpu["world_points_conf"].numpy())
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        conf = depth_conf

    # Point cloud
    colors = images.transpose(0, 2, 3, 1)
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    num_points = save_point_cloud_ply(
        ply_path, world_points, colors, conf,
        conf_threshold_percentile=conf_threshold,
    )
    print(f"      Point cloud: {num_points:,} points")

    # Cameras
    image_names = [os.path.basename(p) for p in image_paths]
    cameras_path = os.path.join(output_dir, "cameras.json")
    save_cameras_json(
        cameras_path, extrinsics, intrinsics,
        image_names=image_names,
        image_size=(depth.shape[1], depth.shape[2]),
    )

    # Depth maps
    depth_path = os.path.join(output_dir, "depth_maps.npz")
    save_depth_maps(depth_path, depth, depth_conf)


def run_segment_tracking(output_dir: str, predictions_cpu: dict,
                         image_paths: List[str], masks_data: dict,
                         srt_path: Optional[str], frame_indices: List[int],
                         max_distance: float, max_missing_frames: int,
                         dormant_timeout: int, use_point_map: bool):
    """Run tracking pipeline on a segment's predictions."""
    from demo_viser_tracking import (
        compute_instance_bboxes,
        ImprovedTracker,
        save_tracking_summary,
        save_trajectory_plots,
        save_kitti_labels,
        project_bboxes_to_2d,
        parse_dji_logs,
    )
    import cv2

    def maybe_squeeze_batch(arr):
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr.squeeze(0)
        return arr

    # Convert to numpy
    if isinstance(predictions_cpu["extrinsic"], torch.Tensor):
        extrinsics = maybe_squeeze_batch(predictions_cpu["extrinsic"].cpu().numpy())
        intrinsics = maybe_squeeze_batch(predictions_cpu["intrinsic"].cpu().numpy())
        depth = maybe_squeeze_batch(predictions_cpu["depth"].cpu().numpy())
        depth_conf = maybe_squeeze_batch(predictions_cpu["depth_conf"].cpu().numpy())
        images = maybe_squeeze_batch(predictions_cpu["images"].cpu().numpy())
    else:
        extrinsics = maybe_squeeze_batch(predictions_cpu["extrinsic"])
        intrinsics = maybe_squeeze_batch(predictions_cpu["intrinsic"])
        depth = maybe_squeeze_batch(predictions_cpu["depth"])
        depth_conf = maybe_squeeze_batch(predictions_cpu["depth_conf"])
        images = maybe_squeeze_batch(predictions_cpu["images"])

    if use_point_map and "world_points" in predictions_cpu:
        if isinstance(predictions_cpu["world_points"], torch.Tensor):
            world_points = maybe_squeeze_batch(predictions_cpu["world_points"].cpu().numpy())
        else:
            world_points = maybe_squeeze_batch(predictions_cpu["world_points"])
    else:
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)

    # Parse DJI SRT for gimbal grounding
    gimbal_data = None
    if srt_path:
        gimbal_data = parse_dji_logs(srt_path, frame_indices)
        if gimbal_data:
            print(f"      DJI gimbal data loaded ({len(gimbal_data)} frames)")
        else:
            print(f"      WARNING: Could not parse DJI data from {srt_path}")

    # Load original images for 2D projection
    original_images = [cv2.imread(p) for p in image_paths]

    # Tracker
    tracker = ImprovedTracker(
        max_distance=max_distance,
        mask_iou_threshold=0.15,
        max_missing_frames=max_missing_frames,
        dormant_timeout=dormant_timeout,
    )

    # Compute 3D bounding boxes
    model_size = (depth.shape[1], depth.shape[2])
    bounding_boxes = compute_instance_bboxes(
        world_points, masks_data, original_images, model_size, tracker,
        gimbal_data=gimbal_data,
    )

    # Collect track IDs
    all_track_ids = []
    for frame_bboxes in bounding_boxes:
        for bbox in frame_bboxes:
            if bbox.track_id is not None and bbox.track_id >= 0:
                if bbox.track_id not in all_track_ids:
                    all_track_ids.append(bbox.track_id)
    all_track_ids = sorted(all_track_ids)

    print(f"      Tracks: {len(all_track_ids)} unique")

    if bounding_boxes and all_track_ids:
        frame_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

        save_tracking_summary(
            output_dir, bounding_boxes, all_track_ids,
            extrinsics=extrinsics, intrinsics=intrinsics,
            image_size=model_size,
        )

        save_kitti_labels(
            output_dir, bounding_boxes,
            extrinsics, intrinsics,
            model_size, frame_names,
        )

        save_trajectory_plots(output_dir, bounding_boxes, all_track_ids)

        project_bboxes_to_2d(
            bounding_boxes, original_images,
            extrinsics, intrinsics,
            output_dir, frame_names, all_track_ids,
            model_size=model_size,
        )

    return len(all_track_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve input → work_dir
    work_dir = os.path.abspath(args.output_dir)

    if args.input_zip:
        if not os.path.isfile(args.input_zip):
            print(f"Input zip not found: {args.input_zip}")
            sys.exit(1)
        os.makedirs(work_dir, exist_ok=True)
        print(f"Extracting {args.input_zip} to {work_dir}...")
        with zipfile.ZipFile(args.input_zip, "r") as zf:
            zf.extractall(work_dir)
    elif args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            print(f"Input directory not found: {input_dir}")
            sys.exit(1)
        if input_dir != work_dir:
            # Copy input to output dir so results are saved alongside data
            print(f"Copying {input_dir} to {work_dir}...")
            if os.path.exists(work_dir):
                # Merge into existing output dir (resume-friendly)
                for item in os.listdir(input_dir):
                    src = os.path.join(input_dir, item)
                    dst = os.path.join(work_dir, item)
                    if os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
                    elif os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
            else:
                shutil.copytree(input_dir, work_dir)
        else:
            print(f"Input and output are the same directory: {work_dir}")

    # Discover segments
    segments = discover_segments(work_dir)
    print(f"Found {len(segments)} segment(s)\n")

    if not segments:
        print("No segments found.")
        shutil.rmtree(work_dir)
        sys.exit(1)

    # Progress
    run_params = {
        "conf_threshold": args.conf_threshold,
        "max_distance": args.max_distance,
        "use_point_map": args.use_point_map,
    }
    progress = load_progress(work_dir, run_params)
    completed = set(progress.get("completed_segments", []))

    # Load VGGT model once
    model = load_vggt_model(device)

    total_processed = 0
    total_skipped = 0
    total_tracks = 0

    try:
        for video_dir, seg_dir, metadata in segments:
            seg_name = os.path.basename(seg_dir)
            video_name = os.path.basename(video_dir)
            seg_id = f"{video_name}/{seg_name}"

            # Check if already complete
            if seg_id in completed and is_segment_vggt_complete(seg_dir):
                print(f"[{seg_id}] SKIP (already complete)")
                total_skipped += 1
                continue

            # Clean up incomplete results
            results_dir = os.path.join(seg_dir, "vggt_results")
            if os.path.isdir(results_dir):
                shutil.rmtree(results_dir)

            frame_numbers = metadata["frame_numbers"]
            pair_label = metadata.get("pair_seg_label", seg_name)
            print(f"[{seg_id}] {len(frame_numbers)} frames ({pair_label})")

            # Collect frame image paths (sorted by frame number)
            image_paths = []
            for fn in sorted(frame_numbers):
                path = os.path.join(seg_dir, f"frame_{fn:06d}.jpg")
                if os.path.isfile(path):
                    image_paths.append(path)

            if not image_paths:
                print(f"    No frame files found, skipping")
                continue

            # --- VGGT Inference ---
            print(f"    Running VGGT inference on {len(image_paths)} frames...")
            t0 = time.time()
            predictions_cpu, timing = run_segment_inference(model, image_paths, device)
            print(f"    Inference: {timing['inference']}s, "
                  f"load: {timing['image_loading']}s, "
                  f"post: {timing['post_processing']}s")

            # --- Save geometry outputs ---
            os.makedirs(results_dir, exist_ok=True)
            save_vggt_outputs(
                results_dir, predictions_cpu, image_paths,
                conf_threshold=args.conf_threshold,
                use_point_map=args.use_point_map,
            )

            # --- Tracking (if masks exist and not skipped) ---
            num_tracks = 0
            sam3_masks_dir = os.path.join(seg_dir, "sam3_masks")

            if not args.skip_tracking and os.path.isdir(sam3_masks_dir):
                # Frame indices are the original video frame numbers
                frame_indices = sorted(frame_numbers)

                # Determine FPS for mask matching
                video_fps = metadata.get("video_fps", 30.0)
                extract_fps = metadata.get("extract_fps", video_fps)

                # Load masks
                format_info = detect_sam3_format(sam3_masks_dir)
                class_names = None
                if args.sam3_class:
                    class_names = [c.strip() for c in args.sam3_class.split(",")]

                if format_info["format"] == "multi_class":
                    masks_data = load_sam3_masks_multi_class(
                        sam3_output_dir=sam3_masks_dir,
                        extracted_frame_indices=list(range(len(image_paths))),
                        video_fps=extract_fps,
                        sam3_fps=extract_fps,
                        class_names=class_names,
                        direct_frame_match=True,
                    )
                else:
                    masks_data = load_sam3_masks(
                        sam3_output_dir=sam3_masks_dir,
                        extracted_frame_indices=list(range(len(image_paths))),
                        video_fps=extract_fps,
                        sam3_fps=extract_fps,
                        class_name=args.object_class,
                        direct_frame_match=True,
                        auto_detect_format=False,
                    )

                if masks_data:
                    # Resolve SRT path for gimbal grounding
                    srt_path = resolve_srt_path(seg_dir, metadata)
                    if srt_path:
                        print(f"    SRT: {srt_path}")

                    print(f"    Running tracking...")
                    num_tracks = run_segment_tracking(
                        output_dir=results_dir,
                        predictions_cpu=predictions_cpu,
                        image_paths=image_paths,
                        masks_data=masks_data,
                        srt_path=srt_path,
                        frame_indices=frame_indices,
                        max_distance=args.max_distance,
                        max_missing_frames=args.max_missing_frames,
                        dormant_timeout=args.dormant_timeout,
                        use_point_map=args.use_point_map,
                    )
                else:
                    print(f"    No masks loaded from {sam3_masks_dir}")
            elif args.skip_tracking:
                print(f"    Tracking skipped (--skip-tracking)")
            else:
                print(f"    No sam3_masks/ found, skipping tracking")

            # --- Save VGGT metadata ---
            vggt_metadata = {
                "segment": seg_id,
                "pair_seg_label": pair_label,
                "num_frames": len(image_paths),
                "frame_numbers": sorted(frame_numbers),
                "video_fps": metadata.get("video_fps"),
                "extract_fps": metadata.get("extract_fps"),
                "srt_path": metadata.get("srt_path"),
                "num_tracks": num_tracks,
                "settings": {
                    "conf_threshold": args.conf_threshold,
                    "max_distance": args.max_distance,
                    "use_point_map": args.use_point_map,
                    "object_class": args.object_class,
                },
                "timing": timing,
                "total_time": round(time.time() - t0, 2),
                "timestamp": datetime.now().isoformat(),
            }
            meta_path = os.path.join(results_dir, "vggt_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(vggt_metadata, f, indent=2)

            # --- Cleanup ---
            del predictions_cpu
            gc.collect()
            torch.cuda.empty_cache()

            total_processed += 1
            total_tracks += num_tracks
            progress["completed_segments"].append(seg_id)
            save_progress(work_dir, progress)
            print()

    finally:
        # Free model
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. Processed {total_processed} segment(s), "
          f"skipped {total_skipped}, {total_tracks} total tracks.")
    print(f"Results saved to: {work_dir}")
    print(f"\nTo zip results: python zip_results.py --input-dir {work_dir} --output-zip output.zip")


if __name__ == "__main__":
    main()
