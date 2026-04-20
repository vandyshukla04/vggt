#!/usr/bin/env python3
"""
Zip a results directory and/or generate dataset statistics.

Can compute comprehensive dataset-level statistics from segment metadata,
SAM3 mask metadata, VGGT results, tracking summaries, and KITTI labels.

Usage:
    # Zip results
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip

    # Zip and delete the source folder
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip --delete-after

    # Stats only (no zipping)
    python zip_results.py --input-dir /path/to/results/ --stats-only

    # Zip + stats (saves dataset_summary.json alongside the zip)
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip --stats
"""

import argparse
import collections
import json
import os
import shutil
import sys
import zipfile
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zip a results directory and/or compute dataset statistics"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the directory to zip / analyze")
    parser.add_argument("--output-zip", type=str, default=None,
                        help="Path for the output zip file (omit for --stats-only)")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete the input directory after successful zipping")
    parser.add_argument("--include-predictions", action="store_true",
                        help="Include predictions.pt files (large, excluded by default)")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="Additional file patterns to exclude (e.g., '*.tmp')")
    parser.add_argument("--stats", action="store_true",
                        help="Compute and save dataset statistics alongside the zip")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics (no zipping)")
    parser.add_argument("--include-video", type=str, nargs="*", default=None,
                        help="Include only these video names (all segments of each). "
                             "Optional — if omitted, all videos are included.")
    parser.add_argument("--include-segment", type=str, nargs="*", default=None,
                        help="Include only these segments, format 'video/segN'. "
                             "Optional — if omitted, all segments of included videos are kept. "
                             "Use with --include-video to pick specific segments within a video.")
    return parser.parse_args()


def should_exclude(filename: str, exclude_patterns: list, include_predictions: bool) -> bool:
    """Check if a file should be excluded from the zip."""
    if not include_predictions and filename == "predictions.pt":
        return True
    for pattern in exclude_patterns:
        if pattern.startswith("*."):
            ext = pattern[1:]
            if filename.endswith(ext):
                return True
        elif filename == pattern:
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def load_json_safe(path):
    """Load a JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def count_kitti_labels(kitti_dir):
    """Count detections and classes from KITTI label files."""
    if not os.path.isdir(kitti_dir):
        return 0, 0, collections.Counter()

    total_detections = 0
    frames_with_labels = 0
    class_counts = collections.Counter()

    for fname in os.listdir(kitti_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(kitti_dir, fname)
        frame_dets = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 15:
                    class_name = parts[0]
                    class_counts[class_name] += 1
                    frame_dets += 1
                    total_detections += 1
        if frame_dets > 0:
            frames_with_labels += 1

    return total_detections, frames_with_labels, class_counts


def segment_passes_filter(video_name: str, seg_name: str,
                          include_videos: Optional[list],
                          include_segments: Optional[list]) -> bool:
    """
    Return True if this (video, segment) passes the include filters.
    - If both filters are None, everything passes.
    - If include_videos given, video must be in it.
    - If include_segments given, 'video/seg' must be in it.
    - Both can be combined (AND).
    """
    seg_id = f"{video_name}/{seg_name}"
    if include_videos is not None and video_name not in include_videos:
        return False
    if include_segments is not None and seg_id not in include_segments:
        return False
    return True


def discover_segment_paths(input_dir: str) -> list:
    """
    Walk input_dir to find all valid segment directories.
    Returns list of (video_name, seg_name, seg_abs_path, video_abs_path) tuples.
    Works at any depth — uses metadata.json + frame_numbers as the identifier.
    """
    found = []
    for root, dirs, files in os.walk(input_dir):
        if "sam3_masks" in root or "vggt_results" in root:
            continue
        if "metadata.json" not in files:
            continue
        meta = load_json_safe(os.path.join(root, "metadata.json"))
        if not meta or "frame_numbers" not in meta:
            continue
        seg_abs = os.path.abspath(root)
        video_abs = os.path.dirname(seg_abs)
        found.append((
            os.path.basename(video_abs),
            os.path.basename(seg_abs),
            seg_abs,
            video_abs,
        ))
    return found


def build_included_paths(input_dir: str,
                         include_videos: Optional[list],
                         include_segments: Optional[list]) -> tuple:
    """
    Build the set of absolute paths (segment dirs + video dirs for SRT/etc.)
    that should be included based on filters.

    Returns (included_seg_dirs, included_video_dirs) as two sets of absolute paths.
    """
    if include_videos is None and include_segments is None:
        return None, None  # no filtering

    all_segs = discover_segment_paths(input_dir)
    included_segs = set()
    included_videos = set()

    for video_name, seg_name, seg_abs, video_abs in all_segs:
        if segment_passes_filter(video_name, seg_name, include_videos, include_segments):
            included_segs.add(seg_abs)
            included_videos.add(video_abs)

    return included_segs, included_videos


def file_passes_filter(file_path: str,
                       included_seg_dirs: Optional[set],
                       included_video_dirs: Optional[set]) -> bool:
    """
    Determine if a file should be included based on pre-computed passing directories.
    - If included_seg_dirs is None, no filter is active → include everything.
    - Otherwise include if the file is inside a passing segment directory, or is a
      direct child of a passing video directory (video-level files like SRT).
    """
    if included_seg_dirs is None:
        return True

    fpath_abs = os.path.abspath(file_path)
    parent = os.path.dirname(fpath_abs)

    # Inside a passing segment directory?
    for seg_dir in included_seg_dirs:
        if fpath_abs.startswith(seg_dir + os.sep) or parent == seg_dir:
            return True

    # Video-level file (e.g., SRT directly in the video dir)?
    if parent in included_video_dirs:
        return True

    return False


def compute_dataset_stats(input_dir: str,
                          include_videos: Optional[list] = None,
                          include_segments: Optional[list] = None) -> dict:
    """
    Walk the results directory and compute comprehensive dataset statistics
    from all available metadata files. Optionally filtered to specific
    videos or segments.
    """
    videos = set()
    segments = []
    total_frames = 0
    total_extracted_frames = 0
    total_jpg_files = 0

    # SAM3
    sam3_total_objects = 0
    sam3_frames_with_detections = 0
    sam3_all_object_ids = set()
    sam3_prompts = set()

    # VGGT
    vggt_total_inference_time = 0.0
    vggt_total_time = 0.0
    vggt_segments_processed = 0

    # Tracking
    tracking_total_detections = 0
    tracking_total_tracks = 0
    track_lengths = []

    # KITTI
    kitti_total_detections = 0
    kitti_frames_with_labels = 0
    kitti_class_counts = collections.Counter()

    # Source video info
    source_videos = set()
    fps_values = set()
    extract_fps_values = set()
    resolutions = set()

    video_segment_counts = collections.Counter()

    # Per-video aggregations
    per_video = collections.defaultdict(lambda: {
        "num_segments": 0,
        "total_frames": 0,
        "kitti_bboxes": 0,
        "kitti_labeled_frames": 0,
        "classes": collections.Counter(),
        "sam3_objects": 0,
        "vggt_tracks": 0,
        "segments": [],
    })

    for root, dirs, files in os.walk(input_dir):
        if "sam3_masks" in root or "vggt_results" in root:
            continue

        if "metadata.json" not in files:
            continue

        meta = load_json_safe(os.path.join(root, "metadata.json"))
        if not meta or "frame_numbers" not in meta:
            continue

        seg_name = os.path.basename(root)
        video_dir = os.path.dirname(root)
        video_name = os.path.basename(video_dir)

        # Apply include filters
        if not segment_passes_filter(video_name, seg_name, include_videos, include_segments):
            continue

        videos.add(video_name)
        video_segment_counts[video_name] += 1

        num_frames = meta.get("num_frames", len(meta["frame_numbers"]))
        num_extracted = meta.get("num_extracted", num_frames)
        total_frames += num_frames
        total_extracted_frames += num_extracted

        jpg_count = len([f for f in files if f.endswith(".jpg")])
        total_jpg_files += jpg_count

        if meta.get("source_video"):
            source_videos.add(meta["source_video"])
        if meta.get("video_fps"):
            fps_values.add(meta["video_fps"])
        if meta.get("extract_fps"):
            extract_fps_values.add(meta["extract_fps"])
        if meta.get("output_resolution"):
            resolutions.add(meta["output_resolution"])

        seg_info = {
            "segment": f"{video_name}/{seg_name}",
            "pair_seg_label": meta.get("pair_seg_label", seg_name),
            "num_frames": num_frames,
            "first_frame": meta.get("first_original_frame"),
            "last_frame": meta.get("last_original_frame"),
        }

        # SAM3 masks
        sam3_meta = load_json_safe(os.path.join(root, "sam3_masks", "metadata.json"))
        if sam3_meta:
            n_objects = sam3_meta.get("unique_objects_tracked", 0)
            n_det_frames = sam3_meta.get("frames_with_detections", 0)
            sam3_total_objects += n_objects
            sam3_frames_with_detections += n_det_frames
            for oid in sam3_meta.get("object_ids", []):
                sam3_all_object_ids.add(f"{video_name}/{seg_name}/obj_{oid}")
            if sam3_meta.get("text_prompt"):
                sam3_prompts.add(sam3_meta["text_prompt"])
            seg_info["sam3_objects"] = n_objects
            seg_info["sam3_detection_frames"] = n_det_frames

        # VGGT results
        vggt_meta = load_json_safe(os.path.join(root, "vggt_results", "vggt_metadata.json"))
        if vggt_meta:
            vggt_segments_processed += 1
            n_tracks = vggt_meta.get("num_tracks", 0)
            timing = vggt_meta.get("timing", {})
            vggt_total_inference_time += timing.get("inference", 0)
            vggt_total_time += vggt_meta.get("total_time", 0)
            seg_info["vggt_tracks"] = n_tracks
            seg_info["vggt_inference_time"] = timing.get("inference", 0)

        # Tracking summary
        tracking = load_json_safe(os.path.join(root, "vggt_results", "tracking_summary.json"))
        if tracking:
            tracking_total_detections += tracking.get("total_detections", 0)
            tracking_total_tracks += tracking.get("total_tracks", 0)
            for tid, tdata in tracking.get("tracks", {}).items():
                track_lengths.append(tdata.get("length", 0))
            seg_info["tracking_detections"] = tracking.get("total_detections", 0)

        # KITTI labels
        kitti_dir = os.path.join(root, "vggt_results", "kitti_labels")
        dets, labeled_frames, cls_counts = count_kitti_labels(kitti_dir)
        kitti_total_detections += dets
        kitti_frames_with_labels += labeled_frames
        kitti_class_counts += cls_counts
        seg_info["kitti_detections"] = dets
        seg_info["kitti_labeled_frames"] = labeled_frames
        seg_info["kitti_classes"] = dict(cls_counts)

        # Per-video aggregation
        pv = per_video[video_name]
        pv["num_segments"] += 1
        pv["total_frames"] += num_frames
        pv["kitti_bboxes"] += dets
        pv["kitti_labeled_frames"] += labeled_frames
        pv["classes"] += cls_counts
        pv["sam3_objects"] += seg_info.get("sam3_objects", 0)
        pv["vggt_tracks"] += seg_info.get("vggt_tracks", 0)
        pv["segments"].append({
            "seg": seg_name,
            "num_frames": num_frames,
            "kitti_bboxes": dets,
            "kitti_labeled_frames": labeled_frames,
        })

        segments.append(seg_info)

    summary = {
        "dataset_overview": {
            "total_videos": len(videos),
            "total_segments": len(segments),
            "total_frames": total_frames,
            "total_extracted_frames": total_extracted_frames,
            "total_jpg_files_on_disk": total_jpg_files,
        },
        "source_info": {
            "unique_source_videos": len(source_videos),
            "video_fps_values": sorted(fps_values),
            "extract_fps_values": sorted(extract_fps_values),
            "resolutions": sorted(resolutions),
        },
        "segments_per_video": {
            "videos": dict(video_segment_counts.most_common()),
            "min_segments": min(video_segment_counts.values()) if video_segment_counts else 0,
            "max_segments": max(video_segment_counts.values()) if video_segment_counts else 0,
            "avg_segments": round(sum(video_segment_counts.values()) / max(len(video_segment_counts), 1), 1),
        },
        "frames_per_segment": {
            "min": min(s["num_frames"] for s in segments) if segments else 0,
            "max": max(s["num_frames"] for s in segments) if segments else 0,
            "avg": round(sum(s["num_frames"] for s in segments) / max(len(segments), 1), 1),
        },
        "sam3_masks": {
            "text_prompts_used": sorted(sam3_prompts),
            "total_unique_object_instances": len(sam3_all_object_ids),
            "total_frames_with_detections": sam3_frames_with_detections,
            "detection_rate_percent": round(sam3_frames_with_detections / max(total_frames, 1) * 100, 1),
        },
        "vggt_inference": {
            "segments_processed": vggt_segments_processed,
            "total_inference_time_seconds": round(vggt_total_inference_time, 1),
            "total_processing_time_seconds": round(vggt_total_time, 1),
            "avg_inference_per_segment_seconds": round(vggt_total_inference_time / max(vggt_segments_processed, 1), 1),
        },
        "tracking": {
            "total_unique_tracks": tracking_total_tracks,
            "total_3d_detections": tracking_total_detections,
            "avg_track_length_frames": round(sum(track_lengths) / max(len(track_lengths), 1), 1) if track_lengths else 0,
            "min_track_length": min(track_lengths) if track_lengths else 0,
            "max_track_length": max(track_lengths) if track_lengths else 0,
            "median_track_length": sorted(track_lengths)[len(track_lengths) // 2] if track_lengths else 0,
        },
        "kitti_labels": {
            "total_bounding_boxes": kitti_total_detections,
            "frames_with_labels": kitti_frames_with_labels,
            "labels_per_frame": round(kitti_total_detections / max(kitti_frames_with_labels, 1), 2),
            "classes": dict(kitti_class_counts.most_common()),
            "num_classes": len(kitti_class_counts),
        },
        "per_video": {
            vname: {
                "num_segments": pv["num_segments"],
                "total_frames": pv["total_frames"],
                "kitti_bboxes": pv["kitti_bboxes"],
                "kitti_labeled_frames": pv["kitti_labeled_frames"],
                "bboxes_per_labeled_frame": round(pv["kitti_bboxes"] / max(pv["kitti_labeled_frames"], 1), 2),
                "classes": dict(pv["classes"].most_common()),
                "sam3_objects": pv["sam3_objects"],
                "vggt_tracks": pv["vggt_tracks"],
                "segments": pv["segments"],
            }
            for vname, pv in sorted(per_video.items())
        },
        "per_segment": segments,
    }

    return summary


def print_stats(summary: dict):
    """Print dataset statistics in a readable format."""
    ov = summary["dataset_overview"]
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Videos:            {ov['total_videos']}")
    print(f"  Segments:          {ov['total_segments']}")
    print(f"  Total frames:      {ov['total_frames']:,}")
    print(f"  JPG files on disk: {ov['total_jpg_files_on_disk']:,}")

    src = summary["source_info"]
    print(f"\n  Source videos:     {src['unique_source_videos']}")
    print(f"  Video FPS:         {src['video_fps_values']}")
    print(f"  Extract FPS:       {src['extract_fps_values']}")
    print(f"  Resolutions:       {src['resolutions']}")

    fpv = summary["segments_per_video"]
    print(f"\n  Segments/video:    min={fpv['min_segments']}, max={fpv['max_segments']}, avg={fpv['avg_segments']}")

    fps = summary["frames_per_segment"]
    print(f"  Frames/segment:    min={fps['min']}, max={fps['max']}, avg={fps['avg']}")

    sam = summary["sam3_masks"]
    print(f"\n  SAM3 Masks")
    print(f"  {'─'*40}")
    print(f"  Prompts:           {sam['text_prompts_used']}")
    print(f"  Object instances:  {sam['total_unique_object_instances']}")
    print(f"  Detection frames:  {sam['total_frames_with_detections']:,} ({sam['detection_rate_percent']}%)")

    vg = summary["vggt_inference"]
    print(f"\n  VGGT Inference")
    print(f"  {'─'*40}")
    print(f"  Segments done:     {vg['segments_processed']}")
    print(f"  Total inference:   {vg['total_inference_time_seconds']:.1f}s")
    print(f"  Avg per segment:   {vg['avg_inference_per_segment_seconds']:.1f}s")

    tr = summary["tracking"]
    print(f"\n  3D Tracking")
    print(f"  {'─'*40}")
    print(f"  Unique tracks:     {tr['total_unique_tracks']}")
    print(f"  Total 3D dets:     {tr['total_3d_detections']:,}")
    print(f"  Track length:      min={tr['min_track_length']}, max={tr['max_track_length']}, "
          f"avg={tr['avg_track_length_frames']}, median={tr['median_track_length']}")

    ki = summary["kitti_labels"]
    print(f"\n  KITTI Labels")
    print(f"  {'─'*40}")
    print(f"  Bounding boxes:    {ki['total_bounding_boxes']:,}")
    print(f"  Labeled frames:    {ki['frames_with_labels']:,}")
    print(f"  Labels/frame:      {ki['labels_per_frame']}")
    print(f"  Classes ({ki['num_classes']}):       {ki['classes']}")

    pv = summary.get("per_video", {})
    if pv:
        print(f"\n  Per-Video Breakdown")
        print(f"  {'─'*60}")
        for vname, v in pv.items():
            cls_str = ", ".join(f"{k}={n}" for k, n in v["classes"].items()) or "none"
            print(f"  {vname}")
            print(f"    segments={v['num_segments']}, frames={v['total_frames']:,}, "
                  f"bboxes={v['kitti_bboxes']:,}, labeled_frames={v['kitti_labeled_frames']:,}, "
                  f"bboxes/frame={v['bboxes_per_labeled_frame']}")
            print(f"    classes: {cls_str}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    include_videos = args.include_video
    include_segments = args.include_segment
    if include_videos or include_segments:
        print(f"Filters: videos={include_videos}, segments={include_segments}")

    # --- Stats ---
    if args.stats or args.stats_only:
        print(f"Computing dataset statistics for {input_dir}...")
        summary = compute_dataset_stats(
            input_dir,
            include_videos=include_videos,
            include_segments=include_segments,
        )
        print_stats(summary)

        if args.output_zip:
            summary_path = os.path.splitext(args.output_zip)[0] + "_summary.json"
        else:
            summary_path = os.path.join(input_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved dataset summary to {summary_path}")

        if args.stats_only:
            return

    # --- Zip ---
    if not args.output_zip:
        print("No --output-zip specified and not --stats-only. Nothing to do.")
        sys.exit(1)

    # Build the set of passing dirs up front (depth-agnostic)
    included_seg_dirs, included_video_dirs = build_included_paths(
        input_dir, include_videos, include_segments
    )
    if included_seg_dirs is not None:
        print(f"Matched {len(included_seg_dirs)} segment(s) in "
              f"{len(included_video_dirs)} video dir(s) for zipping")

    total_files = 0
    excluded_files = 0
    filtered_files = 0
    total_size = 0

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if should_exclude(f, args.exclude, args.include_predictions):
                excluded_files += 1
                continue
            if not file_passes_filter(fpath, included_seg_dirs, included_video_dirs):
                filtered_files += 1
                continue
            total_files += 1
            total_size += os.path.getsize(fpath)

    print(f"\nInput: {input_dir}")
    print(f"Files to zip: {total_files} ({total_size / (1024*1024):.1f} MB)")
    if excluded_files:
        print(f"Files excluded (patterns):  {excluded_files}")
    if filtered_files:
        print(f"Files excluded (filters):   {filtered_files}")

    if total_files == 0:
        print("No files match the filters. Exiting.")
        sys.exit(1)

    print(f"\nCreating {args.output_zip}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_zip)), exist_ok=True)

    zipped = 0
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                fpath = os.path.join(root, f)
                if should_exclude(f, args.exclude, args.include_predictions):
                    continue
                if not file_passes_filter(fpath, included_seg_dirs, included_video_dirs):
                    continue
                arcname = os.path.relpath(fpath, input_dir)
                zf.write(fpath, arcname)
                zipped += 1
                if zipped % 500 == 0:
                    print(f"  {zipped}/{total_files} files...")

    zip_size = os.path.getsize(args.output_zip)
    print(f"Done. {args.output_zip} ({zip_size / (1024*1024):.1f} MB, {zipped} files)")

    if args.delete_after:
        print(f"\nDeleting {input_dir}...")
        shutil.rmtree(input_dir)
        print("Deleted.")


if __name__ == "__main__":
    main()
