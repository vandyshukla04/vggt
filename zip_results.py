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


def compute_dataset_stats(input_dir: str) -> dict:
    """
    Walk the results directory and compute comprehensive dataset statistics
    from all available metadata files.
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

    # --- Stats ---
    if args.stats or args.stats_only:
        print(f"Computing dataset statistics for {input_dir}...")
        summary = compute_dataset_stats(input_dir)
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

    total_files = 0
    excluded_files = 0
    total_size = 0

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if should_exclude(f, args.exclude, args.include_predictions):
                excluded_files += 1
            else:
                total_files += 1
                total_size += os.path.getsize(os.path.join(root, f))

    print(f"\nInput: {input_dir}")
    print(f"Files to zip: {total_files} ({total_size / (1024*1024):.1f} MB)")
    if excluded_files:
        print(f"Files excluded: {excluded_files}")

    print(f"\nCreating {args.output_zip}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_zip)), exist_ok=True)

    zipped = 0
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if should_exclude(f, args.exclude, args.include_predictions):
                    continue
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, input_dir)
                zf.write(file_path, arcname)
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
