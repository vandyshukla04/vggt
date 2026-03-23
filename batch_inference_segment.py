"""
Batch inference for video segments with SRT frame offset support.

Processes segment videos extracted by sam3/scripts/extract_segments.py and
correctly maps frame indices to the original video's SRT telemetry data.

The core problem: a segment video starts at frame 0, but the DJI SRT has
FrameCnt values relative to the original video (e.g., 6240). This script
applies a frame offset so VGGT's telemetry matching works correctly.

Usage:
    # Frame offset from segment metadata (reads start_frame only, not paths)
    python batch_inference_segment.py \
        --video /cluster/data/seg1.mp4 \
        --dji_log /cluster/data/original.SRT \
        --segment_metadata /cluster/data/seg1_metadata.json \
        --sam3_masks /cluster/data/sam3_output/ \
        --output_dir ./outputs/

    # Explicit frame offset (no metadata file needed)
    python batch_inference_segment.py \
        --video seg1.mp4 \
        --dji_log /path/to/original.SRT \
        --frame_offset 6240 \
        --sam3_masks /path/to/sam3_output/ \
        --output_dir ./outputs/
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.utils.video_utils import VideoFrameContext, get_video_info
from vggt.utils.sam3_mask_loader import load_sam3_masks

from batch_inference_video import run_inference, save_predictions, run_tracking


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for video segments with SRT frame offset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Frame offset from segment metadata (paths always explicit)
    python batch_inference_segment.py \\
        --video /data/seg1.mp4 \\
        --dji_log /data/original.SRT \\
        --segment_metadata /data/seg1_metadata.json \\
        --sam3_masks /data/sam3_output/ \\
        --output_dir ./outputs/

    # Explicit offset (no metadata file needed)
    python batch_inference_segment.py \\
        --video seg1.mp4 \\
        --dji_log /path/to/original.SRT \\
        --frame_offset 6240 \\
        --output_dir ./outputs/
        """
    )

    # Segment metadata (primary interface)
    parser.add_argument("--segment_metadata", type=str, default=None,
                        help="Path to segment metadata JSON from extract_segments.py. "
                             "Only reads frame positioning (start_frame), not paths.")

    # Overrides / manual mode
    parser.add_argument("--video", type=str, default=None,
                        help="Path to segment video (required)")
    parser.add_argument("--dji_log", type=str, default=None,
                        help="Path to DJI SRT file (overrides metadata)")
    parser.add_argument("--frame_offset", type=int, default=None,
                        help="Frame offset to add for SRT matching "
                             "(overrides metadata start_frame)")
    parser.add_argument("--original_fps", type=float, default=None,
                        help="FPS of the original source video "
                             "(overrides metadata video_fps). Used for SRT/mask matching.")

    # Frame extraction
    parser.add_argument("--extract_fps", type=float, default=None,
                        help="Frame rate for extraction (default: match SAM3 or video native)")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Max number of frames to process")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Don't delete extracted frames after processing")

    # SAM3 masks
    parser.add_argument("--sam3_masks", type=str, default=None,
                        help="Path to SAM3 output directory")
    parser.add_argument("--sam3_fps", type=float, default=None,
                        help="FPS used when generating SAM3 masks")
    parser.add_argument("--sam3_class", type=str, default=None,
                        help="Class name(s) to load (comma-separated)")
    parser.add_argument("--object_class", type=str, default="object",
                        help="Default class name for SAM3 objects")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # Tracking parameters
    parser.add_argument("--use_gps_refinement", action="store_true",
                        help="Enable GPS-based pose refinement")
    parser.add_argument("--conf_threshold", type=float, default=50.0,
                        help="Confidence threshold percentile for point cloud")
    parser.add_argument("--max_distance", type=float, default=8.0,
                        help="Max 3D distance for tracking")
    parser.add_argument("--max_missing_frames", type=int, default=20,
                        help="Frames before track goes dormant")
    parser.add_argument("--dormant_timeout", type=int, default=100,
                        help="Frames before dormant track removed")
    parser.add_argument("--use_point_map", action="store_true",
                        help="Use point map instead of depth-based points")

    args = parser.parse_args()

    # Resolve segment metadata (only frame positioning info, not paths)
    if args.segment_metadata:
        if not os.path.isfile(args.segment_metadata):
            parser.error(f"Segment metadata not found: {args.segment_metadata}")

        with open(args.segment_metadata) as f:
            seg_meta = json.load(f)

        # Only read frame positioning — paths must be provided via CLI
        if args.frame_offset is None:
            args.frame_offset = seg_meta.get("start_frame", 0)

        if args.original_fps is None:
            args.original_fps = seg_meta.get("video_fps")

        args._seg_meta = seg_meta
    else:
        args._seg_meta = None

    if args.frame_offset is None:
        args.frame_offset = 0

    if args.video is None:
        parser.error("--video is required (paths are never read from metadata)")

    return args


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t0 = time.time()

    # Print segment info
    print(f"\n=== Segment Mode ===")
    print(f"Video: {args.video}")
    if args._seg_meta:
        print(f"Time range: {args._seg_meta.get('start_time')} -> {args._seg_meta.get('end_time')}")
        print(f"Frame range: {args._seg_meta.get('start_frame')} -> {args._seg_meta.get('end_frame')}")
    print(f"Frame offset: {args.frame_offset}")
    if args.dji_log:
        print(f"DJI SRT: {args.dji_log}")

    # Get video info
    video_info = get_video_info(args.video)
    segment_fps = video_info["fps"]
    print(f"Segment info: {video_info['width']}x{video_info['height']}, "
          f"{segment_fps:.2f} fps, {video_info['total_frames']} frames")

    # Original video FPS (for SRT/mask matching) — may differ from segment FPS
    original_fps = args.original_fps or segment_fps
    if original_fps != segment_fps:
        print(f"Original video FPS: {original_fps:.2f} (segment FPS: {segment_fps:.2f})")
    else:
        print(f"Original video FPS: {original_fps:.2f}")

    # Determine extraction FPS
    if args.extract_fps:
        target_fps = args.extract_fps
    elif args.sam3_fps:
        target_fps = args.sam3_fps
    elif args.sam3_masks:
        # Search for metadata.json at root or one level down (e.g., seg1/zebra/metadata.json)
        sam3_metadata = None
        for candidate in [
            os.path.join(args.sam3_masks, "metadata.json"),
            *[os.path.join(args.sam3_masks, d, "metadata.json")
              for d in os.listdir(args.sam3_masks)
              if os.path.isdir(os.path.join(args.sam3_masks, d))],
        ]:
            if os.path.exists(candidate):
                with open(candidate, "r") as f:
                    sam3_metadata = json.load(f)
                print(f"Found SAM3 metadata: {candidate}")
                break

        if sam3_metadata:
            # Prefer effective_fps (accounts for frame stride) over raw fps
            target_fps = sam3_metadata.get("effective_fps", sam3_metadata.get("fps", segment_fps))
            print(f"Using FPS from SAM3 metadata: {target_fps}"
                  f" (stride: {sam3_metadata.get('frame_stride', 'N/A')})")
        else:
            print(f"WARNING: No SAM3 metadata.json found in {args.sam3_masks}, using segment FPS")
            target_fps = segment_fps
    else:
        target_fps = segment_fps

    print(f"Extraction FPS: {target_fps}")

    # Extract frames — maintain two index sets:
    #   segment_frame_indices: relative to segment video (for SAM3 mask matching)
    #   srt_frame_indices:     relative to original video (for DJI SRT telemetry)
    with VideoFrameContext(
        args.video,
        target_fps=target_fps,
        max_frames=args.num_images,
        keep_frames=args.keep_frames,
    ) as ctx:
        image_paths = ctx.frame_paths
        segment_frame_indices = ctx.frame_indices

        print(f"Extracted {len(image_paths)} frames")
        print(f"Segment frame indices: {segment_frame_indices[:5]}{'...' if len(segment_frame_indices) > 5 else ''}")

        # Compute SRT-aligned indices by applying frame offset
        if args.frame_offset:
            srt_frame_indices = [idx + args.frame_offset for idx in segment_frame_indices]
            print(f"SRT frame indices (offset {args.frame_offset}): "
                  f"{srt_frame_indices[0]}-{srt_frame_indices[-1]}")
        else:
            srt_frame_indices = segment_frame_indices

        # Run inference
        predictions, timing = run_inference(image_paths, device)

        # Save predictions
        save_predictions(
            args.output_dir,
            predictions,
            image_paths,
            conf_threshold=args.conf_threshold,
            use_point_map=args.use_point_map,
        )

        # Load SAM3 masks if provided
        masks_data = None
        if args.sam3_masks:
            print(f"\n=== Loading SAM3 Masks ===")

            class_names = None
            if args.sam3_class:
                class_names = [c.strip() for c in args.sam3_class.split(",")]

            from vggt.utils.sam3_mask_loader import detect_sam3_format, load_sam3_masks_multi_class

            format_info = detect_sam3_format(args.sam3_masks)

            # Use segment-relative indices for mask matching (masks are named
            # by segment frame numbers, e.g., frame_000000, frame_000003, ...)
            if format_info["format"] == "multi_class":
                if class_names is None:
                    print(f"Auto-loading all {len(format_info['classes'])} class(es)")
                masks_data = load_sam3_masks_multi_class(
                    sam3_output_dir=args.sam3_masks,
                    extracted_frame_indices=segment_frame_indices,
                    video_fps=segment_fps,
                    sam3_fps=args.sam3_fps or target_fps,
                    class_names=class_names,
                    direct_frame_match=True,
                )
            else:
                masks_data = load_sam3_masks(
                    sam3_output_dir=args.sam3_masks,
                    extracted_frame_indices=segment_frame_indices,
                    video_fps=segment_fps,
                    sam3_fps=args.sam3_fps or target_fps,
                    class_name=args.object_class,
                    direct_frame_match=True,
                    auto_detect_format=False,
                )

        # Run tracking — use SRT-aligned indices for DJI telemetry matching
        if masks_data:
            run_tracking(
                args.output_dir,
                predictions,
                image_paths,
                masks_data,
                dji_log=args.dji_log,
                use_gps_refinement=args.use_gps_refinement,
                max_distance=args.max_distance,
                max_missing_frames=args.max_missing_frames,
                dormant_timeout=args.dormant_timeout,
                use_point_map=args.use_point_map,
                frame_indices=srt_frame_indices,
            )

    total_time = time.time() - t0
    timing["total"] = round(total_time, 2)

    # Save metadata
    metadata = {
        "mode": "segment",
        "segment_video": os.path.abspath(args.video),
        "frame_offset": args.frame_offset,
        "output_dir": os.path.abspath(args.output_dir),
        "num_frames": len(image_paths),
        "image_files": [os.path.basename(f) for f in image_paths],
        "segment_frame_indices": segment_frame_indices,
        "srt_frame_indices": srt_frame_indices,
        "segment_fps": segment_fps,
        "original_fps": original_fps,
        "extraction_fps": target_fps,
        "settings": {
            "num_images": args.num_images,
            "conf_threshold": args.conf_threshold,
            "use_point_map": args.use_point_map,
            "sam3_masks": args.sam3_masks,
            "object_class": args.object_class,
            "dji_log": args.dji_log,
            "max_distance": args.max_distance,
            "max_missing_frames": args.max_missing_frames,
            "dormant_timeout": args.dormant_timeout,
        },
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": timing,
    }

    # Include segment metadata if available
    if args._seg_meta:
        metadata["segment_metadata"] = {
            "source_video": args._seg_meta.get("source_video"),
            "source_srt": args._seg_meta.get("source_srt"),
            "start_time": args._seg_meta.get("start_time"),
            "end_time": args._seg_meta.get("end_time"),
            "start_frame": args._seg_meta.get("start_frame"),
            "end_frame": args._seg_meta.get("end_frame"),
        }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    print(f"\n=== Total Processing Time: {total_time:.2f}s ===")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("  - predictions.pt")
    print("  - point_cloud.ply")
    print("  - cameras.json")
    print("  - depth_maps.npz")
    if masks_data:
        print("  - tracking_summary.json")
        print("  - kitti_labels/")
        print("  - annotated_2d/")
        print("  - track_trajectories_*.png")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
