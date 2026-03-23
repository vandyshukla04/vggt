"""
Batch inference for video segments with SRT frame offset support.

Processes segment videos extracted by sam3/scripts/extract_segments.py and
correctly maps frame indices to the original video's SRT telemetry data.

The core problem: a segment video starts at frame 0, but the DJI SRT has
FrameCnt values relative to the original video (e.g., 6240). This script
applies a frame offset so VGGT's telemetry matching works correctly.

Usage:
    # From segment metadata (easiest)
    python batch_inference_segment.py \
        --segment_metadata /path/to/seg1_metadata.json \
        --sam3_masks /path/to/sam3_output/ \
        --output_dir ./outputs/

    # With explicit overrides
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
    # From segment metadata (reads video path, SRT path, frame offset)
    python batch_inference_segment.py \\
        --segment_metadata /path/to/seg1_metadata.json \\
        --sam3_masks /path/to/sam3_output/ \\
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
                             "Auto-sets video, dji_log, and frame_offset.")

    # Overrides / manual mode
    parser.add_argument("--video", type=str, default=None,
                        help="Path to segment video (overrides metadata)")
    parser.add_argument("--dji_log", type=str, default=None,
                        help="Path to DJI SRT file (overrides metadata)")
    parser.add_argument("--frame_offset", type=int, default=None,
                        help="Frame offset to add for SRT matching "
                             "(overrides metadata start_frame)")

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

    # Resolve segment metadata
    if args.segment_metadata:
        if not os.path.isfile(args.segment_metadata):
            parser.error(f"Segment metadata not found: {args.segment_metadata}")

        with open(args.segment_metadata) as f:
            seg_meta = json.load(f)

        # Auto-set from metadata (CLI flags take precedence)
        if args.video is None:
            # Segment video is next to the metadata file
            meta_dir = os.path.dirname(os.path.abspath(args.segment_metadata))
            args.video = os.path.join(meta_dir, seg_meta["output_file"])

        if args.dji_log is None:
            args.dji_log = seg_meta.get("source_srt")

        if args.frame_offset is None:
            args.frame_offset = seg_meta.get("start_frame", 0)

        # Store full metadata for later
        args._seg_meta = seg_meta
    else:
        args._seg_meta = None

    if args.frame_offset is None:
        args.frame_offset = 0

    if args.video is None:
        parser.error("Either --segment_metadata or --video must be provided")

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
        print(f"Source video: {args._seg_meta.get('source_video', 'unknown')}")
        print(f"Time range: {args._seg_meta.get('start_time')} -> {args._seg_meta.get('end_time')}")
    print(f"Frame offset: {args.frame_offset}")
    if args.dji_log:
        print(f"DJI SRT: {args.dji_log}")

    # Get video info
    video_info = get_video_info(args.video)
    print(f"Segment info: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.2f} fps, {video_info['total_frames']} frames")

    # Determine extraction FPS
    if args.extract_fps:
        target_fps = args.extract_fps
    elif args.sam3_fps:
        target_fps = args.sam3_fps
    elif args.sam3_masks:
        metadata_path = os.path.join(args.sam3_masks, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                sam3_metadata = json.load(f)
            target_fps = sam3_metadata.get("fps", sam3_metadata.get("effective_fps", video_info["fps"]))
            print(f"Using FPS from SAM3 metadata: {target_fps}")
        else:
            target_fps = video_info["fps"]
    else:
        target_fps = video_info["fps"]

    print(f"Extraction FPS: {target_fps}")

    # Extract frames and apply offset
    with VideoFrameContext(
        args.video,
        target_fps=target_fps,
        max_frames=args.num_images,
        keep_frames=args.keep_frames,
    ) as ctx:
        image_paths = ctx.frame_paths
        frame_indices = ctx.frame_indices

        print(f"Extracted {len(image_paths)} frames")
        print(f"Segment frame indices: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")

        # Apply frame offset for SRT matching
        if args.frame_offset:
            frame_indices = [idx + args.frame_offset for idx in frame_indices]
            print(f"Applied frame offset {args.frame_offset}: "
                  f"SRT-aligned indices {frame_indices[0]}-{frame_indices[-1]}")

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

            if format_info["format"] == "multi_class":
                if class_names is None:
                    print(f"Auto-loading all {len(format_info['classes'])} class(es)")
                masks_data = load_sam3_masks_multi_class(
                    sam3_output_dir=args.sam3_masks,
                    extracted_frame_indices=frame_indices,
                    video_fps=video_info["fps"],
                    sam3_fps=args.sam3_fps or target_fps,
                    class_names=class_names,
                    direct_frame_match=True,
                )
            else:
                masks_data = load_sam3_masks(
                    sam3_output_dir=args.sam3_masks,
                    extracted_frame_indices=frame_indices,
                    video_fps=video_info["fps"],
                    sam3_fps=args.sam3_fps or target_fps,
                    class_name=args.object_class,
                    direct_frame_match=True,
                    auto_detect_format=False,
                )

        # Run tracking
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
                frame_indices=frame_indices,
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
        "frame_indices": frame_indices,
        "video_fps": video_info["fps"],
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
