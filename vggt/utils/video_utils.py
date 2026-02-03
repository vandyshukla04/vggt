# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for video frame extraction with automatic cleanup.
"""

import os
import cv2
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using cv2.VideoCapture.

    Args:
        video_path: Path to input video file

    Returns:
        Dict with keys: fps, total_frames, duration_seconds, width, height

    Raises:
        ValueError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration_seconds,
            'width': width,
            'height': height
        }
    finally:
        cap.release()


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    frame_pattern: str = "frame_{:06d}.png"
) -> Tuple[List[str], List[int]]:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Frame rate for extraction (None = use video's native FPS)
        max_frames: Maximum number of frames to extract (None = all)
        start_frame: First video frame number to extract (0-indexed, inclusive)
        end_frame: Last video frame number to extract (0-indexed, inclusive)
        frame_pattern: Naming pattern for output frames

    Returns:
        Tuple of (list of frame paths, list of original video frame indices)
        Note: video_frame_indices contains the ACTUAL video frame numbers,
        which is important for DJI telemetry matching.

    Raises:
        ValueError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply frame range limits
        effective_start = start_frame if start_frame is not None else 0
        effective_end = end_frame if end_frame is not None else total_frames - 1
        effective_end = min(effective_end, total_frames - 1)

        if effective_start > effective_end:
            raise ValueError(f"start_frame ({effective_start}) > end_frame ({effective_end})")

        print(f"Frame range: {effective_start} to {effective_end} "
              f"(total video frames: {total_frames})")

        # Seek to start frame if needed
        if effective_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start)

        # Calculate time offset for the start frame
        start_time = effective_start / video_fps if video_fps > 0 else effective_start

        frame_paths = []
        video_frame_indices = []
        extracted_count = 0
        next_extract_time = start_time

        for frame_idx in range(effective_start, effective_end + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if this frame should be extracted based on timing
            current_time = frame_idx / video_fps if video_fps > 0 else frame_idx

            if current_time >= next_extract_time:
                # Save frame - use the actual video frame number in the filename
                # This ensures compatibility with SAM3 mask naming
                frame_filename = frame_pattern.format(frame_idx)
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                frame_paths.append(frame_path)
                video_frame_indices.append(frame_idx)
                extracted_count += 1

                # Calculate next extraction time
                if target_fps is not None and target_fps > 0:
                    next_extract_time = start_time + (extracted_count / target_fps)
                else:
                    next_extract_time = current_time + (1.0 / video_fps if video_fps > 0 else 1.0)

                if max_frames is not None and extracted_count >= max_frames:
                    break

        return frame_paths, video_frame_indices

    finally:
        cap.release()


def create_temp_frame_directory(prefix: str = 'vggt_frames_') -> str:
    """
    Create a unique temporary directory for frame extraction.

    Args:
        prefix: Prefix for the temporary directory name

    Returns:
        Path to temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_frame_directory(frame_dir: str) -> bool:
    """
    Remove temporary frame directory and all contents.

    Args:
        frame_dir: Path to directory to remove

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        if frame_dir and os.path.exists(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)
            return True
    except Exception as e:
        print(f"Warning: Failed to cleanup frame directory {frame_dir}: {e}")
    return False


class VideoFrameContext:
    """
    Context manager for temporary frame extraction with automatic cleanup.

    Usage:
        with VideoFrameContext(video_path, target_fps=5, max_frames=20) as ctx:
            frame_paths = ctx.frame_paths
            frame_indices = ctx.frame_indices  # Original video frame numbers
            video_info = ctx.video_info
            # ... process frames ...
        # frames automatically cleaned up

    Note:
        frame_indices contains the ACTUAL video frame numbers (0-indexed),
        which are used for:
        - DJI telemetry matching (SRT FrameCnt)
        - SAM3 mask file matching (frame_{idx:06d}.png)
    """

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        frame_pattern: str = "frame_{:06d}.png",
        keep_frames: bool = False
    ):
        """
        Initialize the context manager.

        Args:
            video_path: Path to input video file
            target_fps: Frame rate for extraction (None = use video's native FPS)
            max_frames: Maximum number of frames to extract (None = all)
            start_frame: First video frame number to extract (0-indexed, inclusive)
            end_frame: Last video frame number to extract (0-indexed, inclusive)
            frame_pattern: Naming pattern for output frames
            keep_frames: If True, don't delete frames on exit (useful for debugging)
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_pattern = frame_pattern
        self.keep_frames = keep_frames

        self.temp_dir: Optional[str] = None
        self.frame_paths: List[str] = []
        self.frame_indices: List[int] = []
        self.video_info: Dict[str, Any] = {}

    def __enter__(self) -> 'VideoFrameContext':
        """Extract frames and return self."""
        # Get video info first
        self.video_info = get_video_info(self.video_path)

        # Create temp directory
        self.temp_dir = create_temp_frame_directory()

        # Extract frames
        self.frame_paths, self.frame_indices = extract_frames_from_video(
            self.video_path,
            self.temp_dir,
            self.target_fps,
            self.max_frames,
            self.start_frame,
            self.end_frame,
            self.frame_pattern
        )

        print(f"Extracted {len(self.frame_paths)} frames to {self.temp_dir}")
        if self.frame_indices:
            print(f"Frame indices: {self.frame_indices[0]} to {self.frame_indices[-1]}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Clean up temporary directory."""
        if not self.keep_frames:
            if cleanup_frame_directory(self.temp_dir):
                print(f"Cleaned up temporary frames from {self.temp_dir}")
        else:
            print(f"Keeping frames at {self.temp_dir}")

        # Don't suppress exceptions
        return False
