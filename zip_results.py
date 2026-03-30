#!/usr/bin/env python3
"""
Zip a results directory into a zip file, optionally deleting the source after.

Excludes large intermediate files (predictions.pt) by default.

Usage:
    # Zip results
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip

    # Zip and delete the source folder
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip --delete-after

    # Include predictions.pt (large, usually not needed)
    python zip_results.py --input-dir /path/to/results/ --output-zip /path/to/output.zip --include-predictions
"""

import argparse
import os
import shutil
import sys
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zip a results directory, optionally deleting the source"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the directory to zip")
    parser.add_argument("--output-zip", type=str, required=True,
                        help="Path for the output zip file")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete the input directory after successful zipping")
    parser.add_argument("--include-predictions", action="store_true",
                        help="Include predictions.pt files (large, excluded by default)")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="Additional file patterns to exclude (e.g., '*.tmp')")
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


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Count files
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

    print(f"Input: {input_dir}")
    print(f"Files to zip: {total_files} ({total_size / (1024*1024):.1f} MB)")
    if excluded_files:
        print(f"Files excluded: {excluded_files}")

    # Create zip
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
