#!/usr/bin/env python3
"""Delete the rgb/depth frames of an extracted release that no clip uses. Dry run unless --delete.

  python prune_to_clips.py --clips clips/longleat_v2_part2.json            # report only
  python prune_to_clips.py --clips clips/longleat_v2_part2.json --delete

Only files inside rgb/ and depthm/ folders are considered; metadata, calibration and
everything else stay. Every method and the evaluator read clip frames only, so pruning
changes no result. Changing the clip protocol later means re-fetching the release.
"""
import argparse
from pathlib import Path

from wdcommon import load_clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    C = load_clips(a.clips)
    root = C["root"]
    keep = {f[k] for c in C["clips"] for f in c["frames"] for k in ("rgb", "depth")}
    folders = {Path(f[k]).parent.name for c in C["clips"] for f in c["frames"] for k in ("rgb", "depth")}
    victims = [p for p in root.rglob("*") if p.is_file() and p.parent.name in folders
               and p.relative_to(root).as_posix() not in keep]
    size = sum(p.stat().st_size for p in victims)
    print(f"{C['release']}: keep {len(keep)} clip files; {len(victims)} unused files in {sorted(folders)} "
          f"folders, {size/2**30:.1f} GB")
    if a.delete:
        for p in victims:
            p.unlink()
        print(f"deleted {len(victims)} files, freed {size/2**30:.1f} GB")
    else:
        print("dry run: add --delete to remove them")


if __name__ == "__main__":
    main()
