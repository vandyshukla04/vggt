#!/usr/bin/env python3
"""Fix the evaluation clips for one WildDepth release. Run once; every method uses the result.

  python make_clips.py --root /storage3/.../WildDepth/longleat_v2_part2 --out clips/longleat_v2_part2.json
  python make_clips.py --root <dir> --listing listing.txt --out ...   # from a `tar tf` listing, before extraction

A clip is `clip_len` consecutive synchronised frames (default 96, about 10 s at 10 FPS). Each
snippet contributes at most `max_clips` clips, evenly spaced, so one long snippet cannot dominate.
Snippets with min_len..clip_len-1 frames are used whole; shorter ones are dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from wdcommon import detect_globs, find_sequences, pair_relative, read_metadata


def clip_starts(n: int, clip_len: int, max_clips: int) -> list[int]:
    if n <= clip_len:
        return [0]
    k = min(max_clips, n // clip_len)
    if k == 1:
        return [(n - clip_len) // 2]
    return [round(i * (n - clip_len) / (k - 1)) for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--release", default="", help="name; default = basename of --root")
    ap.add_argument("--listing", default="", help="file with one archive member per line (tar tf); "
                    "then only metadata.txt and one rgb frame per snippet need to exist under --root")
    ap.add_argument("--rgb_glob", default="auto", help='e.g. "*/*/rgb/*.jpg"; auto = detect the layout')
    ap.add_argument("--depth_glob", default="auto")
    ap.add_argument("--depth_scale", type=float, default=0.001, help="raw depth value -> metres")
    ap.add_argument("--clip_len", type=int, default=96)
    ap.add_argument("--max_clips", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=48)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    release = a.release or root.name
    if a.listing:
        rel = [ln.strip() for ln in Path(a.listing).read_text().splitlines() if ln.strip() and not ln.endswith("/")]
        rg, dg = detect_globs(rel, a.rgb_glob, a.depth_glob)
        seqs = {seq: [(root / r, root / d) for r, d in pairs]
                for seq, pairs in pair_relative(rel, rg, dg).items()}
    else:
        seqs = find_sequences(root, a.rgb_glob, a.depth_glob)
    if not seqs:
        raise SystemExit(f"no rgb/depth pairs under {root} with {a.rgb_glob} / {a.depth_glob}")

    clips, dropped = [], []
    for seq, pairs in seqs.items():
        n = len(pairs)
        if n < a.min_len:
            dropped.append((seq, n))
            continue
        meta = read_metadata(root / seq)
        first = next((r for r, _ in pairs if r.is_file()), None)
        if first is None:
            raise SystemExit(f"{seq}: no rgb frame present under {root} to read the image size from")
        with Image.open(first) as im:
            W, H = im.size
        for ci, s0 in enumerate(clip_starts(n, a.clip_len, a.max_clips)):
            sel = pairs[s0: s0 + a.clip_len]
            clips.append({
                "id": f"{seq}/{ci:02d}", "seq": seq, "start_index": s0, "n_frames": len(sel),
                "image_size": [W, H], "meta": meta,
                "frames": [{"stem": r.stem, "rgb": r.relative_to(root).as_posix(),
                            "depth": d.relative_to(root).as_posix()} for r, d in sel],
            })

    out = {"release": release, "root": str(root), "depth_scale": a.depth_scale,
           "clip_len": a.clip_len, "max_clips": a.max_clips, "min_len": a.min_len,
           "n_sequences": len(seqs), "n_clips": len(clips), "clips": clips}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

    n_fr = sum(c["n_frames"] for c in clips)
    print(f"{release}: {len(seqs)} snippets -> {len(clips)} clips, {n_fr} frames")
    for seq, pairs in seqs.items():
        cs = [c for c in clips if c["seq"] == seq]
        m = read_metadata(root / seq)
        print(f"  {seq:<36s} {len(pairs):6d} frames  {len(cs)} clip(s)  "
              f"vehicle={m.get('vehicle', '?')}  {m.get('species', '')}")
    if dropped:
        print(f"  dropped (< {a.min_len} frames): {dropped}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
