#!/usr/bin/env python3
"""Evaluate VGGT depth against WildDepth LiDAR depth.

Two subcommands:

  inspect   Walk a dataset root, report the folder layout, file types per folder, and the
            dtype / value range of a few depth files, so --rgb_glob, --depth_glob and
            --depth_scale can be set correctly.

  eval      For each sequence: run VGGT on windows of consecutive frames, sample the
            predicted depth at every valid LiDAR pixel, align scale, and report metrics.

VGGT depth is defined up to one scale per window (it is normalised to the first frame), so
metrics are reported under two alignments:

  frame    one median scale per frame          -> per-frame relative depth quality
  window   one median scale per window         -> multi-frame consistency, VGGT's strength

Metrics: AbsRel, RMSE (m), SILog, delta<1.25, delta<1.25^2, delta<1.25^3, computed over
valid LiDAR pixels and pooled per sequence and over the whole dataset.

Usage:
  python eval_vggt_wilddepth.py inspect --root /path/to/WildDepth/longleat_v2_part2
  python eval_vggt_wilddepth.py eval --root ... --rgb_glob "*/rgb/*.jpg" \
      --depth_glob "*/depth/*.png" --depth_scale 0.001 --out results/longleat_v2_p2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

IMG_EXT = {".jpg", ".jpeg", ".png"}
DEPTH_EXT = {".npy", ".npz", ".png", ".tif", ".tiff", ".exr", ".pfm"}


# ----------------------------------------------------------------------------- inspect
def cmd_inspect(a):
    root = Path(a.root)
    per_dir = defaultdict(Counter)
    for p in root.rglob("*"):
        if p.is_file():
            per_dir[p.parent.relative_to(root)][p.suffix.lower()] += 1
    print(f"{len(per_dir)} directories with files under {root}\n")
    for d in sorted(per_dir)[: a.max_dirs]:
        print(f"  {str(d):<70s} {dict(per_dir[d])}")
    if len(per_dir) > a.max_dirs:
        print(f"  ... {len(per_dir) - a.max_dirs} more")

    print("\nSample files that look like depth (by name or extension):")
    cand = [p for p in root.rglob("*") if p.is_file() and (
        "depth" in str(p).lower() or "lidar" in str(p).lower()) and p.suffix.lower() in DEPTH_EXT]
    for p in cand[:6]:
        try:
            d = read_depth(p, 1.0)
            v = d[np.isfinite(d) & (d > 0)]
            print(f"  {p.relative_to(root)}  shape={d.shape} dtype={read_raw(p).dtype} "
                  f"valid={v.size}/{d.size} ({100*v.size/d.size:.2f}%) "
                  f"raw min/median/max={v.min() if v.size else 'na'}/"
                  f"{np.median(v) if v.size else 'na'}/{v.max() if v.size else 'na'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.relative_to(root)}  could not read: {e}")
    others = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in
              {".json", ".yaml", ".yml", ".txt", ".csv"}][:10]
    if others:
        print("\nText / calibration-like files (first 10):")
        for p in others:
            print(f"  {p.relative_to(root)}  ({p.stat().st_size} B)")
    print("\nIf the depth is uint16 in millimetres, use --depth_scale 0.001. If it is float in "
          "metres, use --depth_scale 1.")


# ----------------------------------------------------------------------------- io
def read_raw(p: Path) -> np.ndarray:
    s = p.suffix.lower()
    if s == ".npy":
        return np.load(p)
    if s == ".npz":
        z = np.load(p)
        return z[z.files[0]]
    if s in {".exr", ".pfm"}:
        import cv2
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    from PIL import Image
    return np.asarray(Image.open(p))


def read_depth(p: Path, scale: float) -> np.ndarray:
    d = read_raw(p).astype(np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    return d * scale


# ----------------------------------------------------------------------------- pairing
def find_sequences(root: Path, rgb_glob: str, depth_glob: str):
    """Pair RGB and depth files by filename stem, grouped by the RGB file's sequence dir.

    The sequence dir is the first path component under root, which matches the
    one-folder-per-snippet layout of the WildDepth releases.
    """
    rgb = sorted(root.glob(rgb_glob))
    dep = {}
    for p in root.glob(depth_glob):
        seq = p.relative_to(root).parts[0]
        dep[(seq, p.stem)] = p
    seqs = defaultdict(list)
    for p in rgb:
        seq = p.relative_to(root).parts[0]
        q = dep.get((seq, p.stem))
        if q is not None:
            seqs[seq].append((p, q))
    return dict(sorted(seqs.items()))


# ----------------------------------------------------------------------------- geometry
def model_geometry(W: int, H: int, target: int = 518):
    """Reproduce load_and_preprocess_images(mode='crop'): width -> 518, height rounded to a
    multiple of 14, centre crop to 518 if taller. Returns (new_w, new_h, crop_top, out_h)."""
    new_w = target
    new_h = round(H * (new_w / W) / 14) * 14
    crop_top = (new_h - target) // 2 if new_h > target else 0
    out_h = min(new_h, target)
    return new_w, new_h, crop_top, out_h


def sample_pred(pred: np.ndarray, us: np.ndarray, vs: np.ndarray, W: int, H: int):
    """Bilinear sample of the model-resolution depth at original-resolution pixel centres.
    Returns values and a mask for pixels that fall inside the (possibly cropped) model frame."""
    new_w, new_h, crop_top, out_h = model_geometry(W, H)
    x = (us + 0.5) * new_w / W - 0.5
    y = (vs + 0.5) * new_h / H - 0.5 - crop_top
    ok = (x >= 0) & (x <= new_w - 1) & (y >= 0) & (y <= out_h - 1)
    x, y = x[ok], y[ok]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, new_w - 1), np.minimum(y0 + 1, out_h - 1)
    fx, fy = x - x0, y - y0
    v = (pred[y0, x0] * (1 - fx) * (1 - fy) + pred[y0, x1] * fx * (1 - fy)
         + pred[y1, x0] * (1 - fx) * fy + pred[y1, x1] * fx * fy)
    return v, ok


# ----------------------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    if gt.size == 0:
        return {}
    r = np.maximum(pred / gt, gt / pred)
    ld = np.log(pred) - np.log(gt)
    return {
        "n": int(gt.size),
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "silog": float(np.sqrt(max(np.mean(ld ** 2) - np.mean(ld) ** 2, 0.0))),
        "d1": float(np.mean(r < 1.25)),
        "d2": float(np.mean(r < 1.25 ** 2)),
        "d3": float(np.mean(r < 1.25 ** 3)),
    }


# ----------------------------------------------------------------------------- eval
def cmd_eval(a):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = find_sequences(root, a.rgb_glob, a.depth_glob)
    if a.sequences:
        seqs = {k: v for k, v in seqs.items() if k in set(a.sequences.split(","))}
    if not seqs:
        sys.exit("No RGB/depth pairs found. Run `inspect` and fix --rgb_glob / --depth_glob.")
    print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs.values())} paired frames")

    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()

    pooled = {"frame": ([], []), "window": ([], [])}
    per_seq = {}
    for seq, pairs in seqs.items():
        pairs = pairs[:: a.frame_stride]
        if a.max_frames:
            pairs = pairs[: a.max_frames]
        acc = {"frame": ([], []), "window": ([], [])}
        t0 = time.time()
        for w0 in range(0, len(pairs), a.window):
            win = pairs[w0: w0 + a.window]
            if len(win) < 2:
                continue
            imgs = load_and_preprocess_images([str(p) for p, _ in win], mode="crop").to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                pr = model(imgs[None])
            depth = pr["depth"][0, ..., 0].float().cpu().numpy()          # (S, h, w)
            conf = pr["depth_conf"][0].float().cpu().numpy()

            win_pred, win_gt = [], []
            for k, (rp, dp) in enumerate(win):
                gt = read_depth(dp, a.depth_scale)
                H, W = gt.shape
                ok = np.isfinite(gt) & (gt > a.min_depth)
                if a.max_depth:
                    ok &= gt < a.max_depth
                vs, us = np.nonzero(ok)
                if us.size < a.min_points:
                    continue
                p, inside = sample_pred(depth[k], us, vs, W, H)
                g = gt[vs[inside], us[inside]]
                if a.conf_pct > 0:
                    c, _ = sample_pred(conf[k], us, vs, W, H)
                    keep = c >= np.percentile(c, a.conf_pct)
                    p, g = p[keep], g[keep]
                good = p > 1e-6
                p, g = p[good], g[good]
                if g.size < a.min_points:
                    continue
                s = np.median(g) / np.median(p)
                acc["frame"][0].append(p * s)
                acc["frame"][1].append(g)
                win_pred.append(p)
                win_gt.append(g)
            if win_gt:
                P, G = np.concatenate(win_pred), np.concatenate(win_gt)
                s = np.median(G) / np.median(P)
                acc["window"][0].append(P * s)
                acc["window"][1].append(G)

        res = {}
        for mode, (P, G) in acc.items():
            if G:
                P, G = np.concatenate(P), np.concatenate(G)
                res[mode] = metrics(P, G)
                pooled[mode][0].append(P)
                pooled[mode][1].append(G)
        per_seq[seq] = res
        f = res.get("window", {})
        print(f"  {seq:<40s} frames={len(pairs):5d}  "
              f"AbsRel={f.get('abs_rel', float('nan')):.3f}  RMSE={f.get('rmse', float('nan')):.2f}  "
              f"d1={f.get('d1', float('nan')):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    overall = {m: metrics(np.concatenate(P), np.concatenate(G))
               for m, (P, G) in pooled.items() if G}
    report = {"config": vars(a), "overall": overall, "per_sequence": per_seq}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print("\nOverall (pixels pooled over all sequences)")
    print(f"{'alignment':<8s} {'n':>10s} {'AbsRel':>7s} {'RMSE':>6s} {'SILog':>6s} {'d1':>6s} {'d2':>6s} {'d3':>6s}")
    for m, r in overall.items():
        print(f"{m:<8s} {r['n']:>10d} {r['abs_rel']:7.3f} {r['rmse']:6.2f} {r['silog']:6.3f} "
              f"{r['d1']:6.3f} {r['d2']:6.3f} {r['d3']:6.3f}")
    print(f"\nwrote {out / 'metrics.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect")
    i.add_argument("--root", required=True)
    i.add_argument("--max_dirs", type=int, default=60)

    e = sub.add_parser("eval")
    e.add_argument("--root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--rgb_glob", required=True, help='e.g. "*/rgb/*.jpg", relative to --root')
    e.add_argument("--depth_glob", required=True, help='e.g. "*/depth/*.png"; paired by file stem')
    e.add_argument("--depth_scale", type=float, required=True, help="raw value -> metres")
    e.add_argument("--sequences", default="", help="comma-separated subset of sequence folders")
    e.add_argument("--window", type=int, default=24, help="frames per VGGT forward pass")
    e.add_argument("--frame_stride", type=int, default=1, help="use every k-th paired frame")
    e.add_argument("--max_frames", type=int, default=0, help="cap per sequence (0 = all)")
    e.add_argument("--min_depth", type=float, default=0.1)
    e.add_argument("--max_depth", type=float, default=0.0, help="0 = no cap")
    e.add_argument("--min_points", type=int, default=20, help="skip frames with fewer LiDAR px")
    e.add_argument("--conf_pct", type=float, default=0.0,
                   help="drop pixels below this percentile of VGGT depth confidence (0 = keep all)")
    a = ap.parse_args()
    cmd_inspect(a) if a.cmd == "inspect" else cmd_eval(a)


if __name__ == "__main__":
    main()
