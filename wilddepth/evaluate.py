#!/usr/bin/env python3
"""Score saved predictions against WildDepth LiDAR depth. CPU only.

  python evaluate.py --clips clips/*.json --methods vggt cut3r --preds_dir preds --out results/eval.json

Per clip, the prediction is sampled at every valid LiDAR pixel and scored under three alignments:
  clip     one median scale for the whole clip   (primary; the video-depth protocol)
  frame    one median scale per frame            (diagnostic)
  metric   no alignment                          (only methods that claim metric scale)
Per clip also: scale drift = std of the log per-frame scale; scale error = |log clip scale|.

Aggregation: clip -> snippet (mean of its clips) -> dataset (mean over snippets, each snippet
weighted equally, with a bootstrap CI over snippets). Pixel-pooled numbers and depth-range
bins (clip alignment) are reported alongside.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from wdcommon import Stats, load_clips, pred_path, read_depth, sample_at

ALIGN = ("clip", "frame", "metric")
BINS = ((0, 15), (15, 40), (40, 1e9))
KEYS = ("abs_rel", "rmse", "silog", "d1", "d2", "d3")
CI_KEYS = ("clip_abs_rel", "clip_d1", "clip_rmse", "frame_abs_rel", "metric_abs_rel", "drift", "scale_err")


def load_gt(clip, root, depth_scale, a):
    """Valid LiDAR pixels of every frame of a clip: [(us, vs, depth_m) or None]. Read once per clip."""
    out = []
    for f in clip["frames"]:
        gt = read_depth(root / f["depth"], depth_scale)
        ok = np.isfinite(gt) & (gt > a.min_depth)
        if a.max_depth:
            ok &= gt < a.max_depth
        vs, us = np.nonzero(ok)
        out.append((us, vs, gt[vs, us]) if us.size >= a.min_points else None)
    return out


def score_clip(clip, gt_frames, pred, a):
    depth = pred["depth"].astype(np.float32)
    geom = pred["geom"]
    assert list(pred["frames"]) == [f["stem"] for f in clip["frames"]], f"frame order differs: {clip['id']}"
    P, G, log_sf = [], [], []
    for k, fr in enumerate(gt_frames):
        if fr is None:
            continue
        us, vs, gv = fr
        p, inside = sample_at(depth[k], us, vs, geom)
        g = gv[inside]
        good = p > 1e-6
        p, g = p[good], g[good]
        if g.size < a.min_points:
            continue
        log_sf.append(float(np.log(np.median(g) / np.median(p))))
        P.append(p)
        G.append(g)
    if not G:
        return None
    stats = {k: Stats() for k in ALIGN}
    for p, g, ls in zip(P, G, log_sf):
        stats["frame"].add(p * np.exp(ls), g)
    P, G = np.concatenate(P), np.concatenate(G)
    log_sc = float(np.log(np.median(G) / np.median(P)))
    Pc = P * np.exp(log_sc)
    stats["clip"].add(Pc, G)
    metric = bool(pred["metric"])
    if metric:
        stats["metric"].add(P, G)
    bins = []
    for lo, hi in BINS:
        m = (G >= lo) & (G < hi)
        b = Stats()
        b.add(Pc[m], G[m])
        bins.append(b)
    return {"stats": stats, "bins": bins, "metric": metric, "log_sc": log_sc,
            "drift": float(np.std(log_sf)) if len(log_sf) > 1 else float("nan"),
            "n_frames_used": len(log_sf)}


def eval_clip(job):
    """One clip, all methods. Returns {method: None (no prediction) | False (unusable) | result}."""
    clip, root, depth_scale, release, a = job
    paths = {m: pred_path(a.preds_dir, m, release, clip["id"]) for m in a.methods}
    have = {m: p for m, p in paths.items() if p.is_file()}
    out = {m: None for m in a.methods}
    if not have:
        return release, clip, out
    gt_frames = load_gt(clip, root, depth_scale, a)
    for m, pp in have.items():
        pred = np.load(pp)
        r = score_clip(clip, gt_frames, pred, a)
        if r is None:
            out[m] = False
            continue
        row = {"release": release, "seq": clip["seq"], "id": clip["id"],
               "vehicle": clip["meta"].get("vehicle", "?"), "species": clip["meta"].get("species", ""),
               "n_frames": clip["n_frames"], "n_frames_used": r["n_frames_used"],
               "drift": r["drift"], "log_scale": r["log_sc"],
               "scale_err": abs(r["log_sc"]) if r["metric"] else float("nan"),
               "ms_per_frame": 1e3 * float(pred["forward_s"]) / clip["n_frames"],
               "peak_gb": float(pred["peak_gb"])}
        for al in ALIGN:
            res = r["stats"][al].result()
            for k in KEYS:
                row[f"{al}_{k}"] = res.get(k, float("nan"))
        out[m] = {"row": row, "stats": r["stats"], "bins": r["bins"]}
    return release, clip, out


def mean_of(rows, key):
    v = [r[key] for r in rows if np.isfinite(r.get(key, float("nan")))]
    return float(np.mean(v)) if v else float("nan")


def bootstrap(vals, n_boot=1000, seed=0):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(rows, num_keys):
    """Mean over snippets of the per-snippet mean of each numeric key, with CIs over snippets."""
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)
    snippet = [{k: mean_of(rs, k) for k in num_keys} for rs in by_seq.values()]
    out = {"n_clips": len(rows), "n_snippets": len(by_seq)}
    for k in num_keys:
        out[k] = mean_of(snippet, k)
        if k in CI_KEYS:
            out[k + "_ci"] = bootstrap([s[k] for s in snippet])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--preds_dir", default="preds")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=0.0, help="0 = no cap")
    ap.add_argument("--min_points", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="processes; 0 = the CPUs this job may use, at most 8")
    a = ap.parse_args()

    clipsets = [load_clips(c) for c in a.clips]
    jobs = [(clip, C["root"], C["depth_scale"], C["release"], a) for C in clipsets for clip in C["clips"]]
    acc = {m: {"rows": [], "missing": 0, "pooled": {k: Stats() for k in ALIGN}, "bins": [Stats() for _ in BINS]}
           for m in a.methods}
    workers = a.workers or min(8, len(os.sched_getaffinity(0)))
    print(f"{len(jobs)} clips, {len(a.methods)} methods, {workers} worker process(es); ground truth is read once per clip",
          flush=True)
    t0 = time.time()
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    results = pool.imap(eval_clip, jobs) if pool else map(eval_clip, jobs)
    for i, (release, clip, out) in enumerate(results, 1):
        for m, r in out.items():
            if r is None:
                acc[m]["missing"] += 1
            elif r:
                acc[m]["rows"].append(r["row"])
                for al in ALIGN:
                    acc[m]["pooled"][al].merge(r["stats"][al])
                for j, b in enumerate(r["bins"]):
                    acc[m]["bins"][j].merge(b)
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} clips  ({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.close()

    report = {"config": vars(a), "methods": {}}
    for method in a.methods:
        rows, missing, pooled, bins = (acc[method][k] for k in ("rows", "missing", "pooled", "bins"))
        if not rows:
            print(f"\n== {method}: no predictions under {a.preds_dir}/{method}")
            continue

        num_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
        summary = summarise(rows, num_keys)
        splits = {}
        for key in ("vehicle", "release"):
            for val in sorted({r[key] for r in rows}):
                splits[f"{key}={val}"] = summarise([r for r in rows if r[key] == val], num_keys)
        bin_res = {f"{lo}-{int(hi) if hi < 1e9 else 'inf'} m": b.result() for (lo, hi), b in zip(BINS, bins)}
        report["methods"][method] = {
            "summary": summary, "splits": splits,
            "pooled_pixels": {al: pooled[al].result() for al in ALIGN if pooled[al].s["n"]},
            "depth_bins_pooled": bin_res, "missing_clips": missing, "clips": rows}

        s = summary
        ci = lambda k: f"[{s[k + '_ci'][0]:.3f}, {s[k + '_ci'][1]:.3f}]"
        print(f"\n== {method}: {s['n_clips']} clips from {s['n_snippets']} snippets; {missing} clips without predictions")
        print(f"   clip scale    AbsRel {s['clip_abs_rel']:.3f} {ci('clip_abs_rel')}   "
              f"d1 {s['clip_d1']:.3f} {ci('clip_d1')}   RMSE {s['clip_rmse']:.2f} m")
        print(f"   frame scale   AbsRel {s['frame_abs_rel']:.3f}   d1 {s['frame_d1']:.3f}")
        if np.isfinite(s["metric_abs_rel"]):
            print(f"   metric        AbsRel {s['metric_abs_rel']:.3f}   d1 {s['metric_d1']:.3f}   "
                  f"scale error {s['scale_err']:.3f} {ci('scale_err')}")
        print(f"   scale drift   {s['drift']:.3f} {ci('drift')}   {s['ms_per_frame']:.0f} ms/frame   "
              f"peak {s['peak_gb']:.1f} GB")
        for name, sp in splits.items():
            print(f"   {name:<30s} {sp['n_clips']:3d} clips   AbsRel {sp['clip_abs_rel']:.3f}   d1 {sp['clip_d1']:.3f}")
        for name, br in bin_res.items():
            if br:
                print(f"   depth {name:<10s} {br['n']:>11d} px   AbsRel {br['abs_rel']:.3f}   d1 {br['d1']:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
