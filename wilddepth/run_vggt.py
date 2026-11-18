#!/usr/bin/env python3
"""VGGT and PAGE-4D runner: one prediction file per clip, in the common format (see wdcommon.py).

  python run_vggt.py --method vggt   --clips clips/*.json --preds_dir preds --gpu_mem_gb 32
  python run_vggt.py --method page4d --repo /storage2/.../repos/PAGE4D --ckpt .../checkpoint_nomask.pt \
      --clips clips/*.json --preds_dir preds --gpu_mem_gb 32

PAGE-4D is VGGT retrained for dynamic scenes; it keeps VGGT's preprocessing and output format,
so both share this runner. Resumable: clips whose prediction file exists are skipped. The process
caps its own GPU allocations at the memory the slurm job reserved (shards x 4 GB) minus 1 GB, so
it can never use memory reserved by other jobs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wdcommon import gpu_budget_gb, load_clips, pred_path, vggt_crop_geometry  # noqa: E402

METRIC = False          # VGGT and PAGE-4D depth is normalised per scene; neither claims metric scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="vggt", choices=["vggt", "page4d"])
    ap.add_argument("--repo", default="", help="page4d: PAGE4D repository root")
    ap.add_argument("--ckpt", default="", help="page4d: checkpoint_nomask.pt")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--preds_dir", default="preds")
    ap.add_argument("--gpu_mem_gb", type=float, default=0.0,
                    help="GPU memory reserved for this job; default: read the shards from slurm")
    ap.add_argument("--model", default="facebook/VGGT-1B")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many clips (0 = all)")
    ap.add_argument("--weights", default="fp32", choices=["fp32", "half"],
                    help="half casts the weights to the autocast dtype (halves memory; only for small GPUs)")
    a = ap.parse_args()

    import torch
    if a.method == "page4d":
        if not (a.repo and a.ckpt):
            sys.exit("--method page4d needs --repo and --ckpt")
        sys.path.insert(0, str(Path(a.repo) / "model"))
        from page.models.vggt import VGGT
        from page.utils.load_fn import load_and_preprocess_images   # identical to VGGT's
    else:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    budget_gb, source = gpu_budget_gb(a.gpu_mem_gb)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gb / total_gb), 0)
    print(f"{torch.cuda.get_device_name(0)} {total_gb:.1f} GB; capped at {budget_gb:.1f} GB ({source})", flush=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if a.method == "page4d":
        model = VGGT()
        state = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)   # 6.4 GB file
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"page4d checkpoint: {len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
        if any(k.startswith(("aggregator", "depth_head")) for k in missing):
            sys.exit("page4d checkpoint does not cover the aggregator/depth head: " + str(missing[:5]))
    else:
        model = VGGT.from_pretrained(a.model)
    model.camera_head = model.point_head = model.track_head = None      # depth only
    if a.weights == "half":
        model.aggregator = model.aggregator.to(dtype)   # the depth head runs in fp32 either way
    model = model.cuda().eval()

    done = 0
    for cpath in a.clips:
        C = load_clips(cpath)
        root, release = C["root"], C["release"]
        for clip in C["clips"]:
            out = pred_path(a.preds_dir, a.method, release, clip["id"])
            if out.is_file():
                continue
            if a.limit and done >= a.limit:
                return
            W, H = clip["image_size"]
            geom, (h, w) = vggt_crop_geometry(W, H)

            t0 = time.time()
            imgs = load_and_preprocess_images([str(root / f["rgb"]) for f in clip["frames"]],
                                              mode="crop").cuda()
            load_s = time.time() - t0
            assert tuple(imgs.shape[-2:]) == (h, w), (imgs.shape, h, w)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    pr = model(imgs[None])
            except torch.cuda.OutOfMemoryError:
                sys.exit(f"out of memory within the {budget_gb:.1f} GB cap on {clip['id']} "
                         f"({clip['n_frames']} frames); request more shards. Other jobs unaffected.")
            torch.cuda.synchronize()
            forward_s = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 2**30

            depth = pr["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16)
            conf = pr["depth_conf"][0].float().cpu().numpy().astype(np.float16)
            del pr, imgs
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, depth=depth, conf=conf, geom=geom,
                     frames=np.array([f["stem"] for f in clip["frames"]]),
                     metric=METRIC, forward_s=forward_s, load_s=load_s, peak_gb=peak_gb)
            done += 1
            print(f"  {release}/{clip['id']:<28s} {clip['n_frames']:3d} fr  load {load_s:5.1f}s  "
                  f"fwd {forward_s:5.1f}s  peak {peak_gb:4.1f} GB", flush=True)
    print(f"done: {done} new clips")


if __name__ == "__main__":
    main()
