#!/usr/bin/env python3
"""π³ runner: one prediction file per clip, in the common format (see wdcommon.py).

  python run_pi3.py --repo /storage2/.../repos/Pi3 --clips clips/*.json --preds_dir preds --gpu_mem_gb 32

Preprocessing follows pi3.utils.basic.load_images_as_tensor: every frame is resized (no crop) to the
largest multiple-of-14 size with at most 255,000 pixels that keeps the aspect ratio, 672x378 for
1080p. Depth is the z of the per-view local point map. Resumable; GPU allocations capped at
--gpu_mem_gb minus 1 GB.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wdcommon import load_clips, pred_path  # noqa: E402

METHOD = "pi3"
METRIC = False          # π³ predicts scale-invariant local point maps
PIXEL_LIMIT = 255000


def pi3_geometry(W: int, H: int):
    """Target size of load_images_as_tensor: aspect kept, multiples of 14, <= PIXEL_LIMIT pixels."""
    scale = math.sqrt(PIXEL_LIMIT / (W * H))
    Wt, Ht = W * scale, H * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    tw, th = max(1, k) * 14, max(1, m) * 14
    return np.array([tw / W, th / H, 0.0, 0.0], dtype=np.float32), (th, tw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Pi3 repository root")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--preds_dir", default="preds")
    ap.add_argument("--gpu_mem_gb", type=float, required=True, help="shards x 4 GB")
    ap.add_argument("--model", default="yyfz233/Pi3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default="fp32", choices=["fp32", "half"],
                    help="half casts the weights to the autocast dtype (small GPUs only)")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms
    sys.path.insert(0, a.repo)
    from pi3.models.pi3 import Pi3

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    budget_gb = a.gpu_mem_gb - 1.0
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gb / total_gb), 0)
    print(f"{torch.cuda.get_device_name(0)} {total_gb:.1f} GB; capped at {budget_gb:.1f} GB", flush=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = Pi3.from_pretrained(a.model)
    if a.weights == "half":                      # transformers only; the heads run in fp32 either way
        for name in ("encoder", "decoder", "point_decoder", "conf_decoder", "camera_decoder"):
            getattr(model, name).to(dtype)
    model = model.cuda().eval()
    to_tensor = transforms.ToTensor()

    done = 0
    for cpath in a.clips:
        C = load_clips(cpath)
        root, release = C["root"], C["release"]
        for clip in C["clips"]:
            out = pred_path(a.preds_dir, METHOD, release, clip["id"])
            if out.is_file():
                continue
            if a.limit and done >= a.limit:
                return
            W, H = clip["image_size"]
            geom, (h, w) = pi3_geometry(W, H)

            t0 = time.time()
            frames = []
            for f in clip["frames"]:
                with Image.open(root / f["rgb"]) as im:
                    frames.append(to_tensor(im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)))
            imgs = torch.stack(frames).cuda()                               # (S, 3, h, w) in [0, 1]
            load_s = time.time() - t0

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    res = model(imgs[None])
            except torch.cuda.OutOfMemoryError:
                sys.exit(f"out of memory within the {budget_gb:.1f} GB cap on {clip['id']} "
                         f"({clip['n_frames']} frames); request more shards. Other jobs unaffected.")
            torch.cuda.synchronize()
            forward_s = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 2**30

            depth = res["local_points"][0, ..., 2].float().cpu().numpy().astype(np.float16)
            conf = torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy().astype(np.float16)
            del res, imgs
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
