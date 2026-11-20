#!/usr/bin/env python3
"""CUT3R / TTT3R runner: one prediction file per clip, in the common format (see wdcommon.py).

  python run_cut3r.py --method cut3r --repo /storage3/.../repos/CUT3R --ckpt .../cut3r_512_dpt_4_64.pth \
      --clips clips/*.json --preds_dir preds --gpu_mem_gb 16
  python run_cut3r.py --method ttt3r --repo /storage3/.../repos/TTT3R --ckpt <same file> ...

Both methods load the same checkpoint. TTT3R is the CUT3R code with a different state-update
rule, switched on through the model config. Each method runs through the inference routine its
authors use in their demo and video-depth evaluation: CUT3R `inference`, TTT3R
`inference_recurrent_lighter` (no state resets within a clip, as in TTT3R's evaluation).
Depth is the z of each frame's point map in its own camera frame (pts3d_in_self_view).
The rotary position embedding runs in PyTorch when the compiled kernel is absent (see use_direct_rope).
Resumable; the process caps its own GPU allocations at what the slurm job reserved, minus 1 GB.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wdcommon import gpu_budget_gb, load_clips, pred_path  # noqa: E402

METRIC = True           # CUT3R is trained with metric supervision and claims metric-scale output


def cut3r_geometry(W: int, H: int, size: int = 512):
    """Geometry of dust3r.utils.image.load_images(size=512): long side -> 512, then a centre
    crop to multiples of 16. Returns (geom, (h, w))."""
    S = max(W, H)
    new_w, new_h = int(round(W * size / S)), int(round(H * size / S))
    cx, cy = new_w // 2, new_h // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if new_w == new_h:
        halfh = 3 * halfw / 4
    left, top = cx - halfw, cy - halfh
    geom = np.array([new_w / W, new_h / H, float(left), float(top)], dtype=np.float32)
    return geom, (int(2 * halfh), int(2 * halfw))


def use_direct_rope(model) -> str:
    """Make the PyTorch RoPE2D usable for CUT3R; returns which implementation the model runs.

    CUT3R's rotary position embedding has a compiled CUDA kernel (cuRoPE2D) and a PyTorch fallback.
    The fallback looks cos/sin up in a table indexed by position, which fails for the pose token's
    position -1. Here its forward is replaced by the direct computation the kernel performs
    (angle = position x inverse frequency), valid for any integer position. Checked against the
    compiled kernel on full CUT3R runs: depth mean relative difference 7e-5 (STUDY.md, 2026-11-20).
    """
    import torch

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=tokens.device).float() / D))

        def rope1d(x, pos):
            ang = pos[:, :, None].float() * inv_freq
            ang = torch.cat((ang, ang), dim=-1)[:, None]
            return x * ang.cos().to(x.dtype) + self.rotate_half(x) * ang.sin().to(x.dtype)

        y, x = tokens.chunk(2, dim=-1)
        return torch.cat((rope1d(y, positions[:, :, 0]), rope1d(x, positions[:, :, 1])), dim=-1)

    classes = {type(m) for m in model.modules() if type(m).__name__ in ("RoPE2D", "cuRoPE2D")}
    if not classes:
        sys.exit("no RoPE module found in the model; the position-embedding code has changed")
    kinds = set()
    for c in classes:
        if hasattr(c, "get_cos_sin"):               # the PyTorch fallback class
            c.forward = forward
            kinds.add("pytorch-direct")
        else:
            kinds.add("compiled-kernel")
    return "+".join(sorted(kinds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["cut3r", "ttt3r"])
    ap.add_argument("--repo", required=True, help="CUT3R or TTT3R repository root")
    ap.add_argument("--ckpt", required=True, help="cut3r_512_dpt_4_64.pth")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--preds_dir", default="preds")
    ap.add_argument("--gpu_mem_gb", type=float, default=0.0,
                    help="GPU memory reserved for this job; default: read the shards from slurm")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rope", default="auto", choices=["auto", "python"],
                    help="auto: compiled cuRoPE2D kernel if the repo has it, else PyTorch; python: force PyTorch")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(a.repo) / "src"))
    if a.rope == "python":
        sys.modules["models.curope"] = None       # the import of the compiled kernel then fails -> PyTorch RoPE
    if a.method == "ttt3r":
        # TTT3R's own demo and video-depth evaluation call this frame-by-frame routine
        from dust3r.inference import inference_recurrent_lighter as inference
    else:
        # CUT3R's demo and video-depth evaluation (and the WildLIFT pipeline) call this one
        from dust3r.inference import inference
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.image import load_images

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    budget_gb, source = gpu_budget_gb(a.gpu_mem_gb)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gb / total_gb), 0)
    print(f"{torch.cuda.get_device_name(0)} {total_gb:.1f} GB; capped at {budget_gb:.1f} GB ({source})", flush=True)

    model = ARCroco3DStereo.from_pretrained(a.ckpt).cuda().eval()
    if a.method == "ttt3r":
        model.config.model_update_type = "ttt3r"         # as in the TTT3R demo
    elif hasattr(model.config, "model_update_type"):
        model.config.model_update_type = "cut3r"
    rope = use_direct_rope(model)
    print(f"{a.method}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters, RoPE: {rope}", flush=True)

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
            geom, (h, w) = cut3r_geometry(W, H, a.size)

            t0 = time.time()
            imgs = load_images([str(root / f["rgb"]) for f in clip["frames"]], size=a.size, verbose=False)
            views = []
            for i, im in enumerate(imgs):                 # same view dicts as the demo's prepare_input
                assert tuple(im["img"].shape[-2:]) == (h, w), (im["img"].shape, h, w)
                views.append({
                    "img": im["img"],
                    "ray_map": torch.full((1, 6, h, w), torch.nan),
                    "true_shape": torch.from_numpy(im["true_shape"]),
                    "idx": i, "instance": str(i),
                    "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
                    "img_mask": torch.tensor(True)[None], "ray_mask": torch.tensor(False)[None],
                    "update": torch.tensor(True)[None], "reset": torch.tensor(False)[None]})
            load_s = time.time() - t0

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                with torch.no_grad():
                    outputs, _ = inference(views, model, "cuda", verbose=False)
            except torch.cuda.OutOfMemoryError:
                sys.exit(f"out of memory within the {budget_gb:.1f} GB cap on {clip['id']} "
                         f"({clip['n_frames']} frames); request more shards. Other jobs unaffected.")
            torch.cuda.synchronize()
            forward_s = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 2**30

            depth = np.stack([p["pts3d_in_self_view"][0, ..., 2].float().cpu().numpy() for p in outputs["pred"]])
            conf = np.stack([p["conf_self"][0].float().cpu().numpy() for p in outputs["pred"]])
            del outputs, views, imgs
            torch.cuda.empty_cache()
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, depth=depth.astype(np.float16), conf=conf.astype(np.float16), geom=geom,
                     frames=np.array([f["stem"] for f in clip["frames"]]),
                     metric=METRIC, forward_s=forward_s, load_s=load_s, peak_gb=peak_gb, rope=rope)
            done += 1
            print(f"  {release}/{clip['id']:<28s} {clip['n_frames']:3d} fr  load {load_s:5.1f}s  "
                  f"fwd {forward_s:5.1f}s  peak {peak_gb:4.1f} GB", flush=True)
    print(f"done: {done} new clips")


if __name__ == "__main__":
    main()
