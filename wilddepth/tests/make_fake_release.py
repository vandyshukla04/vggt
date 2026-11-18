#!/usr/bin/env python3
"""Synthetic WildDepth-like release for local tests: tiny images, sparse millimetre depth PNGs.
  python tests/make_fake_release.py <out_dir>   (writes <out_dir>/fake_release/...)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(sys.argv[1]) / "fake_release"
rng = np.random.default_rng(0)
(R / "calibration").mkdir(parents=True, exist_ok=True)
for seq, n, veh in (("Jul/zebra_1", 14, "yes"), ("Jul/lion_1", 6, "no"), ("Jul/tiny_1", 3, "yes")):
    (R / seq / "rgb").mkdir(parents=True, exist_ok=True)
    (R / seq / "depthm").mkdir(exist_ok=True)
    (R / seq / "metadata.txt").write_text(f"species: {seq.split('/')[1].split('_')[0]}\nvehicle: {veh}\n")
    W, H = 640, 360
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        fid = 1000 + i
        img = np.stack([xx * 255 // W, yy * 255 // H, (xx + yy + 5 * i) % 255], -1).astype(np.uint8)
        Image.fromarray(img).save(R / seq / "rgb" / f"{fid}.jpg", quality=90)
        d = np.zeros((H, W), np.uint16)
        m = rng.random((H, W)) < 0.01
        d[m] = (8000 + 20000 * yy[m] / H).astype(np.uint16)       # 8..28 m in millimetres
        Image.fromarray(d).save(R / seq / "depthm" / f"{fid}.png")
print(R)
