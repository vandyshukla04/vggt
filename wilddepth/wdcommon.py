"""Shared code for the WildDepth 4D-reconstruction study.

Clip list (make_clips.py) -> one runner per method (run_<method>.py) -> evaluate.py.
Every runner writes the same prediction format, one .npz per clip:

  depth      float16 (S, h, w)   camera-space depth at the model's working resolution
  conf       float16 (S, h, w)   model confidence, or absent
  geom       float32 (4,)        sx, sy, ox, oy: original pixel centre (u+.5, v+.5) lands at
                                 model pixel coords x = (u+.5)*sx - ox - .5, y = (v+.5)*sy - oy - .5
  frames     str (S,)            frame stems, in order, to check alignment with the clip list
  metric     bool                whether the method claims metric scale
  forward_s  float               model time for the clip, GPU synchronised
  load_s     float               image decode + preprocessing time
  peak_gb    float               peak GPU allocation during the clip
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- files
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


def stem_key(p):
    """Sort frames numerically when the stem is a number (91540.jpg), else by name."""
    s = p.stem
    return (0, int(s)) if s.isdigit() else (1, s)


def _depth_match(rel: str, pattern: str) -> bool:
    """Match a relative path against a glob with the same component count ('*' never crosses '/')."""
    parts = PurePosixPath(rel).parts
    return len(parts) == len(PurePosixPath(pattern).parts) and PurePosixPath(rel).match(pattern)


def pair_relative(rel_paths, rgb_glob: str, depth_glob: str) -> dict[str, list[tuple[str, str]]]:
    """{sequence: [(rgb, depth), ...]} from relative paths (a directory walk or a tar listing),
    paired by stem and sorted numerically. The sequence is the folder holding the modality
    folders, e.g. July08/graffe_3 for July08/graffe_3/rgb/91540.jpg."""
    rgb, dep = [], {}
    for r in rel_paths:
        r = r[2:] if r.startswith("./") else r
        if _depth_match(r, rgb_glob):
            rgb.append(r)
        elif _depth_match(r, depth_glob):
            p = PurePosixPath(r)
            dep[(p.parent.parent.as_posix(), p.stem)] = r
    seqs = defaultdict(list)
    for r in sorted(rgb, key=lambda x: stem_key(PurePosixPath(x))):
        p = PurePosixPath(r)
        q = dep.get((p.parent.parent.as_posix(), p.stem))
        if q is not None:
            seqs[p.parent.parent.as_posix()].append((r, q))
    return dict(sorted(seqs.items()))


GLOB_CANDIDATES = (("*/*/rgb/*.jpg", "*/*/depthm/*.png"),     # Longleat: <day>/<snippet>/rgb
                   ("*/rgb/*.jpg", "*/depthm/*.png"),         # Bubye: <snippet>/rgb
                   ("*/*/*/rgb/*.jpg", "*/*/*/depthm/*.png"))


def detect_globs(rel_paths, rgb_glob: str = "auto", depth_glob: str = "auto") -> tuple[str, str]:
    """Return (rgb_glob, depth_glob): the given ones, or the first candidate layout that pairs."""
    if rgb_glob != "auto" and depth_glob != "auto":
        return rgb_glob, depth_glob
    rel_paths = list(rel_paths)
    for rg, dg in GLOB_CANDIDATES:
        if pair_relative(rel_paths, rg, dg):
            return rg, dg
    raise SystemExit("no rgb/depth layout recognised; pass --rgb_glob and --depth_glob")


def find_sequences(root: Path, rgb_glob: str, depth_glob: str) -> dict[str, list[tuple[Path, Path]]]:
    """pair_relative over a directory tree, with absolute paths."""
    rel = [p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()]
    rgb_glob, depth_glob = detect_globs(rel, rgb_glob, depth_glob)
    return {seq: [(root / r, root / d) for r, d in pairs]
            for seq, pairs in pair_relative(rel, rgb_glob, depth_glob).items()}


def read_metadata(seq_dir: Path) -> dict:
    """WildDepth's metadata.txt: 'key: value' per line."""
    out = {}
    f = seq_dir / "metadata.txt"
    if f.is_file():
        for line in f.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def load_clips(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["root"] = Path(d["root"])
    return d


def pred_path(preds_dir: Path, method: str, release: str, clip_id: str) -> Path:
    return Path(preds_dir) / method / release / (clip_id.replace("/", "__") + ".npz")


# ----------------------------------------------------------------------------- geometry
def vggt_crop_geometry(W: int, H: int, target: int = 518):
    """Geometry of vggt.utils.load_fn.load_and_preprocess_images(mode='crop'): width -> 518,
    height rounded to a multiple of 14, centre-cropped to 518 if taller.
    Returns (geom, (h, w)) with geom = [sx, sy, ox, oy]."""
    new_w = target
    new_h = round(H * (new_w / W) / 14) * 14
    crop_top = (new_h - target) // 2 if new_h > target else 0
    geom = np.array([new_w / W, new_h / H, 0.0, float(crop_top)], dtype=np.float32)
    return geom, (min(new_h, target), new_w)


def sample_at(pred: np.ndarray, us: np.ndarray, vs: np.ndarray, geom) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample of a model-resolution map at original pixel centres (us, vs).
    Returns (values, inside): `inside` marks pixels that land within the model frame."""
    sx, sy, ox, oy = [float(g) for g in geom]
    h, w = pred.shape
    x = (us + 0.5) * sx - ox - 0.5
    y = (vs + 0.5) * sy - oy - 0.5
    ok = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x, y = x[ok], y[ok]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v = (pred[y0, x0] * (1 - fx) * (1 - fy) + pred[y0, x1] * fx * (1 - fy)
         + pred[y1, x0] * (1 - fx) * fy + pred[y1, x1] * fx * fy)
    return v, ok


# ----------------------------------------------------------------------------- metrics
class Stats:
    """Running sums for AbsRel, RMSE, SILog and the delta accuracies: constant memory, exact
    pooling (every metric is a mean of per-pixel terms)."""
    KEYS = ("n", "abs_rel", "sq_err", "ld", "ld2", "d1", "d2", "d3")

    def __init__(self):
        self.s = dict.fromkeys(self.KEYS, 0.0)

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        if gt.size == 0:
            return
        r = np.maximum(pred / gt, gt / pred)
        ld = np.log(pred) - np.log(gt)
        self.s["n"] += gt.size
        self.s["abs_rel"] += float(np.sum(np.abs(pred - gt) / gt))
        self.s["sq_err"] += float(np.sum((pred - gt) ** 2))
        self.s["ld"] += float(np.sum(ld))
        self.s["ld2"] += float(np.sum(ld ** 2))
        for k, t in (("d1", 1.25), ("d2", 1.25 ** 2), ("d3", 1.25 ** 3)):
            self.s[k] += float(np.sum(r < t))

    def merge(self, other: "Stats") -> None:
        for k in self.KEYS:
            self.s[k] += other.s[k]

    def result(self) -> dict:
        n = self.s["n"]
        if n == 0:
            return {}
        m_ld = self.s["ld"] / n
        return {"n": int(n), "abs_rel": self.s["abs_rel"] / n,
                "rmse": float(np.sqrt(self.s["sq_err"] / n)),
                "silog": float(np.sqrt(max(self.s["ld2"] / n - m_ld ** 2, 0.0))),
                "d1": self.s["d1"] / n, "d2": self.s["d2"] / n, "d3": self.s["d3"] / n}
