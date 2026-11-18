#!/bin/bash
# Environment for the VGGT-family methods beyond VGGT itself: pi3 and PAGE-4D (VGGT-Omega later).
# A copy of the working vggt environment plus a few packages, so the vggt environment stays untouched.
# CPU only; run on the CPU node inside tmux (15-30 min).      bash envs/setup_geo.sh
set -euo pipefail
ENV=/storage2/3DOM/vshukla/envs/geo
REPOS=/storage2/3DOM/vshukla/repos
CKPT=/storage2/3DOM/vshukla/checkpoints/page4d
export HF_HOME=/storage2/3DOM/vshukla/hf_cache
source /opt/miniforge3/etc/profile.d/conda.sh
mkdir -p "$REPOS" "$CKPT" "$HF_HOME"

echo "== 1/4 code"
[ -d "$REPOS/Pi3" ]    || git clone https://github.com/yyfz/Pi3.git "$REPOS/Pi3"
[ -d "$REPOS/PAGE4D" ] || git clone https://github.com/kaichen-z/PAGE4D.git "$REPOS/PAGE4D"

echo "== 2/4 environment (copy of vggt + plyfile, safetensors, opencv, einops)"
[ -d "$ENV" ] || conda create -y -p "$ENV" --clone /storage2/3DOM/vshukla/envs/vggt
conda activate "$ENV"
pip install plyfile safetensors opencv-python-headless einops "huggingface_hub[cli]"

echo "== 3/4 weights: pi3 into the HF cache, PAGE-4D checkpoint_nomask.pt"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download("yyfz233/Pi3"))
print(hf_hub_download("zhouk777/PAGE4D", "checkpoint_nomask.pt", repo_type="dataset",
                      local_dir="/storage2/3DOM/vshukla/checkpoints/page4d"))
PY

echo "== 4/4 check: import both models and load their weights on the CPU"
python - <<'PY'
import sys, torch
sys.path.insert(0, "/storage2/3DOM/vshukla/repos/Pi3")
from pi3.models.pi3 import Pi3
m = Pi3.from_pretrained("yyfz233/Pi3"); print(f"pi3 ok: {sum(p.numel() for p in m.parameters())/1e6:.0f}M parameters")
sys.path.insert(0, "/storage2/3DOM/vshukla/repos/PAGE4D/model")
from page.models.vggt import VGGT
m = VGGT(); st = torch.load("/storage2/3DOM/vshukla/checkpoints/page4d/checkpoint_nomask.pt", map_location="cpu")
st = st.get("model", st); miss, unexp = m.load_state_dict(st, strict=False)
bad = [k for k in miss if k.startswith(("aggregator", "depth_head"))]
print(f"page4d ok: {len(st)} tensors loaded, {len(miss)} missing ({len(bad)} in aggregator/depth head), {len(unexp)} unexpected")
assert not bad, bad[:5]
print("torch", torch.__version__, "python", sys.version.split()[0])
PY
echo "done: env $ENV, code $REPOS/Pi3 and $REPOS/PAGE4D, weights in $HF_HOME and $CKPT"
