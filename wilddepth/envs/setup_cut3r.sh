#!/bin/bash
# Create the CUT3R environment on the cluster and fetch its checkpoint. CPU only: run it on a
# login node, inside tmux (it takes 20-40 minutes).      bash envs/setup_cut3r.sh
#
# Versions match the machine that produced the WildLIFT results: python 3.11, torch 2.5.1
# (CUDA 12.1 build), numpy 1.26.4, transformers 4.52.4.
set -euo pipefail
ENV=/storage2/3DOM/vshukla/envs/cut3r
REPOS=/storage2/3DOM/vshukla/repos
CKPT=/storage2/3DOM/vshukla/checkpoints/cut3r
export PIP_CACHE_DIR=/storage2/3DOM/vshukla/pip_cache     # keep pip downloads out of /home (group quota nearly full)
GDRIVE_ID="1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD"      # cut3r_512_dpt_4_64.pth, from the CUT3R README

source /opt/miniforge3/etc/profile.d/conda.sh
mkdir -p "$REPOS" "$CKPT"

echo "== 1/4 code"
[ -d "$REPOS/CUT3R" ] || git clone https://github.com/CUT3R/CUT3R.git "$REPOS/CUT3R"
[ -d "$REPOS/TTT3R" ] || git clone https://github.com/Inception3D/TTT3R.git "$REPOS/TTT3R"

echo "== 2/4 environment"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$REPOS/CUT3R/requirements.txt"
# requirements.txt leaves transformers unpinned and its newest release demands torch >= 2.6;
# pin the versions of the WildLIFT machine, which work with torch 2.5.1
pip install "transformers==4.52.4" "huggingface_hub==0.32.4" "accelerate==1.7.0"
conda install -y 'llvm-openmp<16'
pip install gdown

echo "== 3/4 checkpoint (3 GB)"
F="$CKPT/cut3r_512_dpt_4_64.pth"
if [ -f "$F" ] && [ "$(stat -c %s "$F")" -lt 3000000000 ]; then
    echo "removing incomplete checkpoint ($(stat -c %s "$F") bytes)"; rm -f "$F"; fi
if [ ! -f "$F" ]; then
    gdown "$GDRIVE_ID" -O "$F" || {
        echo "gdown could not fetch the checkpoint (see its message above). Copy the file from the laptop instead:"
        echo "  scp /home/shuklva/CUT3R/src/cut3r_512_dpt_4_64.pth fbk-cluster:$CKPT/"
        exit 1; }
fi

# The RoPE CUDA kernel (README step "python setup.py build_ext") is NOT built. On this cluster nvcc 12.4
# clashes with the nodes' gcc 14 and glibc headers; run_cut3r.py instead uses a PyTorch RoPE with direct
# computation, checked against the compiled kernel (STUDY.md, 2026-11-20).

echo "== 4/4 check: import the model and load the weights on the CPU"
cd "$REPOS/CUT3R"
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
from dust3r.model import ARCroco3DStereo
m = ARCroco3DStereo.from_pretrained("/storage2/3DOM/vshukla/checkpoints/cut3r/cut3r_512_dpt_4_64.pth")
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"CUT3R ok: {n:.0f}M parameters, torch {torch.__version__}, python {sys.version.split()[0]}")
PY
echo "done: env $ENV, code $REPOS/CUT3R and $REPOS/TTT3R, weights $CKPT"
