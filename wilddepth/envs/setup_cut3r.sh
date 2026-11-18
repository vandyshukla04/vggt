#!/bin/bash
# Create the CUT3R environment on the cluster and fetch its checkpoint. CPU only: run it on a
# login node, inside tmux (it takes 20-40 minutes).      bash envs/setup_cut3r.sh
#
# Versions match the machine that produced the WildLIFT results: python 3.11, torch 2.5.1
# (CUDA 12.1 build), numpy 1.26.4. The RoPE CUDA kernels are not compiled; CUT3R then uses
# its PyTorch implementation of the same operation.
set -euo pipefail
ENV=/storage3/3DOM/vshukla/envs/cut3r
REPOS=/storage3/3DOM/vshukla/repos
CKPT=/storage3/3DOM/vshukla/checkpoints/cut3r
GDRIVE="https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"

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
conda install -y 'llvm-openmp<16'
pip install gdown

echo "== 3/4 checkpoint (3 GB)"
if [ ! -s "$CKPT/cut3r_512_dpt_4_64.pth" ]; then
    gdown --fuzzy "$GDRIVE" -O "$CKPT/cut3r_512_dpt_4_64.pth" || {
        echo "gdown failed (Google Drive quota). Copy the file from the laptop instead:"
        echo "  scp /home/shuklva/CUT3R/src/cut3r_512_dpt_4_64.pth fbk-cluster:$CKPT/"
        exit 1; }
fi

echo "== 4/4 check: import the model and load the weights on the CPU"
cd "$REPOS/CUT3R"
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
from dust3r.model import ARCroco3DStereo
m = ARCroco3DStereo.from_pretrained("/storage3/3DOM/vshukla/checkpoints/cut3r/cut3r_512_dpt_4_64.pth")
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"CUT3R ok: {n:.0f}M parameters, torch {torch.__version__}, python {sys.version.split()[0]}")
PY
echo "done: env $ENV, code $REPOS/CUT3R and $REPOS/TTT3R, weights $CKPT"
