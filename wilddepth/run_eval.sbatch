#!/bin/bash
#SBATCH --job-name=vggt-wilddepth
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wilddepth_%j.out
# Set these four after running `inspect`:
ROOT=${ROOT:-/storage3/3DOM/vshukla/data/WildDepth/longleat_v2_part2}
RGB_GLOB=${RGB_GLOB:?set from inspect}
DEPTH_GLOB=${DEPTH_GLOB:?set from inspect}
DEPTH_SCALE=${DEPTH_SCALE:?set from inspect}

source ~/.bashrc && conda activate vggt
cd "$(dirname "$0")" && mkdir -p logs
python eval_vggt_wilddepth.py eval --root "$ROOT" \
    --rgb_glob "$RGB_GLOB" --depth_glob "$DEPTH_GLOB" --depth_scale "$DEPTH_SCALE" \
    --window ${WINDOW:-24} --frame_stride ${STRIDE:-1} ${EXTRA:-} \
    --out results/$(basename "$ROOT")
