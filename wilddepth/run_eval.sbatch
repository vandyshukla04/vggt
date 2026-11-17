#!/bin/bash
#SBATCH --job-name=vggt-wilddepth
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=wilddepth_%j.out
# Cluster policy: request GPU shards, --mem in MB. node82: 12 shards per A40 (~3.8 GB each);
# 4 shards (~15 GB) covers a 24-frame window. Override with sbatch --gres=shard:N.
# and never set CUDA_VISIBLE_DEVICES by hand.
# Set these four after running `inspect`:
ROOT=${ROOT:-/storage3/3DOM/vshukla/data/WildDepth/longleat_v2_part2}
RGB_GLOB=${RGB_GLOB:?set from inspect}
DEPTH_GLOB=${DEPTH_GLOB:?set from inspect}
DEPTH_SCALE=${DEPTH_SCALE:?set from inspect}

source ~/.bashrc && conda activate ${ENV:-/storage2/3DOM/vshukla/envs/vggt}
export HF_HOME=${HF_HOME:-/storage3/3DOM/vshukla/hf_cache}
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from wilddepth/; slurm runs a spooled copy of this file
echo "job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES node=$(hostname)"   # never set CUDA_VISIBLE_DEVICES by hand
python eval_vggt_wilddepth.py eval --root "$ROOT" \
    --rgb_glob "$RGB_GLOB" --depth_glob "$DEPTH_GLOB" --depth_scale "$DEPTH_SCALE" \
    --window ${WINDOW:-24} --frame_stride ${STRIDE:-1} ${EXTRA:-} \
    --out results/$(basename "$ROOT")
