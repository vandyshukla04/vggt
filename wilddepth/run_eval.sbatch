#!/bin/bash
# Full VGGT evaluation on WildDepth Longleat v2 part 2.   Submit from this folder:  sbatch run_eval.sbatch
#SBATCH --job-name=vggt-wilddepth
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=wilddepth_%j.out

# 4 shards = 16 GB of one A40. The script caps itself at that amount, so it can never use
# GPU memory reserved by other jobs. It uses only the GPU slurm assigns.
# Load conda directly: ~/.bashrc does nothing in a non-interactive batch job.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/vggt || { echo "conda activate failed"; exit 1; }
echo "python: $(which python)  job=$SLURM_JOB_ID  gpu=$CUDA_VISIBLE_DEVICES"
export HF_HOME=/storage3/3DOM/vshukla/hf_cache
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth

python eval_vggt_wilddepth.py eval \
    --root /storage3/3DOM/vshukla/data/WildDepth/longleat_v2_part2 \
    --rgb_glob "*/*/rgb/*.jpg" --depth_glob "*/*/depthm/*.png" --depth_scale 0.001 \
    --window 24 --gpu_mem_gb 16 \
    --out results/longleat_v2_part2
