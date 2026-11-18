#!/bin/bash
# VGGT over every clip list in clips/. Resumable: rerun after adding releases.   sbatch run_vggt.sbatch
#SBATCH --job-name=wd-vggt
#SBATCH --partition=gpu-A40,gpu-L40S-open
#SBATCH --gres=shard:8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=vggt_%j.out
# 8 shards = 32 GB of one card; the runner caps itself at 31 GB and reports the real peak.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/vggt || { echo "conda activate failed"; exit 1; }
export HF_HOME=/storage3/3DOM/vshukla/hf_cache
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES node=$(hostname) python=$(which python)"
python -u run_vggt.py --clips clips/*.json --preds_dir preds --gpu_mem_gb 32
