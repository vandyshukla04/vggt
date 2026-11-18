#!/bin/bash
# PAGE-4D over every clip list in clips/. Resumable.   sbatch run_page4d.sbatch
#SBATCH --job-name=wd-page4d
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=page4d_%j.out
# 8 shards = 32 GB of one card; the runner reads the granted shards from slurm and caps itself 1 GB below.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/geo || { echo "conda activate failed"; exit 1; }
export HF_HOME=/storage2/3DOM/vshukla/hf_cache
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES node=$(hostname) python=$(which python)"
ls clips/*.json >/dev/null 2>&1 || { echo "no clip lists in clips/: run make_clips.py first"; exit 1; }
python -u run_vggt.py --method page4d --repo /storage2/3DOM/vshukla/repos/PAGE4D --ckpt /storage2/3DOM/vshukla/checkpoints/page4d/checkpoint_nomask.pt --clips clips/*.json --preds_dir preds
