#!/bin/bash
# pi3 over every clip list in clips/. Resumable.   sbatch run_pi3.sbatch
#SBATCH --job-name=wd-pi3
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=pi3_%j.out
# 4 shards = 16 GB of one card (measured peak for 96 frames: 10.2 GB); the runner reads the granted shards from slurm and caps itself 1 GB below.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/geo || { echo "conda activate failed"; exit 1; }
export HF_HOME=/storage2/3DOM/vshukla/hf_cache
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES node=$(hostname) python=$(which python)"
ls clips/*.json >/dev/null 2>&1 || { echo "no clip lists in clips/: run make_clips.py first"; exit 1; }
python -u run_pi3.py --repo /storage2/3DOM/vshukla/repos/Pi3 --clips clips/*.json --preds_dir preds
