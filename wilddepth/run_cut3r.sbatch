#!/bin/bash
# CUT3R over every clip list in clips/. Resumable.   sbatch run_cut3r.sbatch
#SBATCH --job-name=wd-cut3r
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=cut3r_%j.out
# 4 shards = 16 GB of one card; the runner reads the granted shards from slurm and caps itself 1 GB below.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/cut3r || { echo "conda activate failed"; exit 1; }
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES node=$(hostname) python=$(which python)"
ls clips/*.json >/dev/null 2>&1 || { echo "no clip lists in clips/: run make_clips.py first"; exit 1; }
python -u run_cut3r.py --method cut3r --repo /storage2/3DOM/vshukla/repos/CUT3R \
    --ckpt /storage2/3DOM/vshukla/checkpoints/cut3r/cut3r_512_dpt_4_64.pth \
    --clips clips/*.json --preds_dir preds
