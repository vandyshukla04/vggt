#!/bin/bash
# pi3 over every clip list in clips/. Resumable.   sbatch run_pi3.sbatch
#SBATCH --job-name=wd-pi3
#SBATCH --partition=gpu-A40
#SBATCH --gres=shard:8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --output=pi3_%j.out
# 8 shards = 32 GB of one card; the runner caps itself at 31 GB and reports the real peak.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/geo || { echo "conda activate failed"; exit 1; }
export HF_HOME=/storage2/3DOM/vshukla/hf_cache
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES node=$(hostname) python=$(which python)"
ls clips/*.json >/dev/null 2>&1 || { echo "no clip lists in clips/: run make_clips.py first"; exit 1; }
python -u run_pi3.py --repo /storage2/3DOM/vshukla/repos/Pi3 --clips clips/*.json --preds_dir preds --gpu_mem_gb 32
