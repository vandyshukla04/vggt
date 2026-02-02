#!/bin/bash
#SBATCH --job-name=vggt
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=02:00:00
#SBATCH --output=vggt_%j.out
#SBATCH --error=vggt_%j.err

# Load CUDA (check available versions with: module avail)
module load cuda/12.4  # or closest available version

# Activate conda environment
source ~/.bashrc
conda activate vggt  # adjust to your env name

# Change to vggt directory
cd /path/to/vggt

# Run batch inference
python batch_inference.py \
    --scene_dir /storage2/3DOM/vshukla/repos/vggt/wd_data/rhinos/rhin-12 \
    --output_dir ./outputs/rhin-12 \
    --num_images 30 \
    --conf_threshold 50

echo "Job completed at $(date)"