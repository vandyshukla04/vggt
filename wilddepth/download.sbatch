#!/bin/bash
# Download and extract one WildDepth release as a CPU batch job.
#   sbatch download.sbatch 19152060 longleat_v2_part1
#SBATCH --job-name=wd-download
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048
#SBATCH --time=12:00:00
#SBATCH --output=download_%j.out
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
bash download.sh "$1" "$2"
