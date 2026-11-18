#!/bin/bash
# Run the CUT3R environment setup as a CPU batch job.   sbatch envs/setup_cut3r.sbatch
#SBATCH --job-name=wd-setup-cut3r
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=04:00:00
#SBATCH --output=setup_cut3r_%j.out
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
bash envs/setup_cut3r.sh
