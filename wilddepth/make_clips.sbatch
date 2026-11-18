#!/bin/bash
# Build the clip list for every extracted WildDepth release that lacks one. CPU job.
#   sbatch make_clips.sbatch
#SBATCH --job-name=wd-clips
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --time=02:00:00
#SBATCH --output=clips_%j.out
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /storage2/3DOM/vshukla/envs/vggt || { echo "conda activate failed"; exit 1; }
cd /storage2/3DOM/vshukla/repos/vggt/wilddepth
mkdir -p clips
for r in /storage3/3DOM/vshukla/data/WildDepth/*/; do
    name=$(basename "$r")
    if [ -f "clips/$name.json" ]; then echo "have clips/$name.json"; continue; fi
    if [ -z "$(find "$r" -maxdepth 3 -type d -name rgb | head -1)" ]; then
        echo "skip $name: no rgb folders within 3 levels (not extracted yet, or a different layout)"; continue; fi
    python -u make_clips.py --root "$r" --out "clips/$name.json" || echo "FAILED: $name"
done
echo "clip lists:"; ls -l clips/
