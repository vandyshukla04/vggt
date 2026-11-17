#!/bin/bash
# Download one WildDepth release into $DATA and list its layout.
# Run on a cluster login or transfer node (zenodo blocks some networks).
#   bash download.sh 19154614 longleat_v2_part2      # 21 GB, smallest 1080p release
set -euo pipefail
REC=${1:?zenodo record id}; NAME=${2:?local name}
DATA=${DATA:-/storage3/3DOM/vshukla/data/WildDepth}
mkdir -p "$DATA/$NAME" && cd "$DATA/$NAME"

# list every file in the record, then fetch the tar archives (the mp4 previews are skipped)
curl -s "https://zenodo.org/api/records/$REC" | python3 -c '
import json,sys
for f in json.load(sys.stdin)["files"]:
    print(f["key"], f["size"], f["links"]["self"])' | tee files.txt
grep -E '\.(tar|zip|tar\.gz)\s' files.txt | while read -r key size url; do
    [ -f "$key" ] || wget -c -O "$key" "$url"
done

for t in *.tar; do
    echo "== $t (first 40 entries)"; tar tf "$t" | head -40
    tar xf "$t"
done
echo "done: $DATA/$NAME"
