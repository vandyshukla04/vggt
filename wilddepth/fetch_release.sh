#!/bin/bash
# Fetch one WildDepth release and keep only the frames the clip protocol uses.
#   bash fetch_release.sh 19152060 longleat_v2_part1          (CPU node, in tmux; about 1 h per release)
# Steps: download the archive -> list its members -> extract metadata, calibration and one frame per
# snippet -> build the clip list from the listing -> extract just the clip frames -> delete the archive.
# The archive is downloaded to $STAGE (default: next to the data). Set STAGE=$HOME/wd_stage to keep it off
# storage3; the extracted release then adds only a few GB there. A local .tar path works in place of a record id.
set -euo pipefail
REC=${1:?zenodo record id, or a local .tar}; NAME=${2:?local release name}
DATA=${DATA:-/storage3/3DOM/vshukla/data/WildDepth}
STAGE=${STAGE:-$DATA}
CLIP_ARGS=${CLIP_ARGS:-}                     # extra make_clips.py options (tests only)
WD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR="$DATA/$NAME"; SDIR="$STAGE/$NAME"
mkdir -p "$DIR" "$SDIR" "$WD/clips"

extract_members() {                          # $1 archive, $2 file of member names (without ./), into $DIR
    local t=$1 want=$2 pfx=''
    head -1 "$DIR/$(basename "$t").list" | grep -q '^\./' && pfx='./'
    sed 's|^\./||' "$DIR/$(basename "$t").list" | grep -Fxf "$want" | sed "s|^|$pfx|" > "$DIR/.members" || true
    [ -s "$DIR/.members" ] && tar xf "$t" -C "$DIR" -T "$DIR/.members"
    rm -f "$DIR/.members"
}

echo "== 1/5 archive"
TARS=()
if [ -f "$REC" ]; then
    TARS=("$(readlink -f "$REC")")
else
    cd "$SDIR"
    curl -s "https://zenodo.org/api/records/$REC" | python3 -c '
import json,sys
for f in json.load(sys.stdin)["files"]:
    print(f["key"], f["size"], f["links"]["self"])' | tee "$DIR/files.txt"
    while read -r key size url; do
        case "$key" in *.tar)
            [ -f "$key" ] && [ "$(stat -c %s "$key")" -eq "$size" ] || wget -q --show-progress -c -O "$key" "$url"
            TARS+=("$SDIR/$key");; esac
    done < "$DIR/files.txt"
fi
[ ${#TARS[@]} -gt 0 ] || { echo "no .tar in record $REC"; exit 1; }

echo "== 2/5 listing"
: > "$DIR/listing.txt"
for t in "${TARS[@]}"; do tar tf "$t" > "$DIR/$(basename "$t").list"; sed 's|^\./||' "$DIR/$(basename "$t").list" >> "$DIR/listing.txt"; done
echo "$(grep -vc '/$' "$DIR/listing.txt") files in $(( ${#TARS[@]} )) archive(s)"

echo "== 3/5 metadata, calibration, one frame per snippet"
python3 - "$WD" "$DIR/listing.txt" <<'PY' > "$DIR/.want"
import sys
sys.path.insert(0, sys.argv[1]); from wdcommon import detect_globs, pair_relative
lines = [l.strip() for l in open(sys.argv[2]) if l.strip() and not l.strip().endswith("/")]
want = [l for l in lines if l.endswith(("metadata.txt", "README.txt")) or l.startswith("calibration/")]
want += [pairs[0][0] for pairs in pair_relative(lines, *detect_globs(lines)).values()]
print("\n".join(want))
PY
for t in "${TARS[@]}"; do extract_members "$t" "$DIR/.want"; done

echo "== 4/5 clip list"
(cd "$WD" && python3 make_clips.py --root "$DIR" --listing "$DIR/listing.txt" --out "clips/$NAME.json" $CLIP_ARGS)
python3 -c '
import json,sys
C=json.load(open(sys.argv[1])); n=len(C["clips"])
print("\n".join(sorted({f[k] for c in C["clips"] for f in c["frames"] for k in ("rgb","depth")})))
sys.exit(0 if n else 1)' "$WD/clips/$NAME.json" > "$DIR/.want" || { echo "no clips were built; nothing extracted"; exit 1; }

echo "== 5/5 clip frames"
for t in "${TARS[@]}"; do extract_members "$t" "$DIR/.want"; done
MISSING=$(while read -r f; do [ -f "$DIR/$f" ] || echo "$f"; done < "$DIR/.want" | wc -l)
[ "$MISSING" -eq 0 ] || { echo "$MISSING clip files missing after extraction; archive kept"; exit 1; }
for t in "${TARS[@]}"; do case "$t" in "$SDIR"/*) rm -f "$t";; esac; rm -f "$DIR/$(basename "$t").list"; done
rmdir "$SDIR" 2>/dev/null || true
N=$(wc -l < "$DIR/.want"); rm -f "$DIR/.want"
echo "done: $NAME  $N clip files, $(du -sh "$DIR" | cut -f1) on disk, clip list $WD/clips/$NAME.json"
