#!/usr/bin/env bash
# Run from the pos-pipeline folder. Downloads into the parent folder of paths.data_root in config.yaml.
# Usage: bash scripts/download_data.sh [phase]
#   phase 1 (default): fMRI (4 subjects), atlases, transcripts S4-S6, Season 6 videos  (~10 GB)
#   phase 2          : Season 4 and 5 videos                                           (~15 GB)
#   phase 3          : Season 3 videos + transcripts (optional extra training data)    (~23 GB)
# Only per-run split files (friends_sXXeYY[a-d].mkv) are fetched. Safe to re-run: finished files are skipped.
set -uo pipefail
PHASE="${1:-1}"
[ -f ~/pos/venv/bin/activate ] && source ~/pos/venv/bin/activate
hash -r
command -v datalad >/dev/null || { echo "datalad not found: finish the environment setup first"; exit 1; }
[ -f config.yaml ] || { echo "run this from the pos-pipeline folder (config.yaml not found)"; exit 1; }
DATA_ROOT=$(python -c "import yaml,os;print(os.path.abspath(os.path.expanduser(yaml.safe_load(open('config.yaml'))['paths']['data_root'])))")
PARENT=$(dirname "$DATA_ROOT")
mkdir -p "$PARENT"
echo "Dataset folder: $DATA_ROOT"; df -h "$PARENT" | tail -1
if [ ! -d "$DATA_ROOT/.git" ] && [ ! -f "$DATA_ROOT/.git" ]; then
  datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git "$DATA_ROOT" || exit 1
fi
cd "$DATA_ROOT"
FAIL=0
get() { datalad get -J8 "$@" || FAIL=1; }
if [ "$PHASE" = "1" ]; then
  get fmri/sub-0*/func/*task-friends*_bold.h5 fmri/sub-0*/atlas/*.nii.gz
  get stimuli/transcripts/friends/s4 stimuli/transcripts/friends/s5 stimuli/transcripts/friends/s6
  get stimuli/movies/friends/s6/friends_s06e*[a-d].mkv
elif [ "$PHASE" = "2" ]; then
  get stimuli/movies/friends/s4/friends_s04e*[a-d].mkv stimuli/movies/friends/s5/friends_s05e*[a-d].mkv
elif [ "$PHASE" = "3" ]; then
  get stimuli/transcripts/friends/s3 stimuli/movies/friends/s3/friends_s03e*[a-d].mkv
fi
du -sh fmri stimuli 2>/dev/null
if [ "$FAIL" = "1" ]; then echo "SOME FILES FAILED: re-run the same command (finished files are skipped)"; exit 1; fi
echo "phase $PHASE complete"
