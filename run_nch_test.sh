#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=4:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_Extract
#PBS -J 1-5

# One array task per age bin. PBS_ARRAY_INDEX is 1-based.
AGE_BINS=("1-2y" "3-5y" "6-12y" "13-18y" "19-100y")
AGE_BIN="${AGE_BINS[$((PBS_ARRAY_INDEX - 1))]}"

INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
MODEL_DIR="/srv/scratch/z5423210/StanleyThesis2026/out_eegmamba/nch/${AGE_BIN}/out/model_weights"

# NOTE: no blanket `rm -f *.[oe]*` cleanup step here (unlike
# run_thesis_prelim.sh) — 5 array tasks share this script, and a shared
# wildcard delete would race across tasks and could clobber a sibling
# task's still-in-progress log. If you want log cleanup, scope it to this
# task's own log file only, e.g.:
#   rm -f "EEGMamba_NCH_Extract.o${PBS_JOBID%%.*}"

cd "$REPO_DIR" || exit 1

echo "Starting extraction for age bin: ${AGE_BIN} (array index ${PBS_ARRAY_INDEX})"

python extract_predictions_nch.py \
    --age_bin "${AGE_BIN}" \
    --index_path "${INDEX_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --cuda 0

# Capture $? IMMEDIATELY after the command whose status we care about —
# this is the exact bug flagged as still-open in finetune_trainer.py's PBS
# script (a preceding blank `echo ""` was overwriting $? with 0 before it
# got logged, so every job silently reported success regardless of outcome).
STATUS=$?
echo "Finished age bin: ${AGE_BIN}, exit code ${STATUS}"

if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: extraction failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
