#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=7:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_HMM_Trained
#
# MMI-trained HMM refinement (mode=hmm_trained): A and alpha are learned discriminatively
# on the TRAIN-split scores, warm-started from the alpha selected on the VAL split by
# select_alpha.sh (read from alpha_selection_<bin>_seqlen<N>.json).
#
# Needs, for this arm:
#   1. test_nch.py (new version) --split test   -> <subject>_<study>_probs.npy + _run_idx.json
#   2. test_nch.py (new version) --split train  -> training_scores/
#   3. test_nch.py (new version) --split val, then select_alpha.sh  -> alpha_selection json
#
#   qsub -v bin=1-2y run_nch_hmm_trained.sh
#   qsub -v bin=6-12y,train_tag=pooled-6-12y-13-18y run_nch_hmm_trained.sh
#   qsub -v bin=1-2y,exp=ms_g0,seed=3407 run_nch_hmm_trained.sh
#
# data_source (default auto): source of the TRAINING labels for the prior; auto follows
# <PRED_DIR>/run_meta.json so the prior and the test windows are trimmed the same way.
#

AGE_BIN="${bin:?Must pass test age bin}"
seq_len="${seq_len:-20}"
train_tag="${train_tag:-$AGE_BIN}"     # same value you used for run_nch_test.sh
exp="${exp:-}"
seed="${seed:-3407}"
DATA_SOURCE="${data_source:-auto}"

SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"

if [ -n "$exp" ]; then
    PRED_DIR="$REPO_DIR/predictions_joint/NCH_${train_tag}_seqlen${seq_len}/${exp}_seed${seed}_on-${AGE_BIN}"
elif [ "$train_tag" = "$AGE_BIN" ]; then
    if [ "$seq_len" -eq 20 ]; then SUFFIX=""; else SUFFIX="_seqlen${seq_len}"; fi
    PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}${SUFFIX}"
else
    PRED_DIR="$REPO_DIR/predictions/NCH_${train_tag}_seqlen${seq_len}_on-${AGE_BIN}"
fi
ls "$PRED_DIR"/[0-9]*_[0-9]*_probs.npy >/dev/null 2>&1 || {
    echo "ERROR: no <subject>_<study>_probs.npy in $PRED_DIR (run the new test_nch.py first)" >&2; exit 1; }
ls "$PRED_DIR"/training_scores/[0-9]*_[0-9]*_probs.npy >/dev/null 2>&1 || {
    echo "ERROR: no train-split scores in $PRED_DIR/training_scores (run the new test_nch.py --split train first)" >&2; exit 1; }

IDX_DIR="/srv/scratch/z5423210/StanleyThesis2026/nch_index"
if [ "$seq_len" -eq 20 ]; then
    if [ -f "$IDX_DIR/nch_index_nch_v2_seqlen20.parquet" ]; then
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen20.parquet"
    else
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2.parquet"   # legacy
    fi
else
    INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen${seq_len}.parquet"
fi
[ -f "$INDEX_PATH" ] || { echo "ERROR: missing $INDEX_PATH" >&2; exit 1; }

# alpha warm start from the val-split selection. Files written before the run-aware
# select_alpha_eval.py have no 'n_recordings' field and are rejected instead of silently reused.
ALPHA_JSON="${PRED_DIR}/alpha_selection_${AGE_BIN}_seqlen${seq_len}.json"
[ -f "$ALPHA_JSON" ] || { echo "ERROR: no $ALPHA_JSON (run select_alpha.sh first)" >&2; exit 1; }
ALPHA_INIT=$(apptainer exec -B /srv:/srv "$SIF" python3 - "$ALPHA_JSON" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if 'n_recordings' not in d:
    sys.exit("alpha_selection json predates the run-aware select_alpha_eval.py; rerun select_alpha.sh")
print(d['selected_alpha'])
PYEOF
) || { echo "ERROR: could not read a valid selected_alpha from ${ALPHA_JSON}" >&2; exit 1; }

echo "Warm-starting hmm_trained with alpha_init=${ALPHA_INIT} (from ${ALPHA_JSON})"

cd "$REPO_DIR" || exit 1

echo "Starting HMM refinement (mode=hmm_trained) for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  pred_dir=${PRED_DIR}  (refined output lands in \${PRED_DIR}/hmmRefined/)"
echo "  data_source=${DATA_SOURCE} exp=${exp:-none} seed=${seed}"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python hmm_refine_nch.py \
    --age_bin "${AGE_BIN}" \
    --mode hmm_trained \
    --seq_len "$seq_len" \
    --pi_source uniform \
    --data_source "${DATA_SOURCE}" \
    --pred_dir "${PRED_DIR}" \
    --index_path "${INDEX_PATH}" \
    --alpha_init "${ALPHA_INIT}"
STATUS=$?

echo "Finished HMM refinement (mode=hmm_trained): ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: HMM refinement failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
