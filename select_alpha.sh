#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=05:00:00
#PBS -q eleceng
#PBS -N EEGMamba_select_alpha
#
# Selects alpha on the VAL split (reads PRED_DIR/eval_scores/, written by
# run_nch_eval.sh). Never touches test.
# Usage:
#   per-bin model:  qsub -v bin=6-12y,seq_len=40 select_alpha.sh
#   pooled model:   qsub -v bin=6-12y,train_tag=pooled-all,seq_len=20 select_alpha.sh
#   (train_tag must equal the one used for run_nch_eval.sh / run_nch_test.sh)
#   hmm_trained check: add mode=hmm_trained

AGE_BIN="${bin:?Must pass test age bin, e.g. qsub -v bin=1-2y select_alpha.sh}"
seq_len="${seq_len:-20}"
train_tag="${train_tag:-$AGE_BIN}"
pi_source="${pi_source:-uniform}"
mode="${mode:-hmm}"

REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"

if [ "$train_tag" = "$AGE_BIN" ]; then
    if [ "$seq_len" -eq 20 ]; then SUFFIX=""; else SUFFIX="_seqlen${seq_len}"; fi
    PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}${SUFFIX}"
else
    PRED_DIR="$REPO_DIR/predictions/NCH_${train_tag}_seqlen${seq_len}_on-${AGE_BIN}"
fi
[ -d "$PRED_DIR/eval_scores" ] || { echo "ERROR: no $PRED_DIR/eval_scores (run run_nch_eval.sh first)" >&2; exit 1; }

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

cd "$REPO_DIR" || exit 1
echo "select_alpha: bin=${AGE_BIN} train_tag=${train_tag} seq_len=${seq_len} mode=${mode}"
echo "  pred_dir=${PRED_DIR}"
echo "  index=${INDEX_PATH}"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec \
    -B /srv:/srv "$SIF" \
    python select_alpha_eval.py \
    --age_bin "${AGE_BIN}" \
    --pred_dir "${PRED_DIR}" \
    --seq_len "$seq_len" \
    --index_path "${INDEX_PATH}" \
    --selection_metric jsd \
    --pi_source "$pi_source" \
    --mode "$mode"
STATUS=$?

echo ""
echo "=========================================="
echo "select_alpha finished, exit code ${STATUS}"
echo "=========================================="
exit $STATUS
