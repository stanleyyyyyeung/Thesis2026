#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=3:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_Finetuned_Train
#
# Generates the model's own softmax scores on the TRAINING split, needed
# only for hmm_trained mode (MMI training of the transition matrix in
# hmm_refine_nch.py). NOT needed for mode="hmm" (the paper-faithful arm,
# which only needs training-split ground truth, not model scores on it)
# -- skip this script until hmm_trained is actually being run.
#
# Longer walltime than the test-split job since the training split is
# larger; adjust per age bin if a given bin's train split is unusually big
# (see katana_command_reference.md's note on n3's 95,600-step run needing
# its own dedicated job -- the same imbalance can show up here).
#
# Pooled over 2 bins (trained with: qsub -v bins=6-12y+13-18y run_nch.sh):
#   train_tag=pooled-6-12y-13-18y
#   qsub -v bin=6-12y,train_tag=pooled-6-12y-13-18y,seq_len=20 run_nch_test.sh
#   -> loads NCH_pooled-6-12y-13-18y_seqlen20, writes
#      predictions/NCH_pooled-6-12y-13-18y_seqlen20_on-6-12y
#   (test bin must be one of the two trained bins, e.g. 6-12y or 13-18y)
#
# Pooled over all 5 bins (trained with: qsub -v bins=all run_nch.sh):
#   train_tag=pooled-all
#   qsub -v bin=6-12y,train_tag=pooled-all,seq_len=20 run_nch_test.sh
#   -> loads NCH_pooled-all_seqlen20, writes
#      predictions/NCH_pooled-all_seqlen20_on-6-12y
#   (repeat with bin=1-2y, 3-5y, 13-18y, 19-100y for the other groups)
#

AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_finetuned_eval.sh}"
seq_len="${seq_len:-20}"
train_tag="${train_tag:-$AGE_BIN}" 

REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"

if [ "$train_tag" = "$AGE_BIN" ]; then
    # per-bin model: keep the legacy no-suffix convention at seq_len 20
    if [ "$seq_len" -eq 20 ]; then SUFFIX=""; else SUFFIX="_seqlen${seq_len}"; fi
    MODEL_DIR="$REPO_DIR/model_weights/NCH_${AGE_BIN}${SUFFIX}"
    PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}${SUFFIX}"
else
    # pooled model: always explicit suffix (matches run_nch.sh)
    MODEL_DIR="$REPO_DIR/model_weights/NCH_${train_tag}_seqlen${seq_len}"
    PRED_DIR="$REPO_DIR/predictions/NCH_${train_tag}_seqlen${seq_len}_on-${AGE_BIN}"
fi
[ -d "$MODEL_DIR" ] || { echo "ERROR: missing $MODEL_DIR" >&2; exit 1; }

IDX_DIR="/srv/scratch/z5423210/StanleyThesis2026/nch_index"
if [ "$seq_len" -eq 20 ]; then
    if [ -f "$IDX_DIR/nch_index_nch_v2_seqlen20.parquet" ]; then
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen20.parquet"
    else
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2.parquet"      # legacy, built before the suffix existed
    fi
else
    INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen${seq_len}.parquet"
fi
[ -f "$INDEX_PATH" ] || { echo "ERROR: missing $INDEX_PATH" >&2; exit 1; }

cd "$REPO_DIR" || exit 1

echo "Starting FINETUNED train-split score extraction for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  model_dir=${MODEL_DIR}"
echo "  pred_dir=${PRED_DIR}  (train-split scores land in \${PRED_DIR}/training_scores/)"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python test_nch.py \
    --age_bin "${AGE_BIN}" \
    --seq_len "$seq_len" \
    --index_path "${INDEX_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --pred_dir "${PRED_DIR}" \
    --split val \
    --cuda 0
STATUS=$?

echo "Finished FINETUNED train-split extraction: ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: finetuned train-split extraction failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
