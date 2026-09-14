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
AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_finetuned_train.sh}"
INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
MODEL_DIR="$REPO_DIR/model_weights/NCH_${AGE_BIN}"
PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}"

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
    --index_path "${INDEX_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --pred_dir "${PRED_DIR}" \
    --split train \
    --cuda 0
STATUS=$?

echo "Finished FINETUNED train-split extraction: ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: finetuned train-split extraction failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
