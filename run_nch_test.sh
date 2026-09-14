#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=02:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_Finetuned_Test
#
# Main arm: run the NCH-FINETUNED checkpoint for this age bin over its own
# test split. Produces {rec_id}_ypred.npy / _ytrue.npy / _probs.npy flat
# under predictions/NCH_<bin>/ -- this is the "raw" (pre-HMM) result the
# thesis's transition-dynamics analysis and the HMM refinement both build
# on. Same shape as run_nch_zeroshot.sh, pointed at the finetuned checkpoint
# instead of the ISRUC-pretrained one, and with no inference_only subdir
# since this IS the primary predictions directory for this age bin.
#
AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_finetuned_test.sh}"
INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
MODEL_DIR="$REPO_DIR/model_weights/NCH_${AGE_BIN}"
PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}"

cd "$REPO_DIR" || exit 1

echo "Starting FINETUNED test-split extraction for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  model_dir=${MODEL_DIR}"
echo "  pred_dir=${PRED_DIR}"

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
    --split test \
    --cuda 0
STATUS=$?

echo "Finished FINETUNED test-split extraction: ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: finetuned test-split extraction failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
