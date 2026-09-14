#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=1:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_HMM_Refine
#
# Paper-faithful HMM refinement (mode=hmm): fixed transition matrix A
# estimated from NCH training-split ground truth, uniform initial
# distribution pi (per the Neuro-Explicit DNN-HMM paper this method is
# based on -- see hmm_refine_nch.py's module docstring), alpha swept over
# a fixed grid at inference time. No model inference needed here beyond
# what test_nch.py already produced -- this reads {rec_id}_probs.npy from
# the finetuned test-split run and decodes.
#
# CPU-only: no --nv, no ngpus request. This script only needs the training
# split's ground-truth labels (via NCHIndexDataset) plus the already-saved
# test-split scores -- no forward pass through the model happens here.
#
# Requires run_nch_finetuned_test.sh to have already been run for this age
# bin (predictions/NCH_<bin>/{rec_id}_probs.npy must exist).
#
AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_hmm_refine.sh}"
INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/tf22_py3.sif"
PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}"

cd "$REPO_DIR" || exit 1

echo "Starting HMM refinement (mode=hmm) for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  pred_dir=${PRED_DIR}  (refined output lands in \${PRED_DIR}/hmmRefined/)"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python hmm_refine_nch.py \
    --age_bin "${AGE_BIN}" \
    --mode hmm \
    --pi_source uniform \
    --pred_dir "${PRED_DIR}" \
    --index_path "${INDEX_PATH}"
STATUS=$?

echo "Finished HMM refinement (mode=hmm): ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: HMM refinement failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
