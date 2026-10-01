#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=3:00:00
#PBS -q eleceng
#PBS -N SleepTransformer_NCH_HMM_Refine

AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_hmm_refine.sh}"
MODE="${mode:-hmm}"
ALPHA_SOURCE="${alpha_source:-bin}"

REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
TEST_INFERENCE_DIR="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch/run1"
LIST_DIR="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
PRED_DIR="${TEST_INFERENCE_DIR}/predictions/${AGE_BIN}"
 
cd "$REPO_DIR" || exit 1
 
echo "Starting HMM refinement (mode=hmm) for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  pred_dir=${PRED_DIR}  (refined output lands in ${PRED_DIR}/hmmRefined/)"
 
PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python hmm_refine_nch.py \
    --age_bin "${AGE_BIN}" \
    --mode "${MODE}" \
    --pi_source uniform \
    --pred_dir "${PRED_DIR}" \
    --test_inference_dir "${TEST_INFERENCE_DIR}" \
    --alpha_source "${ALPHA_SOURCE}" \
    --list_dir "${LIST_DIR}"
STATUS=$?
 
echo "Finished HMM refinement (mode=hmm): ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: HMM refinement failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
