#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=02:00:00
#PBS -q eleceng
#PBS -N ST_SleepEDF_HMM_Refine

DATASET="${dataset:?Must pass dataset, e.g. qsub -v dataset=sleepedf-78 run_sleepedf_hmm_refine.sh}"
MODE="${mode:-hmm}"
FOLD="${fold:-all}"
RUN="${run:-1}"
ALPHA_INIT="${alpha_init:-0.7}"
 
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
 
# IMPORTANT: list files contain relative paths (../../mat_30min/...). Run from
# the SAME working directory you used for the previous SleepEDF HMM scripts so
# these resolve. Set WORKDIR accordingly if it differs from REPO_DIR.
WORKDIR="${workdir:-$REPO_DIR}"
 
cd "$WORKDIR" || exit 1
 
echo "Starting HMM refinement: dataset=${DATASET} mode=${MODE} fold=${FOLD} run=${RUN} (job ${PBS_JOBID})"
 
PYTHONPATH=/srv/scratch/z5423210/python_packages:$REPO_DIR \
apptainer exec -B /srv:/srv "$SIF" \
    python "$REPO_DIR/hmm_refine_sleepedf.py" \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --fold "${FOLD}" \
    --run "${RUN}" \
    --pi_source uniform \
    --alpha_init "${ALPHA_INIT}"
STATUS=$?
 
echo "Finished HMM refinement: ${DATASET}/${MODE}/${FOLD}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: HMM refinement failed (exit ${STATUS})" >&2
    exit "$STATUS"
fi
 
