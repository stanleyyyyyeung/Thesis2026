#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=1:00:00
#PBS -q eleceng
#PBS -N ST_Select_Alpha_Eval

# Usage:
#   qsub -v dataset=sleepedf-78 run_select_alpha_eval.sh                   # all folds, metric=jsd
#   qsub -v dataset=sleepedf-20,metric=accuracy run_select_alpha_eval.sh
#   qsub -v dataset=nch,unit=6-12y run_select_alpha_eval.sh                # one age bin
#
DATASET="${dataset:?Must pass dataset: nch | sleepedf-20 | sleepedf-78}"
UNIT="${unit:-all}"
METRIC="${metric:-jsd}"
RUN="${run:-1}"
 
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
 
# List files contain relative paths (../../mat_30min/...) for SleepEDF: run from
# the same working directory as the refinement runs. Override with -v workdir=...
WORKDIR="${workdir:-$REPO_DIR}"
cd "$WORKDIR" || exit 1
 
echo "Starting alpha selection: dataset=${DATASET} unit=${UNIT} metric=${METRIC} run=${RUN} (job ${PBS_JOBID})"
 
PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python "$REPO_DIR/select_alpha_eval.py" \
    --dataset "${DATASET}" \
    --unit "${UNIT}" \
    --run "${RUN}" \
    --selection_metric "${METRIC}"
STATUS=$?
 
echo "Finished alpha selection: ${DATASET}/${UNIT}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: alpha selection failed (exit ${STATUS})" >&2
    exit "$STATUS"
fi
