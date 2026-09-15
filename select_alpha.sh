#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=1:00:00
#PBS -q eleceng
#PBS -N EEGMamba_select_alpha

AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_hmm_refine.sh}"
INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}"

cd "$REPO_DIR" || exit 1

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python select_alpha_eval.py \
    --age_bin "${AGE_BIN}" \
    --pred_dir "${PRED_DIR}" \
    --index_path "${INDEX_PATH}"
    --selection_metric jsd

echo ""
echo "=========================================="
echo "Run finished, exit code $?"
echo "=========================================="
