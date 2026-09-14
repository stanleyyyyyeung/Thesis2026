#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=30:00:00
#PBS -N NCH_EEGMamba_Finetune

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 1. Age bin selection ("1-2y" "3-5y" "6-12y" "13-18y" "19-100y")---
bin="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y NCH_EEGMamba_Finetune.sh}"
echo "=========================================="
echo "NCH EEGMamba finetuning — age bin: ${bin}"
echo "=========================================="

# -- Learning rate for different modules --
#   qsub -v bin=6-12y,seq_lr_mult=1.0 run_nch.sh
seq_lr_mult="${seq_lr_mult:-2.0}"
head_lr_mult="${head_lr_mult:-5.0}"

# --- 2. Paths ---
PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/pytorch_cu128.sif
DATASETS_DIR=$PROJECT_ROOT/nch_index/nch_index_nch_v2.parquet
MODEL_DIR="$PROJECT_ROOT/EEGMamba/model_weights/NCH_${bin}"

PYTHONPATH_FULL=$EEGMAMBA_DIR:/srv/scratch/z5423210/python_packages

cd "$EEGMAMBA_DIR"

echo "Overwriting previous checkpoint in $MODEL_DIR"
mkdir -p "$MODEL_DIR"
rm -f "$MODEL_DIR"/*.pth

export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

# --- 3. Run ---
PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python3 finetune_main.py \
    --downstream_dataset NCH \
    --datasets_dir "$DATASETS_DIR" \
    --age_bin "$bin" \
    --num_of_classes 5 \
    --model_dir "$MODEL_DIR" \
    --cuda 0 \
    --epochs 50 \
    --frozen False \
    --seq_lr_mult "$seq_lr_mult" \
    --head_lr_mult "$head_lr_mult" \
    --num_workers 4

exit_code=$?
echo ""
echo "=========================================="
echo "Age bin ${bin} finished, exit code ${exit_code}"
echo "=========================================="
