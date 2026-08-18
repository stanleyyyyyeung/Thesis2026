#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=12:00:00
#PBS -N NCH_EEGMamba_Finetune
#PBS -J 1-5

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 1. Age bin selection ---
bins=("1-2y" "3-5y" "6-12y" "13-18y" "19-100y")
i=${PBS_ARRAY_INDEX}
bin=${bins[$((i-1))]}
echo "=========================================="
echo "NCH EEGMamba finetuning — age bin: ${bin} (array index ${i})"
echo "=========================================="

# --- 2. Paths ---
PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/tf22_py3.sif
DATASETS_DIR=$PROJECT_ROOT/nch_index/nch_index_nch_v2.parquet
MODEL_DIR="$PROJECT_ROOT/EEGMamba/model_weights/NCH_${bin}"

PYTHONPATH_FULL=$EEGMAMBA_DIR:/srv/scratch/z5423210/python_packages

cd "$EEGMAMBA_DIR"

echo "Overwriting previous checkpoint in $MODEL_DIR"
mkdir -p "$MODEL_DIR"
rm -f "$MODEL_DIR"/*.pth

export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

# --- 3. Run ---
PYTHONPATH=$PYTHONPATH_FULL \
apptainer exec --nv -B /srv:/srv "$SIF" \
    python3 finetune_main.py \
    --downstream_dataset NCH \
    --datasets_dir "$DATASETS_DIR" \
    --age_bin "$bin" \
    --num_of_classes 5 \
    --model_dir "$MODEL_DIR" \
    --cuda 0 \
    --epochs 50 \
    --num_workers 8

exit_code=$?
echo ""
echo "=========================================="
echo "Age bin ${bin} finished, exit code ${exit_code}"
echo "=========================================="
