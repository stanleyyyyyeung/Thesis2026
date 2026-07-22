#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -N ISRUC_Sanity

rm -f /srv/scratch/z5423210/StanleyThesis2026/ISRUC_Sanity.[oe]*

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/tf22_py3.sif
DATASETS_DIR=/srv/scratch/speechdata/sleep_data/ISRUC

PYTHONPATH_FULL=$EEGMAMBA_DIR:/srv/scratch/z5423210/python_packages

cd "$EEGMAMBA_DIR"

echo "=========================================="
echo "ISRUC sanity run: 2 epochs"
echo "=========================================="

export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

PYTHONPATH=$PYTHONPATH_FULL \
apptainer exec --nv -B /srv:/srv "$SIF" \
    python3 finetune_main.py \
    --downstream_dataset ISRUC \
    --datasets_dir "$DATASETS_DIR" \
    --num_of_classes 5 \
    --model_dir "$PROJECT_ROOT/EEGMamba/model_weights/ISRUC_sanity" \
    --cuda 0 \
    --epochs 2 \
    --num_workers 8

echo ""
echo "=========================================="
echo "Sanity run finished, exit code $?"
echo "=========================================="
