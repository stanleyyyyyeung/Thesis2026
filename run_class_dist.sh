#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=01:00:00
#PBS -N NCH_ClassDist

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# Usage: qsub -v bins=all run_class_dist.sh
#        qsub -v bins=6-12y,13-18y run_class_dist.sh
bins="${bins:-all}"

PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/pytorch_cu128.sif
OUT_DIR=$PROJECT_ROOT/analysis/class_dist/${bins//,/+}

mkdir -p "$OUT_DIR"
cd "$EEGMAMBA_DIR"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec -B /srv:/srv "$SIF" \
    python3 class_distribution.py \
    --age_bins "${bins//+/,}" \
    --out_dir "$OUT_DIR" \
    --workers 8

exit $?
