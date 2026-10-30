#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=00:30:00
#PBS -N ISRUC_ClassDist

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# Usage: qsub -v datasets_dir=/path/to/ISRUC/processed run_class_dist_isruc.sh
#        (add label_offset=1 only if labels turn out to be 1-based)
datasets_dir="/srv/scratch/speechdata/sleep_data/ISRUC"
label_offset="${label_offset:-0}"
if [ -z "$datasets_dir" ]; then echo "ERROR: pass datasets_dir=<path>" >&2; exit 1; fi

PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
SIF=/srv/scratch/z5423210/pytorch_cu128.sif
OUT_DIR=$PROJECT_ROOT/analysis/class_dist/ISRUC

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT/EEGMamba"

PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec -B /srv:/srv "$SIF" \
    python3 class_distribution_isruc.py \
    --datasets_dir "$datasets_dir" \
    --label_offset "$label_offset" \
    --out_dir "$OUT_DIR"

exit $?
