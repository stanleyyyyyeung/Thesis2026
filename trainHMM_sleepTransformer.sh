#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=05:00:00
#PBS -q eleceng
#PBS -N HMM_Trained

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec --nv -B /srv:/srv /srv/scratch/z5423210/tf22_py3.sif \
    python /srv/scratch/z5423210/StanleyThesis2026/extract_predictions_sleepTransformer.py
