#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:gpu_model=A100
#PBS -l walltime=01:00:00
#PBS -N ISRUC_Test_Inference
# Set this to match the run number used for the corresponding finetuning job
RUN_NUMBER=1
# --- 1. Cleanup old logs ---
rm -f /srv/scratch/z5423210/StanleyThesis2026/ISRUC_Test_Inference.[oe]*
# --- 2. Apptainer environment ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp
export APPTAINERENV_LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH}"
# --- 3. Paths ---
PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/tf22_py3.sif
PYTHONPATH_FULL=$EEGMAMBA_DIR:/srv/scratch/z5423210/python_packages
cd "$EEGMAMBA_DIR"
# --- 4. Run inference ---
PYTHONPATH=$PYTHONPATH_FULL \
apptainer exec --nv -B /srv:/srv "$SIF" \
    python3 test_isruc.py \
    --cuda 0 \
    --run_number "$RUN_NUMBER" \
    --model_dir "$EEGMAMBA_DIR/model_weights/ISRUC_full"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Inference FAILED (exit code $EXIT_CODE)."
    exit 1
fi
echo "Inference complete. Predictions saved to /srv/scratch/z5423210/StanleyThesis2026/out_eegmamba/isruc/run${RUN_NUMBER}/out/predictions/"
