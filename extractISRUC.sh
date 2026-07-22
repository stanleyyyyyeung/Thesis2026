#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=08:00:00
#PBS -N ISRUC_Extraction

# CPU-only job: this extraction script never imports torch or touches the
# GPU (mne filtering/reshaping is CPU-bound), so no ngpus request and no
# --nv flag on apptainer exec below.

# --- 1. Cleanup old logs ---
rm -f /srv/scratch/z5423210/StanleyThesis2026/ISRUC_Extraction.[oe]*

# --- 2. Apptainer environment (matches existing project convention) ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 3. Paths ---
PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
ISRUC_MODULE_DIR=$EEGMAMBA_DIR/preprocessing/preprocessing_for_finetuning/ISRUC
SIF=/srv/scratch/z5423210/tf22_py3.sif

PYTHONPATH_FULL=$EEGMAMBA_DIR:$ISRUC_MODULE_DIR:/srv/scratch/z5423210/python_packages

cd "$EEGMAMBA_DIR"

# --- 4. Smoke test first ---
echo "=========================================="
echo "STEP 1: Smoke test (subjects 1, 8, 25, 40)"
echo "=========================================="
PYTHONPATH=$PYTHONPATH_FULL \
apptainer exec -B /srv:/srv "$SIF" \
    python3 prepareISRUC_test.py

SMOKE_EXIT=$?

if [ $SMOKE_EXIT -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "SMOKE TEST FAILED (exit code $SMOKE_EXIT) - ABORTING."
    echo "Full extraction will NOT run. Check the log above before retrying."
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Smoke test passed. Proceeding to full extraction."
echo "=========================================="

# --- 5. Full extraction (only runs if smoke test succeeded) ---
echo ""
echo "=========================================="
echo "STEP 2: Full extraction (all 100 subjects, minus excluded subject 8)"
echo "=========================================="
PYTHONPATH=$PYTHONPATH_FULL \
apptainer exec -B /srv:/srv "$SIF" \
    python3 prepareISRUC.py

FULL_EXIT=$?

if [ $FULL_EXIT -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "FULL EXTRACTION REPORTED FAILURES (exit code $FULL_EXIT)."
    echo "Check the [FAILED] lines above for which subjects failed and why."
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "All done. Both smoke test and full extraction completed successfully."
echo "=========================================="
