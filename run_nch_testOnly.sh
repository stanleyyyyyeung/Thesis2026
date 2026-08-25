#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=0:30:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_ZeroShot

# Zero-shot generalisation check: run the ISRUC-pretrained (NOT NCH-finetuned)
# checkpoint against each NCH age bin's test split, to see how well the
# adult-sleep-trained model's learned transition dynamics transfer to an
# unseen paediatric population.
#
AGE_BIN="${bin:?Must pass age bin, e.g. qsub -v bin=1-2y run_nch_zeroshot.sh}"

INDEX_PATH="/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
SIF="/srv/scratch/z5423210/tf22_py3.sif"

MODEL_DIR="$REPO_DIR/model_weights/ISRUC_full"

OUT_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}/inference_only"

cd "$REPO_DIR" || exit 1

echo "Starting ZERO-SHOT extraction for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  model_dir=${MODEL_DIR}"
echo "  out_dir=${OUT_DIR}"

PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec --nv -B /srv:/srv "$SIF" \
    python test_nch.py \
    --age_bin "${AGE_BIN}" \
    --index_path "${INDEX_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --out_dir "${OUT_DIR}" \
    --cuda 0

STATUS=$?
echo "Finished ZERO-SHOT age bin: ${AGE_BIN}, exit code ${STATUS}"

if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: zero-shot extraction failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
