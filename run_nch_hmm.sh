#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=6:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_HMM_Refine
#
# Paper-faithful HMM refinement (mode=hmm): fixed transition matrix A
# estimated from NCH training-split ground truth, uniform initial
# distribution pi (per the Neuro-Explicit DNN-HMM paper this method is
# based on -- see hmm_refine_nch.py's module docstring), alpha swept over
# a fixed grid at inference time. No model inference needed here beyond
# what test_nch.py already produced -- this reads {rec_id}_probs.npy from
# the finetuned test-split run and decodes.
#
# CPU-only: no --nv, no ngpus request. This script only needs the training
# split's ground-truth labels (via NCHIndexDataset) plus the already-saved
# test-split scores -- no forward pass through the model happens here.
#
# Requires run_nch_test.sh (new version) to have been run for this arm first.
#
#   qsub -v bin=1-2y run_nch_hmm.sh                                  # baseline folders
#   qsub -v bin=6-12y,train_tag=pooled-6-12y-13-18y run_nch_hmm.sh   # pooled model
#   qsub -v bin=1-2y,exp=ms_g0,seed=3407 run_nch_hmm.sh              # joint-trained arm
#
# data_source (default auto) selects where the TRAINING labels for the prior come from;
# auto follows the source recorded in <PRED_DIR>/run_meta.json so the prior and the
# test windows are trimmed the same way.
#
 
AGE_BIN="${bin:?Must pass test age bin}"
seq_len="${seq_len:-20}"
train_tag="${train_tag:-$AGE_BIN}"     # same value you used for run_nch_test.sh
exp="${exp:-}"
seed="${seed:-3407}"
DATA_SOURCE="${data_source:-auto}"
 
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"
REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
 
if [ -n "$exp" ]; then
    PRED_DIR="$REPO_DIR/predictions_joint/NCH_${train_tag}_seqlen${seq_len}/${exp}_seed${seed}_on-${AGE_BIN}"
elif [ "$train_tag" = "$AGE_BIN" ]; then
    if [ "$seq_len" -eq 20 ]; then SUFFIX=""; else SUFFIX="_seqlen${seq_len}"; fi
    PRED_DIR="$REPO_DIR/predictions/NCH_${AGE_BIN}${SUFFIX}"
else
    PRED_DIR="$REPO_DIR/predictions/NCH_${train_tag}_seqlen${seq_len}_on-${AGE_BIN}"
fi
ls "$PRED_DIR"/[0-9]*_[0-9]*_probs.npy >/dev/null 2>&1 || {
    echo "ERROR: no <subject>_<study>_probs.npy in $PRED_DIR (run the new test_nch.py first)" >&2; exit 1; }
 
IDX_DIR="/srv/scratch/z5423210/StanleyThesis2026/nch_index"
if [ "$seq_len" -eq 20 ]; then
    if [ -f "$IDX_DIR/nch_index_nch_v2_seqlen20.parquet" ]; then
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen20.parquet"
    else
        INDEX_PATH="$IDX_DIR/nch_index_nch_v2.parquet"   # legacy
    fi
else
    INDEX_PATH="$IDX_DIR/nch_index_nch_v2_seqlen${seq_len}.parquet"
fi
[ -f "$INDEX_PATH" ] || { echo "ERROR: missing $INDEX_PATH" >&2; exit 1; }
 
cd "$REPO_DIR" || exit 1
 
echo "Starting HMM refinement (mode=hmm) for age bin: ${AGE_BIN} (job ${PBS_JOBID})"
echo "  pred_dir=${PRED_DIR}  (refined output lands in \${PRED_DIR}/hmmRefined/)"
echo "  data_source=${DATA_SOURCE} exp=${exp:-none} seed=${seed}"
 
PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec -B /srv:/srv "$SIF" \
    python hmm_refine_nch.py \
    --age_bin "${AGE_BIN}" \
    --seq_len "$seq_len" \
    --mode hmm \
    --pi_source uniform \
    --data_source "${DATA_SOURCE}" \
    --pred_dir "${PRED_DIR}" \
    --index_path "${INDEX_PATH}"
STATUS=$?
 
echo "Finished HMM refinement (mode=hmm): ${AGE_BIN}, exit code ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: HMM refinement failed for age bin ${AGE_BIN} (exit ${STATUS})" >&2
    exit "$STATUS"
fi
