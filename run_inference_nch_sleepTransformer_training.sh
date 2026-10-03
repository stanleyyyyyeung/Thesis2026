#!/bin/bash
#PBS -J 1-5
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N SleepTrans_NCH_Inference_Train

# Training-split score generation for hmm_trained (MMI) on NCH.
# Same raw SHHS-pretrained checkpoint as the NCH TEST run; the only change is
# that the "test" list is the TRAIN list. Uses the shared test_sleeptransformer.py.
#
# NOTE: the train split is much larger than the test split, so this takes far
# longer than the test run. If a bin times out, resubmit just that bin, e.g.
#   qsub -J 3-3 run_inference_sleepTransformer_nch_training.sh

# ===== PBS job array maps 1-5 -> age bin =====
AGE_BINS=(1-2y 3-5y 6-12y 13-18y 19-100y)
AGE_BIN=${AGE_BINS[$((PBS_ARRAY_INDEX - 1))]}

RUN_NUMBER=1

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# FOLDER containing best_model_acc.* (test_sleeptransformer.py appends "/best_model_acc")
CHECKPOINT="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch/run${RUN_NUMBER}/${AGE_BIN}/checkpoint"

NCH_FILE_LIST="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
NCH_TRAIN_LIST="${NCH_FILE_LIST}/train_list_${AGE_BIN}.txt"

# Must match DEFAULT_TRAIN_INFERENCE_BASE in hmm_refine_nch.py
OUT_DIR="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch/run${RUN_NUMBER}/training_scores/${AGE_BIN}"

CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

mkdir -p "$OUT_DIR"

echo "=== NCH TRAIN-split inference (SHHS-pretrained, no finetuning): age bin ${AGE_BIN} ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Train list (normalisation AND inference): ${NCH_TRAIN_LIST}"

if [ ! -f "$NCH_TRAIN_LIST" ]; then
    echo "ERROR: train list not found at $NCH_TRAIN_LIST" >&2
    exit 1
fi

PYTHONPATH=$PYTHONPATH_DIR \
apptainer exec --nv -B /srv:/srv $CONTAINER \
  bash -c "
    cd $CODE && \
    python test_sleeptransformer.py \
      --eeg_train_data '${NCH_TRAIN_LIST}' \
      --eeg_test_data '${NCH_TRAIN_LIST}' \
      --eog_train_data '' \
      --eog_test_data '' \
      --emg_train_data '' \
      --emg_test_data '' \
      --checkpoint_dir '${CHECKPOINT}' \
      --out_dir '${OUT_DIR}/' \
      --seq_len 21 \
      --num_blocks 4
  " \
  > "$OUT_DIR/train_inference_log.txt" 2>&1

if [ ! -f "$OUT_DIR/test_ret.mat" ]; then
    echo "ERROR: no test_ret.mat for age bin ${AGE_BIN} (see $OUT_DIR/train_inference_log.txt)" >&2
    exit 1
fi

echo "=== Train-split inference complete for age bin ${AGE_BIN} ==="
