#!/bin/bash
#PBS -J 1-5
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=4:00:00
#PBS -q eleceng
#PBS -N SleepTrans_NCH_Inference_Eval

# ===== PBS job array maps 1-5 -> age bin =====
AGE_BINS=(1-2y 3-5y 6-12y 13-18y 19-100y)
AGE_BIN=${AGE_BINS[$((PBS_ARRAY_INDEX - 1))]}

RUN_NUMBER=1

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# Same raw SHHS-pretrained checkpoint folder as the TEST inference run
# (must be the FOLDER containing best_model_acc.*).
CHECKPOINT="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch/run${RUN_NUMBER}/${AGE_BIN}/checkpoint"

NCH_FILE_LIST="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
# IMPORTANT: the TRAIN list stays the train list -- test_sleeptransformer.py
# computes normalisation mean/std from it and applies them to the test data.
# Using the eval list there would normalise with the wrong statistics and the
# eval scores would no longer be comparable to the test scores.
NCH_TRAIN_LIST="${NCH_FILE_LIST}/train_list_${AGE_BIN}.txt"
NCH_EVAL_LIST="${NCH_FILE_LIST}/eval_list_${AGE_BIN}.txt"

# Must match EVAL_SCORE_BASE_NCH in select_alpha_eval.py
OUT_DIR="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference_pretrained_eval/${AGE_BIN}"

CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

mkdir -p "$OUT_DIR"

echo "=== NCH EVAL inference (SHHS-pretrained, no finetuning): age bin ${AGE_BIN} ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Train list (normalisation only): ${NCH_TRAIN_LIST}"
echo "Eval list:  ${NCH_EVAL_LIST}"

for f in "$NCH_TRAIN_LIST" "$NCH_EVAL_LIST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: list not found at $f" >&2
        exit 1
    fi
done

PYTHONPATH=$PYTHONPATH_DIR \
apptainer exec --nv -B /srv:/srv $CONTAINER \
  bash -c "
    cd $CODE && \
    python test_sleeptransformer.py \
      --eeg_train_data '${NCH_TRAIN_LIST}' \
      --eeg_test_data '${NCH_EVAL_LIST}' \
      --eog_train_data '' \
      --eog_test_data '' \
      --emg_train_data '' \
      --emg_test_data '' \
      --checkpoint_dir '${CHECKPOINT}' \
      --out_dir '${OUT_DIR}/' \
      --seq_len 21 \
      --num_blocks 4
  " \
  > "$OUT_DIR/eval_inference_log.txt" 2>&1

if [ ! -f "$OUT_DIR/test_ret.mat" ]; then
    echo "ERROR: no test_ret.mat for age bin ${AGE_BIN} (see $OUT_DIR/eval_inference_log.txt)" >&2
    exit 1
fi

echo "=== Eval inference complete for age bin ${AGE_BIN} ==="
