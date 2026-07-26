#!/bin/bash
#PBS -J 1-5
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=6:00:00
#PBS -q eleceng
#PBS -N SleepTrans_NCH_Inference_Pretrained

# ===== PBS job array maps 1-4 -> age bin =====
AGE_BINS=(1-2y 3-5y 6-12y 13-18y 19-100y)
AGE_BIN=${AGE_BINS[$((PBS_ARRAY_INDEX - 1))]}

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# The raw SHHS-pretrained checkpoint - no finetuning, no per-fold ensemble.
# NOTE: --checkpoint_dir must be the FOLDER containing best_model_acc.* files
# (test_sleeptransformer.py appends "/best_model_acc" internally) - passing
# the file prefix itself here duplicates the path and fails to find the file.
CHECKPOINT="${CODE}"

# NCH file lists - used for BOTH normalization stats (train) and the actual
# test-set inference (test), per age bin
NCH_FILE_LIST="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
NCH_TRAIN_LIST="${NCH_FILE_LIST}/train_list_${AGE_BIN}.txt"
NCH_TEST_LIST="${NCH_FILE_LIST}/test_list_${AGE_BIN}.txt"

OUT_DIR="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference_pretrained/${AGE_BIN}"

CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

mkdir -p "$OUT_DIR"

echo "=== NCH inference (SHHS-pretrained, no finetuning): age bin ${AGE_BIN} ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Test list:  ${NCH_TEST_LIST}"

if [ ! -f "$NCH_TEST_LIST" ]; then
    echo "ERROR: test list not found at $NCH_TEST_LIST"
    exit 1
fi

PYTHONPATH=$PYTHONPATH_DIR \
apptainer exec --nv -B /srv:/srv $CONTAINER \
  bash -c "
    cd $CODE && \
    python test_sleeptransformer.py \
      --eeg_train_data '${NCH_TRAIN_LIST}' \
      --eeg_test_data '${NCH_TEST_LIST}' \
      --eog_train_data '' \
      --eog_test_data '' \
      --emg_train_data '' \
      --emg_test_data '' \
      --checkpoint_dir '${CHECKPOINT}' \
      --out_dir '${OUT_DIR}/' \
      --seq_len 21 \
      --num_blocks 4
  " \
  > "$OUT_DIR/test_log.txt" 2>&1

echo "=== Inference complete for age bin ${AGE_BIN} ==="
