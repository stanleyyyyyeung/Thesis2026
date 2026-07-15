#!/bin/bash
#PBS -J 1-4
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N SleepTrans_NCH_Inference

# ===== PBS job array maps 1-4 -> age bin =====
# Index:   1       2       3        4
# Bin:     1-2y    3-5y    6-12y    13-18y
AGE_BINS=(1-2y 3-5y 6-12y 13-18y)
AGE_BIN=${AGE_BINS[$((PBS_ARRAY_INDEX - 1))]}

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# SleepEDF-78 file lists (used ONLY for --eeg_train_data, i.e. normalization stats
# matching the fold's own finetuning data — NOT the NCH train data)
SLEEPEDF78_FILE_LIST="${CODE}/sleepedf-78/file_list_30min/eeg"

# NCH age-bin-specific test list (the actual external data being inferred on)
NCH_FILE_LIST="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
NCH_TEST_LIST="${NCH_FILE_LIST}/test_list_${AGE_BIN}.txt"

# Where each fold's finetuned SleepEDF-78 checkpoint already lives
CKPT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-78/run1/out"

# Where NCH inference outputs go, organised by age bin then fold
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference/${AGE_BIN}"

CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

echo "=== NCH ensemble inference: age bin ${AGE_BIN} (array index ${PBS_ARRAY_INDEX}) ==="
echo "Test list: ${NCH_TEST_LIST}"

if [ ! -f "$NCH_TEST_LIST" ]; then
    echo "ERROR: test list not found at $NCH_TEST_LIST"
    exit 1
fi

# ===== Loop through all 10 SleepEDF-78 finetuned folds for the ensemble =====
for i in {1..10}
do
    FOLD_CKPT_DIR="${CKPT_BASE}/n${i}"
    OUT_DIR="${OUT_BASE}/n${i}"
    mkdir -p "$OUT_DIR"

    echo "  [${AGE_BIN}] Starting inference with fold n${i} checkpoint..."

    PYTHONPATH=$PYTHONPATH_DIR \
    apptainer exec --nv -B /srv:/srv $CONTAINER \
      bash -c "
        cd $CODE && \
        python test_sleeptransformer.py \
          --eeg_train_data '${SLEEPEDF78_FILE_LIST}/train_list_n${i}.txt' \
          --eeg_test_data '${NCH_TEST_LIST}' \
          --eog_train_data '' \
          --eog_test_data '' \
          --emg_train_data '' \
          --emg_test_data '' \
          --out_dir '${FOLD_CKPT_DIR}/' \
          --seq_len 21 \
          --num_blocks 4
      " \
      > "$OUT_DIR/test_log.txt" 2>&1

    echo "  [${AGE_BIN}] Finished fold n${i}"
done

echo "=== All 10 folds complete for age bin ${AGE_BIN} ==="
