#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=5:00:00
#PBS -q eleceng
#PBS -N SleepTrans_Inference

# ===== EDIT THIS BEFORE EACH RUN =====
RUN_NUMBER=1

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
FILE_LIST="${CODE}/sleepedf-78/file_list_30min/eeg"
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-78/run${RUN_NUMBER}/out"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

echo "Running inference for run${RUN_NUMBER}..."

for i in {1..10}
do
  OUT_DIR="$OUT_BASE/n${i}"
  mkdir -p "$OUT_DIR"

  echo "Starting inference n${i}..."

  PYTHONPATH=$PYTHONPATH_DIR \
  apptainer exec --nv -B /srv:/srv $CONTAINER \
    bash -c "
      cd $CODE && \
      python test_sleeptransformer.py \
        --eeg_train_data '${FILE_LIST}/train_list_n${i}.txt' \
        --eeg_test_data '${FILE_LIST}/test_list_n${i}.txt' \
        --eog_train_data '' \
        --eog_test_data '' \
        --emg_train_data '' \
        --emg_test_data '' \
        --out_dir '${OUT_DIR}/' \
        --checkpoint_dir '${CODE}/sleepedf-78/' \
        --seq_len 21 \
        --num_blocks 4
    " \
    > "$OUT_DIR/test_log.txt" 2>&1

  echo "Finished n${i}"
done

echo "All inference complete for run${RUN_NUMBER}."
