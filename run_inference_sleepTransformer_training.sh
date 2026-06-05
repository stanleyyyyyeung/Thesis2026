#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=5:00:00
#PBS -q eleceng
#PBS -N LSeq_Inference

# ===== EDIT THIS BEFORE EACH RUN =====
RUN_NUMBER=3

# ===== PATHS =====
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
FILE_LIST="${CODE}"
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-20/run${RUN_NUMBER}/training_scores"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

echo "Running inference for run${RUN_NUMBER}..."

# Change depending on what dataset is using (20 for sleepedf-20 but 10 for sleepedf-78)
for i in {1..20}
do
  FINETUNED="/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-20/run${RUN_NUMBER}/out/n${i}/checkpoint"
  OUT_DIR="$OUT_BASE/n${i}"
  mkdir -p "$OUT_DIR"

  echo "Starting inference n${i}..."

  PYTHONPATH=$PYTHONPATH_DIR \
  apptainer exec --nv -B /srv:/srv $CONTAINER \
    bash -c "
      cd $SCRIPT_DIR && \
      python test_sleeptransformer.py \
        --eeg_train_data '${FILE_LIST}/train_list_n${i}.txt' \
        --eeg_test_data '${FILE_LIST}/train_list_n${i}.txt' \
        --eog_train_data '' \
        --eog_test_data '' \
        --emg_train_data '' \
        --emg_test_data '' \
        --out_dir '${OUT_DIR}/' \
        --seq_len 21 \
        --num_blocks 4
    " \
    > "$OUT_DIR/test_log.txt" 2>&1

  echo "Finished n${i}"
done

echo "All inference complete for run${RUN_NUMBER}."
