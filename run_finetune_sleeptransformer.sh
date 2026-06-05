#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N SleepTrans_Finetune
#PBS -J 1-20%5

# --- 1. Environment ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 2. Subject and run index ---
i=${PBS_ARRAY_INDEX}
echo "Running subject n${i}"

runIndex=1
echo "Run ${runIndex}"

# --- 3. Paths ---
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
PRETRAINED="${CODE}/best_model_acc"
FILE_LIST="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-20/run${runIndex}/out"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"

cd ${CODE}

# --- 4. Run ---
PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec --nv -B /srv:/srv $CONTAINER \
  bash -c "
    cd $CODE && \
    python finetune_sleeptransformer.py \
      --eeg_train_data '${FILE_LIST}/train_list_n${i}.txt' \
      --eeg_eval_data '${FILE_LIST}/eval_list_n${i}.txt' \
      --eog_train_data '' \
      --eog_eval_data '' \
      --emg_train_data '' \
      --emg_eval_data '' \
      --pretrained_model '$PRETRAINED' \
      --out_dir '${OUT_BASE}/n${i}' \
      --seq_len 21 \
      --num_blocks 4 \
      --early_stopping True
  "
