#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N SleepTrans_NCH_Finetune
#PBS -J 1-4

# --- 1. Environment ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 2. Age bin and run index ---
bins=("1-2y" "3-5y" "6-12y" "13-18y" "19-100y")
i=${PBS_ARRAY_INDEX}
bin=${bins[$((i-1))]}
echo "Running age bin: ${bin}"

runIndex=1
echo "Run ${runIndex}"

# --- 3. Paths ---
CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# TODO: fill in the actual path to the SHHS-pretrained checkpoint before
# submitting - this is NOT the same file the SleepEDF-78 script pointed at
PRETRAINED="${CODE}/best_model_acc"

FILE_LIST="/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch/run${runIndex}"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"

cd ${CODE}

# --- 4. Run ---
PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec --nv -B /srv:/srv $CONTAINER \
  bash -c "
    cd ${FILE_LIST} && \
      python ${CODE}/finetune_sleeptransformer.py \
      --eeg_train_data '${FILE_LIST}/train_list_${bin}.txt' \
      --eeg_eval_data '${FILE_LIST}/eval_list_${bin}.txt' \
      --eog_train_data '' \
      --eog_eval_data '' \
      --emg_train_data '' \
      --emg_eval_data '' \
      --pretrained_model '${PRETRAINED}' \
      --out_dir '${OUT_BASE}/${bin}' \
      --seq_len 21 \
      --num_blocks 4 \
      --early_stopping True
  "
