#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=5:00:00
#PBS -q eleceng
#PBS -N LSeq_Inference

# ===== EDIT THIS BEFORE EACH RUN =====
RUN_NUMBER=1

# ===== PATHS =====
BASE_DIR="/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20"
SCRIPT_DIR="$BASE_DIR/network/lseqsleepnet"
OUT_BASE="/srv/scratch/z5423210/StanleyThesis2026/prev_runs/run${RUN_NUMBER}/out"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

echo "Running inference for run${RUN_NUMBER}..."

for i in {1..20}
do
  FINETUNED="/srv/scratch/z5423210/StanleyThesis2026/prev_runs/run${RUN_NUMBER}/out/n${i}/checkpoint"
  OUT_DIR="$OUT_BASE/n${i}"
  mkdir -p "$OUT_DIR"

  echo "Starting inference n${i}..."

  PYTHONPATH=$PYTHONPATH_DIR \
  apptainer exec --nv -B /srv:/srv $CONTAINER \
    bash -c "
      cd $SCRIPT_DIR && \
      python test_lseqsleepnet.py \
        --eeg_train_data '../../file_list_20sub/eeg/train_list_n${i}.txt' \
        --eeg_test_data '../../file_list_20sub/eeg/test_list_n${i}.txt' \
        --out_dir '$OUT_DIR/' \
        --checkpoint_dir '$FINETUNED' \
        --dropout_rnn 0.9 \
        --sub_seq_len 10 \
        --nfilter 32 \
        --nhidden1 64 \
        --nhidden2 64 \
        --attention_size 64 \
        --nsubseq 20 \
        --dualrnn_blocks 1 \
        --gpu_usage 0.9
    " \
    > "$OUT_DIR/test_log.txt" 2>&1

  echo "Finished n${i}"
done

echo "All inference complete for run${RUN_NUMBER}."
