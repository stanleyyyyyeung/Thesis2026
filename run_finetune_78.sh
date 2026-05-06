#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N LSeq_78sub
#PBS -J 1-10%5

# --- 1. Cleanup old logs ---
rm -f /srv/scratch/z5423210/StanleyThesis2026/LSeq_78sub.[oe]*

# --- 2. Environment ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 3. Fold and run index ---
i=${PBS_ARRAY_INDEX}
echo "Running fold n${i}"

runIndex=1
echo "Run ${runIndex}"

# --- 4. Path fix ---
cd /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-78/network/lseqsleepnet/

# --- 5. Pretrained model ---
PRETRAINED="/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-78/__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks/best_model_acc"

# --- 6. Skip if already complete ---
OUT="/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-78/run${runIndex}/n${i}"
if [ -f "${OUT}/current_best.txt" ]; then
    echo "Fold n${i} already complete — skipping."
    exit 0
fi

# --- 7. Run ---
apptainer exec --nv -B /srv:/srv /srv/scratch/z5423210/tf22_py3.sif \
    python finetune_lseqsleepnet.py \
    --eeg_train_data "../../file_list_30min/eeg/train_list_n${i}.txt" \
    --eeg_eval_data  "../../file_list_30min/eeg/eval_list_n${i}.txt" \
    --pretrained_model "$PRETRAINED" \
    --out_dir "${OUT}" \
    --dropout_rnn 0.9 \
    --sub_seq_len 10 \
    --nfilter 32 \
    --nhidden1 64 \
    --nhidden2 64 \
    --attention_size 64 \
    --nsubseq 20 \
    --dualrnn_blocks 1 \
    --gpu_usage 0.9 \
    --early_stopping True
