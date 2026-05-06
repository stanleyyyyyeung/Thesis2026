
#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N LSeq_Finetune
#PBS -J 1-20%5

# --- 1. Environment ---
export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 2. Subject and run index ---
i=${PBS_ARRAY_INDEX}
echo "Running subject n${i}"

runIndex=1
echo "Run ${runIndex}"

# --- 3. Path Fix ---
cd /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20/network/lseqsleepnet/
mkdir -p ../../mat_30min
ln -sf /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20/mat_30min/* ../../mat_30min/

# --- 4. Pretrained model ---
PRETRAINED="/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20/__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks/best_model_acc"

# --- 5. Run ---
apptainer exec --nv -B /srv:/srv /srv/scratch/z5423210/tf22_py3.sif \
    python finetune_lseqsleepnet.py \
    --eeg_train_data "../../file_list_20sub/eeg/train_list_n${i}.txt" \
    --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n${i}.txt" \
    --pretrained_model "$PRETRAINED" \
    --out_dir "/srv/scratch/z5423210/StanleyThesis2026/prev_runs/run${runIndex}/out/n${i}" \
    --log_dir "/srv/scratch/z5423210/StanleyThesis2026/log_final/n${i}" \
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
