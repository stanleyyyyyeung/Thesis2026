#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=6:00:00
#PBS -q eleceng
#PBS -N LSeq_Inf78_n3

cd /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-78/network/lseqsleepnet/

PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec --nv -B /srv:/srv /srv/scratch/z5423210/tf22_py3.sif \
    python test_lseqsleepnet.py \
    --eeg_train_data "../../file_list_30min/eeg/train_list_n3.txt" \
    --eeg_test_data "../../file_list_30min/eeg/test_list_n3.txt" \
    --out_dir "/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-78/run1/n3" \
    --dropout_rnn 0.9 \
    --sub_seq_len 10 \
    --nfilter 32 \
    --nhidden1 64 \
    --nhidden2 64 \
    --attention_size 64 \
    --nsubseq 20 \
    --dualrnn_blocks 1 \
    --gpu_usage 0.9
