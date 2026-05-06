#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N EDF_Download

wget -r -N -c -np -nd \
    --accept "SC4*.edf,SC4*.txt" \
    -P /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-78/edfFiles/ \
    https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
