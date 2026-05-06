#!/bin/bash
#PBS -l select=1:ncpus=8:mem=46gb
#PBS -l walltime=12:00:00
#PBS -q eleceng
#PBS -N PrepareData

module load matlab/R2025b   # adjust to whatever version is available

cd /srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-78

matlab -nodisplay -nosplash -r "addpath(genpath('data_processing')); prepare_data_30min; quit"
