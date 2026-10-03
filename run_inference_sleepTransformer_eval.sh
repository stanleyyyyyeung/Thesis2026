#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=3:00:00
#PBS -q eleceng
#PBS -N SleepTrans_Inference_Eval
#
# Usage:
#   qsub -v dataset=sleepedf-78 run_inference_sleepTransformer_eval.sh
#   qsub -v dataset=sleepedf-20 run_inference_sleepTransformer_eval.sh
#
DATASET="${dataset:?Must pass dataset: sleepedf-78 | sleepedf-20}"
RUN_NUMBER="${run:-1}"

CODE="/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"
RUN_ROOT="/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/${DATASET}/run${RUN_NUMBER}"
OUT_BASE="${RUN_ROOT}/eval_scores"
CONTAINER="/srv/scratch/z5423210/tf22_py3.sif"
PYTHONPATH_DIR="/srv/scratch/z5423210/python_packages"

# List locations / fold counts, matching hmm_refine_sleepedf.py
if [ "$DATASET" = "sleepedf-78" ]; then
    FILE_LIST="${CODE}/sleepedf-78/file_list_30min/eeg"
    NFOLDS=10
elif [ "$DATASET" = "sleepedf-20" ]; then
    FILE_LIST="${CODE}"
    NFOLDS=20
else
    echo "Unknown dataset: $DATASET" >&2; exit 1
fi

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

echo "Running EVAL inference: dataset=${DATASET} run=${RUN_NUMBER} folds=1..${NFOLDS}"

FAILED=""
for i in $(seq 1 $NFOLDS)
do
  OUT_DIR="$OUT_BASE/n${i}"
  mkdir -p "$OUT_DIR"
  echo "--- fold n${i} ---"

  PYTHONPATH=$PYTHONPATH_DIR \
  apptainer exec --nv -B /srv:/srv $CONTAINER \
    bash -c "
      cd $CODE && \
      python test_sleeptransformer.py \
        --eeg_train_data '${FILE_LIST}/train_list_n${i}.txt' \
        --eeg_test_data '${FILE_LIST}/eval_list_n${i}.txt' \
        --eog_train_data '' \
        --eog_test_data '' \
        --emg_train_data '' \
        --emg_test_data '' \
        --out_dir '${OUT_DIR}/' \
        --checkpoint_dir '${RUN_ROOT}/out/n${i}/checkpoint' \
        --seq_len 21 \
        --num_blocks 4
    " \
    > "$OUT_DIR/eval_inference_log.txt" 2>&1

  if [ ! -f "$OUT_DIR/test_ret.mat" ]; then
      echo "ERROR: no test_ret.mat for fold n${i} (see $OUT_DIR/eval_inference_log.txt)" >&2
      FAILED="$FAILED n${i}"
  fi
done

echo "Finished eval inference for ${DATASET}."
if [ -n "$FAILED" ]; then
    echo "Failed folds:$FAILED" >&2
    exit 1
fi
