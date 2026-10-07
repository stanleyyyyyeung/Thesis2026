#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=02:00:00
#PBS -q eleceng
#PBS -N EEGMamba_NCH_Staged_Test
#
# Inference with a checkpoint from run_nch_staged.sh (multi-scale patch-encoder experiment).
# Writes {rec_id}_ypred/_ytrue/_probs.npy + _seq_starts.json in the same format as
# run_nch_finetuned_test.sh, but into predictions_staged/ (the baseline predictions/ tree is
# never touched).
#
# `bin` / `bins` identify the TRAINING run (same as run_nch_staged.sh); the bin to TEST on is
# `test_bin` (defaults to `bin`; required for pooled runs).
#
#   qsub -v bin=6-12y run_nch_staged_test.sh                                   # gated run, stage 2
#   qsub -v bin=6-12y,stage=both -l walltime=04:00:00 run_nch_staged_test.sh   # stage 1 and 2
#   qsub -v bin=6-12y,mode=full_pe,no_new_branch=1 run_nch_staged_test.sh      # control B
#   qsub -v bin=6-12y,no_new_branch=1,stage1_epochs=0 run_nch_staged_test.sh   # control A (stage 2 only)
#   qsub -v bins=6-12y+13-18y,test_bin=6-12y run_nch_staged_test.sh            # pooled model, per-bin test
#   Other options: split (test|val|train), seed, kernel_size, seq_len, run_dir, overwrite=1
#   The run settings (mode, no_new_branch, stage1_epochs, kernel_size, seed) must match the
#   training job so the run folder is found; or pass run_dir=/full/path/to/run folder.

bin="${bin:-}"; bins="${bins:-}"
if [ -n "$bin" ] && [ -n "$bins" ]; then echo "ERROR: pass bin OR bins" >&2; exit 1; fi
if [ -z "$bin" ] && [ -z "$bins" ]; then echo "ERROR: pass bin or bins" >&2; exit 1; fi

if [ -n "$bins" ]; then
    BIN_TAG="pooled-${bins//+/-}"
    TEST_BIN="${test_bin:?pooled run: pass test_bin=<one age bin> to choose the bin to evaluate}"
else
    BIN_TAG="$bin"
    TEST_BIN="${test_bin:-$bin}"
fi

seq_len="${seq_len:-20}"
mode="${mode:-gated}"
no_new_branch="${no_new_branch:-0}"
stage1_epochs="${stage1_epochs:-8}"
kernel_size="${kernel_size:-99}"
seed="${seed:-0}"
stage="${stage:-2}"                       # 1 | 2 | both
split="${split:-test}"
overwrite="${overwrite:-0}"

# MUST match the RUN_TAG logic in run_nch_staged.sh
if [ "$no_new_branch" = "1" ]; then
    if [ "$stage1_epochs" -eq 0 ]; then RUN_TAG="A_continue"; else RUN_TAG="B_${mode}_nobranch"; fi
else
    RUN_TAG="C_${mode}_k${kernel_size}"
fi

REPO_DIR="/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
PROJECT_ROOT="/srv/scratch/z5423210/StanleyThesis2026"
SIF="/srv/scratch/z5423210/pytorch_cu128.sif"

# same index selection as run_nch_staged.sh (training)
if [ "$seq_len" -eq 20 ] && [ ! -f "$PROJECT_ROOT/nch_index/nch_index_nch_v2_seqlen20.parquet" ]; then
    INDEX_PATH="$PROJECT_ROOT/nch_index/nch_index_nch_v2.parquet"
else
    INDEX_PATH="$PROJECT_ROOT/nch_index/nch_index_nch_v2_seqlen${seq_len}.parquet"
fi
if [ ! -f "$INDEX_PATH" ]; then echo "ERROR: index not found: $INDEX_PATH" >&2; exit 1; fi

RUN_DIR="${run_dir:-$REPO_DIR/model_weights_staged/NCH_${BIN_TAG}_seqlen${seq_len}/${RUN_TAG}_seed${seed}}"
if [ ! -d "$RUN_DIR" ]; then echo "ERROR: run folder not found: $RUN_DIR" >&2; exit 1; fi

case "$stage" in
    1|2) STAGE_LIST=("$stage") ;;
    both) STAGE_LIST=(1 2) ;;
    *) echo "ERROR: stage must be 1, 2 or both" >&2; exit 1 ;;
esac

cd "$REPO_DIR" || exit 1
export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

echo "Staged ${split}-split extraction | trained on: ${BIN_TAG} | test bin: ${TEST_BIN} | run: ${RUN_TAG} seed ${seed} | seq_len=${seq_len} (job ${PBS_JOBID})"
echo "  index_path=${INDEX_PATH}"
echo "  run_dir=${RUN_DIR}"

OVERWRITE_ARG=()
[ "$overwrite" = "1" ] && OVERWRITE_ARG=(--overwrite)

FINAL_STATUS=0
for st in "${STAGE_LIST[@]}"; do
    if [ ! -f "$RUN_DIR/stage${st}_best.pt" ]; then
        echo "ERROR: $RUN_DIR/stage${st}_best.pt not found" >&2
        FINAL_STATUS=1
        continue
    fi
    PRED_DIR="$REPO_DIR/predictions_staged/NCH_${BIN_TAG}_seqlen${seq_len}/${RUN_TAG}_seed${seed}/stage${st}/test-${TEST_BIN}"
    echo "--- stage ${st} -> ${PRED_DIR}"

    PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
    apptainer exec --nv \
        --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
        --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
        -B /srv:/srv "$SIF" \
        python test_nch_staged.py \
        --age_bin "${TEST_BIN}" \
        --seq_len "$seq_len" \
        --index_path "${INDEX_PATH}" \
        --run_dir "${RUN_DIR}" \
        --stage "${st}" \
        --pred_dir "${PRED_DIR}" \
        --split "${split}" \
        "${OVERWRITE_ARG[@]}" \
        --cuda 0
    STATUS=$?
    echo "Finished stage ${st}: exit code ${STATUS}"
    if [ "$STATUS" -ne 0 ]; then
        echo "ERROR: staged extraction failed (stage ${st}, test bin ${TEST_BIN}, exit ${STATUS})" >&2
        FINAL_STATUS=$STATUS
    fi
done

exit $FINAL_STATUS
