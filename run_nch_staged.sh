#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=60:00:00
#PBS -N NCH_EEGMamba_MS

# Staged finetuning (multi-scale patch encoder) on top of an EXISTING finetuned baseline.
# Same interface as the baseline script: pass `bin` OR `bins` (pooled, '+'-separated, or 'all').
#
#   Proposed (gated branch):   qsub -v bin=6-12y run_nch_staged.sh
#   Switch (full patch enc.):  qsub -v bin=6-12y,mode=full_pe run_nch_staged.sh
#   Control B (no new branch): qsub -v bin=6-12y,mode=full_pe,no_new_branch=1 run_nch_staged.sh
#   Control A (continue only): qsub -v bin=6-12y,no_new_branch=1,stage1_epochs=0 run_nch_staged.sh
#   Pooled bins:               qsub -v bins=6-12y+13-18y run_nch_staged.sh     (or bins=all)
#   Extra options (all optional env vars): baseline_ckpt, kernel_size, gate_init, stage1_epochs,
#     stage2_epochs, lr_new1, lr_pe1, lr_new2, lr_pe2, lr_rest2, seq_lr_mult, head_lr_mult,
#     batch_size, num_workers, data_source (auto|cache|live), seed, verify (1|0), overwrite (0|1)
#

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 1. Age bin selection ("1-2y" "3-5y" "6-12y" "13-18y" "19-100y") ---
bin="${bin:-}"; bins="${bins:-}"

if [ -n "$bin" ] && [ -n "$bins" ]; then echo "ERROR: pass bin OR bins" >&2; exit 1; fi
if [ -z "$bin" ] && [ -z "$bins" ]; then echo "ERROR: pass bin or bins" >&2; exit 1; fi

if [ -n "$bins" ]; then
    AGE_VAL="${bins//+/,}"
    BIN_TAG="pooled-${bins//+/-}"        # same tag as the baseline script -> finds its checkpoint folder
else
    AGE_VAL="$bin"
    BIN_TAG="$bin"
fi

seq_len="${seq_len:-20}"

# --- 2. Experiment settings ---
mode="${mode:-gated}"                    # gated | full_pe  (what stage 1 trains)
no_new_branch="${no_new_branch:-0}"      # 1 = control arms (original architecture)
stage1_epochs="${stage1_epochs:-8}"
stage2_epochs="${stage2_epochs:-20}"
kernel_size="${kernel_size:-99}"
gate_init="${gate_init:-0.0}"

# PLACEHOLDER LRs: set lr_rest2 to ~0.1x the base LR of your baseline finetune (see finetune_main.py default)
lr_new1="${lr_new1:-1e-3}"
lr_pe1="${lr_pe1:-1e-4}"
lr_new2="${lr_new2:-1e-4}"
lr_pe2="${lr_pe2:-1e-5}"
lr_rest2="${lr_rest2:-1e-5}"
# same multipliers as the baseline run (applied on top of lr_rest2 in stage 2)
seq_lr_mult="${seq_lr_mult:-2.0}"
head_lr_mult="${head_lr_mult:-5.0}"

batch_size="${batch_size:-32}"           # set to your baseline's batch size
num_workers="${num_workers:-0}"          # baseline used 0
data_source="${data_source:-auto}"       # auto = preprocessed cache if usable, else on-the-fly
seed="${seed:-0}"
verify="${verify:-1}"                    # 1 = check gate=0 reproduces the baseline before training
overwrite="${overwrite:-0}"              # 1 = clear previous files in THIS run's output folder only

if [ "$mode" != "gated" ] && [ "$mode" != "full_pe" ]; then echo "ERROR: mode must be gated or full_pe" >&2; exit 1; fi
if [ "$mode" = "gated" ] && [ "$no_new_branch" = "1" ] && [ "$stage1_epochs" -gt 0 ]; then
    echo "ERROR: mode=gated needs the new branch. Use mode=full_pe with no_new_branch=1 (control B)." >&2; exit 1
fi

if [ "$no_new_branch" = "1" ]; then
    if [ "$stage1_epochs" -eq 0 ]; then RUN_TAG="A_continue"; else RUN_TAG="B_${mode}_nobranch"; fi
else
    RUN_TAG="C_${mode}_k${kernel_size}"
fi

echo "=========================================="
echo "NCH EEGMamba STAGED finetuning — bins: ${AGE_VAL}, seq_len: ${seq_len}, run: ${RUN_TAG}, seed: ${seed}"
echo "=========================================="

# --- 3. Paths ---
PROJECT_ROOT=/srv/scratch/z5423210/StanleyThesis2026
EEGMAMBA_DIR=$PROJECT_ROOT/EEGMamba
SIF=/srv/scratch/z5423210/pytorch_cu128.sif

if [ "$seq_len" -eq 20 ] && [ ! -f "$PROJECT_ROOT/nch_index/nch_index_nch_v2_seqlen20.parquet" ]; then
    DATASETS_DIR="$PROJECT_ROOT/nch_index/nch_index_nch_v2.parquet"
else
    DATASETS_DIR="$PROJECT_ROOT/nch_index/nch_index_nch_v2_seqlen${seq_len}.parquet"
fi
if [ ! -f "$DATASETS_DIR" ]; then
    echo "ERROR: no index parquet found for seq_len=${seq_len} at $DATASETS_DIR" >&2
    exit 1
fi
echo "Using index: $DATASETS_DIR"

# baseline checkpoint (READ ONLY): the baseline script's output folder for this bin/pool
BASE_DIR="$EEGMAMBA_DIR/model_weights/NCH_${BIN_TAG}_seqlen${seq_len}"
if [ -n "${baseline_ckpt:-}" ]; then
    BASELINE_CKPT="$baseline_ckpt"
else
    shopt -s nullglob; cands=("$BASE_DIR"/*.pth); shopt -u nullglob
    if [ "${#cands[@]}" -eq 1 ]; then
        BASELINE_CKPT="${cands[0]}"
    else
        echo "ERROR: expected exactly 1 .pth in $BASE_DIR, found ${#cands[@]}: ${cands[*]}" >&2
        echo "       Pass the one to use:  qsub -v ...,baseline_ckpt=/full/path/model.pth" >&2
        exit 1
    fi
fi
if [ ! -f "$BASELINE_CKPT" ]; then echo "ERROR: baseline checkpoint not found: $BASELINE_CKPT" >&2; exit 1; fi
echo "Baseline checkpoint (read-only): $BASELINE_CKPT"

# outputs go to a NEW folder tree, never the baseline folder
OUT_DIR="$EEGMAMBA_DIR/model_weights_staged/NCH_${BIN_TAG}_seqlen${seq_len}/${RUN_TAG}_seed${seed}"
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
    if [ "$overwrite" = "1" ]; then
        echo "overwrite=1: clearing previous results in $OUT_DIR"
        rm -f "$OUT_DIR"/stage1_best.pt "$OUT_DIR"/stage2_best.pt "$OUT_DIR"/results.json "$OUT_DIR"/log.txt
    else
        echo "ERROR: $OUT_DIR already has files. Use overwrite=1 or another seed." >&2
        exit 1
    fi
fi

cd "$EEGMAMBA_DIR"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

COMMON_ARGS=(
    --baseline_ckpt "$BASELINE_CKPT"
    --out_dir "$OUT_DIR"
    --datasets_dir "$DATASETS_DIR"
    --age_bin "$AGE_VAL"
    --seq_len "$seq_len"
    --num_of_classes 5
    --mode "$mode"
    --kernel_size "$kernel_size"
    --gate_init "$gate_init"
    --seed "$seed"
    --cuda 0
)
[ "$no_new_branch" = "1" ] && COMMON_ARGS+=(--no_new_branch)

TRAIN_ARGS=(
    --stage1_epochs "$stage1_epochs"
    --stage2_epochs "$stage2_epochs"
    --lr_new1 "$lr_new1" --lr_pe1 "$lr_pe1"
    --lr_new2 "$lr_new2" --lr_pe2 "$lr_pe2" --lr_rest2 "$lr_rest2"
    --seq_lr_mult "$seq_lr_mult" --head_lr_mult "$head_lr_mult"
    --batch_size "$batch_size"
    --num_workers "$num_workers"
    --data_source "$data_source"
)

run_py () {
    PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
    apptainer exec --nv \
        --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
        --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
        -B /srv:/srv "$SIF" \
        python3 finetune_staged.py "$@"
}

# --- 4. Sanity check: ModelMS at gate=0 must reproduce the baseline exactly ---
if [ "$verify" = "1" ]; then
    echo "--- verify: baseline vs ModelMS ---"
    run_py "${COMMON_ARGS[@]}" --verify
    vcode=$?
    if [ $vcode -ne 0 ]; then
        echo "ERROR: verify failed (exit ${vcode}); not starting training." >&2
        exit $vcode
    fi
fi

# --- 5. Run ---
run_py "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"

exit_code=$?
echo ""
echo "=========================================="
echo "Bins ${AGE_VAL}, run ${RUN_TAG}, seed ${seed} finished, exit code ${exit_code}"
echo "Results: $OUT_DIR/results.json"
echo "=========================================="
exit $exit_code
