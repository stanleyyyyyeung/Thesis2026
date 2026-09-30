#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=90:00:00
#PBS -N NCH_EEGMamba_Finetune

export APPTAINER_CACHEDIR=/srv/scratch/z5423210/.apptainer_cache
export APPTAINER_TMPDIR=/srv/scratch/z5423210/.apptainer_tmp

# --- 1. Age bin selection ("1-2y" "3-5y" "6-12y" "13-18y" "19-100y")---
bin="${bin:-}"; bins="${bins:-}"

if [ -n "$bin" ] && [ -n "$bins" ]; then echo "ERROR: pass bin OR bins" >&2; exit 1; fi
if [ -z "$bin" ] && [ -z "$bins" ]; then echo "ERROR: pass bin or bins" >&2; exit 1; fi

if [ -n "$bins" ]; then
    AGE_ARGS=(--age_bins "${bins//+/,}")
    BIN_TAG="pooled-${bins//+/-}"        # e.g. pooled-6-12y-13-18y, pooled-all
else
    AGE_ARGS=(--age_bin "$bin")
    BIN_TAG="$bin"
fi

seq_len="${seq_len:-20}"
echo "=========================================="
echo "NCH EEGMamba finetuning — age bin: ${bin}, seq_len: ${seq_len}"
echo "=========================================="

# -- Learning rate for different modules --
#   qsub -v bin=6-12y,seq_lr_mult=1.0 run_nch.sh
seq_lr_mult="${seq_lr_mult:-2.0}"
head_lr_mult="${head_lr_mult:-5.0}"

# --- 2. Paths ---
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
    echo "Run build_nch_index.py --seq-len ${seq_len} (non-dry-run) first." >&2
    exit 1
fi

echo "Using index: $DATASETS_DIR"

MODEL_DIR="$PROJECT_ROOT/EEGMamba/model_weights/NCH_${BIN_TAG}_seqlen${seq_len}"

PYTHONPATH_FULL=$EEGMAMBA_DIR:/srv/scratch/z5423210/python_packages

cd "$EEGMAMBA_DIR"

echo "Overwriting previous checkpoint in $MODEL_DIR"
mkdir -p "$MODEL_DIR"
rm -f "$MODEL_DIR"/*.pth

export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

# --- 3. Run ---
PYTHONPATH=/srv/scratch/z5423210/python_packages_py310:/srv/scratch/z5423210/python_packages \
apptainer exec --nv \
    --env TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib" \
    --env LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH" \
    -B /srv:/srv "$SIF" \
    python3 finetune_main.py \
    --downstream_dataset NCH \
    --datasets_dir "$DATASETS_DIR" \
    --age_bin "${AGE_ARGS[@]}" \
    --seq_len "$seq_len" \
    --num_of_classes 5 \
    --model_dir "$MODEL_DIR" \
    --epochs 50 \
    --frozen False \
    --seq_lr_mult "$seq_lr_mult" \
    --head_lr_mult "$head_lr_mult" \
    --num_workers 0 \
    --cuda 0

exit_code=$?
echo ""
echo "=========================================="
echo "Age bin ${bin}, seq_len ${seq_len} finished, exit code ${exit_code}"
echo "=========================================="
exit $exit_code
