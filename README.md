# L-SeqSleepNet — SleepEDF-20

## What Is Here
- `lseqsleepnet/` — all network scripts (finetune, test, train, config etc.)
- `file_list_20sub/` — train/eval/test splits for all 20 folds
- `run_inference.sh` — PBS script for test inference
- `run_thesis_prelim.sh` — PBS script for finetuning
- `extract_predictions.py` — extracts .npy pred/label files from test_ret.mat

## Recovery — Full Environment From Scratch

### Step 1 — Clone this repo
```bash
cd /srv/scratch/z5423210
git clone --single-branch --branch L-SeqSleepNet/sleepedf-20 \
    https://github.com/stanleyyyyyeung/Thesis2026.git sleepedf20_code
```

### Step 2 — Sparse clone original repo
```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/pquochuy/L-SeqSleepNet.git tmp_original
cd tmp_original
git sparse-checkout set sleepedf-20/mat_30min \
                        sleepedf-20/file_list_20sub \
                        sleepedf-20/network/lseqsleepnet \
                        sleepedf-20/__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks
cd ..
```

### Step 3 — Reconstruct directory
```bash
DEST="/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20"
mkdir -p $DEST/network/lseqsleepnet

cp -r tmp_original/sleepedf-20/mat_30min $DEST/
cp -r tmp_original/sleepedf-20/file_list_20sub $DEST/
cp -r tmp_original/sleepedf-20/__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks $DEST/

for f in tmp_original/sleepedf-20/network/lseqsleepnet/*; do
    fname=$(basename $f)
    if [ ! -f "$DEST/network/lseqsleepnet/$fname" ]; then
        cp "$f" "$DEST/network/lseqsleepnet/"
    fi
done

cp sleepedf20_code/lseqsleepnet/* $DEST/network/lseqsleepnet/
cp sleepedf20_code/run_inference.sh /srv/scratch/z5423210/StanleyThesis2026/
cp sleepedf20_code/run_thesis_prelim.sh /srv/scratch/z5423210/StanleyThesis2026/
cp sleepedf20_code/extract_predictions.py /srv/scratch/z5423210/StanleyThesis2026/

rm -rf tmp_original sleepedf20_code
```

### Step 4 — Pull container (inside interactive job)
```bash
qsub -I -l select=1:ncpus=8:mem=46gb,walltime=1:00:00
apptainer pull /srv/scratch/z5423210/tf22_py3.sif \
    docker://nvcr.io/nvidia/tensorflow:22.05-tf1-py3
```

### Step 5 — Reinstall Python packages
```bash
rm -rf /srv/scratch/z5423210/python_packages/numpy*
apptainer exec /srv/scratch/z5423210/tf22_py3.sif pip3 install \
    hdf5storage "scikit-learn==0.24.2" "imbalanced-learn==0.8.1" \
    --target=/srv/scratch/z5423210/python_packages/
```

### Step 6 — Run finetuning then inference
```bash
qsub /srv/scratch/z5423210/StanleyThesis2026/run_thesis_prelim.sh
qsub /srv/scratch/z5423210/StanleyThesis2026/run_inference.sh
```

## Notes
- mat_30min is NOT in this repo (848MB) — recovered via sparse clone in Step 2
- prev_runs is NOT in this repo — regenerated via Step 6
- Never install numpy into python_packages — breaks TensorFlow
- test_ret.mat predictions for all 20 folds are saved locally on Mac
