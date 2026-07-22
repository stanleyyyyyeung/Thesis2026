"""
test_isruc.py — Run inference on the ISRUC test split (subjects 91-100),
save raw per-window predictions + labels per subject for reconstruction
into chronological hypnograms.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from models import model_for_isruc

DATASETS_DIR = '/srv/scratch/speechdata/sleep_data/ISRUC'
SEQ_DIR = os.path.join(DATASETS_DIR, 'seq')
LABEL_DIR = os.path.join(DATASETS_DIR, 'labels')
MODEL_DIR = '/srv/scratch/z5423210/StanleyThesis2026/EEGMamba/model_weights/ISRUC_full'

TEST_SUBJECT_NUMS = list(range(91, 101))  # matches split_dataset(): i in [90,99] -> subject_num i+1
FNAME_RE = re.compile(r'-(\d+)\.npy$')


def numeric_sort_key(fname):
    """Numeric sort on the trailing global index. isruc_dataset.py uses
    sorted(os.listdir(...)) -- a STRING sort that would misorder e.g.
    '...-999.npy' before '...-1000.npy'. Harmless for training/eval (metrics
    are order-invariant) but would corrupt chronological reconstruction here."""
    m = FNAME_RE.search(fname)
    if not m:
        raise ValueError(f"Unexpected filename format: {fname}")
    return int(m.group(1))


def build_param(cuda_index, model_dir, foundation_dir):
    """Matches finetune_main.py's argparse defaults/overrides for the ISRUC
    run, minus training-only args the Model/backbone don't need at inference.
    use_pretrained_weights=False: skips loading foundation_dir into the
    backbone before we immediately overwrite everything via load_state_dict
    with the finetuned checkpoint below -- avoids an unnecessary dependency
    on foundation_dir being valid/present for this script."""
    p = argparse.Namespace()
    p.cuda = cuda_index
    p.downstream_dataset = 'ISRUC'
    p.datasets_dir = DATASETS_DIR
    p.num_of_classes = 5
    p.model_dir = model_dir
    p.use_pretrained_weights = False
    p.foundation_dir = foundation_dir  # unused when use_pretrained_weights=False, kept for completeness
    return p


def find_checkpoint(model_dir):
    ckpts = glob.glob(os.path.join(model_dir, '*.pth'))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .pth checkpoint found in {model_dir}")
    if len(ckpts) > 1:
        raise RuntimeError(
            f"Multiple checkpoints found in {model_dir}, expected exactly one:\n" +
            "\n".join(ckpts) +
            "\nSpecify --checkpoint explicitly."
        )
    return ckpts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--run_number', type=int, required=True,
                         help='Matches the run number used during finetuning, '
                              'e.g. 1 for out_eegmamba/isruc/run1/')
    parser.add_argument('--model_dir', type=str, default=None,
                         help='Overrides the derived model_dir if set')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Explicit checkpoint path; auto-discovered from --model_dir if omitted')
    parser.add_argument('--foundation_dir', type=str,
                         default='pretrained_weights/pretrained_EEGMamba.pth')
    args = parser.parse_args()

    run_root = f'/srv/scratch/z5423210/StanleyThesis2026/out_eegmamba/isruc/run{args.run_number}/out'
    model_dir = args.model_dir or os.path.join(run_root, 'model_weights')
    out_dir = os.path.join(run_root, 'predictions')
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or find_checkpoint(model_dir)
    print(f"Loading checkpoint: {checkpoint_path}")

    param = build_param(args.cuda, model_dir, args.foundation_dir)
    torch.cuda.set_device(param.cuda)

    model = model_for_isruc.Model(param).cuda()
    state_dict = torch.load(checkpoint_path, map_location=f'cuda:{param.cuda}')
    model.load_state_dict(state_dict)
    model.eval()

    for subject_num in TEST_SUBJECT_NUMS:
        # ... unchanged loop body ...
        np.save(os.path.join(out_dir, f'{subject_num}_ypred.npy'), subj_preds)
        np.save(os.path.join(out_dir, f'{subject_num}_ytrue.npy'), subj_truths)
        with open(os.path.join(out_dir, f'{subject_num}_window_order.json'), 'w') as f:
            json.dump(window_index_order, f)

    print(f"Done. Predictions saved to {out_dir}")

if __name__ == '__main__':
    main()
