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
    p = argparse.Namespace()
    p.cuda = cuda_index
    p.downstream_dataset = 'ISRUC'
    p.datasets_dir = DATASETS_DIR
    p.num_of_classes = 5
    p.model_dir = model_dir
    p.use_pretrained_weights = False
    p.foundation_dir = foundation_dir
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
    parser.add_argument('--run_number', type=int, required=True)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
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
        subj_id = f'ISRUC-group1-{subject_num}'
        subj_seq_dir = os.path.join(SEQ_DIR, subj_id)
        subj_label_dir = os.path.join(LABEL_DIR, subj_id)

        if not os.path.isdir(subj_seq_dir):
            print(f"WARNING: {subj_id} seq dir missing, skipping "
                  f"(expected only for excluded subject 8, not in 91-100 range).")
            continue

        seq_fnames = sorted(os.listdir(subj_seq_dir), key=numeric_sort_key)
        label_fnames = sorted(os.listdir(subj_label_dir), key=numeric_sort_key)
        assert seq_fnames == label_fnames, \
            f"{subj_id}: seq/label filename mismatch after numeric sort"

        subj_preds = []
        subj_truths = []
        window_index_order = []

        with torch.no_grad():
            for fname in seq_fnames:
                seq = np.load(os.path.join(subj_seq_dir, fname))     # (20, 6, 6000)
                label = np.load(os.path.join(subj_label_dir, fname)) # (20,)

                x = torch.from_numpy(seq / 100).float().unsqueeze(0).cuda()  # (1, 20, 6, 6000)
                pred = model(x)                                              # (1, 20, 5)
                pred_y = torch.max(pred, dim=-1)[1]                          # (1, 20) -- matches Evaluator

                subj_preds += pred_y.cpu().squeeze().numpy().tolist()
                subj_truths += label.tolist()
                window_index_order.append(numeric_sort_key(fname))

        subj_preds = np.array(subj_preds, dtype=int)
        subj_truths = np.array(subj_truths, dtype=int)
        assert subj_truths.shape == subj_preds.shape

        np.save(os.path.join(out_dir, f'{subject_num}_ypred.npy'), subj_preds)
        np.save(os.path.join(out_dir, f'{subject_num}_ytrue.npy'), subj_truths)
        with open(os.path.join(out_dir, f'{subject_num}_window_order.json'), 'w') as f:
            json.dump(window_index_order, f)

        print(f"{subj_id}: {len(seq_fnames)} windows -> {len(subj_preds)} epochs saved.")

    print(f"Done. Predictions saved to {out_dir}")


if __name__ == '__main__':
    main()
