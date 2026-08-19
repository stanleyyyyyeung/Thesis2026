"""
tests_nch.py — Run inference on the NCH test split for ONE
age bin, save raw per-window predictions + labels per recording for
reconstruction into chronological hypnograms.

Usage (one call per age bin, e.g. as a PBS array task — see
run_extract_nch.sh):
    python test_nch.py \
        --age_bin 6-12y \
        --index_path /srv/scratch/z5423210/StanleyThesis2026/nch_index.parquet \
        --model_dir /srv/scratch/z5423210/StanleyThesis2026/out_eegmamba/nch/6-12y/out/model_weights
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from datasets.nch_dataset import NCHIndexDataset
from models import model_for_isruc

AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y", "19-100y"]


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


def build_param(cuda_index, model_dir, foundation_dir):
    p = argparse.Namespace()
    p.cuda = cuda_index
    p.downstream_dataset = 'NCH'
    p.num_of_classes = 5
    p.model_dir = model_dir
    p.use_pretrained_weights = False
    p.foundation_dir = foundation_dir
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--age_bin', type=str, required=True, choices=AGE_BINS)
    parser.add_argument('--index_path', type=str, required=True,
                         help='Path to the parquet produced by build_nch_index.py')
    parser.add_argument('--model_dir', type=str, default=None,
                         help='Defaults to out_eegmamba/nch/<age_bin>/out/model_weights')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--foundation_dir', type=str,
                         default='pretrained_weights/pretrained_EEGMamba.pth')
    args = parser.parse_args()

    run_root = f'/srv/scratch/z5423210/StanleyThesis2026/out_eegmamba/nch/{args.age_bin}/out'
    model_dir = args.model_dir or os.path.join(run_root, 'model_weights')
    out_dir = os.path.join(run_root, 'predictions')
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or find_checkpoint(model_dir)
    print(f"[{args.age_bin}] Loading checkpoint: {checkpoint_path}")

    param = build_param(args.cuda, model_dir, args.foundation_dir)
    torch.cuda.set_device(param.cuda)

    model = model_for_isruc.Model(param).cuda()
    state_dict = torch.load(checkpoint_path, map_location=f'cuda:{param.cuda}')
    model.load_state_dict(state_dict)
    model.eval()

    test_set = NCHIndexDataset(args.index_path, split='test', age_bin=args.age_bin)

    # Sort so each recording's windows are visited consecutively and in
    # chronological (seq_start_sec) order. This is what keeps the dataset's
    # internal _raw_cache warm across consecutive __getitem__ calls, and
    # what makes the concatenated per-recording predictions a valid
    # chronological hypnogram rather than an arbitrarily-ordered bag of
    # windows.
    sorted_df = test_set.df.sort_values(['edf_path', 'seq_start_sec'])

    # .groups (not .indices!) returns original row labels in appearance
    # order per group. Since NCHIndexDataset.__init__ does
    # df.reset_index(drop=True), these labels are exactly the positional
    # indices __getitem__ expects, so no further remapping is needed.
    recordings = sorted_df.groupby('edf_path', sort=False).groups

    print(f"[{args.age_bin}] {len(test_set)} windows across {len(recordings)} recordings.")

    with torch.no_grad():
        for edf_path, row_positions in recordings.items():
            subj_preds = []
            subj_truths = []
            seq_starts = []

            for pos in row_positions:
                x, y = test_set[pos]            # x: (20,6,6000) float, y: (20,) long
                x = x.unsqueeze(0).cuda()        # (1,20,6,6000) -- already scaled by
                                                  # SCALE_TO_MODEL_INPUT inside
                                                  # NCHIndexDataset; do not rescale here
                pred = model(x)                  # (1, 20, 5)
                pred_y = torch.max(pred, dim=-1)[1]  # (1, 20)

                subj_preds += pred_y.cpu().squeeze(0).numpy().tolist()
                subj_truths += y.numpy().tolist()
                seq_starts.append(float(test_set.df.loc[pos, 'seq_start_sec']))

            subj_preds = np.array(subj_preds, dtype=int)
            subj_truths = np.array(subj_truths, dtype=int)
            assert subj_preds.shape == subj_truths.shape

            # Filesystem-safe recording identifier. Swap this for a
            # subject_id column if the index has one AND subject != recording
            # (e.g. multiple nights per subject that should still be kept
            # separate for hypnogram reconstruction) — in that case this is
            # still the right granularity, just rename the variable.
            rec_id = os.path.splitext(os.path.basename(edf_path))[0]

            np.save(os.path.join(out_dir, f'{rec_id}_ypred.npy'), subj_preds)
            np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), subj_truths)
            with open(os.path.join(out_dir, f'{rec_id}_seq_starts.json'), 'w') as f:
                json.dump(seq_starts, f)

            print(f"[{args.age_bin}] {rec_id}: {len(row_positions)} windows -> "
                  f"{len(subj_preds)} epochs saved.")

    print(f"[{args.age_bin}] Done. Predictions saved to {out_dir}")


if __name__ == '__main__':
    main()
