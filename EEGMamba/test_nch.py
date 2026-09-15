"""
tests_nch.py — Run inference on the NCH test split for ONE
age bin, save raw per-window predictions + labels + softmax
probabilities per recording for reconstruction into chronological
hypnograms and downstream HMM/Viterbi refinement.

Usage (one call per age bin, e.g. as a PBS array task — see
run_extract_nch.sh):
    python test_nch.py \
        --age_bin 6-12y \
        --index_path /srv/scratch/z5423210/StanleyThesis2026/nch_index.parquet \
        --pred_dir /srv/scratch/z5423210/StanleyThesis2026/EEGMamba/predictions/NCH_6-12y \
        --model_dir /srv/scratch/z5423210/StanleyThesis2026/EEGMamba/model_weights/NCH_6-12y
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

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
    parser.add_argument('--pred_dir', type=str, required=True,
                         help='Base predictions directory for this age bin, e.g. '
                              'EEGMamba/predictions/NCH_1-2y. Test-split outputs are '
                              'written directly here (flat, matching the existing '
                              'convention); --split train/eval write into a nested '
                              'subdirectory instead — see --split.')
    parser.add_argument('--model_dir', type=str, required=True,
                         help='Directory containing the .pth checkpoint for this age bin.')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--foundation_dir', type=str,
                         default='pretrained_weights/pretrained_EEGMamba.pth')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'eval', 'test'],
                         help="Which split to run inference on. Defaults to 'test'. "
                              "Set to 'train' when generating scores for HMM MMI training "
                              "(hmm_trained refinement mode) — see hmm_refine_nch.py.")
    args = parser.parse_args()

    model_dir = args.model_dir

    # Test-split outputs are written flat into --pred_dir, matching the
    # existing on-disk convention (rec_id_ypred.npy etc. sitting directly
    # in EEGMamba/predictions/NCH_<age_bin>/). Train/eval-split outputs go
    # into a named subdirectory so they never collide with or get mistaken
    # for the real test-split predictions.
    if args.split == 'test':
        out_dir = args.pred_dir
    elif args.split == 'train':
        out_dir = os.path.join(args.pred_dir, 'training_scores')
    else:  # eval
        out_dir = os.path.join(args.pred_dir, 'eval_scores')
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or find_checkpoint(model_dir)
    print(f"[{args.age_bin}] Loading checkpoint: {checkpoint_path}")

    param = build_param(args.cuda, model_dir, args.foundation_dir)
    torch.cuda.set_device(param.cuda)

    model = model_for_isruc.Model(param).cuda()
    state_dict = torch.load(checkpoint_path, map_location=f'cuda:{param.cuda}')
    model.load_state_dict(state_dict)
    model.eval()

    test_set = NCHIndexDataset(args.index_path, split=args.split, age_bin=args.age_bin)

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

    print(f"[{args.age_bin}] {len(test_set)} windows across {len(recordings)} recordings "
          f"(split={args.split}).")

    with torch.no_grad():
        for edf_path, row_positions in recordings.items():
            subj_preds = []
            subj_truths = []
            subj_probs = []
            seq_starts = []

            for pos in row_positions:
                x, y = test_set[pos]            # x: (20,6,6000) float, y: (20,) long
                x = x.unsqueeze(0).cuda()        # (1,20,6,6000) -- already scaled by
                                                  # SCALE_TO_MODEL_INPUT inside
                                                  # NCHIndexDataset; do not rescale here
                pred = model(x)                  # (1, 20, 5) raw logits
                probs = F.softmax(pred, dim=-1)  # (1, 20, 5) -- emission probabilities
                pred_y = torch.max(pred, dim=-1)[1]  # (1, 20)

                subj_preds += pred_y.cpu().squeeze(0).numpy().tolist()
                subj_truths += y.numpy().tolist()
                subj_probs.append(probs.cpu().squeeze(0).numpy())  # each (20, 5)
                seq_starts.append(float(test_set.df.loc[pos, 'seq_start_sec']))

            subj_preds = np.array(subj_preds, dtype=int)
            subj_truths = np.array(subj_truths, dtype=int)
            subj_probs = np.concatenate(subj_probs, axis=0).astype(np.float32)  # (n_epochs, 5)
            assert subj_preds.shape == subj_truths.shape
            assert subj_probs.shape[0] == subj_preds.shape[0]
            assert subj_probs.shape[1] == 5

            # Filesystem-safe recording identifier. Swap this for a
            # subject_id column if the index has one AND subject != recording
            # (e.g. multiple nights per subject that should still be kept
            # separate for hypnogram reconstruction) — in that case this is
            # still the right granularity, just rename the variable.
            rec_id = os.path.splitext(os.path.basename(edf_path))[0]

            np.save(os.path.join(out_dir, f'{rec_id}_ypred.npy'), subj_preds)
            np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), subj_truths)
            np.save(os.path.join(out_dir, f'{rec_id}_probs.npy'), subj_probs)
            with open(os.path.join(out_dir, f'{rec_id}_seq_starts.json'), 'w') as f:
                json.dump(seq_starts, f)

            print(f"[{args.age_bin}] {rec_id}: {len(row_positions)} windows -> "
                  f"{len(subj_preds)} epochs saved (ypred, ytrue, probs).")

    print(f"[{args.age_bin}] Done. Predictions saved to {out_dir}")


if __name__ == '__main__':
    main()
