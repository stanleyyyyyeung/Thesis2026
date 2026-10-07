"""
test_nch_staged.py — same job as test_nch.py

Usage:
    python test_nch_staged.py --age_bin 6-12y --index_path <index.parquet> \
        --run_dir model_weights_staged/NCH_6-12y_seqlen20/C_gated_k99_seed0 --stage 2 \
        --pred_dir predictions_staged/NCH_6-12y_seqlen20/C_gated_k99_seed0/stage2/test-6-12y
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.nch_dataset import NCHIndexDataset
from models.model_ms import ModelMS

AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y", "19-100y"]
BRANCH_KEY = 'backbone.patch_embedding.proj_in_b.0.weight'   # Conv2d weight (25, 1, 1, k)
GATE_KEY = 'backbone.patch_embedding.gate'


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    if not args.run_dir:
        raise SystemExit('Give --checkpoint, or --run_dir together with --stage.')
    path = os.path.join(args.run_dir, f'stage{args.stage}_best.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found (does this run have a stage {args.stage}? '
                                'control A has no stage 1).')
    return path


def build_param(cuda_index, use_new_branch, kernel_size):
    p = argparse.Namespace()
    p.cuda = cuda_index
    p.downstream_dataset = 'NCH'
    p.num_of_classes = 5
    p.use_pretrained_weights = False      # weights come from the staged checkpoint
    p.foundation_dir = None
    p.ms_use_new_branch = use_new_branch
    p.ms_kernel_size = kernel_size
    p.ms_gate_init = 0.0
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--age_bin', type=str, required=True, choices=AGE_BINS,
                        help='Bin to TEST on (a pooled model is tested one bin at a time).')
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # same convention as test_nch.py: test flat in pred_dir, train/val in named subdirs
    if args.split == 'test':
        out_dir = args.pred_dir
    elif args.split == 'train':
        out_dir = os.path.join(args.pred_dir, 'training_scores')
    else:
        out_dir = os.path.join(args.pred_dir, 'eval_scores')
    if glob.glob(os.path.join(out_dir, '*_ypred.npy')) and not args.overwrite:
        raise SystemExit(f'{out_dir} already contains predictions. Use --overwrite or another --pred_dir.')
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = resolve_checkpoint(args)
    print(f"[{args.age_bin}] Loading checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location='cpu')
    use_new_branch = BRANCH_KEY in sd
    kernel_size = int(sd[BRANCH_KEY].shape[-1]) if use_new_branch else 99
    gate = float(sd[GATE_KEY].item()) if GATE_KEY in sd else None
    print(f"[{args.age_bin}] new branch: {use_new_branch}"
          + (f" (kernel {kernel_size}, gate {gate:.4f})" if use_new_branch else ""))

    param = build_param(args.cuda, use_new_branch, kernel_size)
    torch.cuda.set_device(param.cuda)
    model = ModelMS(param)
    model.load_state_dict(sd)             # strict: any mismatch is an error
    model = model.cuda().eval()

    test_set = NCHIndexDataset(args.index_path, seq_len=args.seq_len,
                               split=args.split, age_bin=args.age_bin)

    with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
        json.dump(dict(checkpoint=checkpoint_path, index=args.index_path, seq_len=args.seq_len,
                       split=args.split, test_age_bin=args.age_bin, model='ModelMS',
                       use_new_branch=use_new_branch, kernel_size=kernel_size, gate=gate,
                       run_dir=args.run_dir, stage=args.stage), f, indent=2)

    # Visit each recording's windows consecutively and in chronological order (keeps the
    # dataset's raw cache warm and makes the concatenated predictions a valid hypnogram).
    sorted_df = test_set.df.sort_values(['edf_path', 'seq_start_sec'])
    # .groups returns original row labels in order; after reset_index these are the
    # positional indices __getitem__ expects.
    recordings = sorted_df.groupby('edf_path', sort=False).groups

    print(f"[{args.age_bin}] {len(test_set)} windows across {len(recordings)} recordings "
          f"(split={args.split}).")

    with torch.no_grad():
        for edf_path, row_positions in recordings.items():
            subj_preds, subj_truths, subj_probs, seq_starts = [], [], [], []

            for pos in row_positions:
                x, y = test_set[pos]             # x: (20,6,6000), already scaled inside the dataset
                x = x.unsqueeze(0).cuda()
                pred = model(x)                  # (1, 20, 5) logits
                probs = F.softmax(pred, dim=-1)
                pred_y = torch.max(pred, dim=-1)[1]

                subj_preds += pred_y.cpu().squeeze(0).numpy().tolist()
                subj_truths += y.numpy().tolist()
                subj_probs.append(probs.cpu().squeeze(0).numpy())
                seq_starts.append(float(test_set.df.loc[pos, 'seq_start_sec']))

            subj_preds = np.array(subj_preds, dtype=int)
            subj_truths = np.array(subj_truths, dtype=int)
            subj_probs = np.concatenate(subj_probs, axis=0).astype(np.float32)
            assert subj_preds.shape == subj_truths.shape
            assert subj_probs.shape[0] == subj_preds.shape[0]
            assert subj_probs.shape[1] == 5

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
