"""
hmm_refine_nch.py — HMM/Viterbi refinement of EEGMamba NCH predictions.

Two refinement modes:
  "hmm"          Empirical transition matrix A estimated via MLE from NCH
                 training-split ground truth (no model inference on the
                 train split required). Alpha swept over ALPHA_VALUES at
                 inference time; single-best Viterbi decode per alpha.
  "hmm_trained"  A and alpha jointly learned via MMI discriminative
                 training (k-best Viterbi + MMI loss, kNearestViterbi.py),
                 warm-started from the same empirical A. Needs the model's
                 own softmax scores on the training split -- run
                 `test_nch.py --split train` first for this age bin.

Sequences are handled PER RUN: a run is a stretch of epochs with no gap between them
(a gap = an unscored '?' epoch or similar). Transitions are only counted inside a run, and
Viterbi is decoded separately for each run (each starting from the uniform pi), so the
transition prior is never applied across a gap. Run boundaries come from
<rec>_run_idx.json written by test_nch.py; recordings are '<subject>_<study>'.

Usage (call once per age bin):
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm \
        --pred_dir .../EEGMamba/predictions/NCH_6-12y --index_path .../nch_index_nch_v2.parquet
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm_trained \
        --pred_dir .../EEGMamba/predictions/NCH_6-12y --index_path .../nch_index_nch_v2.parquet
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy.stats import entropy

from kNearestViterbi import train_hmm_mmi

# ============================================================
# CONFIGURATION
# ============================================================
AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y", "19-100y"]

STAGES = [0, 1, 2, 3, 4]   # W, N1, N2, N3, REM -- 0-indexed, matching test_nch.py's pred_y / y_true.
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
K = len(STAGES)

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]

REC_PROB_RE = re.compile(r'^(\d+_\d+)_probs\.npy$')   # <subject>_<study>_probs.npy


# ============================================================
# LABEL SANITY CHECK
# ============================================================
def verify_label_range(y_true_list, context=""):
    """Raise loudly if labels fall outside 0..K-1 (checks the 0-indexed assumption every run)."""
    all_vals = np.concatenate(y_true_list)
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 0 or hi > K - 1:
        raise ValueError(
            f"[{context}] labels outside expected 0..{K-1} range: "
            f"observed min={lo}, max={hi}. Check the label convention before trusting any HMM output."
        )
    print(f"[{context}] label range check passed: min={lo}, max={hi}")


# ============================================================
# FILE / RUN HELPERS
# ============================================================
def list_recordings(directory):
    """[(rec_id, probs_path)] for '<subject>_<study>_probs.npy' files; others are reported and skipped."""
    ok, stale = [], []
    for p in sorted(glob.glob(os.path.join(directory, '*_probs.npy'))):
        m = REC_PROB_RE.match(os.path.basename(p))
        if m:
            ok.append((m.group(1), p))
        else:
            stale.append(os.path.basename(p))
    if stale:
        print(f"[WARN] {len(stale)} file(s) in {directory} are not '<subject>_<study>_probs.npy' "
              f"(e.g. {stale[0]}) -- skipped. Remove stale outputs from older runs.")
    return ok


def read_source(directory):
    """data_source_used recorded by test_nch.py ('cache' | 'parquet'), or None."""
    p = os.path.join(directory, 'run_meta.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get('data_source_used')


def infer_data_source(pred_dir, requested):
    """Estimate A from the same windowing the predictions were made with."""
    used = {'cache': 'cache', 'parquet': 'live'}.get(read_source(pred_dir))
    if requested == 'auto':
        if used:
            print(f"[data] predictions came from the {used} path (run_meta.json) -> estimating A from the same")
            return used
        return 'auto'
    if used and used != requested:
        print(f"[WARN] --data_source={requested} but the predictions in {pred_dir} came from '{used}'; "
              "A and the decoded windows use different window grids.")
    return requested


def run_bounds(n_epochs, run_idx_path, seq_len, rec_id=""):
    """[(start, end)] epoch ranges, one per run. run_idx is stored per window (seq_len epochs each)."""
    if not os.path.exists(run_idx_path):
        print(f"[WARN] {rec_id}: no run_idx file ({os.path.basename(run_idx_path)}); decoding as ONE "
              "sequence, so gaps between runs are NOT respected (rerun test_nch.py to get it)")
        return [(0, n_epochs)]
    runs = np.repeat(np.asarray(json.load(open(run_idx_path)), dtype=int), seq_len)
    if len(runs) != n_epochs or (runs < 0).all():
        print(f"[WARN] {rec_id}: run_idx unusable (len {len(runs)} vs {n_epochs} epochs); decoding as ONE sequence")
        return [(0, n_epochs)]
    cut = np.where(np.diff(runs) != 0)[0] + 1
    edges = [0, *cut.tolist(), n_epochs]
    return list(zip(edges[:-1], edges[1:]))


# ============================================================
# HMM PARAMETER ESTIMATION FROM TRAINING GROUND TRUTH
# ============================================================
def get_uniform_pi():
    """Fixed uniform initial-state distribution (paper: pi is NOT estimated from data)."""
    return np.ones(K) / K


def estimate_hmm_parameters_from_gt(y_true_list, label="", verbose=True):
    """Transition matrix A via pooled MLE counts over per-run training label sequences.
    Also returns an empirical pi (first label of each sequence) for comparison only."""
    pi_counts = np.zeros(K)
    for y in y_true_list:
        pi_counts[y[0]] += 1
    pi_empirical = (pi_counts + 1e-6) / (pi_counts + 1e-6).sum()

    A_counts = np.zeros((K, K))
    for y in y_true_list:
        for t in range(len(y) - 1):
            A_counts[y[t], y[t + 1]] += 1
    A_counts += 1e-6
    A = A_counts / A_counts.sum(axis=1, keepdims=True)

    if verbose:
        print(f"\n{'='*70}\nHMM PARAMETERS ({label}) -- from NCH training ground truth\n{'='*70}")
        print(pd.DataFrame(np.round(A, 4),
                           index=[STAGE_NAMES[s] for s in STAGES],
                           columns=[STAGE_NAMES[s] for s in STAGES]))
        print("\nInitial State Distribution pi (empirical alternative -- NOT necessarily "
              "what gets used; see --pi_source)")
        for i in STAGES:
            print(f"  {STAGE_NAMES[i]} : {pi_empirical[i]:.4f}")
        print("=" * 70 + "\n")

    return A, pi_empirical


# ============================================================
# VITERBI WITH SOFTMAX EMISSIONS (single-best; used at inference time)
# ============================================================
def viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha):
    T, k = obs_probs.shape
    assert k == K
    log_emit = np.log(obs_probs + 1e-10)

    viterbi = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)
    viterbi[0] = log_pi + log_emit[0]

    for t in range(1, T):
        scores = viterbi[t - 1][:, None] + alpha * log_A
        best_prev = np.argmax(scores, axis=0)
        viterbi[t] = scores[best_prev, np.arange(K)] + log_emit[t]
        backptr[t] = best_prev

    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(viterbi[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path   # 0-indexed, matches STAGES directly


def decode_by_runs(obs_probs, bounds, log_A, log_pi, alpha):
    """Viterbi decoded separately on each run; every run starts from the uniform pi."""
    path = np.empty(len(obs_probs), dtype=int)
    for s, e in bounds:
        path[s:e] = viterbi_hmm_softmax(obs_probs[s:e], log_A, log_pi, alpha)
    return path


# ============================================================
# LOADING GROUND TRUTH FOR THE TRAINING SPLIT (one label sequence per RUN)
# ============================================================
def load_nch_train_sequences(index_path, age_bin, seq_len=20, data_source='auto'):
    """Per-run chronological training label sequences. Cache: label files only (x never loaded).
    Parquet: labels come from the index's label_seq column (no EDF decode)."""
    if data_source != 'live':
        from argparse import Namespace
        from datasets.nch_cached_dataset import NCHCachedDataset, cache_status, CACHE_ROOT, LABEL_OFFSET
        ok, reasons = cache_status(Namespace(seq_len=seq_len, age_bin=age_bin))
        if ok:
            ds = NCHCachedDataset(CACHE_ROOT, 'train', age_bin, require_meta=True)
            groups = {}
            for (sp, lp), (subj, study, run_idx, block_idx) in zip(ds.items, ds.meta):
                groups.setdefault((subj, study, run_idx), []).append((block_idx, lp))
            y_true_list = [
                np.concatenate([np.load(lp).astype(np.int64).reshape(-1) - LABEL_OFFSET
                                for _, lp in sorted(v)])
                for v in groups.values()
            ]
            print(f"[{age_bin}] (cache) {len(y_true_list)} training runs, "
                  f"{sum(len(y) for y in y_true_list)} epochs for GT transition estimation.")
            return y_true_list
        if data_source == 'cache':
            raise RuntimeError(f"data_source=cache but cache unusable: {reasons[:5]}")
        print(f"[{age_bin}] cache not usable -> parquet. Reason: {reasons[:3]}")

    from datasets.nch_dataset import NCHIndexDataset   # local import: keeps this module CPU-friendly
    df = NCHIndexDataset(index_path, split='train', age_bin=age_bin, seq_len=seq_len).df
    if 'run_idx' in df.columns:
        keys = ['edf_path', 'run_idx']
    else:
        print("[WARN] index has no run_idx column; treating each recording as one sequence")
        keys = ['edf_path']
    df = df.sort_values(keys + ['seq_start_sec'])
    y_true_list = [np.concatenate([np.asarray(l, dtype=int) for l in g.label_seq])
                   for _, g in df.groupby(keys, sort=False)]
    print(f"[{age_bin}] (parquet) {len(y_true_list)} training runs, "
          f"{sum(len(y) for y in y_true_list)} epochs for GT transition estimation.")
    return y_true_list


# ============================================================
# LOADING TRAIN-SPLIT MODEL SCORES (only needed for hmm_trained)
# ============================================================
def load_nch_train_probs_and_gt(pred_dir, age_bin, seq_len=20):
    """Per-run (probs, ytrue) segments from <pred_dir>/training_scores/, as produced by
    `test_nch.py --age_bin <bin> --split train --pred_dir <pred_dir>`."""
    train_pred_dir = os.path.join(pred_dir, 'training_scores')
    recs = list_recordings(train_pred_dir)
    if not recs:
        raise FileNotFoundError(
            f"No train-split scores found at {train_pred_dir}.\n"
            f"hmm_trained mode needs the model's own softmax scores on the training split. Run:\n"
            f"  python test_nch.py --age_bin {age_bin} --split train "
            f"--pred_dir {pred_dir} --model_dir <checkpoint dir> --index_path <index.parquet>\n"
            f"first, then retry."
        )

    obs_probs_list, y_true_list = [], []
    for rec_id, prob_path in recs:
        probs = np.load(prob_path)
        ytrue = np.load(os.path.join(train_pred_dir, f'{rec_id}_ytrue.npy'))
        bounds = run_bounds(len(ytrue), os.path.join(train_pred_dir, f'{rec_id}_run_idx.json'),
                            seq_len, rec_id)
        for s, e in bounds:
            obs_probs_list.append(probs[s:e])
            y_true_list.append(ytrue[s:e])

    print(f"[{age_bin}] Loaded {len(recs)} train-split recordings -> {len(obs_probs_list)} runs "
          f"from {train_pred_dir} for MMI training.")
    return obs_probs_list, y_true_list


# ============================================================
# MAIN REFINEMENT LOOP
# ============================================================
def refine_age_bin(pred_dir, age_bin, mode, index_path, seq_len=20, pi_source="uniform",
                   alpha_init=0.7, data_source='auto'):
    assert mode in ("hmm", "hmm_trained")
    assert pi_source in ("uniform", "empirical")

    test_pred_dir = pred_dir      # test-split files sit flat in pred_dir (see test_nch.py)
    out_dir = os.path.join(pred_dir, 'hmmRefined')
    os.makedirs(out_dir, exist_ok=True)
    data_source = infer_data_source(pred_dir, data_source)

    # --------------------------------------------------------
    # 1. Estimate (or train) the transition prior from the NCH train split
    # --------------------------------------------------------
    train_gt = load_nch_train_sequences(index_path, age_bin, seq_len=seq_len, data_source=data_source)
    verify_label_range(train_gt, context=f"{age_bin} train GT")
    A_init, pi_empirical = estimate_hmm_parameters_from_gt(train_gt, label=f"NCH {age_bin} (train split)")
    pi_init = get_uniform_pi() if pi_source == "uniform" else pi_empirical
    print(f"[{age_bin}] pi_source={pi_source} -> pi={np.round(pi_init, 4)}")

    if mode == "hmm":
        A, pi, alpha = A_init, pi_init, None
    else:  # hmm_trained
        train_src, test_src = read_source(os.path.join(pred_dir, 'training_scores')), read_source(pred_dir)
        if train_src and test_src and train_src != test_src:
            print(f"[WARN] training scores came from '{train_src}' but test predictions from '{test_src}'")
        obs_probs_list, ytrue_list = load_nch_train_probs_and_gt(pred_dir, age_bin, seq_len=seq_len)
        verify_label_range(ytrue_list, context=f"{age_bin} train GT (from predictions)")
        A, pi, alpha = train_hmm_mmi(obs_probs_list, ytrue_list, A_init, pi_init, alpha_init=alpha_init)
        print(f"\n{'='*70}\nNCH TRAINED HMM PRIOR -- {age_bin}\n{'='*70}")
        print(pd.DataFrame(np.round(A, 4),
                           index=[STAGE_NAMES[s] for s in STAGES],
                           columns=[STAGE_NAMES[s] for s in STAGES]))
        print(f"\nTrained alpha: {alpha:.4f}")
        print("=" * 70 + "\n")

        trained_params_path = os.path.join(out_dir, f"hmm_trained_params_{age_bin}.npz")
        np.savez(trained_params_path, A=A, pi=pi, alpha=alpha, pi_source=pi_source, alpha_init=alpha_init)
        print(f"[{age_bin}] Saved trained HMM parameters to {trained_params_path}")

    log_A = np.log(A + 1e-300)
    log_pi = np.log(pi + 1e-300)

    # --------------------------------------------------------
    # 2. Refine every test-split recording, one decode per run
    # --------------------------------------------------------
    recs = list_recordings(test_pred_dir)
    if not recs:
        raise FileNotFoundError(
            f"No '<subject>_<study>_probs.npy' files found at {test_pred_dir}. "
            f"Run test_nch.py for {age_bin} first (default --split test)."
        )

    for rec_id, prob_path in recs:
        obs_probs = np.load(prob_path)                                        # (T, K)
        y_true = np.load(os.path.join(test_pred_dir, f'{rec_id}_ytrue.npy'))  # (T,)
        y_pred_raw = np.load(os.path.join(test_pred_dir, f'{rec_id}_ypred.npy'))

        assert obs_probs.shape[0] == len(y_true) == len(y_pred_raw), (
            f"{rec_id}: length mismatch between probs ({obs_probs.shape[0]}), "
            f"ytrue ({len(y_true)}), ypred ({len(y_pred_raw)})"
        )
        bounds = run_bounds(len(y_true), os.path.join(test_pred_dir, f'{rec_id}_run_idx.json'),
                            seq_len, rec_id)

        max_prob = obs_probs.max(axis=1)
        ent = entropy(obs_probs, axis=1, base=2)
        print(f"  {rec_id} ({len(bounds)} run(s)) -- mean max-prob: {max_prob.mean():.3f}, "
              f"mean entropy: {ent.mean():.3f} bits (max={np.log2(K):.2f})")

        # ytrue / timing files are copied into hmmRefined/ so downstream scripts only
        # need to look in one directory per experimental arm.
        np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), y_true)
        for ext in ('seq_starts', 'run_idx'):
            src = os.path.join(test_pred_dir, f'{rec_id}_{ext}.json')
            if os.path.exists(src):
                shutil.copy(src, out_dir)

        if mode == "hmm":
            last_path = None
            for a in ALPHA_VALUES:
                path = decode_by_runs(obs_probs, bounds, log_A, log_pi, alpha=a)
                np.save(os.path.join(out_dir, f'{rec_id}_ypred_alpha{a:.1f}.npy'), path)
                last_path = path
            num_changed = np.sum(y_pred_raw != last_path)
            print(f"    (alpha={ALPHA_VALUES[-1]:.1f}) changed epochs vs raw argmax: "
                  f"{num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")
        else:  # hmm_trained
            path = decode_by_runs(obs_probs, bounds, log_A, log_pi, alpha=alpha)
            np.save(os.path.join(out_dir, f'{rec_id}_ypred_trained.npy'), path)
            num_changed = np.sum(y_pred_raw != path)
            print(f"    HMM-trained changed epochs vs raw argmax: "
                  f"{num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")

    print(f"[{age_bin}] Done. Refined predictions saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age_bin', type=str, required=True, choices=AGE_BINS)
    parser.add_argument('--mode', type=str, required=True, choices=['hmm', 'hmm_trained'])
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Base predictions directory, must match the --pred_dir used for test_nch.py.')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Parquet from build_nch_index.py (train-split GT when the parquet path is used)')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Must match the index parquet and the seq_len the predictions were made at.')
    parser.add_argument('--pi_source', type=str, default='uniform', choices=['uniform', 'empirical'],
                        help="'uniform' (default) matches the Neuro-Explicit DNN-HMM paper; 'empirical' is "
                             "available for comparison only.")
    parser.add_argument('--alpha_init', type=float, default=0.7,
                        help="Warm-start alpha for MMI training (only --mode hmm_trained). Pass the alpha "
                             "selected by the eval-set sweep for this age bin.")
    parser.add_argument('--data_source', type=str, default='auto', choices=['auto', 'cache', 'live'],
                        help="Source of the train-split labels used for A. 'auto' follows the data source "
                             "recorded in <pred_dir>/run_meta.json, so A and the predictions share one "
                             "window grid.")
    args = parser.parse_args()

    print(f"\n{'='*60}\n  Age bin: {args.age_bin} | Mode: {args.mode} | seq_len: {args.seq_len} "
          f"| pred_dir: {args.pred_dir}\n{'='*60}\n")
    if args.mode == "hmm_trained":
        print(f"Initial alpha: {args.alpha_init}")
    print(f"{'='*60}\n")
    refine_age_bin(args.pred_dir, args.age_bin, args.mode, args.index_path,
                   seq_len=args.seq_len, pi_source=args.pi_source, alpha_init=args.alpha_init,
                   data_source=args.data_source)


if __name__ == '__main__':
    main()
