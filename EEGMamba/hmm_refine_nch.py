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

Usage (call once per age bin -- loop in your shell script for all bins):
    # empirical prior, alpha sweep
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm \
        --pred_dir /srv/scratch/z5423210/StanleyThesis2026/EEGMamba/predictions/NCH_6-12y \
        --index_path /srv/scratch/z5423210/StanleyThesis2026/nch_index.parquet

    # MMI-trained prior (requires test_nch.py --split train to have
    # already been run for this bin, producing pred_dir/training_scores/)
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm_trained \
        --pred_dir /srv/scratch/z5423210/StanleyThesis2026/EEGMamba/predictions/NCH_6-12y \
        --index_path /srv/scratch/z5423210/StanleyThesis2026/nch_index.parquet
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy

from kNearestViterbi import train_hmm_mmi

# ============================================================
# CONFIGURATION
# ============================================================
AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y", "19-100y"]

STAGES = [0, 1, 2, 3, 4]   # W, N1, N2, N3, REM -- 0-indexed, matching
                           # test_nch.py's pred_y / y_true convention.
                           # This was inferred (not confirmed) during the
                           # SleepTransformer->EEGMamba port -- do not
                           # trust it silently, see verify_label_range().
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
K = len(STAGES)

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]


# ============================================================
# LABEL SANITY CHECK
# ============================================================
def verify_label_range(y_true_list, context=""):
    """
    Raise loudly if labels fall outside 0..K-1. Direct check on the
    0-indexed assumption flagged during the port: SleepTransformer's .mat
    labels are 1-indexed and require `label - 1`; EEGMamba's test_nch.py
    compares pred_y to y with no offset, suggesting NCH is already
    0-indexed -- but that was an inference, not a verified fact, so check
    it here every run rather than trusting it silently.
    """
    all_vals = np.concatenate(y_true_list)
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 0 or hi > K - 1:
        raise ValueError(
            f"[{context}] labels outside expected 0..{K-1} range: "
            f"observed min={lo}, max={hi}. Check NCHIndexDataset's label "
            f"convention before trusting any HMM output from this run."
        )
    print(f"[{context}] label range check passed: min={lo}, max={hi}")


# ============================================================
# HMM PARAMETER ESTIMATION FROM TRAINING GROUND TRUTH
# ============================================================
def get_uniform_pi():
    """
    Fixed uniform initial-state distribution, per the paper this method is
    based on ("A Neuro-Explicit DNN-HMM Approach..." Section 2.2): pi is
    NOT estimated from data, it is set uniform over the K states so that
    only the DNN's own posteriorgram influences the very first epoch's
    decoding. This is PI_SOURCE="uniform" (the default).
    """
    return np.ones(K) / K


def estimate_hmm_parameters_from_gt(y_true_list, label="", verbose=True):
    """
    Estimates the transition matrix A via pooled MLE counts over the
    training ground-truth sequences (matches the paper's "transition
    matrix initialized with the averaged transitions over the entire
    dataset"). Also returns an empirically-estimated pi (from the first
    label of each sequence) for comparison purposes only -- the paper
    itself does NOT use this; see get_uniform_pi() and PI_SOURCE.
    """
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
        print(pd.DataFrame(
            np.round(A, 4),
            index=[STAGE_NAMES[s] for s in STAGES],
            columns=[STAGE_NAMES[s] for s in STAGES]
        ))
        print("\nInitial State Distribution pi (empirical alternative -- NOT necessarily "
              "what gets used; see PI_SOURCE / --pi_source)")
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

    return path   # 0-indexed, matches STAGES directly -- no +1 offset
                   # needed here (unlike the SleepTransformer version,
                   # which re-added 1 for its 1-indexed label convention)


# ============================================================
# LOADING GROUND TRUTH FOR THE TRAINING SPLIT
# ============================================================
def load_nch_train_sequences(index_path, age_bin):
    """
    Reconstruct per-recording, chronologically-ordered label sequences for
    the NCH training split, for empirical transition-matrix estimation.

    NOTE: this currently goes through NCHIndexDataset.__getitem__, which
    decodes EEG (x) even though only the label (y) is needed here -- that
    is wasted work. If NCHIndexDataset's underlying dataframe already
    carries per-epoch labels without requiring EDF decode (worth checking
    directly against its source), swap this for a direct dataframe read.
    As written this is correct but should be run once as a CPU-only job,
    not folded into a GPU inference run.
    """
    from datasets.nch_dataset import NCHIndexDataset  # local import: this
                                                        # module has no
                                                        # other torch/cuda
                                                        # dependency, keep
                                                        # it that way so it
                                                        # can run on a
                                                        # CPU-only node
    train_set = NCHIndexDataset(index_path, split='train', age_bin=age_bin)
    sorted_df = train_set.df.sort_values(['edf_path', 'seq_start_sec'])
    recordings = sorted_df.groupby('edf_path', sort=False).groups

    y_true_list = []
    for edf_path, row_positions in recordings.items():
        labels = []
        for pos in row_positions:
            _, y = train_set[pos]
            labels += y.numpy().tolist()
        y_true_list.append(np.array(labels, dtype=int))

    print(f"[{age_bin}] Loaded {len(y_true_list)} training recordings "
          f"({sum(len(y) for y in y_true_list)} total epochs) for GT transition estimation.")
    return y_true_list


# ============================================================
# LOADING TRAIN-SPLIT MODEL SCORES (only needed for hmm_trained)
# ============================================================
def load_nch_train_probs_and_gt(pred_dir, age_bin):
    """
    Loads {rec_id}_probs.npy / {rec_id}_ytrue.npy pairs from
    <pred_dir>/training_scores/, as produced by
    `test_nch.py --age_bin <bin> --split train --pred_dir <pred_dir>`.

    Raises with a clear, actionable message (does not silently skip) if
    that hasn't been run yet -- hmm_trained is not usable without it.
    """
    train_pred_dir = os.path.join(pred_dir, 'training_scores')

    prob_files = sorted(glob.glob(os.path.join(train_pred_dir, '*_probs.npy')))
    if not prob_files:
        raise FileNotFoundError(
            f"No train-split scores found at {train_pred_dir}.\n"
            f"hmm_trained mode needs the model's own softmax scores on the "
            f"training split. Run:\n"
            f"  python test_nch.py --age_bin {age_bin} --split train "
            f"--pred_dir {pred_dir} --model_dir <checkpoint dir> "
            f"--index_path <index.parquet>\n"
            f"first, then retry."
        )

    obs_probs_list = []
    y_true_list = []
    for prob_path in prob_files:
        rec_id = os.path.basename(prob_path).replace('_probs.npy', '')
        ytrue_path = os.path.join(train_pred_dir, f'{rec_id}_ytrue.npy')
        obs_probs_list.append(np.load(prob_path))
        y_true_list.append(np.load(ytrue_path))

    print(f"[{age_bin}] Loaded {len(obs_probs_list)} train-split score/label pairs "
          f"from {train_pred_dir} for MMI training.")
    return obs_probs_list, y_true_list


# ============================================================
# MAIN REFINEMENT LOOP
# ============================================================
def refine_age_bin(pred_dir, age_bin, mode, index_path, pi_source="uniform"):
    assert mode in ("hmm", "hmm_trained")
    assert pi_source in ("uniform", "empirical")

    # Test-split files (ypred/ytrue/probs) sit flat in pred_dir, matching
    # the existing on-disk convention -- see test_nch.py.
    test_pred_dir = pred_dir
    out_dir = os.path.join(pred_dir, 'hmmRefined')
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Estimate (or train) the transition prior from NCH train split
    # --------------------------------------------------------
    train_gt = load_nch_train_sequences(index_path, age_bin)
    verify_label_range(train_gt, context=f"{age_bin} train GT")
    A_init, pi_empirical = estimate_hmm_parameters_from_gt(
        train_gt, label=f"NCH {age_bin} (train split)"
    )
    pi_init = get_uniform_pi() if pi_source == "uniform" else pi_empirical
    print(f"[{age_bin}] pi_source={pi_source} -> pi={np.round(pi_init, 4)}")

    if mode == "hmm":
        A, pi, alpha = A_init, pi_init, None
    else:  # hmm_trained
        obs_probs_list, ytrue_list = load_nch_train_probs_and_gt(pred_dir, age_bin)
        verify_label_range(ytrue_list, context=f"{age_bin} train GT (from predictions)")
        A, pi, alpha = train_hmm_mmi(obs_probs_list, ytrue_list, A_init, pi_init)
        print(f"\n{'='*70}\nNCH TRAINED HMM PRIOR -- {age_bin}\n{'='*70}")
        print(pd.DataFrame(np.round(A, 4),
                            index=[STAGE_NAMES[s] for s in STAGES],
                            columns=[STAGE_NAMES[s] for s in STAGES]))
        print(f"\nTrained alpha: {alpha:.4f}")
        print("=" * 70 + "\n")

    log_A = np.log(A + 1e-300)
    log_pi = np.log(pi + 1e-300)

    # --------------------------------------------------------
    # 2. Refine every test-split recording
    # --------------------------------------------------------
    prob_files = sorted(glob.glob(os.path.join(test_pred_dir, '*_probs.npy')))
    if not prob_files:
        raise FileNotFoundError(
            f"No test-split predictions found at {test_pred_dir}. "
            f"Run test_nch.py for {age_bin} first (default --split test)."
        )

    for prob_path in prob_files:
        rec_id = os.path.basename(prob_path).replace('_probs.npy', '')
        obs_probs = np.load(prob_path)                                       # (T, K)
        y_true = np.load(os.path.join(test_pred_dir, f'{rec_id}_ytrue.npy'))  # (T,)
        y_pred_raw = np.load(os.path.join(test_pred_dir, f'{rec_id}_ypred.npy'))

        assert obs_probs.shape[0] == len(y_true) == len(y_pred_raw), (
            f"{rec_id}: length mismatch between probs ({obs_probs.shape[0]}), "
            f"ytrue ({len(y_true)}), ypred ({len(y_pred_raw)})"
        )

        max_prob = obs_probs.max(axis=1)
        ent = entropy(obs_probs, axis=1, base=2)
        print(f"  {rec_id} -- mean max-prob: {max_prob.mean():.3f}, "
              f"mean entropy: {ent.mean():.3f} bits (max={np.log2(K):.2f})")

        # ytrue copied into hmmRefined/ too, so downstream analysis scripts
        # only ever need to look in one directory per experimental arm.
        np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), y_true)

        if mode == "hmm":
            last_path = None
            for a in ALPHA_VALUES:
                path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=a)
                np.save(os.path.join(out_dir, f'{rec_id}_ypred_alpha{a:.1f}.npy'), path)
                last_path = path
            num_changed = np.sum(y_pred_raw != last_path)  # last (largest) alpha, for the log line
            print(f"    (alpha={ALPHA_VALUES[-1]:.1f}) changed epochs vs raw argmax: "
                  f"{num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")

        else:  # hmm_trained
            path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=alpha)
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
                         help='Base predictions directory for this age bin, e.g. '
                              'EEGMamba/predictions/NCH_1-2y. Must match the --pred_dir '
                              'used for the corresponding test_nch.py run(s).')
    parser.add_argument('--index_path', type=str, required=True,
                         help='Path to the parquet produced by build_nch_index.py '
                              '(needed to reconstruct train-split GT sequences)')
    parser.add_argument('--pi_source', type=str, default='uniform',
                         choices=['uniform', 'empirical'],
                         help="Initial-state distribution. 'uniform' (default) matches "
                              "the Neuro-Explicit DNN-HMM paper this method is based on "
                              "-- pi is fixed uniform over states, not estimated from data. "
                              "'empirical' (pi from the first label of each training "
                              "sequence) is available for comparison but is NOT what the "
                              "paper does.")
    args = parser.parse_args()

    print(f"\n{'='*60}\n  Age bin: {args.age_bin} | Mode: {args.mode} | pred_dir: {args.pred_dir}\n{'='*60}\n")
    refine_age_bin(args.pred_dir, args.age_bin, args.mode, args.index_path, args.pi_source)


if __name__ == '__main__':
    main()
