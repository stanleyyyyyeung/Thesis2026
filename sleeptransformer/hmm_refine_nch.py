"""
hmm_refine_nch.py (SleepTransformer) -- HMM/Viterbi refinement of NCH predictions.
Mirrors the EEGMamba hmm_refine_nch.py: one age bin per call, CLI-driven.

Two refinement modes:
  "hmm"          Transition matrix A estimated via MLE from NCH training-split
                 ground truth. pi is FIXED UNIFORM by default (per the
                 Neuro-Explicit DNN-HMM paper). Alpha swept over ALPHA_VALUES.
  "hmm_trained"  A and alpha jointly learned via MMI (kNearestViterbi.train_hmm_mmi),
                 warm-started from the empirical A and --alpha_init. Needs
                 SHHS-pretrained scores on the NCH TRAIN split (--train_inference_dir).

INDEXING CONVENTION (SleepTransformer):
  On disk (.mat labels, saved ytrue/ypred .npy): 1-indexed  W=1 N1=2 N2=3 N3=4 REM=5
  Internally (A, pi, Viterbi, MMI):              0-indexed  W=0 ... REM=4
  Conversion happens in exactly two places: load (label - 1) and save (path + 1).

Usage:
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm
    python hmm_refine_nch.py --age_bin 6-12y --mode hmm_trained --alpha_init 0.7
"""

import argparse
import os

import hdf5storage
import numpy as np
import pandas as pd
import json
from scipy.stats import entropy

from kNearestViterbi import train_hmm_mmi

# ============================================================
# CONFIGURATION
# ============================================================
AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y"]

STAGES = [0, 1, 2, 3, 4]   # internal 0-indexed: W, N1, N2, N3, REM
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
K = len(STAGES)
SEQ_LEN = 21               # SleepTransformer context length

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]

BASE = "/srv/scratch/z5423210/StanleyThesis2026"
DEFAULT_TEST_INFERENCE_BASE = f"{BASE}/out_sleeptransformer/nch/run1"
DEFAULT_TRAIN_INFERENCE_BASE = f"{BASE}/out_sleeptransformer/nch/run1/training_scores"
DEFAULT_LIST_DIR = "/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"
DEFAULT_PRED_ROOT = f"{BASE}/out_sleeptransformer/nch/run1/predictions"


# ============================================================
# LABEL SANITY CHECKS
# ============================================================
def verify_raw_labels_one_indexed(y_list, context=""):
    """Raw .mat labels MUST be in 1..K. Fails loudly otherwise (guards the
    1-indexed assumption before we subtract 1)."""
    all_vals = np.concatenate([np.asarray(y).ravel() for y in y_list])
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 1 or hi > K:
        raise ValueError(
            f"[{context}] raw labels outside expected 1..{K}: min={lo}, max={hi}. "
            f"SleepTransformer labels should be 1-indexed."
        )
    print(f"[{context}] raw label range check passed (1-indexed): min={lo}, max={hi}")


def verify_label_range(y_list, context=""):
    """Internal (post label-1) labels MUST be in 0..K-1."""
    all_vals = np.concatenate(y_list)
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 0 or hi > K - 1:
        raise ValueError(
            f"[{context}] internal labels outside 0..{K-1}: min={lo}, max={hi}. "
            f"Did the 1-indexed -> 0-indexed conversion run?"
        )
    print(f"[{context}] internal label range check passed (0-indexed): min={lo}, max={hi}")


# ============================================================
# AGGREGATION (unchanged from original SleepTransformer pipeline)
# ============================================================
def softmax(z):
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def aggregate_probs(score):
    """Multiply-aggregate across seq_len context positions -> (N, K) probs.
    score: (SEQ_LEN, N_valid, K); returns N_valid + SEQ_LEN - 1 rows."""
    fused_score = None
    for i in range(SEQ_LEN):
        prob_i = np.log10(softmax(np.squeeze(score[i, :, :])))
        prob_i = np.concatenate((np.ones((SEQ_LEN - 1, K)), prob_i), axis=0)
        prob_i = np.roll(prob_i, -(SEQ_LEN - i - 1), axis=0)
        fused_score = prob_i if fused_score is None else fused_score + prob_i
    fused_score -= np.max(fused_score, axis=1, keepdims=True)
    probs = np.exp(fused_score)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


# ============================================================
# HMM PARAMETERS
# ============================================================
def get_uniform_pi():
    """Fixed uniform pi, per the paper (Section 2.2): pi is NOT estimated."""
    return np.ones(K) / K


def estimate_hmm_parameters_from_gt(y_true_list, label="", verbose=True):
    """Pooled-MLE transition matrix A over 0-indexed training GT. Also returns
    an empirical pi for comparison only (NOT what the paper uses)."""
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
# VITERBI (single-best; inference time)
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
    return path   # 0-indexed


# ============================================================
# DATA LOADING
# ============================================================
def load_list(list_path):
    """Returns [(mat_path, n_epochs), ...]."""
    out = []
    with open(list_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                out.append((parts[0], int(parts[1])))
    return out


def load_labels_raw(files):
    """Raw 1-indexed labels, one array per recording."""
    ys = []
    for fpath in files:
        data = hdf5storage.loadmat(file_name=fpath)
        ys.append(np.array(data['label']).squeeze().astype(int))
    return ys


def scores_to_probs_list(mat_path, lengths):
    """Split a concatenated test_ret.mat score tensor into per-recording
    aggregated (n_epochs, K) probability arrays. `lengths` = full epoch
    counts per recording, in list order."""
    mat = hdf5storage.loadmat(mat_path)
    score = np.transpose(mat['score'], (1, 0, 2))   # (SEQ_LEN, N_total, K)

    probs_list, sum_size = [], 0
    for n in lengths:
        valid_len = n - (SEQ_LEN - 1)
        probs_list.append(aggregate_probs(score[:, sum_size:sum_size + valid_len, :]))
        sum_size += valid_len
    assert sum_size == score.shape[1], (
        f"Score/list misalignment in {mat_path}: consumed {sum_size} positions, "
        f"score tensor has {score.shape[1]}. Lists and inference run do not match."
    )
    return probs_list


def rec_id_from_path(fpath):
    parts = os.path.basename(fpath).replace("_eeg.mat", "").split("_")
    return f"{parts[0]}_night{parts[1]}"

def resolve_alpha_init(args, age_bin):
    """Warm-start alpha for MMI: manual override, else the eval-split selection JSON."""
    if args.alpha_source == "manual":
        if args.alpha_init is None:
            raise ValueError("--alpha_source manual requires --alpha_init")
        return args.alpha_init, "manual"

    path = os.path.join(args.pred_dir, f"alpha_selection_nch_{age_bin}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `select_alpha_eval.py --dataset nch --unit {age_bin}` first, "
            f"or pass --alpha_source manual --alpha_init X.")
    with open(path) as f:
        d = json.load(f)
    alpha = float(d["selected_alpha"])
    metric = d.get("selection_metric")
    print(f"[{age_bin}] alpha_init={alpha} from {path} (selection_metric={metric})")
    if alpha < 0.1:
        print(f"[{age_bin}] WARNING: selected alpha={alpha} switches the transition prior off; "
              f"clamping warm start to 0.1")
        alpha = 0.1
    return alpha, metric

# ============================================================
# MAIN REFINEMENT
# ============================================================
def refine_age_bin(args):
    age_bin, mode = args.age_bin, args.mode
    pred_dir = args.pred_dir
    out_dir = os.path.join(pred_dir, 'hmmRefined')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Estimate (or train) the transition prior from NCH train split
    train_entries = load_list(os.path.join(args.list_dir, f"train_list_{age_bin}.txt"))
    train_raw = load_labels_raw([p for p, _ in train_entries])
    verify_raw_labels_one_indexed(train_raw, context=f"{age_bin} train GT")
    train_gt = [y - 1 for y in train_raw]
    verify_label_range(train_gt, context=f"{age_bin} train GT")

    A_init, pi_empirical = estimate_hmm_parameters_from_gt(
        train_gt, label=f"NCH {age_bin} (train split)")
    pi_init = get_uniform_pi() if args.pi_source == "uniform" else pi_empirical
    print(f"[{age_bin}] pi_source={args.pi_source} -> pi={np.round(pi_init, 4)}")

    if mode == "hmm":
        A, pi, alpha = A_init, pi_init, None
    else:
        train_score_path = os.path.join(args.train_inference_dir, age_bin, "test_ret.mat")
        if not os.path.exists(train_score_path):
            raise FileNotFoundError(
                f"No train-split scores at {train_score_path}. hmm_trained needs the "
                f"SHHS-pretrained model's scores on the NCH training split; run that "
                f"inference job first."
            )
        obs_probs_list = scores_to_probs_list(train_score_path, [len(y) for y in train_gt])
        alpha_init, alpha_sel_metric = resolve_alpha_init(args, age_bin)
        A, pi, alpha = train_hmm_mmi(obs_probs_list, train_gt, A_init, pi_init,
                                     alpha_init=args.alpha_init)
        print(f"\n{'='*70}\nNCH TRAINED HMM PRIOR -- {age_bin}\n{'='*70}")
        print(pd.DataFrame(np.round(A, 4),
                           index=[STAGE_NAMES[s] for s in STAGES],
                           columns=[STAGE_NAMES[s] for s in STAGES]))
        print(f"\nTrained alpha: {alpha:.4f}\n{'='*70}\n")
        params_path = os.path.join(out_dir, f"hmm_trained_params_{age_bin}.npz")
        np.savez(params_path, A=A, pi=pi, alpha=alpha, pi_source=args.pi_source,
                 alpha_init=alpha_init, alpha_source=args.alpha_source,
                 alpha_selection_metric=str(alpha_sel_metric))
        print(f"[{age_bin}] Saved trained HMM parameters to {params_path}")

    log_A = np.log(A + 1e-300)
    log_pi = np.log(pi + 1e-300)

    # 2. Refine every test-split recording
    test_score_path = os.path.join(args.test_inference_dir, age_bin, "test_ret.mat")
    if not os.path.exists(test_score_path):
        raise FileNotFoundError(f"No test-split scores at {test_score_path}.")

    test_entries = load_list(os.path.join(args.list_dir, f"test_list_{age_bin}.txt"))
    test_raw = load_labels_raw([p for p, _ in test_entries])
    verify_raw_labels_one_indexed(test_raw, context=f"{age_bin} test GT")
    for (p, n), y in zip(test_entries, test_raw):
        assert len(y) == n, f"{p}: list says {n} epochs, label has {len(y)}"

    test_probs = scores_to_probs_list(test_score_path, [n for _, n in test_entries])

    for (fpath, _), y_raw, obs_probs in zip(test_entries, test_raw, test_probs):
        rec_id = rec_id_from_path(fpath)
        y_true0 = y_raw - 1
        y_pred_raw0 = np.argmax(obs_probs, axis=-1)

        assert obs_probs.shape[0] == len(y_true0) == len(y_pred_raw0), (
            f"{rec_id}: length mismatch probs ({obs_probs.shape[0]}), "
            f"ytrue ({len(y_true0)}), ypred ({len(y_pred_raw0)})"
        )

        max_prob = obs_probs.max(axis=1)
        ent = entropy(obs_probs, axis=1, base=2)
        print(f"  {rec_id} -- mean max-prob: {max_prob.mean():.3f}, "
              f"mean entropy: {ent.mean():.3f} bits (max={np.log2(K):.2f})")

        # Raw baseline artefacts (1-indexed on disk) into pred_dir
        os.makedirs(pred_dir, exist_ok=True)
        np.save(os.path.join(pred_dir, f'{rec_id}_ytrue.npy'), y_true0 + 1)
        np.save(os.path.join(pred_dir, f'{rec_id}_ypred.npy'), y_pred_raw0 + 1)
        np.save(os.path.join(pred_dir, f'{rec_id}_probs.npy'), obs_probs)
        # ytrue also copied into hmmRefined/ so downstream needs one dir per arm
        np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), y_true0 + 1)

        if mode == "hmm":
            last_path = None
            for a in ALPHA_VALUES:
                path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=a)
                np.save(os.path.join(out_dir, f'{rec_id}_ypred_alpha{a:.1f}.npy'), path + 1)
                last_path = path
            num_changed = np.sum(y_pred_raw0 != last_path)
            print(f"    (alpha={ALPHA_VALUES[-1]:.1f}) changed epochs vs raw argmax: "
                  f"{num_changed}/{len(y_pred_raw0)} ({100*num_changed/len(y_pred_raw0):.2f}%)")
        else:
            path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=alpha)
            np.save(os.path.join(out_dir, f'{rec_id}_ypred_trained.npy'), path + 1)
            num_changed = np.sum(y_pred_raw0 != path)
            print(f"    HMM-trained changed epochs vs raw argmax: "
                  f"{num_changed}/{len(y_pred_raw0)} ({100*num_changed/len(y_pred_raw0):.2f}%)")

    print(f"[{age_bin}] Done. Refined predictions saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age_bin', type=str, required=True, choices=AGE_BINS)
    parser.add_argument('--mode', type=str, required=True, choices=['hmm', 'hmm_trained'])
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Output dir for this age bin. Default: '
                             f'{DEFAULT_PRED_ROOT}/<age_bin>. Refined output goes to <pred_dir>/hmmRefined/')
    parser.add_argument('--test_inference_dir', type=str, default=DEFAULT_TEST_INFERENCE_BASE,
                        help='Base dir containing <age_bin>/test_ret.mat for the test split')
    parser.add_argument('--train_inference_dir', type=str, default=DEFAULT_TRAIN_INFERENCE_BASE,
                        help='Base dir containing <age_bin>/test_ret.mat for the TRAIN split '
                             '(only needed for hmm_trained)')
    parser.add_argument('--list_dir', type=str, default=DEFAULT_LIST_DIR)
    parser.add_argument('--pi_source', type=str, default='uniform', choices=['uniform', 'empirical'],
                        help="'uniform' (default) matches the Neuro-Explicit DNN-HMM paper; "
                             "'empirical' is for comparison only.")
    parser.add_argument('--alpha_source', default='bin', choices=['bin', 'manual'],
                        help="'bin': read the age bin's eval-selected alpha from its JSON; "
                             "'manual': use --alpha_init")
    parser.add_argument('--alpha_init', type=float, default=None,
                        help="Only used with --alpha_source manual")
    args = parser.parse_args()
    if args.pred_dir is None:
        args.pred_dir = os.path.join(DEFAULT_PRED_ROOT, args.age_bin)

    print(f"\n{'='*60}\n  Age bin: {args.age_bin} | Mode: {args.mode} | pred_dir: {args.pred_dir}\n{'='*60}\n")
    if args.mode == "hmm_trained":
        print(f"Initial alpha: {args.alpha_init}")
    refine_age_bin(args)


if __name__ == '__main__':
    main()
