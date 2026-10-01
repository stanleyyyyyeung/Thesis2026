"""
hmm_refine_sleepedf.py (SleepTransformer) -- HMM/Viterbi refinement for
SleepEDF-20 and SleepEDF-78, mirroring the EEGMamba hmm_refine_nch.py.

Modes:
  hmm          A via pooled MLE from the fold's TRAIN-split GT. pi is FIXED
               UNIFORM by default (Neuro-Explicit DNN-HMM paper, Sec 2.2).
               Alpha swept over ALPHA_VALUES at inference.
  hmm_trained  A and alpha learned via MMI (kNearestViterbi.train_hmm_mmi),
               warm-started from empirical A and --alpha_init. Needs
               <run_root>/training_scores/<fold>/test_ret.mat.

INDEXING: on disk (.mat labels, saved ytrue/ypred) = 1-indexed (W=1..REM=5).
Internally (A, pi, Viterbi, MMI) = 0-indexed. Conversion happens only at
load (label - 1) and save (path + 1).

Usage:
    python hmm_refine_sleepedf.py --dataset sleepedf-78 --mode hmm --fold n1
    python hmm_refine_sleepedf.py --dataset sleepedf-20 --mode hmm_trained --fold all --alpha_init 0.7
"""
import argparse
import os

import hdf5storage
import numpy as np
import pandas as pd
from scipy.stats import entropy

from kNearestViterbi import train_hmm_mmi

# ============================================================
# CONFIGURATION
# ============================================================
BASE = "/srv/scratch/z5423210/StanleyThesis2026"
CODE = f"{BASE}/sleeptransformer"

# Where each dataset keeps its train/test list files
LIST_DIRS = {
    "sleepedf-78": f"{CODE}/sleepedf-78/file_list_30min/eeg",
    "sleepedf-20": f"{CODE}",
}

STAGES = [0, 1, 2, 3, 4]   # internal 0-indexed: W, N1, N2, N3, REM
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
K = len(STAGES)
SEQ_LEN = 21

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]


# ============================================================
# LABEL SANITY CHECKS
# ============================================================
def verify_raw_labels_one_indexed(y_list, context=""):
    all_vals = np.concatenate([np.asarray(y).ravel() for y in y_list])
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 1 or hi > K:
        raise ValueError(f"[{context}] raw labels outside expected 1..{K}: min={lo}, max={hi}.")
    print(f"[{context}] raw label range check passed (1-indexed): min={lo}, max={hi}")


def verify_label_range(y_list, context=""):
    all_vals = np.concatenate(y_list)
    lo, hi = int(all_vals.min()), int(all_vals.max())
    if lo < 0 or hi > K - 1:
        raise ValueError(f"[{context}] internal labels outside 0..{K-1}: min={lo}, max={hi}.")
    print(f"[{context}] internal label range check passed (0-indexed): min={lo}, max={hi}")


# ============================================================
# AGGREGATION
# ============================================================
def softmax(z):
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def aggregate_probs(score):
    """score: (SEQ_LEN, N_valid, K) -> (N_valid + SEQ_LEN - 1, K) normalised probs.
    argmax of this equals the old aggregate_mul argmax, so raw predictions are unchanged."""
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
    return np.ones(K) / K


def estimate_hmm_parameters_from_gt(y_true_list, label="", verbose=True):
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
        print(f"\n{'='*70}\nHMM PARAMETERS ({label}) -- from training ground truth\n{'='*70}")
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
# VITERBI
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
    """Returns file paths in LIST ORDER (must match the inference run's order;
    do not sort -- the score tensor is concatenated in list order)."""
    with open(list_path, "r") as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]


def load_list_counts(list_path):
    """[(path, n_epochs_declared_in_list_or_None)] in list order. The score tensor
    was produced using THESE counts, so slicing must use them."""
    out = []
    with open(list_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                out.append((parts[0], int(parts[1]) if len(parts) > 1 else None))
    return out


def load_labels_raw(files):
    ys = []
    for fpath in files:
        data = hdf5storage.loadmat(file_name=fpath)
        ys.append(np.array(data['label']).squeeze().astype(int))
    return ys


def scores_to_probs_list(mat_path, entries, label_lengths):
    """Split concatenated test_ret.mat scores into per-recording (n, K) probs.

    Slices with the LIST-declared epoch counts (what inference used). If a
    recording's list count != its actual label length (stale list vs regenerated
    .mat), the scores cannot be aligned to labels: that recording yields None
    and is skipped/reported, but sum_size still advances so later nights stay
    aligned. The final assert then checks the tensor is fully consumed.
    """
    mat = hdf5storage.loadmat(mat_path)
    score = np.transpose(mat['score'], (1, 0, 2))   # (SEQ_LEN, N_total, K)

    probs_list, sum_size, mismatched = [], 0, []
    for (fpath, n_list), n_label in zip(entries, label_lengths):
        n = n_list if n_list is not None else n_label
        valid_len = n - (SEQ_LEN - 1)
        if n != n_label:
            mismatched.append((os.path.basename(fpath), n, n_label))
            probs_list.append(None)
        else:
            probs_list.append(aggregate_probs(score[:, sum_size:sum_size + valid_len, :]))
        sum_size += valid_len
    if mismatched:
        print(f"  WARNING: {len(mismatched)} recording(s) with list count != label length "
              f"(skipped; regenerate lists + rerun inference to fix):")
        for name, nl, nb in mismatched:
            print(f"    {name}: list={nl}, label={nb}, diff={nl - nb}")
    assert sum_size == score.shape[1], (
        f"Score/list misalignment in {mat_path}: consumed {sum_size} positions, "
        f"tensor has {score.shape[1]}. The inference run used a different list than this one."
    )
    return probs_list


def rec_id_from_path(fpath):
    parts = os.path.basename(fpath).replace("_eeg.mat", "").split("_")
    return f"{parts[0]}_night{parts[1]}"


def discover_folds(pred_base):
    folds = [d for d in os.listdir(pred_base)
             if os.path.exists(os.path.join(pred_base, d, "test_ret.mat"))]
    return sorted(folds, key=lambda d: int(d[1:]))


def resolve_alpha_init(args, fold, pred_dir):
    """Warm-start alpha for MMI: manual override, else the eval-split selection JSON."""
    if args.alpha_source == "manual":
        if args.alpha_init is None:
            raise ValueError("--alpha_source manual requires --alpha_init")
        return args.alpha_init, "manual"

    name = f"alpha_selection_{args.dataset}_{'global' if args.alpha_source == 'global' else fold}.json"
    path = os.path.join(pred_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run select_alpha_eval.py first "
            f"(--unit all for the global file), or pass --alpha_source manual --alpha_init X.")
    with open(path) as f:
        d = json.load(f)
    alpha = float(d["selected_alpha"])
    print(f"[{fold}] alpha_init={alpha} from {path} (selection_metric={d.get('selection_metric')})")

    if alpha < 0.1:
        print(f"[{age_bin}] WARNING: selected alpha={alpha} switches the transition prior off; "
              f"clamping warm start to 0.1")
        alpha = 0.1

    return alpha, d.get("selection_metric")


# ============================================================
# PER-FOLD REFINEMENT
# ============================================================
def refine_fold(fold, args, paths):
    fold_id = fold[1:]                     # "n15" -> "15"
    list_dir = paths["list_dir"]
    out_dir = os.path.join(paths["pred_dir"], "hmmRefined")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}\n  {args.dataset} | fold {fold} | mode {args.mode}\n{'='*60}\n")

    # 1. Prior from this fold's TRAIN split
    train_entries = load_list_counts(os.path.join(list_dir, f"train_list_n{fold_id}.txt"))
    train_files = [p for p, _ in train_entries]
    train_raw = load_labels_raw(train_files)
    verify_raw_labels_one_indexed(train_raw, context=f"{fold} train GT")
    train_gt = [y - 1 for y in train_raw]
    verify_label_range(train_gt, context=f"{fold} train GT")

    A_init, pi_empirical = estimate_hmm_parameters_from_gt(train_gt, label=f"{args.dataset} {fold} (train split)")
    pi_init = get_uniform_pi() if args.pi_source == "uniform" else pi_empirical
    print(f"[{fold}] pi_source={args.pi_source} -> pi={np.round(pi_init, 4)}")

    if args.mode == "hmm":
        A, pi, alpha = A_init, pi_init, None
    else:
        train_score_path = os.path.join(paths["train_score_base"], fold, "test_ret.mat")
        if not os.path.exists(train_score_path):
            raise FileNotFoundError(f"No train-split scores at {train_score_path}; run the training-score inference first.")
        all_probs = scores_to_probs_list(train_score_path, train_entries, [len(y) for y in train_gt])
        keep = [i for i, p in enumerate(all_probs) if p is not None]
        obs_probs_list = [all_probs[i] for i in keep]
        mmi_gt = [train_gt[i] for i in keep]
        print(f"[{fold}] MMI training on {len(keep)}/{len(all_probs)} train recordings")
        alpha_init, alpha_sel_metric = resolve_alpha_init(args, fold, paths["pred_dir"])
        A, pi, alpha = train_hmm_mmi(obs_probs_list, mmi_gt, A_init, pi_init, alpha_init=alpha_init)
        print(f"\n{'='*70}\nTRAINED HMM PRIOR -- {args.dataset} {fold}\n{'='*70}")
        print(pd.DataFrame(np.round(A, 4),
                           index=[STAGE_NAMES[s] for s in STAGES],
                           columns=[STAGE_NAMES[s] for s in STAGES]))
        print(f"\nTrained alpha: {alpha:.4f}\n{'='*70}\n")
        params_path = os.path.join(out_dir, f"hmm_trained_params_{fold}.npz")
        np.savez(params_path, A=A, pi=pi, alpha=alpha, pi_source=args.pi_source,
                 alpha_init=alpha_init, alpha_source=args.alpha_source,
                 alpha_selection_metric=str(alpha_sel_metric))
        print(f"[{fold}] Saved trained HMM parameters to {params_path}")

    log_A = np.log(A + 1e-300)
    log_pi = np.log(pi + 1e-300)

    # 2. Refine every test recording of this fold
    test_score_path = os.path.join(paths["pred_base"], fold, "test_ret.mat")
    test_entries = load_list_counts(os.path.join(list_dir, f"test_list_n{fold_id}.txt"))
    test_files = [p for p, _ in test_entries]
    test_raw = load_labels_raw(test_files)
    verify_raw_labels_one_indexed(test_raw, context=f"{fold} test GT")
    test_probs = scores_to_probs_list(test_score_path, test_entries, [len(y) for y in test_raw])

    for fpath, y_raw, obs_probs in zip(test_files, test_raw, test_probs):
        rec_id = rec_id_from_path(fpath)
        if obs_probs is None:
            print(f"  {rec_id} -- SKIPPED (list/label length mismatch)")
            continue
        y_true0 = y_raw - 1
        y_pred_raw0 = np.argmax(obs_probs, axis=-1)

        assert obs_probs.shape[0] == len(y_true0) == len(y_pred_raw0), (
            f"{rec_id}: length mismatch probs ({obs_probs.shape[0]}), "
            f"ytrue ({len(y_true0)}), ypred ({len(y_pred_raw0)})")

        max_prob = obs_probs.max(axis=1)
        ent = entropy(obs_probs, axis=1, base=2)
        print(f"  {rec_id} -- mean max-prob: {max_prob.mean():.3f}, "
              f"mean entropy: {ent.mean():.3f} bits (max={np.log2(K):.2f})")

        # Raw baseline (replaces old REFINEMENT_MODE="none"), 1-indexed on disk
        np.save(os.path.join(paths["pred_dir"], f'{rec_id}_ytrue.npy'), y_true0 + 1)
        np.save(os.path.join(paths["pred_dir"], f'{rec_id}_ypred.npy'), y_pred_raw0 + 1)
        np.save(os.path.join(paths["pred_dir"], f'{rec_id}_probs.npy'), obs_probs)
        np.save(os.path.join(out_dir, f'{rec_id}_ytrue.npy'), y_true0 + 1)

        if args.mode == "hmm":
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

    print(f"[{fold}] Done. Refined predictions saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(LIST_DIRS))
    parser.add_argument('--mode', required=True, choices=['hmm', 'hmm_trained'])
    parser.add_argument('--run', type=int, default=1, help='RUN_NUMBER (default 1)')
    parser.add_argument('--fold', default='all',
                        help="Fold name like n1, or 'all' to loop every fold with a test_ret.mat")
    parser.add_argument('--pi_source', default='uniform', choices=['uniform', 'empirical'],
                        help="'uniform' (default) matches the paper; 'empirical' for comparison only.")
    parser.add_argument('--alpha_source', default='global', choices=['global', 'fold', 'manual'])
    parser.add_argument('--alpha_init', type=float, default=None)
    args = parser.parse_args()

    run_root = f"{BASE}/out_sleeptransformer/{args.dataset}/run{args.run}"
    paths = {
        "list_dir": LIST_DIRS[args.dataset],
        "pred_base": f"{run_root}/out",
        "pred_dir": f"{run_root}/predictions",
        "train_score_base": f"{run_root}/training_scores",
    }
    os.makedirs(paths["pred_dir"], exist_ok=True)

    folds = discover_folds(paths["pred_base"]) if args.fold == 'all' else [args.fold]
    if not folds:
        raise FileNotFoundError(f"No folds with test_ret.mat under {paths['pred_base']}")

    print(f"Dataset={args.dataset} run={args.run} mode={args.mode} folds={folds}")

    for fold in folds:
        refine_fold(fold, args, paths)

    print("\nAll requested folds processed.")
    print(f"Results saved to: {paths['pred_dir']}/hmmRefined")


if __name__ == '__main__':
    main()
