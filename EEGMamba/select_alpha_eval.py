"""
select_alpha_eval.py — Select alpha on the EVAL (val) split, never touching test.

Usage (run locally, after the eval job above has completed and synced down):
    python select_alpha_eval.py --age_bin 6-12y \
        --pred_dir /path/to/EEGMamba/predictions/NCH_6-12y \
        --index_path /path/to/nch_index.parquet \
        --selection_metric jsd
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# Reuse the exact same estimation / Viterbi code as the main refinement
# pipeline, rather than reimplementing it -- avoids any risk of the
# eval-side decode subtly diverging from the test-side decode.
from hmm_refine_nch import (
    AGE_BINS, ALPHA_VALUES, STAGES, STAGE_NAMES, K,
    get_uniform_pi, estimate_hmm_parameters_from_gt,
    load_nch_train_sequences, verify_label_range, viterbi_hmm_softmax,
)

EVAL_SPLIT_NAME = "eval"  # change to "val" if that's what NCHIndexDataset calls it


# ============================================================
# LOAD EVAL-SPLIT SCORES (mirrors load_nch_train_probs_and_gt)
# ============================================================
def load_eval_probs_and_gt(pred_dir, age_bin):
    eval_dir = os.path.join(pred_dir, f"{EVAL_SPLIT_NAME}_scores")
    prob_files = sorted(glob.glob(os.path.join(eval_dir, "*_probs.npy")))
    if not prob_files:
        raise FileNotFoundError(
            f"No {EVAL_SPLIT_NAME}-split scores found at {eval_dir}.\n"
            f"Run on Katana first:\n"
            f"  python test_nch.py --age_bin {age_bin} --split {EVAL_SPLIT_NAME} "
            f"--pred_dir {pred_dir} --model_dir <checkpoint dir> "
            f"--index_path <index.parquet>\n"
        )

    obs_probs_list, y_true_list, rec_ids = [], [], []
    for prob_path in prob_files:
        rec_id = os.path.basename(prob_path).replace("_probs.npy", "")
        ytrue_path = os.path.join(eval_dir, f"{rec_id}_ytrue.npy")
        obs_probs_list.append(np.load(prob_path))
        y_true_list.append(np.load(ytrue_path).astype(int))
        rec_ids.append(rec_id)

    print(f"[{age_bin}] Loaded {len(rec_ids)} {EVAL_SPLIT_NAME}-split recordings "
          f"from {eval_dir}.")
    return obs_probs_list, y_true_list, rec_ids


# ============================================================
# PER-STAGE TRANSITION JSD (mirrors the extraction script's logic)
# ============================================================
def get_transition_matrix(seq):
    m = np.zeros((K, K))
    for i in range(len(seq) - 1):
        c, n = seq[i], seq[i + 1]
        if 0 <= c < K and 0 <= n < K:
            m[c, n] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    return np.divide(m, row_sums, out=np.full_like(m, np.nan), where=row_sums != 0)


def per_stage_jsd(y_true, y_pred):
    tm_true = get_transition_matrix(y_true)
    tm_pred = get_transition_matrix(y_pred)
    out = []
    for i in range(K):
        if np.isnan(tm_true[i]).any() or np.isnan(tm_pred[i]).any():
            out.append(np.nan)
        else:
            out.append(jensenshannon(tm_true[i], tm_pred[i]) ** 2)
    return out


# ============================================================
# MAIN SWEEP
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age_bin", required=True, choices=AGE_BINS)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--selection_metric", default="jsd",
                     choices=["jsd", "accuracy", "kappa", "macro_f1"],
                     help="Criterion for picking the 'best' alpha. Default 'jsd' "
                          "minimizes mean per-stage transition JSD, matching the "
                          "thesis's own distributional-fidelity framing rather than "
                          "the paper's accuracy-based Table 1 selection. Pass "
                          "'accuracy' if you want the paper-faithful comparison "
                          "point instead -- ideally report both.")
    args = ap.parse_args()

    # --- A from train split (unchanged from hmm_refine_nch.py) ---
    train_gt = load_nch_train_sequences(args.index_path, args.age_bin)
    verify_label_range(train_gt, context=f"{args.age_bin} train GT")
    A, _ = estimate_hmm_parameters_from_gt(train_gt, label=f"NCH {args.age_bin} (train)")
    pi = get_uniform_pi()
    log_A, log_pi = np.log(A + 1e-300), np.log(pi + 1e-300)

    # --- eval-split probs/labels ---
    obs_probs_list, y_true_list, rec_ids = load_eval_probs_and_gt(args.pred_dir, args.age_bin)
    verify_label_range(y_true_list, context=f"{args.age_bin} eval GT")

    # --- sweep ---
    sweep = []
    for a in ALPHA_VALUES:
        accs, kappas, mf1s, jsds = [], [], [], []
        for obs_probs, y_true in zip(obs_probs_list, y_true_list):
            y_pred = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=a)
            accs.append(accuracy_score(y_true, y_pred))
            kappas.append(cohen_kappa_score(y_true, y_pred))
            mf1s.append(f1_score(y_true, y_pred, average="macro", labels=STAGES, zero_division=0))
            jsds.append(np.nanmean(per_stage_jsd(y_true, y_pred)))

        sweep.append(dict(
            alpha=a,
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
            kappa_mean=float(np.mean(kappas)), kappa_std=float(np.std(kappas)),
            mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
            jsd_mean=float(np.nanmean(jsds)), jsd_std=float(np.nanstd(jsds)),
        ))

    print(f"\n{'='*80}\nALPHA SWEEP on {EVAL_SPLIT_NAME} split -- {args.age_bin} "
          f"({len(rec_ids)} recordings)\n{'='*80}")
    print(f"{'alpha':<7}{'acc%':<16}{'kappa':<16}{'macroF1%':<16}{'JSD (lower=better)':<18}")
    for r in sweep:
        print(f"{r['alpha']:<7.1f}"
              f"{r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f}    "
              f"{r['kappa_mean']:.3f}±{r['kappa_std']:.3f}    "
              f"{r['mf1_mean']*100:.2f}±{r['mf1_std']*100:.2f}    "
              f"{r['jsd_mean']:.4f}±{r['jsd_std']:.4f}")
    print("=" * 80)

    key_map = {
        "jsd": ("jsd_mean", min),
        "accuracy": ("acc_mean", max),
        "kappa": ("kappa_mean", max),
        "macro_f1": ("mf1_mean", max),
    }
    metric_key, best_fn = key_map[args.selection_metric]
    best = best_fn(sweep, key=lambda r: r[metric_key])

    print(f"\nSelected alpha (criterion={args.selection_metric}): {best['alpha']}")
    print(f"  -> apply this to hmmRefined/*_ypred_alpha{best['alpha']:.1f}.npy on the "
          f"TEST split for final reporting (already decoded, no redecode needed)")
    print(f"  -> or pass alpha={best['alpha']} as the MMI warm-start value for hmm_trained\n")

    # Also show what accuracy-based selection would have picked, if different,
    # so both numbers are on record for the thesis appendix.
    acc_best = max(sweep, key=lambda r: r["acc_mean"])
    if acc_best["alpha"] != best["alpha"]:
        print(f"NOTE: accuracy-based selection would have picked alpha="
              f"{acc_best['alpha']} instead of {best['alpha']} -- the two criteria "
              f"disagree, which is itself worth reporting.\n")

    out_path = os.path.join(args.pred_dir, f"alpha_selection_{args.age_bin}.json")
    with open(out_path, "w") as f:
        json.dump(dict(age_bin=args.age_bin, split=EVAL_SPLIT_NAME,
                        selection_metric=args.selection_metric,
                        selected_alpha=best["alpha"], sweep=sweep), f, indent=2)
    print(f"Full sweep + selection saved to {out_path}")


if __name__ == "__main__":
    main()
