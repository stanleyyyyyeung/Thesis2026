"""
select_alpha_eval.py — Select alpha on the EVAL (val) split, never touching test.

Eval-split scores come from `test_nch.py --split val`, which writes them to
<PRED_DIR>/eval_scores/ as <subject>_<study>_{probs,ytrue,ypred}.npy + _run_idx.json.

Usage:
    python select_alpha_eval.py --age_bin 6-12y --pred_dir <PRED_DIR> \
        --index_path <index.parquet> --seq_len 20 --selection_metric jsd

"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# Reuse the exact same estimation / Viterbi code as the main refinement
# pipeline, rather than reimplementing it.
from hmm_refine_nch import (
    AGE_BINS, ALPHA_VALUES, STAGES, STAGE_NAMES, K,
    get_uniform_pi, estimate_hmm_parameters_from_gt,
    load_nch_train_sequences, verify_label_range, viterbi_hmm_softmax,
    list_recordings, run_segments, resolve_data_source,
)

EVAL_SPLIT_NAME = "eval"     # folder name is eval_scores/; test_nch.py's flag for it is --split val


def alpha_json_name(age_bin, seq_len):
    return f"alpha_selection_{age_bin}_seqlen{seq_len}.json"


def check_json_name(age_bin, seq_len):
    return f"hmm_trained_eval_check_{age_bin}_seqlen{seq_len}.json"


# ============================================================
# LOAD EVAL-SPLIT SCORES
# ============================================================
def load_eval_probs_and_gt(pred_dir, age_bin, seq_len):
    eval_dir = os.path.join(pred_dir, f"{EVAL_SPLIT_NAME}_scores")
    rec_ids = list_recordings(eval_dir)
    if not rec_ids:
        raise FileNotFoundError(
            f"No <subject>_<study>_probs.npy found in {eval_dir}.\n"
            f"Run first:\n"
            f"  python test_nch.py --age_bin {age_bin} --split val "
            f"--pred_dir {pred_dir} --model_dir <checkpoint dir> --index_path <index.parquet>\n"
        )
    obs_probs_list, y_true_list, segs_list = [], [], []
    for rec_id in rec_ids:
        probs = np.load(os.path.join(eval_dir, f"{rec_id}_probs.npy"))
        y = np.load(os.path.join(eval_dir, f"{rec_id}_ytrue.npy")).astype(int)
        assert len(probs) == len(y), f"{rec_id}: probs {len(probs)} vs ytrue {len(y)}"
        obs_probs_list.append(probs)
        y_true_list.append(y)
        segs_list.append(run_segments(len(y), os.path.join(eval_dir, f"{rec_id}_run_idx.json"), seq_len))
    print(f"[{age_bin}] Loaded {len(rec_ids)} {EVAL_SPLIT_NAME}-split recordings from {eval_dir} "
          f"({sum(len(s) for s in segs_list)} runs).")
    return obs_probs_list, y_true_list, segs_list, rec_ids


def warn_if_split_sources_differ(pred_dir):
    """Alpha is chosen on eval windows and applied to test windows; both should be trimmed alike."""
    srcs = {}
    for name, d in (("eval", os.path.join(pred_dir, "eval_scores")), ("test", pred_dir)):
        p = os.path.join(d, "run_meta.json")
        if os.path.exists(p):
            srcs[name] = json.load(open(p)).get("data_source_used")
    if len(set(srcs.values())) > 1:
        print(f"[WARN] eval and test predictions came from different data sources {srcs}: "
              "the selected alpha may not transfer. Regenerate both with the same --data_source.")


# ============================================================
# PER-STAGE TRANSITION JSD (transitions counted within runs only)
# ============================================================
def get_transition_matrix(seq, segs=None):
    m = np.zeros((K, K))
    for s, e in ([(0, len(seq))] if segs is None else segs):
        sub = seq[s:e]
        for i in range(len(sub) - 1):
            c, n = sub[i], sub[i + 1]
            if 0 <= c < K and 0 <= n < K:
                m[c, n] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    return np.divide(m, row_sums, out=np.full_like(m, np.nan), where=row_sums != 0)


def per_stage_jsd(y_true, y_pred, segs=None):
    tm_true = get_transition_matrix(y_true, segs)
    tm_pred = get_transition_matrix(y_pred, segs)
    out = []
    for i in range(K):
        if np.isnan(tm_true[i]).any() or np.isnan(tm_pred[i]).any():
            out.append(np.nan)
        else:
            out.append(jensenshannon(tm_true[i], tm_pred[i]) ** 2)
    return out


def decode_runs(obs_probs, segs, log_A, log_pi, alpha):
    path = np.empty(len(obs_probs), dtype=int)
    for s, e in segs:
        path[s:e] = viterbi_hmm_softmax(obs_probs[s:e], log_A, log_pi, alpha=alpha)
    return path


def score_recordings(obs_probs_list, y_true_list, segs_list, log_A, log_pi, alpha):
    accs, kappas, mf1s, jsds = [], [], [], []
    for obs_probs, y_true, segs in zip(obs_probs_list, y_true_list, segs_list):
        y_pred = decode_runs(obs_probs, segs, log_A, log_pi, alpha)
        accs.append(accuracy_score(y_true, y_pred))
        kappas.append(cohen_kappa_score(y_true, y_pred))
        mf1s.append(f1_score(y_true, y_pred, average="macro", labels=STAGES, zero_division=0))
        jsds.append(np.nanmean(per_stage_jsd(y_true, y_pred, segs)))
    return accs, kappas, mf1s, jsds


def summarize(alpha, accs, kappas, mf1s, jsds):
    return dict(
        alpha=alpha,
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        kappa_mean=float(np.mean(kappas)), kappa_std=float(np.std(kappas)),
        mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
        jsd_mean=float(np.nanmean(jsds)), jsd_std=float(np.nanstd(jsds)),
    )


def evaluate_hmm_trained(pred_dir, age_bin, seq_len):
    params_path = os.path.join(pred_dir, "hmmRefined", f"hmm_trained_params_{age_bin}.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"No trained HMM parameters at {params_path}.\n"
            f"Run: python hmm_refine_nch.py --age_bin {age_bin} --mode hmm_trained "
            f"--pred_dir {pred_dir} --index_path <index.parquet>  first."
        )
    saved = np.load(params_path)
    A, pi, alpha = saved["A"], saved["pi"], float(saved["alpha"])
    print(f"[{age_bin}] Loaded trained HMM params from {params_path} (alpha={alpha:.4f})")
    log_A, log_pi = np.log(A + 1e-300), np.log(pi + 1e-300)

    obs_probs_list, y_true_list, segs_list, rec_ids = load_eval_probs_and_gt(pred_dir, age_bin, seq_len)
    verify_label_range(y_true_list, context=f"{age_bin} eval GT (hmm_trained check)")
    res = summarize(alpha, *score_recordings(obs_probs_list, y_true_list, segs_list, log_A, log_pi, alpha))

    print(f"\n{'='*80}\nHMM_TRAINED eval-split check -- {age_bin} ({len(rec_ids)} recordings)\n{'='*80}")
    print(f"  Accuracy : {res['acc_mean']*100:.2f} ± {res['acc_std']*100:.2f} %")
    print(f"  Kappa    : {res['kappa_mean']:.3f} ± {res['kappa_std']:.3f}")
    print(f"  Macro F1 : {res['mf1_mean']*100:.2f} ± {res['mf1_std']*100:.2f} %")
    print(f"  JSD      : {res['jsd_mean']:.4f} ± {res['jsd_std']:.4f}  (lower=better)")
    print("=" * 80)

    out_path = os.path.join(pred_dir, check_json_name(age_bin, seq_len))
    with open(out_path, "w") as f:
        json.dump(dict(age_bin=age_bin, split=EVAL_SPLIT_NAME, mode="hmm_trained", **res), f, indent=2)
    print(f"Saved to {out_path}")


# ============================================================
# MAIN SWEEP
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age_bin", required=True, choices=AGE_BINS)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--seq_len", type=int, default=20,
                    help="Must match the index parquet and the seq_len the eval-split predictions "
                         "were generated at.")
    ap.add_argument("--selection_metric", default="jsd",
                    choices=["jsd", "accuracy", "kappa", "macro_f1"],
                    help="Criterion for picking alpha. 'jsd' minimises mean per-stage transition JSD "
                         "(distributional fidelity); 'accuracy' is the paper-faithful point. Report both.")
    ap.add_argument("--pi_source", type=str, default="uniform", choices=["uniform", "empirical"],
                    help="Must match the --pi_source passed to hmm_refine_nch.py for this bin.")
    ap.add_argument("--mode", default="hmm", choices=["hmm", "hmm_trained"],
                    help="'hmm': sweep ALPHA_VALUES with the empirical train-MLE A. "
                         "'hmm_trained': report the eval-split performance of the MMI-trained (A, pi, alpha).")
    ap.add_argument("--data_source", default="auto", choices=["auto", "cache", "live"],
                    help="Source of the TRAINING labels for the prior. 'auto' follows the source recorded "
                         "in <pred_dir>/eval_scores/run_meta.json.")
    args = ap.parse_args()

    warn_if_split_sources_differ(args.pred_dir)
    data_source = resolve_data_source(args.data_source, os.path.join(args.pred_dir, f"{EVAL_SPLIT_NAME}_scores"))

    if data_source != "cache":      # the parquet will (or may) be used: its seq_len must match
        md = pq.ParquetFile(args.index_path).schema_arrow.metadata or {}
        file_len = int(md[b"seq_len"]) if b"seq_len" in md else 20   # legacy index = 20
        if file_len != args.seq_len:
            raise ValueError(f"{args.index_path} has seq_len={file_len}, but --seq_len={args.seq_len}")

    if args.mode == "hmm_trained":
        evaluate_hmm_trained(args.pred_dir, args.age_bin, args.seq_len)
        return

    # --- A from train split (same code path as hmm_refine_nch.py) ---
    train_gt = load_nch_train_sequences(args.index_path, args.age_bin,
                                        seq_len=args.seq_len, data_source=data_source)
    verify_label_range(train_gt, context=f"{args.age_bin} train GT")
    A, pi_empirical = estimate_hmm_parameters_from_gt(train_gt, label=f"NCH {args.age_bin} (train)")
    pi = get_uniform_pi() if args.pi_source == "uniform" else pi_empirical
    print(f"[{args.age_bin}] pi_source={args.pi_source} -> pi={np.round(pi, 4)}")
    log_A, log_pi = np.log(A + 1e-300), np.log(pi + 1e-300)

    # --- eval-split probs/labels ---
    obs_probs_list, y_true_list, segs_list, rec_ids = load_eval_probs_and_gt(
        args.pred_dir, args.age_bin, args.seq_len)
    verify_label_range(y_true_list, context=f"{args.age_bin} eval GT")

    # --- sweep ---
    sweep = [summarize(a, *score_recordings(obs_probs_list, y_true_list, segs_list, log_A, log_pi, a))
             for a in ALPHA_VALUES]

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
    print(f"  -> apply this to hmmRefined/<subject>_<study>_ypred_alpha{best['alpha']:.1f}.npy on the "
          f"TEST split for final reporting (already decoded, no redecode needed)")
    print(f"  -> or pass alpha={best['alpha']} as the MMI warm-start value for hmm_trained\n")

    acc_best = max(sweep, key=lambda r: r["acc_mean"])
    if acc_best["alpha"] != best["alpha"]:
        print(f"NOTE: accuracy-based selection would have picked alpha={acc_best['alpha']} instead of "
              f"{best['alpha']} -- the two criteria disagree, which is itself worth reporting.\n")

    out_path = os.path.join(args.pred_dir, alpha_json_name(args.age_bin, args.seq_len))
    with open(out_path, "w") as f:
        json.dump(dict(age_bin=args.age_bin, split=EVAL_SPLIT_NAME,
                       selection_metric=args.selection_metric, data_source=data_source,
                       pi_source=args.pi_source,
                       n_recordings=len(rec_ids),
                       selected_alpha=best["alpha"], sweep=sweep), f, indent=2)
    print(f"Full sweep + selection saved to {out_path}")


if __name__ == "__main__":
    main()
