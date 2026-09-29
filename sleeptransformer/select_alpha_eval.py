"""
select_alpha_eval.py (SleepTransformer) -- select alpha on the EVAL (val) split,
never touching test. Works for NCH (per age bin), SleepEDF-20 and SleepEDF-78
(per fold).

Usage:
    python select_alpha_eval.py --dataset sleepedf-78 --unit all --selection_metric jsd
    python select_alpha_eval.py --dataset sleepedf-20 --unit n3
    python select_alpha_eval.py --dataset nch --unit 6-12y
"""
import argparse
import json
import os
import warnings

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from hmm_refine_sleepedf import (
    ALPHA_VALUES, BASE, K, LIST_DIRS, STAGES,
    discover_folds, estimate_hmm_parameters_from_gt, get_uniform_pi,
    load_labels_raw, load_list, scores_to_probs_list,
    verify_label_range, verify_raw_labels_one_indexed, viterbi_hmm_softmax,
)
from hmm_refine_nch import AGE_BINS as NCH_AGE_BINS
from hmm_refine_nch import DEFAULT_LIST_DIR as NCH_LIST_DIR
from hmm_refine_nch import DEFAULT_PRED_ROOT as NCH_PRED_ROOT

# NCH eval-split inference output (mirrors nch_inference_pretrained_train)
EVAL_SCORE_BASE_NCH = f"{BASE}/out_sleeptransformer/nch_inference_pretrained_eval"

KEY_MAP = {
    "jsd": ("jsd_mean", min),
    "accuracy": ("acc_mean", max),
    "kappa": ("kappa_mean", max),
    "macro_f1": ("mf1_mean", max),
}


# ============================================================
# PER-STAGE TRANSITION JSD
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
    tm_true, tm_pred = get_transition_matrix(y_true), get_transition_matrix(y_pred)
    out = []
    for i in range(K):
        if np.isnan(tm_true[i]).any() or np.isnan(tm_pred[i]).any():
            out.append(np.nan)
        else:
            out.append(jensenshannon(tm_true[i], tm_pred[i]) ** 2)
    return out


# ============================================================
# UNIT (fold / age bin) CONFIG
# ============================================================
def list_units(args, run_root):
    if args.dataset == "nch":
        return NCH_AGE_BINS if args.unit == "all" else [args.unit]
    return discover_folds(f"{run_root}/out") if args.unit == "all" else [args.unit]


def unit_config(args, unit, run_root):
    if args.dataset == "nch":
        return dict(
            train_list=os.path.join(NCH_LIST_DIR, f"train_list_{unit}.txt"),
            eval_list=os.path.join(NCH_LIST_DIR, f"eval_list_{unit}.txt"),
            eval_scores=os.path.join(EVAL_SCORE_BASE_NCH, unit, "test_ret.mat"),
            out_dir=os.path.join(NCH_PRED_ROOT, unit),
        )
    fid, list_dir = unit[1:], LIST_DIRS[args.dataset]
    return dict(
        train_list=os.path.join(list_dir, f"train_list_n{fid}.txt"),
        eval_list=os.path.join(list_dir, f"eval_list_n{fid}.txt"),
        eval_scores=os.path.join(run_root, "eval_scores", unit, "test_ret.mat"),
        out_dir=os.path.join(run_root, "predictions"),
    )


# ============================================================
# SWEEP
# ============================================================
def summarize(per_rec):
    """per_rec: {alpha: {acc:[], kappa:[], mf1:[], jsd:[]}} -> list of row dicts."""
    rows = []
    for a in ALPHA_VALUES:
        m = per_rec[a]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN JSD slices
            rows.append(dict(
                alpha=a,
                acc_mean=float(np.mean(m["acc"])), acc_std=float(np.std(m["acc"])),
                kappa_mean=float(np.mean(m["kappa"])), kappa_std=float(np.std(m["kappa"])),
                mf1_mean=float(np.mean(m["mf1"])), mf1_std=float(np.std(m["mf1"])),
                jsd_mean=float(np.nanmean(m["jsd"])), jsd_std=float(np.nanstd(m["jsd"])),
            ))
    return rows


def print_table(rows, title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    print(f"{'alpha':<7}{'acc%':<16}{'kappa':<16}{'macroF1%':<16}{'JSD (lower=better)':<18}")
    for r in rows:
        print(f"{r['alpha']:<7.1f}"
              f"{r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f}    "
              f"{r['kappa_mean']:.3f}±{r['kappa_std']:.3f}    "
              f"{r['mf1_mean']*100:.2f}±{r['mf1_std']*100:.2f}    "
              f"{r['jsd_mean']:.4f}±{r['jsd_std']:.4f}")
    print("=" * 80)


def select(rows, metric):
    key, fn = KEY_MAP[metric]
    best = fn(rows, key=lambda r: r[key])
    acc_best = max(rows, key=lambda r: r["acc_mean"])
    print(f"\nSelected alpha (criterion={metric}): {best['alpha']}")
    if acc_best["alpha"] != best["alpha"]:
        print(f"NOTE: accuracy-based selection would have picked alpha={acc_best['alpha']} "
              f"instead -- the two criteria disagree, which is itself worth reporting.")
    return best


def sweep_unit(args, unit, cfg):
    # A from this unit's TRAIN split, uniform pi (identical to the refine script)
    train_files = load_list(cfg["train_list"])
    train_raw = load_labels_raw(train_files)
    verify_raw_labels_one_indexed(train_raw, context=f"{unit} train GT")
    train_gt = [y - 1 for y in train_raw]
    verify_label_range(train_gt, context=f"{unit} train GT")
    A, _ = estimate_hmm_parameters_from_gt(train_gt, label=f"{args.dataset} {unit} (train)")
    log_A, log_pi = np.log(A + 1e-300), np.log(get_uniform_pi() + 1e-300)

    # Eval-split probs + labels
    eval_files = load_list(cfg["eval_list"])
    eval_raw = load_labels_raw(eval_files)
    verify_raw_labels_one_indexed(eval_raw, context=f"{unit} eval GT")
    eval_gt = [y - 1 for y in eval_raw]
    verify_label_range(eval_gt, context=f"{unit} eval GT")
    obs_probs_list = scores_to_probs_list(cfg["eval_scores"], [len(y) for y in eval_gt])
    print(f"[{unit}] {len(eval_gt)} eval recordings from {cfg['eval_scores']}")

    per_rec = {a: dict(acc=[], kappa=[], mf1=[], jsd=[]) for a in ALPHA_VALUES}
    for a in ALPHA_VALUES:
        for obs_probs, y_true in zip(obs_probs_list, eval_gt):
            assert obs_probs.shape[0] == len(y_true)
            y_pred = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=a)
            per_rec[a]["acc"].append(accuracy_score(y_true, y_pred))
            per_rec[a]["kappa"].append(cohen_kappa_score(y_true, y_pred))
            per_rec[a]["mf1"].append(f1_score(y_true, y_pred, average="macro",
                                              labels=STAGES, zero_division=0))
            per_rec[a]["jsd"].append(np.nanmean(per_stage_jsd(y_true, y_pred))
                                     if not np.all(np.isnan(per_stage_jsd(y_true, y_pred))) else np.nan)
    return per_rec, len(eval_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nch", "sleepedf-20", "sleepedf-78"])
    ap.add_argument("--unit", default="all",
                    help="NCH age bin (e.g. 6-12y) or SleepEDF fold (e.g. n3); 'all' loops every unit")
    ap.add_argument("--run", type=int, default=1, help="RUN_NUMBER (SleepEDF only)")
    ap.add_argument("--selection_metric", default="jsd", choices=list(KEY_MAP),
                    help="Default 'jsd' minimises mean per-stage transition JSD (thesis framing); "
                         "'accuracy' is the paper-faithful comparison point -- ideally report both.")
    args = ap.parse_args()

    run_root = f"{BASE}/out_sleeptransformer/{args.dataset}/run{args.run}"
    units = list_units(args, run_root)
    if not units:
        raise FileNotFoundError("No units found to process.")

    pooled = {a: dict(acc=[], kappa=[], mf1=[], jsd=[]) for a in ALPHA_VALUES}
    selected, skipped, last_out_dir = {}, [], None

    for unit in units:
        cfg = unit_config(args, unit, run_root)
        last_out_dir = cfg["out_dir"]
        if not os.path.exists(cfg["eval_scores"]):
            print(f"[{unit}] SKIPPED: no eval scores at {cfg['eval_scores']} "
                  f"(run the eval-split inference first)")
            skipped.append(unit)
            continue

        per_rec, n_rec = sweep_unit(args, unit, cfg)
        rows = summarize(per_rec)
        print_table(rows, f"ALPHA SWEEP on eval split -- {args.dataset} {unit} ({n_rec} recordings)")
        best = select(rows, args.selection_metric)
        selected[unit] = best["alpha"]
        print(f"  -> use --alpha_init {best['alpha']} for hmm_trained, or read "
              f"hmmRefined/*_ypred_alpha{best['alpha']:.1f}.npy on TEST for final reporting")

        os.makedirs(cfg["out_dir"], exist_ok=True)
        out_path = os.path.join(cfg["out_dir"], f"alpha_selection_{args.dataset}_{unit}.json")
        with open(out_path, "w") as f:
            json.dump(dict(dataset=args.dataset, unit=unit, split="eval",
                           selection_metric=args.selection_metric,
                           selected_alpha=best["alpha"], sweep=rows), f, indent=2)
        print(f"  saved {out_path}")

        for a in ALPHA_VALUES:
            for k in pooled[a]:
                pooled[a][k].extend(per_rec[a][k])

    # Dataset-level alpha (SleepEDF: one alpha across all folds is less noisy than per-fold)
    if args.dataset != "nch" and len(selected) > 1:
        rows = summarize(pooled)
        print_table(rows, f"POOLED eval sweep -- {args.dataset}, {len(selected)} folds")
        best = select(rows, args.selection_metric)
        out_path = os.path.join(last_out_dir, f"alpha_selection_{args.dataset}_global.json")
        with open(out_path, "w") as f:
            json.dump(dict(dataset=args.dataset, split="eval", folds=list(selected),
                           selection_metric=args.selection_metric,
                           selected_alpha=best["alpha"], per_fold_selected=selected,
                           sweep=rows), f, indent=2)
        print(f"  saved {out_path}")

    print(f"\nPer-unit selected alphas: {selected}")
    if skipped:
        raise SystemExit(f"ERROR: skipped units with missing eval scores: {skipped}")


if __name__ == "__main__":
    main()
