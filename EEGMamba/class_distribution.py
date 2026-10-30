"""Per-stage epoch proportions for the NCH cached dataset (train / val / test),
per age bin, pooled, and per subject. Also prints candidate class weights."""

import argparse, csv, glob, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CACHE_ROOT = "/srv/scratch/speechdata/sleep_data/NCH/eegmamba_seq"
LIST_DIR, LABEL_DIR = "split_lists", "labels"
SPLITS = {"train": "train_list", "val": "eval_list", "test": "test_list"}
STAGES = ["W", "N1", "N2", "N3", "REM"]   # ASSUMED order of classes 0..4 -- verify
LABEL_OFFSET = 1                           # cache labels are 1-based (as in nch_cached_dataset.py)
X_KEY, Y_KEY = "x", "y"
K = len(STAGES)


def read_list(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d, n = line.split("\t")
                out.append((d, int(n)))
    return out


def resolve_bins(cache_root, spec):
    if spec.lower() == "all":
        files = glob.glob(os.path.join(cache_root, LIST_DIR, "train_list_*.txt"))
        return sorted(os.path.basename(f)[len("train_list_"):-4] for f in files)
    return [b for b in spec.replace("+", ",").split(",") if b]


def load_labels(seq_path, label_path):
    """Mirror NCHCachedDataset: prefer the separate label file; fall back to a pickled dict."""
    if os.path.isfile(label_path):
        y = np.load(label_path)
    else:
        d = np.load(seq_path, allow_pickle=True).item()
        y = d[Y_KEY]
    y = np.asarray(y).astype(np.int64).reshape(-1) - LABEL_OFFSET
    if y.min() < 0 or y.max() >= K:
        raise ValueError(f"{label_path}: labels out of range: {np.unique(y).tolist()}")
    return np.bincount(y, minlength=K)


def subject_counts(cache_root, subj_dir, workers_inner=1):
    sid = os.path.basename(subj_dir.rstrip("/"))
    label_dir = os.path.join(cache_root, LABEL_DIR, sid)
    files = sorted(f for f in os.listdir(subj_dir) if f.endswith(".npy"))
    c = np.zeros(K, dtype=np.int64)
    for f in files:
        c += load_labels(os.path.join(subj_dir, f), os.path.join(label_dir, f))
    return sid, len(files), c


def fmt(counts):
    tot = counts.sum()
    return "  ".join(f"{s}={100*c/tot:5.2f}%" for s, c in zip(STAGES, counts))

def print_table(title, table):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'group':<22}{'epochs':>10} " + " ".join(f"{s:>7}" for s in STAGES))
    for label, c in table:
        c = np.asarray(c); t = c.sum()
        if t == 0:
            continue
        print(f"{label:<22}{t:>10d} " + " ".join(f"{100 * x / t:>6.2f}%" for x in c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    ap.add_argument("--age_bins", default="all", help='"all" or comma/plus-separated, e.g. 6-12y,13-18y')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    bins = resolve_bins(a.cache_root, a.age_bins)
    print(f"Age bins: {bins}")
    summary, rows = {}, []
    pooled = {s: np.zeros(K, dtype=np.int64) for s in SPLITS}
    table = []

    for b in bins:
        for split, prefix in SPLITS.items():
            lp = os.path.join(a.cache_root, LIST_DIR, f"{prefix}_{b}.txt")
            if not os.path.isfile(lp):
                print(f"[skip] missing {lp}")
                continue
            entries = read_list(lp)
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(lambda e: subject_counts(a.cache_root, e[0]), entries))
            tot = np.zeros(K, dtype=np.int64)
            for sid, nseq, c in res:
                tot += c
                rows.append([b, split, sid, nseq, *c.tolist()])
            pooled[split] += tot
            table.append((f"{b}/{split}", tot))
            summary[f"{b}/{split}"] = {"n_subjects": len(res), "n_epochs": int(tot.sum()),
                                       "counts": dict(zip(STAGES, tot.tolist())),
                                       "proportions": dict(zip(STAGES, (tot / tot.sum()).round(5).tolist()))}
            print(f"[{b:>7s} | {split:5s}] subj={len(res):4d} epochs={tot.sum():9d}\n    {fmt(tot)}")

    print("\n=== POOLED over bins ===")
    for split, c in pooled.items():
        if c.sum() == 0:
            continue
        table.append((f"POOLED/{split}", c))
        summary[f"pooled/{split}"] = {"n_epochs": int(c.sum()),
                                      "counts": dict(zip(STAGES, c.tolist())),
                                      "proportions": dict(zip(STAGES, (c / c.sum()).round(5).tolist()))}
        print(f"[{split:5s}] {fmt(c)}")

    # Candidate class weights from the pooled TRAIN distribution
    c = pooled["train"].astype(float)
    if c.sum() > 0:
        inv = c.sum() / (K * c)                 # sklearn 'balanced'
        sqrt_inv = np.sqrt(inv); sqrt_inv /= sqrt_inv.mean()
        print("\nCandidate weights (pooled train):")
        print("  inverse-freq ('balanced'):", dict(zip(STAGES, inv.round(3).tolist())))
        print("  sqrt-inverse (mean=1)    :", dict(zip(STAGES, sqrt_inv.round(3).tolist())))
        summary["weights_pooled_train"] = {"balanced": dict(zip(STAGES, inv.tolist())),
                                           "sqrt_inverse": dict(zip(STAGES, sqrt_inv.tolist()))}

    print_table("SUMMARY: NCH stage proportions (% of epochs)", table)
    with open(os.path.join(a.out_dir, "class_distribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.out_dir, "class_distribution_per_subject.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bin", "split", "subject", "n_seqs", *STAGES]); w.writerows(rows)
    print(f"\nSaved to {a.out_dir}")


if __name__ == "__main__":
    main()
