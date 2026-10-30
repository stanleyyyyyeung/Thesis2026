"""Per-stage epoch proportions for the ISRUC downstream dataset (train/val/test),
mirroring datasets/isruc_dataset.py: subjects 1..100, split by index 0-79/80-89/90-99.
Reads only labels/*.npy (never the large seq tensors)."""

import argparse, csv, json, os
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]   # ASSUMED order of classes 0..4 -- verify
K = len(STAGES)
SPLIT_RANGES = {"train": range(0, 80), "val": range(80, 90), "test": range(90, 100)}
N_SUBJECTS = 100


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
    ap.add_argument("--datasets_dir", required=True,
                    help="ISRUC processed dir containing seq/ and labels/")
    ap.add_argument("--label_offset", type=int, default=0,
                    help="0 if labels are already 0-based (the ISRUC loader passes them straight "
                         "to CrossEntropy, so they should be). Use 1 if they turn out 1-based.")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    seqs_dir = os.path.join(a.datasets_dir, "seq")
    labels_dir = os.path.join(a.datasets_dir, "labels")

    # per-subject counts, indexed by position 0..99 (same indexing as split_dataset)
    per_subject = {}
    rows, warnings = [], []
    for i in range(N_SUBJECTS):
        sid = f"ISRUC-group1-{i + 1}"
        sd, ld = os.path.join(seqs_dir, sid), os.path.join(labels_dir, sid)
        if not (os.path.isdir(sd) and os.path.isdir(ld)):
            print(f"[warn] subject {i + 1} missing -> 0 samples (expected only for subject 8)")
            per_subject[i] = (sid, 0, np.zeros(K, dtype=np.int64))
            continue
        sf, lf = sorted(os.listdir(sd)), sorted(os.listdir(ld))
        if len(sf) != len(lf):
            warnings.append(f"{sid}: {len(sf)} seq files vs {len(lf)} label files "
                            "(loader zips them -> silent truncation/misalignment!)")
        if [os.path.splitext(x)[0] for x in sf] != [os.path.splitext(x)[0] for x in lf]:
            warnings.append(f"{sid}: seq/label filenames do not match 1:1 after sorting")
        c = np.zeros(K, dtype=np.int64)
        for f in lf[:len(sf)]:                       # mirror zip() truncation
            y = np.load(os.path.join(ld, f)).astype(np.int64).reshape(-1) - a.label_offset
            if y.min() < 0 or y.max() >= K:
                raise ValueError(f"{sid}/{f}: labels out of range after offset "
                                 f"{a.label_offset}: {np.unique(y).tolist()}")
            c += np.bincount(y, minlength=K)
        per_subject[i] = (sid, min(len(sf), len(lf)), c)

    table = []
    summary, pooled = {}, np.zeros(K, dtype=np.int64)
    for split, rng in SPLIT_RANGES.items():
        tot = np.zeros(K, dtype=np.int64)
        n_subj = 0
        for i in rng:
            sid, nseq, c = per_subject[i]
            rows.append([split, sid, nseq, *c.tolist()])
            tot += c
            n_subj += int(nseq > 0)
        pooled += tot
        table.append((f"isruc/{split}", tot))
        summary[f"isruc/{split}"] = {
            "n_subjects": n_subj, "n_epochs": int(tot.sum()),
            "counts": dict(zip(STAGES, tot.tolist())),
            "proportions": dict(zip(STAGES, (tot / tot.sum()).round(5).tolist()))}
        print(f"[{split:5s}] subj={n_subj:3d} epochs={tot.sum():8d}\n    {fmt(tot)}")

    # Candidate class weights from TRAIN
    c = np.array(list(summary["isruc/train"]["counts"].values()), dtype=float)
    inv = c.sum() / (K * c)
    sqrt_inv = np.sqrt(inv); sqrt_inv /= sqrt_inv.mean()
    print("\nCandidate weights (train):")
    print("  inverse-freq ('balanced'):", dict(zip(STAGES, inv.round(3).tolist())))
    print("  sqrt-inverse (mean=1)    :", dict(zip(STAGES, sqrt_inv.round(3).tolist())))
    summary["weights_train"] = {"balanced": dict(zip(STAGES, inv.tolist())),
                                "sqrt_inverse": dict(zip(STAGES, sqrt_inv.tolist()))}

    print_table("SUMMARY: ISRUC stage proportions (% of epochs)", table)
    if warnings:
        print("\n!!! DATA WARNINGS !!!")
        for w in warnings:
            print("  ", w)
    summary["warnings"] = warnings

    with open(os.path.join(a.out_dir, "class_distribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.out_dir, "class_distribution_per_subject.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "subject", "n_seqs", *STAGES]); w.writerows(rows)
    print(f"\nSaved to {a.out_dir}")


if __name__ == "__main__":
    main()
