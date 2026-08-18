"""
NCH EEGMamba dataset diagnostic script.

Covers, in one run:
  1. Index schema dump (so we know exactly what columns exist)
  2. Dataset load + len() sanity check
  3. Value-range sanity (min/max/mean/std/NaN/Inf) stratified by age_bin, with timing
  4. Cache behavior (cold vs warm __getitem__ latency, using whatever recording-id
     column is detected in the index)
  5. Best-effort run-boundary / gap column detection
  6. Summary

Usage (from the EEGMamba dir on Katana, inside the container):

PYTHONPATH=/srv/scratch/z5423210/python_packages \
apptainer exec -B /srv:/srv /srv/scratch/z5423210/tf22_py3.sif \
    python diagnose_nch_dataset.py

Edit INDEX_PATH / N_PER_BIN below if needed. The script is defensive about
column names since the exact index schema wasn't available when this was
written — step 1's printed columns will tell you if any auto-detection
(recording-id column, boundary/gap column) picked the wrong thing.
"""

import time
import sys
import numpy as np
import pandas as pd

INDEX_PATH = "/srv/scratch/z5423210/StanleyThesis2026/nch_index/nch_index_nch_v2.parquet"
EEGMAMBA_DIR = "/srv/scratch/z5423210/StanleyThesis2026/EEGMamba"
N_PER_BIN = 5
SEED = 42

sys.path.insert(0, EEGMAMBA_DIR)
from nch_dataset import NCHIndexDataset  # noqa: E402


def main():
    print("=" * 70)
    print("1. INDEX SCHEMA")
    print("=" * 70)
    df = pd.read_parquet(INDEX_PATH)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(df.dtypes)
    print(df.head(3).to_string())

    if "age_bin" not in df.columns:
        print("\n!! No 'age_bin' column found — cannot stratify by age bin. "
              "Check the column list above and edit this script.")
        age_bins = [None]
    else:
        age_bins = sorted(df["age_bin"].unique())
        print(f"\nage_bin values: {age_bins}")
        print(df["age_bin"].value_counts())

    # Try to find a recording-identifier column for the cache test.
    recording_col_candidates = [
        "study_id", "edf_path", "file", "file_path", "recording_id",
        "subject_id", "edf_file",
    ]
    recording_col = next((c for c in recording_col_candidates if c in df.columns), None)
    if recording_col:
        print(f"\nUsing '{recording_col}' as recording-identifier column for the cache test "
              f"(edit RECORDING_COL_OVERRIDE below if this is wrong).")
    else:
        print("\n!! No obvious recording-identifier column found — cache test will be skipped "
              "unless you set RECORDING_COL_OVERRIDE manually below.")

    print("\n" + "=" * 70)
    print("2. LOADING DATASET")
    print("=" * 70)
    ds = NCHIndexDataset(INDEX_PATH)
    print(f"len(ds) = {len(ds)}  (matches index rows: {len(ds) == len(df)})")

    rng = np.random.default_rng(SEED)

    print("\n" + "=" * 70)
    print("3. VALUE-RANGE SANITY CHECK + TIMING (stratified by age_bin)")
    print("=" * 70)

    all_stats = []
    for ab in age_bins:
        if ab is None:
            sub_idx = df.index.to_numpy()
        else:
            sub_idx = df.index[df["age_bin"] == ab].to_numpy()
        if len(sub_idx) == 0:
            print(f"\n[age_bin={ab}] no rows, skipping")
            continue
        picks = rng.choice(sub_idx, size=min(N_PER_BIN, len(sub_idx)), replace=False)
        print(f"\n--- age_bin = {ab} ({len(sub_idx)} windows total, sampling {len(picks)}) ---")
        for i in picks:
            i = int(i)
            t0 = time.perf_counter()
            x, y = ds[i]
            t1 = time.perf_counter()
            xn = x.numpy()
            nan_ct = int(np.isnan(xn).sum())
            inf_ct = int(np.isinf(xn).sum())
            stats = dict(
                idx=i, age_bin=ab, elapsed_s=t1 - t0,
                min=float(xn.min()), max=float(xn.max()),
                mean=float(xn.mean()), std=float(xn.std()),
                nan_ct=nan_ct, inf_ct=inf_ct,
                y=y.numpy().tolist(),
            )
            all_stats.append(stats)
            flag = ""
            if nan_ct or inf_ct:
                flag += " !!NAN/INF!!"
            # Rough sanity band for filtered scalp EEG in microvolts.
            if abs(stats["mean"]) > 500 or stats["std"] > 2000 or stats["std"] < 1e-6:
                flag += " !!SUSPICIOUS SCALE!!"
            print(f"  idx={i:>7}  t={stats['elapsed_s']*1000:8.1f}ms  "
                  f"min={stats['min']:9.2f} max={stats['max']:9.2f} "
                  f"mean={stats['mean']:8.3f} std={stats['std']:8.3f}  "
                  f"nan={nan_ct} inf={inf_ct}{flag}")
            print(f"           y={stats['y']}")

    print("\n" + "=" * 70)
    print("4. CACHE BEHAVIOR TEST (cold vs warm)")
    print("=" * 70)

    if recording_col is not None:
        counts = df[recording_col].value_counts()
        multi_window_recordings = counts[counts >= 2].index.tolist()
        if multi_window_recordings:
            rec = multi_window_recordings[0]
            rec_idx = df.index[df[recording_col] == rec].to_numpy()
            i0, i1 = int(rec_idx[0]), int(rec_idx[1])
            print(f"Recording '{rec}' has {len(rec_idx)} windows in index. "
                  f"Testing idx={i0} (cold) then idx={i1} (should hit cache).")

            t0 = time.perf_counter(); _ = ds[i0]; t_cold = time.perf_counter() - t0
            t0 = time.perf_counter(); _ = ds[i1]; t_warm = time.perf_counter() - t0
            t0 = time.perf_counter(); _ = ds[i0]; t_warm2 = time.perf_counter() - t0

            print(f"  first access  (cold, new recording):     {t_cold*1000:8.1f} ms")
            print(f"  second access (same rec, next window):   {t_warm*1000:8.1f} ms")
            print(f"  re-access idx0 (should still be cached):  {t_warm2*1000:8.1f} ms")
            if t_warm < t_cold * 0.5:
                print("  -> cache appears to be hit (warm access much faster).")
            else:
                print("  -> !! warm access not much faster than cold — cache may not be "
                      "working as expected. Check the LRU key / CACHE_SIZE in nch_dataset.py.")
        else:
            print(f"No single value in '{recording_col}' covers >=2 windows — can't test "
                  f"warm-cache speedup this way. Try two indices known to share a recording.")
    else:
        print("Skipped: no recording-identifier column found in step 1. "
              "Manually pick two indices known to share a recording and time them.")

    print("\n" + "=" * 70)
    print("5. RUN-BOUNDARY / GAP SPOT CHECK")
    print("=" * 70)
    boundary_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["gap", "boundary", "run", "break", "onset", "start_epoch"]
    )]
    if boundary_cols:
        print(f"Columns that might indicate run/boundary info: {boundary_cols}")
        print("Inspect these manually below — script doesn't assume your exact encoding.")
        print(df[boundary_cols].describe(include="all").to_string())
    else:
        print("No obviously-named boundary/gap/run column found in the index. To check "
              "windowing correctness directly, either:")
        print("  - add a debug assertion inside nch_dataset.py's window-building step that")
        print("    each window's 20 onset timestamps increment by exactly 30.0s (+/- 0.5s), or")
        print("  - dump the 20 onset timestamps for one sampled window here and check spacing")
        print("    by hand.")

    print("\n" + "=" * 70)
    print("6. SUMMARY")
    print("=" * 70)
    if all_stats:
        elapsed = [s["elapsed_s"] for s in all_stats]
        nan_flags = [s for s in all_stats if s["nan_ct"] or s["inf_ct"]]
        print(f"Sampled {len(all_stats)} windows total across {len(age_bins)} age bin(s).")
        print(f"Timing: min={min(elapsed)*1000:.1f}ms  max={max(elapsed)*1000:.1f}ms  "
              f"mean={np.mean(elapsed)*1000:.1f}ms")
        print(f"Windows with NaN/Inf: {len(nan_flags)} / {len(all_stats)}")
        if nan_flags:
            print("  !! Indices with NaN/Inf:", [s["idx"] for s in nan_flags])
    else:
        print("No windows were sampled — check age_bin detection above.")


if __name__ == "__main__":
    main()
