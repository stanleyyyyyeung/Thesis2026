import argparse
import glob
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ============================ ASSUMPTIONS - VERIFY ============================
CACHE_ROOT = "/srv/scratch/speechdata/sleep_data/NCH/eegmamba_seq"
SEQ_SUBDIR = "seq"
LABEL_SUBDIR = "labels"
LIST_DIR_NAME = "split_lists"
SPLIT_TO_LIST = {"train": "train_list", "val": "eval_list", "test": "test_list"}  # confirmed
CACHED_SEQ_LEN = 20            # cache holds 20-epoch blocks
N_CH, EPOCH_SAMPLES = 6, 6000
CACHE_SCALE = 0.01             # ASSUMED: files hold microvolts; model input = uV/100,
                               # which equals the live path's volts * 1e4
X_KEY, Y_KEY = "x", "y"        # only used if a .npy is a pickled dict instead of a plain array
PLAUSIBLE_STD = (0.02, 20.0)   # scale sanity range for per-channel std in model units
# ==============================================================================


def _list_path(cache_root, split, age_bin):
    return os.path.join(cache_root, LIST_DIR_NAME, f"{SPLIT_TO_LIST[split]}_{age_bin}.txt")


def _read_list(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d, n = line.split("\t")
                out.append((d, int(n)))
    return out


def _resolve_bins(cache_root, age_bin):
    if age_bin is None or age_bin == "all":
        files = glob.glob(os.path.join(cache_root, LIST_DIR_NAME, "train_list_*.txt"))
        return sorted(os.path.basename(f)[len("train_list_"):-len(".txt")] for f in files)
    return [age_bin] if isinstance(age_bin, str) else list(age_bin)


def cache_status(params, cache_root=CACHE_ROOT):
    """Returns (usable, reasons). Only 'files not there' conditions make it unusable."""
    reasons = []
    if params.seq_len != CACHED_SEQ_LEN:
        reasons.append(f"seq_len={params.seq_len} but the cache holds {CACHED_SEQ_LEN}-epoch blocks")
    bins = _resolve_bins(cache_root, getattr(params, "age_bin", "all"))
    if not bins:
        reasons.append(f"no list files found under {os.path.join(cache_root, LIST_DIR_NAME)}")
    for b in bins:
        for split in SPLIT_TO_LIST:
            p = _list_path(cache_root, split, b)
            if not os.path.isfile(p):
                reasons.append(f"missing {p}")
                continue
            entries = _read_list(p)
            if not entries:
                reasons.append(f"empty {p}")
                continue
            gone = [d for d, _ in entries if not os.path.isdir(d)]
            if gone:
                reasons.append(f"{len(gone)} subject dirs in {p} do not exist (e.g. {gone[0]})")
    return (not reasons), reasons


class NCHCachedDataset(Dataset):
    """Items are (x [20, 6, 6000] float32, y [20] int64), same contract as NCHIndexDataset."""

    def __init__(self, cache_root, split, age_bin):
        self.items = []  # (seq_path, label_path)
        for b in _resolve_bins(cache_root, age_bin):
            for subj_dir, n_listed in _read_list(_list_path(cache_root, split, b)):
                files = sorted(f for f in os.listdir(subj_dir) if f.endswith(".npy"))
                if len(files) != n_listed:
                    print(f"[cache] WARNING {subj_dir}: list says {n_listed} sequences, found {len(files)}")
                label_dir = os.path.join(cache_root, LABEL_SUBDIR, os.path.basename(subj_dir.rstrip("/")))
                for f in files:
                    self.items.append((os.path.join(subj_dir, f), os.path.join(label_dir, f)))
        if not self.items:
            raise RuntimeError(f"cache: no sequences for split={split}, age_bin={age_bin}")
        self._check_scale()

    def _check_scale(self):
        probe = [self[0][0], self[len(self) // 2][0]]
        std = float(np.median([p.numpy().std(axis=(0, 2)) for p in probe]))
        lo, hi = PLAUSIBLE_STD
        if not (lo <= std <= hi):
            raise ValueError(
                f"cache: median channel std is {std:.4g} after CACHE_SCALE={CACHE_SCALE}, outside "
                f"{PLAUSIBLE_STD}. CACHE_SCALE is probably wrong for how the .npy files were saved "
                "(volts -> 1e4, microvolts -> 0.01, already model units -> 1.0). Fix it, don't fall back.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sp, lp = self.items[i]
        obj = np.load(sp, allow_pickle=True)
        if obj.dtype == object:  # pickled dict
            d = obj.item()
            x, y = d[X_KEY], d[Y_KEY]
        else:
            x, y = obj, np.load(lp)
        x = np.asarray(x, dtype=np.float32) * CACHE_SCALE
        y = np.asarray(y).astype(np.int64).reshape(-1)
        if x.shape != (CACHED_SEQ_LEN, N_CH, EPOCH_SAMPLES) or y.shape[0] != CACHED_SEQ_LEN:
            raise ValueError(f"{sp}: expected x (20,6,6000) and y (20,), got {x.shape} and {y.shape}")
        return torch.from_numpy(x), torch.from_numpy(y)


class LoadDatasetCachedFirst:
    """Drop-in for LoadDataset: get_data_loader() -> {'train','val','test'}.
    mode: 'auto' (cache if usable else live) | 'cache' (error if unusable) | 'live'."""

    def __init__(self, params, mode="auto", cache_root=CACHE_ROOT):
        self.params, self.mode, self.cache_root = params, mode, cache_root
        self.source = None

    def get_data_loader(self):
        if self.mode != "live":
            ok, reasons = cache_status(self.params, self.cache_root)
            if ok:
                self.source = "cache"
                print("[data] using PREPROCESSED cache")
                return self._cached_loaders()
            msg = "; ".join(reasons[:5]) + (f" (+{len(reasons) - 5} more)" if len(reasons) > 5 else "")
            if self.mode == "cache":
                raise RuntimeError(f"data_source=cache but the cache is not usable: {msg}")
            print(f"[data] WARNING cache not usable -> falling back to on-the-fly loading. Reason: {msg}")
        if getattr(self.params, "datasets_dir", None) is None:
            raise RuntimeError("Live loading needs --datasets_dir (the index parquet).")
        from .nch_dataset_loader import LoadDataset as LiveLoadDataset  # existing code, unchanged
        self.source = "live"
        print("[data] using ON-THE-FLY loading")
        return LiveLoadDataset(self.params).get_data_loader()

    def _cached_loaders(self):
        p = self.params
        age = getattr(p, "age_bin", "all")
        sets = {s: NCHCachedDataset(self.cache_root, s, age) for s in SPLIT_TO_LIST}
        print("[data] cached windows train/val/test:", [len(sets[s]) for s in ("train", "val", "test")])
        kw = dict(batch_size=p.batch_size, num_workers=p.num_workers, pin_memory=True)
        return {"train": DataLoader(sets["train"], shuffle=True, **kw),
                "val": DataLoader(sets["val"], shuffle=False, **kw),
                "test": DataLoader(sets["test"], shuffle=False, **kw)}


def compare(params, cache_root=CACHE_ROOT, n_cached=200, n_live=6, num_classes=5):
    """Sanity comparison of cached vs live. Windows are sampled independently (they need not
    correspond one-to-one), so this checks scale/label balance, not window identity."""
    from .nch_dataset import NCHIndexDataset
    age = getattr(params, "age_bin", "all")
    rng = random.Random(0)
    for split in ("train", "val", "test"):
        c = NCHCachedDataset(cache_root, split, age)
        live = NCHIndexDataset(params.datasets_dir, seq_len=CACHED_SEQ_LEN, split=split, age_bin=age)
        cs = [c[i] for i in rng.sample(range(len(c)), min(n_cached, len(c)))]
        ls = [live[i][0] for i in rng.sample(range(len(live)), min(n_live, len(live)))]
        std = lambda xs: np.median(np.stack([x.numpy().std(axis=(0, 2)) for x in xs]), axis=0)
        ratio = std([x for x, _ in cs]) / std(ls)
        frac = lambda ys: np.bincount(np.concatenate(ys), minlength=num_classes) / sum(len(y) for y in ys)
        cf = frac([y.numpy() for _, y in cs])
        lf = frac([np.asarray(r) for r in live.df.label_seq])
        print(f"\n[{split}] windows: cached={len(c)} live={len(live)}")
        print("  channel std ratio cached/live:", np.round(ratio, 3),
              "<-- should be ~1" if np.all((ratio > 0.7) & (ratio < 1.4)) else "<-- SCALE/FILTER MISMATCH")
        print("  label fractions cached:", np.round(cf, 3))
        print("  label fractions live  :", np.round(lf, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", required=True)
    ap.add_argument("--age_bin", default="all")
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    a = ap.parse_args()
    compare(Namespace(datasets_dir=a.datasets_dir, age_bin=a.age_bin, seq_len=CACHED_SEQ_LEN), a.cache_root)
