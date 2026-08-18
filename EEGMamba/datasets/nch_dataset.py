"""
nch_dataset.py

On-the-fly PyTorch Dataset for NCH sleep staging, matching the
ISRUC/EEGMamba finetune input contract: each item is a (20, 6, 6000) tensor
(20 epochs x 6 channels x 30s@200Hz) plus a (20,) label sequence.

Reads exclusively from the index built by build_nch_index.py — no signal
data is EVER written to disk by this file. Signal extraction (EDF read,
resample, filter, channel select, epoch slice) happens live inside
__getitem__, per request, and is discarded after use except for a small
per-worker in-memory cache of already-processed recordings (see
CACHE_SIZE below) — this is what "no files stored in between" means in
practice: nothing survives past the Python process's own memory, and
even that is bounded and evictable.

Everything you're likely to want to tune lives in the CONFIG block below,
each with its own comment on what changing it does and what it costs.
"""

import os
from collections import OrderedDict

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target sample rate after resampling
TARGET_SFREQ = 200.0
# Normalise each data value by multiplying by 100uV (mirroring ISRUC method)
SCALE_TO_MODEL_INPUT = 1e4

# Bandpass + notch filter, applied ONCE per opened recording (see
# CACHE-then-filter note below), matching ISRUC's preprocessing exactly.
FILTER_LOWCUT_HZ = 0.3
FILTER_HIGHCUT_HZ = 35.0
NOTCH_FREQ_HZ = 50.0

# Channel order — LOCKED to match model input convention
CHANNELS_ORDERED = [
    "EEG F3-M2",
    "EEG C3-M2",
    "EEG O1-M2",
    "EEG F4-M1",
    "EEG C4-M1",
    "EEG O2-M1",
]

EPOCH_SEC = 30.0
SEQ_LEN = 20
SAMPLES_PER_EPOCH = int(EPOCH_SEC * TARGET_SFREQ)  # 6000
SAMPLES_PER_WINDOW = SEQ_LEN * SAMPLES_PER_EPOCH    # 120000

# How many fully-processed (resampled+filtered, in-memory) recordings to
# keep resident at once
CACHE_SIZE = 4


class NCHIndexDataset(Dataset):
    """
    Args:
        index_parquet_path: path to a parquet produced by build_nch_index.py
        split: optional filter, e.g. "train" / "eval" / "test"
        age_bin: optional filter, e.g. "6-12y" — pass this to get exactly
            one finetuning group's data; omit to get everything in the index
        exclude_epilepsy: if True, drops rows where epilepsy_flag is truthy
            (the index itself does NOT filter this by default — see
            build_nch_index.py's --exclude-epilepsy flag, which does the
            same filtering at index-build time instead, if you'd rather
            bake it in once than filter on every Dataset construction)
        cache_size: overrides CACHE_SIZE per-instance if you want a
            different cache budget for a specific run
    """

    def __init__(self, index_parquet_path, split=None, age_bin=None,
                 exclude_epilepsy=False, cache_size=CACHE_SIZE):
        df = pd.read_parquet(index_parquet_path)

        if split is not None:
            df = df[df.split == split]
        if age_bin is not None:
            df = df[df.age_bin == age_bin]
        if exclude_epilepsy:
            df = df[~df.epilepsy_flag.astype(str).isin(["1", "True", "true"])]

        if len(df) == 0:
            raise ValueError(
                f"No rows matched split={split!r}, age_bin={age_bin!r}, "
                f"exclude_epilepsy={exclude_epilepsy} in {index_parquet_path}. "
                "Check the filter values against what's actually in the index "
                "(e.g. df.split.unique(), df.age_bin.unique())."
            )

        self.df = df.reset_index(drop=True)
        self.cache_size = cache_size
        # OrderedDict as a simple LRU: move-to-end on access, pop oldest on overflow
        self._raw_cache = OrderedDict()

    def __len__(self):
        return len(self.df)

    def _load_processed_raw(self, edf_path):
        """
        Returns a fully processed (channel-selected, resampled, filtered)
        MNE Raw object for edf_path, using the per-worker cache if present.

        NOTE ON WHERE FILTERING HAPPENS: filtering is applied to the WHOLE
        recording, once, at load time — not to each 20-epoch slice
        independently. This avoids edge-effect artifacts at window
        boundaries that per-slice filtering would introduce (a bandpass
        filter needs some signal on either side of the region you actually
        care about to settle correctly). The cost is that caching now holds
        real signal in memory, not just header metadata — see CACHE_SIZE
        above for the memory/speed tradeoff this implies. If you want the
        lighter-weight (header-only cache, filter-per-slice) tradeoff
        instead, the change is localized to this method plus __getitem__.
        """
        if edf_path in self._raw_cache:
            self._raw_cache.move_to_end(edf_path)
            return self._raw_cache[edf_path]

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        missing = [ch for ch in CHANNELS_ORDERED if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"{edf_path}: missing expected channel(s) {missing}. "
                f"Available channels: {raw.ch_names}. "
                "This study should probably have been excluded at the "
                "manifest stage (see handoff's 9-subject channel-based "
                "exclusion) — if you're seeing this, check whether this "
                "study slipped through that filter."
            )

        raw.pick(CHANNELS_ORDERED)
        raw.reorder_channels(CHANNELS_ORDERED)  # belt-and-suspenders: pick()
                                                  # order isn't guaranteed
                                                  # identical across MNE
                                                  # versions, this is.

        if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6:
            raw.resample(TARGET_SFREQ, verbose=False)

        raw.filter(FILTER_LOWCUT_HZ, FILTER_HIGHCUT_HZ, fir_design="firwin", verbose=False)
        raw.notch_filter(NOTCH_FREQ_HZ, verbose=False)

        self._raw_cache[edf_path] = raw
        self._raw_cache.move_to_end(edf_path)
        if len(self._raw_cache) > self.cache_size:
            self._raw_cache.popitem(last=False)  # evict least-recently-used

        return raw

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = self._load_processed_raw(row.edf_path)

        # seq_start_sec comes from the TSV's onset column, which is a wall-clock
        # offset unaffected by resampling — multiplying by TARGET_SFREQ is
        # correct exactly because raw is guaranteed resampled to TARGET_SFREQ
        # by the time we get here (see assertion below).
        assert abs(raw.info["sfreq"] - TARGET_SFREQ) < 1e-6, (
            f"internal error: raw.info['sfreq']={raw.info['sfreq']} != "
            f"TARGET_SFREQ={TARGET_SFREQ} after _load_processed_raw — resample "
            "step above didn't take effect as expected."
        )
        start_sample = int(round(row.seq_start_sec * TARGET_SFREQ))
        stop_sample = start_sample + SAMPLES_PER_WINDOW

        data = raw.get_data(start=start_sample, stop=stop_sample)  # (6, 120000)
        data = data * SCALE_TO_MODEL_INPUT

        if data.shape[1] != SAMPLES_PER_WINDOW:
            raise ValueError(
                f"{row.edf_path} at seq_start_sec={row.seq_start_sec}: expected "
                f"{SAMPLES_PER_WINDOW} samples, got {data.shape[1]}. This usually "
                "means the window runs past the end of the (resampled) recording — "
                "check whether the index was built against a differently-processed "
                "version of this EDF, or whether resampling introduced an off-by-one "
                "at the recording's tail."
            )

        x = data.reshape(len(CHANNELS_ORDERED), SEQ_LEN, SAMPLES_PER_EPOCH)
        x = x.transpose(1, 0, 2)  # (20, 6, 6000)

        y = np.asarray(row.label_seq, dtype=np.int64)
        if y.shape[0] != SEQ_LEN:
            raise ValueError(
                f"{row.edf_path} at seq_start_sec={row.seq_start_sec}: label_seq "
                f"has length {y.shape[0]}, expected {SEQ_LEN}. Index row is malformed."
            )

        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y).long()
