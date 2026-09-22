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
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_SFREQ = 200.0
SCALE_TO_MODEL_INPUT = 1e4

FILTER_LOWCUT_HZ = 0.3
FILTER_HIGHCUT_HZ = 75.0
NOTCH_FREQ_HZ = 60.0

CHANNELS_ORDERED = [
    "EEG F3-M2",
    "EEG C3-M2",
    "EEG O1-M2",
    "EEG F4-M1",
    "EEG C4-M1",
    "EEG O2-M1",
]

EPOCH_SEC = 30.0
SAMPLES_PER_EPOCH = int(EPOCH_SEC * TARGET_SFREQ)  # 6000, independent of seq_len

# Default only — real Dataset instances always get seq_len explicitly from
# the caller (see LoadDataset), which should itself come from --seq_len.
# This exists so NCHIndexDataset(path) still works in a REPL/notebook
# without forcing you to know the length up front.
DEFAULT_SEQ_LEN = 20

CACHE_SIZE = 4


class NCHIndexDataset(Dataset):
    """
    Args:
        index_parquet_path: path to a parquet produced by build_nch_index.py
        seq_len: number of 30s epochs per window. MUST match the --seq-len
            the parquet was built with — checked against the parquet's own
            file metadata at construction time (see below), not inferred.
        split: optional filter, e.g. "train" / "eval" / "test"
        age_bin: optional filter, e.g. "6-12y"
        exclude_epilepsy: if True, drops rows where epilepsy_flag is truthy
        cache_size: overrides CACHE_SIZE per-instance
    """

    def __init__(self, index_parquet_path, seq_len=DEFAULT_SEQ_LEN, split=None,
                 age_bin=None, exclude_epilepsy=False, cache_size=CACHE_SIZE):
        self.seq_len = seq_len
        self.samples_per_window = self.seq_len * SAMPLES_PER_EPOCH

        # --- fail fast: does this parquet actually match the requested seq_len? ---
        # build_nch_index.py embeds seq_len in the file's pyarrow metadata
        # (not a row column), so we check it via ParquetFile.schema_arrow.metadata
        # rather than pd.read_parquet, which drops file-level metadata.
        schema_metadata = pq.ParquetFile(index_parquet_path).schema_arrow.metadata or {}
        raw_seq_len = schema_metadata.get(b"seq_len")
        if raw_seq_len is None:
            raise ValueError(
                f"{index_parquet_path} has no 'seq_len' key in its file metadata. "
                "Likely built by a pre--seq-len version of build_nch_index.py. "
                "Rebuild the index rather than assuming it's seq_len=20."
            )
        file_seq_len = int(raw_seq_len)
        if file_seq_len != self.seq_len:
            raise ValueError(
                f"{index_parquet_path} was built with seq_len={file_seq_len}, but "
                f"this Dataset was constructed with seq_len={self.seq_len}. Point "
                f"--datasets_dir at nch_index_..._seqlen{self.seq_len}.parquet instead."
            )

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
                f"exclude_epilepsy={exclude_epilepsy}, seq_len={self.seq_len} in "
                f"{index_parquet_path}. Check the filter values against what's "
                "actually in the index (e.g. df.split.unique(), df.age_bin.unique()) "
                "— note this can also legitimately be zero rows at long seq_len "
                "for fragmented age bins, see handoff's window-survival concern."
            )

        self.df = df.reset_index(drop=True)
        self.cache_size = cache_size
        self._raw_cache = OrderedDict()

    def __len__(self):
        return len(self.df)

    def _load_processed_raw(self, edf_path):
        """(unchanged — filtering is seq_len-independent, still whole-recording)"""
        if edf_path in self._raw_cache:
            self._raw_cache.move_to_end(edf_path)
            return self._raw_cache[edf_path]

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        missing = [ch for ch in CHANNELS_ORDERED if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"{edf_path}: missing expected channel(s) {missing}. "
                f"Available channels: {raw.ch_names}. "
                "This study should probably have been excluded at the "
                "manifest stage — check whether it slipped through that filter."
            )

        raw.pick(CHANNELS_ORDERED)
        raw.reorder_channels(CHANNELS_ORDERED)
        raw.load_data(verbose=False)

        if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6:
            raw.resample(TARGET_SFREQ, verbose=False)

        raw.filter(FILTER_LOWCUT_HZ, FILTER_HIGHCUT_HZ, fir_design="firwin", verbose=False)
        raw.notch_filter(NOTCH_FREQ_HZ, verbose=False)

        self._raw_cache[edf_path] = raw
        self._raw_cache.move_to_end(edf_path)
        if len(self._raw_cache) > self.cache_size:
            self._raw_cache.popitem(last=False)

        return raw

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = self._load_processed_raw(row.edf_path)

        assert abs(raw.info["sfreq"] - TARGET_SFREQ) < 1e-6, (
            f"internal error: raw.info['sfreq']={raw.info['sfreq']} != "
            f"TARGET_SFREQ={TARGET_SFREQ} after _load_processed_raw."
        )
        start_sample = int(round(row.seq_start_sec * TARGET_SFREQ))
        stop_sample = start_sample + self.samples_per_window

        data = raw.get_data(start=start_sample, stop=stop_sample)  # (6, samples_per_window)
        data = data * SCALE_TO_MODEL_INPUT

        if data.shape[1] != self.samples_per_window:
            raise ValueError(
                f"{row.edf_path} at seq_start_sec={row.seq_start_sec}: expected "
                f"{self.samples_per_window} samples (seq_len={self.seq_len}), got "
                f"{data.shape[1]}. This usually means the window runs past the end "
                "of the (resampled) recording — check whether the index was built "
                "against a differently-processed version of this EDF, or whether "
                "resampling introduced an off-by-one at the recording's tail."
            )

        x = data.reshape(len(CHANNELS_ORDERED), self.seq_len, SAMPLES_PER_EPOCH)
        x = x.transpose(1, 0, 2)  # (seq_len, 6, 6000)

        y = np.asarray(row.label_seq, dtype=np.int64)
        if y.shape[0] != self.seq_len:
            raise ValueError(
                f"{row.edf_path} at seq_start_sec={row.seq_start_sec}: label_seq "
                f"has length {y.shape[0]}, expected {self.seq_len}. Index row is malformed."
            )

        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y).long()

from .nch_dataset_loader import LoadDataset
