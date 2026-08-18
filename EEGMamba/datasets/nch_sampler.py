import random
from torch.utils.data import Sampler

class RecordingGroupedBatchSampler(Sampler):
    """
    Groups dataset row-indices by edf_path so each batch is drawn from a
    single recording wherever possible. Without this, DataLoader(shuffle=True)
    picks uniformly across all ~168k windows spanning many recordings, which
    thrashes NCHIndexDataset's per-worker LRU cache (every __getitem__ call
    becomes a cold ~8-10s EDF load instead of a cache hit).
    """
    def __init__(self, df, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.groups = df.groupby('edf_path').indices  # {edf_path: array of row positions}

    def __iter__(self):
        group_keys = list(self.groups.keys())
        if self.shuffle:
            random.shuffle(group_keys)

        batches = []
        for key in group_keys:
            idxs = list(self.groups[key])
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i:i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)

        if self.shuffle:
            random.shuffle(batches)  # shuffle batch ORDER, not contents —
                                       # keeps each batch single-recording
        return iter(batches)

    def __len__(self):
        n = sum(len(v) for v in self.groups.values())
        return (n + self.batch_size - 1) // self.batch_size if not self.drop_last \
               else n // self.batch_size
