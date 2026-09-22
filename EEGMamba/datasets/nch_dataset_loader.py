from torch.utils.data import DataLoader
from .nch_dataset import NCHIndexDataset
from .nch_sampler import RecordingGroupedBatchSampler

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.index_path = params.datasets_dir
        self.age_bin = params.age_bin
        self.seq_len = params.seq_len  # new: add --seq_len to train.py argparse

    def get_data_loader(self):
        train_set = NCHIndexDataset(self.index_path, seq_len=self.seq_len, split='train', age_bin=self.age_bin)
        val_set   = NCHIndexDataset(self.index_path, seq_len=self.seq_len, split='val',   age_bin=self.age_bin)
        test_set  = NCHIndexDataset(self.index_path, seq_len=self.seq_len, split='test',  age_bin=self.age_bin)
        print(len(train_set), len(val_set), len(test_set))

        train_sampler = RecordingGroupedBatchSampler(
            train_set.df, batch_size=self.params.batch_size, shuffle=True
        )
        data_loader = {
            'train': DataLoader(train_set, batch_sampler=train_sampler,
                                 num_workers=self.params.num_workers),
            'val': DataLoader(val_set, batch_size=1, shuffle=False,
                               num_workers=self.params.num_workers),
            'test': DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=self.params.num_workers),
        }
        return data_loader
