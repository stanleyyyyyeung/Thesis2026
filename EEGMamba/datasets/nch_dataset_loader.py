from torch.utils.data import DataLoader
from nch_dataset import NCHIndexDataset
from nch_sampler import RecordingGroupedBatchSampler

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        # params.datasets_dir repurposed to hold the parquet path
        self.index_path = params.datasets_dir
        self.age_bin = params.age_bin  # add this arg to argparse

    def get_data_loader(self):
        train_set = NCHIndexDataset(self.index_path, split='train', age_bin=self.age_bin)
        val_set   = NCHIndexDataset(self.index_path, split='val',   age_bin=self.age_bin)
        test_set  = NCHIndexDataset(self.index_path, split='test',  age_bin=self.age_bin)
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
