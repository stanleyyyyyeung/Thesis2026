import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal
from torch.utils.data.sampler import RandomSampler
import math


class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir
    ):
        super(PretrainingDataset, self).__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # if len(self.keys) > 20000:
        #     self.keys = self.keys[:20000]
        # self.keys = self.keys[:64]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        patch = to_tensor(patch)
        # print(patch)
        return patch



class CustomSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.sum_dataset_size = sum([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.sum_dataset_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        # step = self.batch_size * self.number_of_datasets
        # samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        # epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset

        for i in range(self.number_of_datasets):
            cur_batch_sampler = sampler_iterators[i]
            num_batchs = math.ceil(len(self.dataset.datasets[i]) / self.batch_size)
            for j in range(num_batchs):
                batch_samples = []
                for _ in range(self.batch_size):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        batch_samples.append(cur_sample)
                    except StopIteration:
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        batch_samples.append(cur_sample)
                final_samples_list.append(batch_samples)
        # print(final_samples_list)
        random.shuffle(final_samples_list)
        for batch_samples in final_samples_list:
            for sample in batch_samples:
                yield sample



