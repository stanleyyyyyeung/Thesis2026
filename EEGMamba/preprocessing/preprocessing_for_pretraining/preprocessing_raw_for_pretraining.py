import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb


root_dir = r'/data2/datasets/Raw EEG Data/files'
target_dir = r'/data2/datasets/Raw EEG Data/processed_5s_average'

file_paths = os.listdir(root_dir)

channels = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
    'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
    # 'M1', 'M2',
]

db = lmdb.open(target_dir, map_size=40000000000)
file_key_list = []

for file_path in tqdm(file_paths):
    raw = mne.io.read_raw_bdf(os.path.join(root_dir, file_path), preload=True)
    raw.pick_channels(channels, ordered=True)
    # print(raw.info)
    raw.set_eeg_reference(ref_channels='average')
    raw.resample(200)
    # raw.plot(duration=5, n_channels=64, clipping=None)
    raw.filter(l_freq=0.3, h_freq=52)
    raw.notch_filter((60))
    # raw.plot(duration=30, n_channels=27, clipping=None)
    # raw.compute_psd().plot(average=True)
    samples = raw.get_data(units='uV')
    # print(samples.shape)
    temp = samples.shape[1] % 1000
    if temp != 0:
        samples = samples[:, :-temp]
    # print(samples.shape)
    samples = samples.reshape(64, -1, 5, 200)
    # print(samples.shape)
    samples = samples.transpose(1, 0, 2, 3)
    # print(samples.shape)

    for i, sample in enumerate(samples):
        sample_key = f'{file_path[:-4]}_{i}'
        print(sample_key)
        file_key_list.append(sample_key)
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
        txn.commit()


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
txn.commit()
db.close()