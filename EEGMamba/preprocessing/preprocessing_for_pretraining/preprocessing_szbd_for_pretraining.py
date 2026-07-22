import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb


root_dir = r'/data2/datasets/szbd/62_rest_4classes_good_final_set'
target_dir = r'/data2/datasets/szbd/processed_average'

file_paths = os.listdir(root_dir)
print(file_paths)
print(len(file_paths))

channels = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6',
    'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2',
    # 'VEO', 'HEO'
]

db = lmdb.open(target_dir, map_size=30000000000)
file_key_list = []

for file_path in tqdm(file_paths):
    raw = mne.io.read_raw_fif(os.path.join(root_dir, file_path), preload=True)
    raw.pick_channels(channels, ordered=True)
    raw.set_eeg_reference(ref_channels='average')
    raw.resample(200)
    # raw.plot(duration=30, n_channels=27, clipping=None)
    # raw.compute_psd().plot(average=True)
    samples = raw.get_data(units='uV')
    temp = samples.shape[1] % 6000
    if temp != 0:
        samples = samples[:, :-temp]
    samples = samples.reshape(60, -1, 30, 200)
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

