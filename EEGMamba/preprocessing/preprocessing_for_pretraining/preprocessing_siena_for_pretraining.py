import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb


root_dir = r'/data2/datasets/Siena_Scalp_EEG_Database/physionet.org/files/siena-scalp-eeg/1.0.0'
target_dir = r'/data2/datasets/Siena_Scalp_EEG_Database/processed_10s_average'

file_paths = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if '.edf' in file:
            file_paths.append((root, file))


channels = [
    'EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3', 'EEG T5',
    'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fz', 'EEG Pz', 'EEG F4',
    'EEG C4', 'EEG P4', 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6', 'EEG Fc2', 'EEG Fc6',
    'EEG Cp2', 'EEG Cp6', 'EEG F10'
]

db = lmdb.open(target_dir, map_size=80000000000)
file_key_list = []

for root, file in tqdm(file_paths):
    raw = mne.io.read_raw_edf(os.path.join(root, file), preload=True)
    # print(raw.info)
    raw.pick_channels(channels, ordered=True)
    raw.set_eeg_reference(ref_channels='average')
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((50))
    # raw.plot(duration=30, n_channels=27, clipping=None)
    # raw.compute_psd().plot(average=True)
    # print(raw.info)
    samples = raw.get_data(units='uV')
    temp = samples.shape[1] % 2000
    if temp != 0:
        samples = samples[:, :-temp]
    samples = samples.reshape(27, -1, 10, 200)
    samples = samples.transpose(1, 0, 2, 3)
    for i, sample in enumerate(samples):
        sample_key = f'{file[:-4]}_{i}'
        print(sample_key)
        file_key_list.append(sample_key)
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
        txn.commit()

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
txn.commit()
db.close()