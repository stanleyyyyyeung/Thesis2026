import wfdb
from wfdb.processing import resample_sig, resample_multichan
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import signal
import lmdb
import pickle

root_dir = r'/data2/datasets/Physionet2018/training'
target_dir = r'/data2/datasets/Physionet2018/processed'

subject_dirs = os.listdir(root_dir)
subject_dirs.sort()

# for subject_dir in subject_dirs:
#     num = len(os.listdir(os.path.join(root_dir, subject_dir)))
#     if num != 4:
#         print(subject_dir)

file_path_list = [f'{root_dir}/{subject}/{subject}.mat' for subject in subject_dirs]
print(file_path_list)
print(len(file_path_list))

label2id = {'W': 0,
            'N1': 1,
            'N2': 2,
            'N3': 3,
            'R': 4}

db = lmdb.open(target_dir, map_size=253954797509)
file_key_list = []

for file_path in tqdm(file_path_list):
    signals, fields = wfdb.rdsamp(file_path[:-4], channel_names=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'])
    ann = wfdb.rdann(file_path[:-4], 'arousal')
    sleep_signals = signals[ann.sample[0]:ann.sample[-1], :]

    temp = sleep_signals.shape[0] % 6000
    if temp != 0:
        sleep_signals = sleep_signals[:-temp]
    # print(sleep_signals.shape)

    sample_num = sleep_signals.shape[0] // 6000
    # print(sample_num)
    sleep_signals = sleep_signals.reshape(sample_num, 30, 200, 6)
    samples = sleep_signals.transpose((0, 3, 1, 2))
    # print(samples.shape)

    for i, sample in enumerate(samples):
        # print(i, sample.shape)
        sample_key = f'{file_path[-13:-4]}_{i}'
        print(sample_key)
        file_key_list.append(sample_key)
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
        txn.commit()


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
txn.commit()
db.close()
