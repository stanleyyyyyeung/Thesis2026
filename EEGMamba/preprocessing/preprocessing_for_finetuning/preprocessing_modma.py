import scipy
from scipy import signal
import os
import lmdb
import pickle

root_dir = '/data/datasets/BigDownstream/MODMA/files'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

mdd_files = files[:24]
hc_files = files[24:]

files_dict = {
    'train':mdd_files[:15]+hc_files[:18],
    'val':mdd_files[15:19]+hc_files[18:23],
    'test':mdd_files[19:24]+hc_files[23:29],
}
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open('/data/datasets/BigDownstream/MODMA/processed', map_size=4144193024)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        data = scipy.io.loadmat(os.path.join(root_dir, file))
        eeg = data[list(data.keys())[3]][:128, :]/1000
        # print(eeg)
        # print(eeg.shape)
        temp = eeg.shape[1] % (15*250)
        if temp != 0:
            eeg = eeg[:, :-temp]
        # print(eeg.shape)
        eeg = signal.resample(eeg, eeg.shape[1]//250*200, axis=1)
        # print(eeg.shape)
        eeg = eeg.reshape(128, -1, 15, 200)
        # print(eeg.shape)
        eeg = eeg.transpose(1, 0, 2, 3)
        # print(eeg.shape)
        label = 1 if '0201'== file[:4] else 0
        print(label, file[:8])

        for i, sample in enumerate(eeg):
            sample_key = f'{file[:8]}-{i}'
            print(sample_key)
            data_dict = {
                'sample':sample, 'label':label
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()