# %%
"""
ISRUC group 1 extraction: .rec + scorer-1 .txt -> per-subject seq/label .npy

Fixes applied vs. original script:
  1. Paths updated for Katana (were pointing at an old cluster's /data/...).
  2. Channel selection is now NAME-based, not positional. A full 1-100 sweep
     found 9 distinct channel name/order layouts across ISRUC group 1 —
     positional slicing (psg_array[:, 2:8]) only happened to be correct for
     8/9 sampled subjects and would have silently pulled the wrong physical
     channels for others (e.g. subject 40's reordered montage).
  3. Subject 8 is excluded: its recording has only 4 of the 6 target
     channels (F3 and F4 are entirely absent, not just misnamed/reordered).
     This is a deliberate exclusion, not a bug workaround — it shifts the
     paper's train split from 80 to 79 subjects (see isruc_dataset.py note).
  4. raw.resample(sfreq=200) is intentionally left out: confirmed via full
     sweep that all 100 subjects are natively recorded at 200Hz, so this
     is a no-op, not an omission.
  5. Per-subject try/except so one malformed file doesn't kill the whole
     batch run and lose progress on subjects already processed.
"""

import os
from edf_ import read_raw_edf
import numpy as np
from tqdm import tqdm

dir_path = '/srv/scratch/speechdata/sleep_data/ISRUC'
seq_dir = '/srv/scratch/speechdata/sleep_data/ISRUC/seq'
label_dir = '/srv/scratch/speechdata/sleep_data/ISRUC/labels'

# Deliberately excluded — see module docstring point 3
EXCLUDED_SUBJECTS = {8}

TARGET_SITES = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']  # fixed output channel order

label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}

psg_label_f_pairs = []  # list of (subject_id, psg_path, label_path)
for i in range(1, 101):
    if i in EXCLUDED_SUBJECTS:
        continue
    numstr = str(i)
    psg_f_name = f'{dir_path}/{numstr}/{numstr}.rec'
    label_f_name = f'{dir_path}/{numstr}/{numstr}_1.txt'
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((i, psg_f_name, label_f_name))

print(f"{len(psg_label_f_pairs)} subjects queued for extraction "
      f"(excluded: {sorted(EXCLUDED_SUBJECTS)})")


def select_target_channels(raw):
    """Pick the 6 target EEG channels by 10-20 site name, regardless of
    reference-electrode suffix (-A2/-A1/-M2/-M1) or original channel order.
    Raises loudly if a subject is missing a target site, rather than
    silently mis-selecting via position."""
    site_to_actual = {}
    for ch in raw.ch_names:
        site = ch.split('-')[0]
        if site in TARGET_SITES and site not in site_to_actual:
            site_to_actual[site] = ch
    missing = [s for s in TARGET_SITES if s not in site_to_actual]
    if missing:
        raise ValueError(f"Missing target sites {missing} in channels: {raw.ch_names}")
    ordered_names = [site_to_actual[s] for s in TARGET_SITES]
    raw.pick_channels(ordered_names, ordered=True)
    return raw


# %%
num_seqs = 0
num_labels = 0
failed_subjects = []

for subject_num, psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    try:
        labels_list = []
        raw = read_raw_edf(psg_f_name, preload=True)
        raw = select_target_channels(raw)
        raw.filter(0.3, 35, fir_design='firwin')
        raw.notch_filter((50))

        psg_array = raw.to_data_frame().values
        psg_array = psg_array[:, 1:]  # drop the time column; remaining 6 cols are our target channels, in order

        i_trunc = psg_array.shape[0] % (30 * 200)
        if i_trunc > 0:
            psg_array = psg_array[:-i_trunc, :]
        psg_array = psg_array.reshape(-1, 30 * 200, 6)

        a = psg_array.shape[0] % 20
        if a > 0:
            psg_array = psg_array[:-a, :, :]
        psg_array = psg_array.reshape(-1, 20, 30 * 200, 6)
        epochs_seq = psg_array.transpose(0, 1, 3, 2)

        for line in open(label_f_name).readlines():
            line_str = line.strip()
            if line_str != '':
                labels_list.append(label2id[line_str])
        labels_array = np.array(labels_list)
        if a > 0:
            labels_array = labels_array[:-a]
        labels_seq = labels_array.reshape(-1, 20)

        if epochs_seq.shape[0] != labels_seq.shape[0]:
            raise ValueError(f"seq/label count mismatch: {epochs_seq.shape[0]} vs {labels_seq.shape[0]}")

        subj_seq_dir = f'{seq_dir}/ISRUC-group1-{subject_num}'
        subj_label_dir = f'{label_dir}/ISRUC-group1-{subject_num}'
        os.makedirs(subj_seq_dir, exist_ok=True)
        os.makedirs(subj_label_dir, exist_ok=True)

        for seq in epochs_seq:
            np.save(f'{subj_seq_dir}/ISRUC-group1-{subject_num}-{num_seqs}.npy', seq)
            num_seqs += 1
        for label in labels_seq:
            np.save(f'{subj_label_dir}/ISRUC-group1-{subject_num}-{num_labels}.npy', label)
            num_labels += 1

    except Exception as e:
        print(f"\n[FAILED] {psg_f_name}: {e}")
        failed_subjects.append((psg_f_name, str(e)))
        continue

print(f"\nDone. {num_seqs} sequences, {num_labels} labels saved.")
if failed_subjects:
    print(f"\n{len(failed_subjects)} subject(s) FAILED extraction:")
    for f, err in failed_subjects:
        print(f"  {f}: {err}")
