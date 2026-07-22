# %%
"""
Test: runs the fixed ISRUC extraction on a small subject subset
covering each distinct case found during the channel-naming sweep:
  - subject 1  : "A2/A1" naming variant (16 subjects use this pattern)
  - subject 8  : deliberately excluded (missing F3/F4) - should be skipped
                 with a printed reason, not crash
  - subject 25 : "M2/M1" naming variant with numeric aux channels (11 subjects)
  - subject 40 : reordered layout, no reference suffix on channel names
                 (the case positional slicing got wrong)

If this completes cleanly and produces sane shapes for all 3 processed
subjects (8 should be absent from output, not erroring), the full run
in extract_isruc_full.py should be safe to run.
"""

import os
from edf_ import read_raw_edf
import numpy as np
from tqdm import tqdm

dir_path = '/srv/scratch/speechdata/sleep_data/ISRUC'
seq_dir = '/srv/scratch/speechdata/sleep_data/ISRUC/seq_TEST'
label_dir = '/srv/scratch/speechdata/sleep_data/ISRUC/labels_TEST'

SMOKE_TEST_SUBJECTS = [1, 8, 25, 40]
EXCLUDED_SUBJECTS = {8}

TARGET_SITES = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}

psg_label_f_pairs = []
for i in SMOKE_TEST_SUBJECTS:
    if i in EXCLUDED_SUBJECTS:
        print(f"Subject {i}: deliberately excluded (missing F3/F4) - skipping, as expected.")
        continue
    numstr = str(i)
    psg_f_name = f'{dir_path}/{numstr}/{numstr}.rec'
    label_f_name = f'{dir_path}/{numstr}/{numstr}_1.txt'
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((i, psg_f_name, label_f_name))

print(f"{len(psg_label_f_pairs)} smoke-test subjects queued: "
      f"{[p[0] for p in psg_label_f_pairs]}")


def select_target_channels(raw):
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
        psg_array = psg_array[:, 1:]

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

        print(f"Subject {subject_num}: OK -> {epochs_seq.shape[0]} sequences, shape {epochs_seq.shape[1:]}")

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
        print(f"\n[FAILED] subject {subject_num} ({psg_f_name}): {e}")
        failed_subjects.append((subject_num, str(e)))
        continue

print(f"\nSmoke test done. {num_seqs} sequences, {num_labels} labels saved to *_SMOKETEST dirs.")
if failed_subjects:
    print(f"\n{len(failed_subjects)} unexpected failure(s):")
    for s, err in failed_subjects:
        print(f"  subject {s}: {err}")
    raise SystemExit(1)  # non-zero exit so the calling shell script can detect failure
else:
    print("All expected subjects processed successfully. Safe to proceed to full run.")
