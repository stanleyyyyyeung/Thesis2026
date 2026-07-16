import numpy as np
import hdf5storage
import os
import pandas as pd
from scipy.stats import entropy
from kNearestViterbi import calc_viterbi_k_best, calc_MMI_loss, train_hmm_mmi

# ============================================================
# CONFIGURATION — EDIT THESE BEFORE EACH RUN
# ============================================================
AGE_BINS = ["1-2y", "3-5y", "6-12y", "13-18y"]

# REFINEMENT_SOURCE: "none", "adult", or "nch"
REFINEMENT_SOURCE = "none"
# REFINEMENT_MODE: "none", "hmm", or "hmm_trained"  (ignored if REFINEMENT_SOURCE == "none")
REFINEMENT_MODE = "none"

if REFINEMENT_SOURCE == "none":
    REFINEMENT_MODE = "none"
elif REFINEMENT_SOURCE in ("adult", "nch") and REFINEMENT_MODE not in ("hmm", "hmm_trained"):
    raise ValueError("REFINEMENT_MODE must be 'hmm' or 'hmm_trained' when REFINEMENT_SOURCE is 'adult' or 'nch'")

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]

# ============================================================
# PATHS
# ============================================================
CODE = "/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer"

# --- Adult (SleepEDF-78) paths — used only when REFINEMENT_SOURCE == "adult" ---
# Fold 1's train+eval+test lists combined give a complete, non-overlapping
# partition of all 78 subjects — used here purely as a source of ground-truth
# labels (no per-fold finetuned model involved anymore).
ADULT_FILE_LIST = f"{CODE}/sleepedf-78/file_list_30min/eeg"
ADULT_ALL_SUBJECTS_LISTS = [
    f"{ADULT_FILE_LIST}/train_list_n1.txt",
    f"{ADULT_FILE_LIST}/eval_list_n1.txt",
    f"{ADULT_FILE_LIST}/test_list_n1.txt",
]
# NOTE: not yet run — needed only for REFINEMENT_SOURCE == "adult" with REFINEMENT_MODE == "hmm_trained".
# This should be the SHHS-pretrained checkpoint (best_model_acc) run once on all 78 adult subjects.
ADULT_PRETRAINED_SCORE_PATH = "/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf78_pretrained_scores/all/test_ret.mat"

# --- NCH paths ---
NCH_TEST_INFERENCE_BASE = "/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference_pretrained"
# NOTE: not yet built — needed only for REFINEMENT_SOURCE == "nch" with REFINEMENT_MODE == "hmm_trained"
NCH_TRAIN_INFERENCE_BASE = "/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference_pretrained_train"
NCH_LIST_DIR = "/srv/scratch/speechdata/sleep_data/NCH/sleeptransformer"

PRED_DIR = "/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/nch_inference_pretrained/predictions"

if REFINEMENT_SOURCE == "none":
    OUT_SUBDIR = None
elif REFINEMENT_SOURCE == "adult" and REFINEMENT_MODE == "hmm":
    OUT_SUBDIR = "hmmRefined_adult"
elif REFINEMENT_SOURCE == "adult" and REFINEMENT_MODE == "hmm_trained":
    OUT_SUBDIR = "hmmRefined_adult_trained"
elif REFINEMENT_SOURCE == "nch" and REFINEMENT_MODE == "hmm":
    OUT_SUBDIR = "hmmRefined_nch"
elif REFINEMENT_SOURCE == "nch" and REFINEMENT_MODE == "hmm_trained":
    OUT_SUBDIR = "hmmRefined_nch_trained"

# ============================================================
# CONSTANTS
# ============================================================
seq_len = 21
nstage = 5
STAGES = [1, 2, 3, 4, 5]   # W=1, N1=2, N2=3, N3=4, REM=5
STAGE_NAMES = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}

# ============================================================
# AGGREGATION
# ============================================================
def softmax(z):
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def aggregate_probs(score):
    """
    Multiply aggregation across all seq_len context positions, returning
    the full normalised probability vector (N, 5).
    score: (seq_len, N, 5)
    """
    fused_score = None
    for i in range(seq_len):
        prob_i = np.log10(softmax(np.squeeze(score[i, :, :])))
        prob_i = np.concatenate((np.ones((seq_len - 1, nstage)), prob_i), axis=0)
        prob_i = np.roll(prob_i, -(seq_len - i - 1), axis=0)
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    fused_score -= np.max(fused_score, axis=1, keepdims=True)
    probs = np.exp(fused_score)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs   # (N, 5)

# ============================================================
# HMM PARAMETER ESTIMATION FROM TRAINING SUBJECTS
# ============================================================
def estimate_hmm_parameters_from_gt(y_true_list, label="", verbose=True):
    K = nstage

    pi_counts = np.zeros(K)
    for y in y_true_list:
        pi_counts[y[0]] += 1
    pi = (pi_counts + 1e-6) / (pi_counts + 1e-6).sum()

    A_counts = np.zeros((K, K))
    for y in y_true_list:
        for t in range(len(y) - 1):
            A_counts[y[t], y[t + 1]] += 1
    A_counts += 1e-6
    A = A_counts / A_counts.sum(axis=1, keepdims=True)

    if verbose:
        print(f"\n{'='*70}\nHMM PARAMETERS ({label}) — from ground-truth sequences\n{'='*70}")
        print(pd.DataFrame(
            np.round(A, 4),
            index=[STAGE_NAMES[s] for s in STAGES],
            columns=[STAGE_NAMES[s] for s in STAGES]
        ))
        print("\nInitial State Distribution pi")
        for i in range(K):
            print(f"  {STAGE_NAMES[i + 1]} : {pi[i]:.4f}")
        print("=" * 70 + "\n")

    return A, pi

# ============================================================
# VITERBI WITH SOFTMAX EMISSIONS
# ============================================================
def viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha):
    T, K = obs_probs.shape
    log_emit = np.log(obs_probs + 1e-10)

    viterbi = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)
    viterbi[0] = log_pi + log_emit[0]

    for t in range(1, T):
        scores = viterbi[t - 1][:, None] + alpha * log_A
        best_prev = np.argmax(scores, axis=0)
        viterbi[t] = scores[best_prev, np.arange(K)] + log_emit[t]
        backptr[t] = best_prev

    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(viterbi[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path   # 0-indexed

# ============================================================
# HELPERS
# ============================================================
def load_list(list_path):
    with open(list_path, "r") as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]

def load_labels(files):
    ytrue_list = []
    for fpath in files:
        data = hdf5storage.loadmat(file_name=fpath)
        label = np.array(data['label']).squeeze()
        ytrue_list.append((label - 1).astype(int))
    return ytrue_list

# ============================================================
# ADULT (SleepEDF-78) HMM PRIOR — single estimate, no fold averaging needed
# ============================================================
def compute_adult_hmm_prior(trained):
    """
    Ground truth comes from fold 1's train+eval+test lists combined — a
    complete, non-overlapping partition of all 78 subjects. No per-fold
    finetuned model exists anymore, so there's nothing to average across.
    """
    all_files = []
    for list_path in ADULT_ALL_SUBJECTS_LISTS:
        all_files.extend(load_list(list_path))

    ytrue_list = load_labels(all_files)
    A_init, pi_init = estimate_hmm_parameters_from_gt(
        ytrue_list, label="adult (all 78 subjects)", verbose=True
    )

    if not trained:
        return A_init, pi_init, None

    # trained: need the SHHS-pretrained checkpoint's own softmax scores on
    # these same 78 subjects — a single inference run, not yet built.
    mat = hdf5storage.loadmat(ADULT_PRETRAINED_SCORE_PATH)
    score = np.transpose(mat['score'], (1, 0, 2))   # (seq_len, N, 5)

    obs_probs_list = []
    sum_size = 0
    for y in ytrue_list:
        valid_len = len(y) - (seq_len - 1)
        night_score = score[:, sum_size:sum_size + valid_len, :]
        obs_probs_list.append(aggregate_probs(night_score))
        sum_size += valid_len

    A, pi, alpha = train_hmm_mmi(obs_probs_list, ytrue_list, A_init, pi_init)

    print(f"\n{'='*70}\nADULT TRAINED HMM PRIOR (all 78 subjects, SHHS-pretrained scores)\n{'='*70}")
    print(pd.DataFrame(np.round(A, 4),
                        index=[STAGE_NAMES[s] for s in STAGES],
                        columns=[STAGE_NAMES[s] for s in STAGES]))
    print(f"\nTrained alpha: {alpha:.4f}")
    print("=" * 70 + "\n")

    return A, pi, alpha

# ============================================================
# NCH HMM PRIOR — one per age bin, from NCH's own training split
# ============================================================
def compute_nch_hmm_prior(age_bin, trained):
    train_files = load_list(os.path.join(NCH_LIST_DIR, f"train_list_{age_bin}.txt"))
    ytrue_list = load_labels(train_files)

    A_init, pi_init = estimate_hmm_parameters_from_gt(
        ytrue_list, label=f"NCH {age_bin} (train split)", verbose=True
    )

    if not trained:
        return A_init, pi_init, None

    # trained: single SHHS-pretrained-checkpoint inference on this age bin's
    # NCH training subjects (needs a separate inference job, not yet built)
    pred_path = os.path.join(NCH_TRAIN_INFERENCE_BASE, age_bin, "test_ret.mat")
    mat = hdf5storage.loadmat(pred_path)
    score = np.transpose(mat['score'], (1, 0, 2))

    obs_probs_list = []
    sum_size = 0
    for y in ytrue_list:
        valid_len = len(y) - (seq_len - 1)
        night_score = score[:, sum_size:sum_size + valid_len, :]
        obs_probs_list.append(aggregate_probs(night_score))
        sum_size += valid_len

    A, pi, alpha = train_hmm_mmi(obs_probs_list, ytrue_list, A_init, pi_init)

    print(f"\n{'='*70}\nNCH TRAINED HMM PRIOR — {age_bin}\n{'='*70}")
    print(pd.DataFrame(np.round(A, 4),
                        index=[STAGE_NAMES[s] for s in STAGES],
                        columns=[STAGE_NAMES[s] for s in STAGES]))
    print(f"\nTrained alpha: {alpha:.4f}")
    print("=" * 70 + "\n")

    return A, pi, alpha

# ============================================================
# COMPUTE THE ADULT PRIOR ONCE (age-bin-independent)
# ============================================================
adult_prior = None
if REFINEMENT_SOURCE == "adult":
    adult_prior = compute_adult_hmm_prior(trained=(REFINEMENT_MODE == "hmm_trained"))

# ============================================================
# MAIN LOOP — one age bin at a time
# ============================================================
for age_bin in AGE_BINS:
    print(f"\n{'='*60}\n  Age bin: {age_bin} | Source: {REFINEMENT_SOURCE} | Mode: {REFINEMENT_MODE}\n{'='*60}\n")

    age_bin_out_dir = os.path.join(PRED_DIR, age_bin, OUT_SUBDIR) if OUT_SUBDIR else os.path.join(PRED_DIR, age_bin)
    os.makedirs(age_bin_out_dir, exist_ok=True)

    if REFINEMENT_SOURCE == "none":
        A, pi, alpha = None, None, None
    elif REFINEMENT_SOURCE == "adult":
        A, pi, alpha = adult_prior
    elif REFINEMENT_SOURCE == "nch":
        A, pi, alpha = compute_nch_hmm_prior(age_bin, trained=(REFINEMENT_MODE == "hmm_trained"))

    if A is not None:
        log_A = np.log(A + 1e-300)
        log_pi = np.log(pi + 1e-300)

    # --------------------------------------------------------
    # Load the single SHHS-pretrained-checkpoint score matrix for this age bin
    # --------------------------------------------------------
    pred_path = os.path.join(NCH_TEST_INFERENCE_BASE, age_bin, "test_ret.mat")
    if not os.path.exists(pred_path):
        print(f"  SKIPPING {age_bin}: no predictions found at {pred_path}")
        continue

    mat = hdf5storage.loadmat(pred_path)
    score = np.transpose(mat['score'], (1, 0, 2))   # (seq_len, N, 5)

    test_files = []
    test_list_path = os.path.join(NCH_LIST_DIR, f"test_list_{age_bin}.txt")
    with open(test_list_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                test_files.append((parts[0], int(parts[1])))

    # --------------------------------------------------------
    # Per-night prediction
    # --------------------------------------------------------
    sum_size = 0

    for fpath, n_epochs in test_files:
        data = hdf5storage.loadmat(file_name=fpath)
        label = np.array(data['label']).squeeze()
        y_true = label

        valid_len = n_epochs - (seq_len - 1)
        score_night = score[:, sum_size:sum_size + valid_len, :]
        obs_probs = aggregate_probs(score_night)
        y_pred_raw = np.argmax(obs_probs, axis=-1) + 1

        mat_basename = os.path.basename(fpath)
        parts = mat_basename.replace("_eeg.mat", "").split("_")
        patient_id = parts[0]
        night_num = parts[1]

        if REFINEMENT_SOURCE == "none":
            y_pred_final = y_pred_raw

        elif REFINEMENT_MODE == "hmm":
            max_prob = obs_probs.max(axis=1)
            ent = entropy(obs_probs, axis=1, base=2)
            print(
                f"  {patient_id}_night{night_num} — mean max-prob: {max_prob.mean():.3f}, "
                f"mean entropy: {ent.mean():.3f} bits (max={np.log2(nstage):.2f})"
            )

            alpha_preds = {}
            for a in ALPHA_VALUES:
                path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=a)
                alpha_preds[a] = path + 1

            y_pred_final = alpha_preds[ALPHA_VALUES[-1]]
            num_changed = np.sum(y_pred_raw != y_pred_final)
            print(f"    HMM changed epochs: {num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")

        elif REFINEMENT_MODE == "hmm_trained":
            path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=alpha)
            y_pred_final = path + 1
            num_changed = np.sum(y_pred_raw != y_pred_final)
            print(f"  {patient_id}_night{night_num} — HMM-trained changed epochs: "
                  f"{num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")

        if len(y_pred_final) != len(y_true):
            print(f"  SKIPPING {mat_basename}: y_pred length {len(y_pred_final)} != y_true length {len(y_true)}")
            sum_size += valid_len
            continue

        print(f"  {patient_id}_night{night_num}: pred shape={y_pred_final.shape}, true shape={y_true.shape}, "
              f"unique pred={np.unique(y_pred_final)}, unique true={np.unique(y_true)}")

        np.save(os.path.join(age_bin_out_dir, f"{patient_id}_night{night_num}_ytrue.npy"), y_true)

        if REFINEMENT_SOURCE == "none":
            np.save(os.path.join(age_bin_out_dir, f"{patient_id}_night{night_num}_ypred.npy"), y_pred_final)
            np.save(os.path.join(age_bin_out_dir, f"{patient_id}_night{night_num}_probs.npy"), obs_probs)
        elif REFINEMENT_MODE == "hmm":
            for a, y_pred_a in alpha_preds.items():
                np.save(os.path.join(age_bin_out_dir, f"{patient_id}_night{night_num}_ypred_alpha{a:.1f}.npy"), y_pred_a)
        elif REFINEMENT_MODE == "hmm_trained":
            np.save(os.path.join(age_bin_out_dir, f"{patient_id}_night{night_num}_ypred_trained.npy"), y_pred_final)

        sum_size += valid_len

    print(f"  Done {age_bin}\n")

print("All age bins processed.")
print(f"Results saved to: {PRED_DIR}")
