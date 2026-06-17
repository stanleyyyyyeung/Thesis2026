import numpy as np
import hdf5storage
import os
import glob
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy
import torch
from kNearestViterbi import calc_viterbi_k_best, calc_MMI_loss, train_hmm_mmi

# ============================================================
# CONFIGURATION — EDIT THESE BEFORE EACH RUN
# ============================================================
RUN_NUMBER = 1

# Mode: "none", "hmm", or "hmm_trained"
REFINEMENT_MODE = "hmm_trained"

# ============================================================
# PATHS
# ============================================================
PRED_BASE = f"/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-20/run{RUN_NUMBER}/out"
PRED_DIR = f"/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-20/run{RUN_NUMBER}/predictions"
TRAIN_SCORE_BASE = f"/srv/scratch/z5423210/StanleyThesis2026/out_sleeptransformer/sleepedf-20/run{RUN_NUMBER}/training_scores"

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]

# Output directory depends on mode
if REFINEMENT_MODE == "none":
    OUT_DIR = PRED_DIR                          # write to base predictions dir
elif REFINEMENT_MODE == "hmm" or REFINEMENT_MODE == "hmm_trained":
    OUT_DIR = os.path.join(PRED_DIR, "hmmRefined")
else:
    raise ValueError(f"Unknown REFINEMENT_MODE: '{REFINEMENT_MODE}'. Choose 'none', 'hmm', or 'hmm_trained'.")

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
seq_len  = 21
nstage   = 5
STAGES   = [1, 2, 3, 4, 5]   # W=1, N1=2, N2=3, N3=4, REM=5
STAGE_NAMES = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}

# ============================================================
# AGGREGATION (unchanged from original)
# ============================================================
def softmax(z):
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def aggregate_mul(score):
    """
    Multiply aggregation across all 200 context positions.
    score: (200, N, 5)
    returns: y_pred array of shape (N,) with labels in 1..5
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
    return np.argmax(fused_score, axis=-1) + 1   # labels 1..5

def aggregate_probs(score):
    """
    Same as aggregate_mul but returns the full probability vector (N, 5)
    instead of argmax — needed for HMM/HSMM emission scores.
    score: (200, N, 5)
    returns: prob array of shape (N, 5)
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
    # convert log-sum back to normalised probabilities
    fused_score -= np.max(fused_score, axis=1, keepdims=True)
    probs = np.exp(fused_score)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs   # (N, 5)

# ============================================================
# HMM / HSMM PARAMETER ESTIMATION FROM TRAINING SUBJECTS
# ============================================================
def get_run_lengths(sequence):
    """Return list of (stage, run_length) tuples for a label sequence."""
    runs = []
    if len(sequence) == 0:
        return runs
    current = sequence[0]
    count = 1
    for s in sequence[1:]:
        if s == current:
            count += 1
        else:
            runs.append((current, count))
            current = s
            count = 1
    runs.append((current, count))
    return runs

def estimate_hmm_parameters_from_gt(y_true_list):
    """
    Estimate A and pi purely from ground-truth label sequences.
 
    y_true_list : list of (T_i,) int arrays, values in 0..K-1
 
    Returns:
        A  : (K, K)  transition matrix,  A[i,j] = P(next=j | cur=i)
        pi : (K,)    initial distribution
    """
    K = nstage
 
    # pi — empirical first-stage frequency
    pi_counts = np.zeros(K)
    for y in y_true_list:
        pi_counts[y[0]] += 1
    pi = (pi_counts + 1e-6) / (pi_counts + 1e-6).sum()   # small Laplace smooth
 
    # A — empirical bigram transition counts
    A_counts = np.zeros((K, K))
    for y in y_true_list:
        for t in range(len(y) - 1):
            A_counts[y[t], y[t + 1]] += 1
    # Laplace smoothing to avoid zero transitions
    A_counts += 1e-6
    A = A_counts / A_counts.sum(axis=1, keepdims=True)
 
    # ---- Diagnostics ----
    print("\n" + "=" * 70)
    print("HMM PARAMETERS (estimated from ground-truth sequences only)")
    print("=" * 70)
 
    print("\nTransition Matrix A  —  A[i,j] = P(next state j | current state i)\n")
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
    """
    Viterbi decoding using the model's softmax output as emission probabilities.
 
    At each timestep t the emission log-probability for hidden state k is:
        log P(obs_t | state = k) = log softmax[t, k]
 
    This replaces the old B-matrix lookup  (log_B[k, argmax_t])  with the
    full continuous softmax vector, so observed and hidden spaces are no
    longer the same discrete label set.
 
    obs_probs : (T, K)  — softmax probability vectors from aggregate_probs()
    log_A     : (K, K)  — log transition matrix
    log_pi    : (K,)    — log initial distribution
    alpha     : constant - weight to put on ground truth transitions for decoding
 
    Returns:
        path : (T,) int array of decoded states, 0-indexed
    """
    T, K = obs_probs.shape
    log_emit = np.log(obs_probs + 1e-10)   # (T, K) — log softmax, one value per (t, state)
 
    viterbi = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)
 
    # Initialisation
    viterbi[0] = log_pi + log_emit[0]     # pi[k] * softmax[0, k]
 
    # Recursion
    for t in range(1, T):
        # scores[i, k] = viterbi[t-1, i] + log_A[i, k]
        scores = viterbi[t - 1][:, None] + alpha*log_A   # (K, K)
        best_prev = np.argmax(scores, axis=0)       # (K,)
        viterbi[t] = scores[best_prev, np.arange(K)] + log_emit[t]
        backptr[t] = best_prev
 
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(viterbi[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
 
    return path   # 0-indexed

# ============================================================
# MAIN LOOP
# ============================================================
patients = sorted([
    d for d in os.listdir(PRED_BASE)
    if os.path.exists(os.path.join(PRED_BASE, d, "test_ret.mat"))
])

print(f"\n{'='*60}")
print(f"  Run {RUN_NUMBER} | Mode: {REFINEMENT_MODE.upper()}")
print(f"  Output → {OUT_DIR}")
print(f"{'='*60}\n")

for patient in patients:
    print(f"Processing {patient} ...")

    # ----------------------------------------------------------
    # Load score matrix and aggregate to get probs + argmax pred
    # ----------------------------------------------------------
    pred_path  = os.path.join(PRED_BASE, patient, "test_ret.mat")
    mat        = hdf5storage.loadmat(pred_path)
    score      = mat['score']                        # (N, 21, 5)
    score      = np.transpose(score, (1, 0, 2))      # → (21, N, 5)

    test_list_path = f"/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer/test_list_n{patient[1:]}.txt"
    with open(test_list_path, "r") as f:
        test_files = [line.strip().split('\t')[0] for line in f if line.strip()]
    test_files = sorted(test_files)

    # HMM / HSMM: estimate parameters from all OTHER subjects
    if REFINEMENT_MODE in ("hmm", "hmm_trained"):
        obs_probs_list = []
        train_ytrue_list = []

        # UPDATE: Included new inference script to extract softmax probabilities for the training patients as well
        train_mat_path  = os.path.join(TRAIN_SCORE_BASE, patient, "test_ret.mat")
        train_mat       = hdf5storage.loadmat(train_mat_path)
        train_score     = np.transpose(train_mat['score'], (1, 0, 2))  # (21, N, 5)

        # Load ground truth labels for all training subjects of this fold
        train_list_path = f"/srv/scratch/z5423210/StanleyThesis2026/sleeptransformer/train_list_n{patient[1:]}.txt"
        with open(train_list_path, "r") as f:
            train_files = [line.strip().split('\t')[0] for line in f if line.strip()]

        train_sum = 0

        for fpath in train_files:
            data = hdf5storage.loadmat(file_name=fpath)
            label = np.array(data['label']).squeeze()
            n = len(label)
            valid_len = n - (seq_len - 1)
            night_score = train_score[:, train_sum:train_sum + valid_len, :]
            obs_probs_list.append(aggregate_probs(night_score))
            train_ytrue_list.append((label-1).astype(int))
            train_sum += valid_len

       # print(f"  Running Baum-Welch on {len(obs_probs_list)} training nights...")
        A_init, pi_init = estimate_hmm_parameters_from_gt(train_ytrue_list)
        log_A  = np.log(A_init + 1e-300)
        log_pi = np.log(pi_init + 1e-300)

        if REFINEMENT_MODE == "hmm_trained":
            A, pi, trained_alpha = train_hmm_mmi(
                obs_probs_list, train_ytrue_list,
                A_init, pi_init
            )
            log_A  = np.log(A + 1e-300)
            log_pi = np.log(pi + 1e-300)
            print(f"\n  Trained alpha: {trained_alpha:.4f}")
            print(f"\n  Trained Transition Matrix A:")
            print(pd.DataFrame(
                np.round(A, 4),
                index=[STAGE_NAMES[s] for s in STAGES],
                columns=[STAGE_NAMES[s] for s in STAGES]
            ))
    # ----------------------------------------------------------
    # Per-night prediction
    # ----------------------------------------------------------
    sum_size = 0

    for i, fpath in enumerate(test_files):
        data  = hdf5storage.loadmat(file_name=fpath)
        label = np.array(data['label']).squeeze()

        y_true = label

        n = len(label)
        valid_len = n - (seq_len - 1)

        score_night = score[:, sum_size:sum_size + valid_len, :]

        # ----------------------------------------------------------
        # ALWAYS compute raw prediction first
        # ----------------------------------------------------------

        y_pred_raw = aggregate_mul(score_night)

        # ----------------------------------------------------------
        # Extract patient_id / night_num up front
        # ----------------------------------------------------------
        mat_basename = os.path.basename(fpath)                          # "n15_1_eeg.mat"
        parts        = mat_basename.replace("_eeg.mat", "").split("_")  # ["n15", "1"]
        patient_id   = parts[0]                                          # "n15"
        night_num    = parts[1]                                          # "1"

        # ----------------------------------------------------------
        # NO REFINEMENT
        # ----------------------------------------------------------

        if REFINEMENT_MODE == "none":
            y_pred_final = y_pred_raw

        # ----------------------------------------------------------
        # HMM REFINEMENT
        # ----------------------------------------------------------

        elif REFINEMENT_MODE == "hmm":
            obs_probs = aggregate_probs(score_night)

            # Checking confidence in raw softmax output
            max_prob = obs_probs.max(axis=1)         # Max probility for each epoch (effectively predicted stage by the model)
            ent = entropy(obs_probs, axis=1, base=2) # Shannon entropy (high value indicates low confidence in model prediction)
            
            print(
                f"  Emission confidence — "
                f"mean max-prob: {max_prob.mean():.3f}, "
                f"median: {np.median(max_prob):.3f}, "
                f"% epochs >0.95 conf: {100*np.mean(max_prob > 0.95):.1f}%, "
                f"mean entropy: {ent.mean():.3f} bits (max={np.log2(nstage):.2f})"
            )

            alpha_preds = {}

            for alpha in ALPHA_VALUES:
                path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=alpha)
                alpha_preds[alpha] = path + 1

            # use last alpha's result for the summary print/comparison below
            y_pred_final = alpha_preds[ALPHA_VALUES[-1]]
            num_changed = np.sum(y_pred_raw != y_pred_final)
            print(f"  HMM changed epochs: {num_changed}/{len(y_pred_raw)} ({100*num_changed/len(y_pred_raw):.2f}%)")

        elif REFINEMENT_MODE == "hmm_trained":
            obs_probs = aggregate_probs(score_night)
            path = viterbi_hmm_softmax(obs_probs, log_A, log_pi, alpha=trained_alpha)
            y_pred_final = path + 1

            num_changed = np.sum(y_pred_raw != y_pred_final)
            print(
                f"  HMM-trained changed epochs: "
                f"{num_changed}/{len(y_pred_raw)} "
                f"({100*num_changed/len(y_pred_raw):.2f}%)"
            )
             
        # Check shape mismatch
        if len(y_pred_final) != len(y_true):
            print(
                f"  SKIPPING {mat_basename}: "
                f"y_pred length {len(y_pred_final)} != y_true length {len(y_true)}"
            )
            sum_size += valid_len
            continue

        # ----------------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------------

        print(
            f"  Night {i+1}: "
            f"pred shape={y_pred_final.shape}, "
            f"true shape={y_true.shape}"
        )

        print(
            f"  Unique pred: {np.unique(y_pred_final)}, "
            f"Unique true: {np.unique(y_true)}"
        )

        # ----------------------------------------------------------
        # SAVE
        # ----------------------------------------------------------

        # ytrue is alpha-independent, save once regardless of mode
        np.save(
            os.path.join(
                OUT_DIR,
                f"{patient_id}_night{night_num}_ytrue.npy"
            ),
            y_true
        )

        # Only save a single "ypred" file for non-HMM modes.
        # For "hmm", per-alpha files were already saved above.
        if REFINEMENT_MODE == "none":
            np.save(os.path.join(OUT_DIR, f"{patient_id}_night{night_num}_ypred.npy"), y_pred_final)
        elif REFINEMENT_MODE == "hmm":
            for alpha, y_pred_alpha in alpha_preds.items():
                np.save(
                    os.path.join(OUT_DIR, f"{patient_id}_night{night_num}_ypred_alpha{alpha:.1f}.npy"),
                    y_pred_alpha
                )
        elif REFINEMENT_MODE == "hmm_trained":
            np.save(os.path.join(OUT_DIR, f"{patient_id}_night{night_num}_ypred_trained.npy"), y_pred_final)
        
        sum_size += valid_len

    print(f"  Done {patient}\n")

print("All patients processed.")
print(f"Results saved to: {OUT_DIR}")
