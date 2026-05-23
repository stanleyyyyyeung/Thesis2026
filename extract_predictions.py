import numpy as np
import hdf5storage
import os
import glob
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy

# ============================================================
# CONFIGURATION — EDIT THESE BEFORE EACH RUN
# ============================================================
RUN_NUMBER = 1

# Mode: "none", "hmm", or "hsmm"
REFINEMENT_MODE = "hmm"

# ============================================================
# PATHS
# ============================================================
BASE         = "/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20"
PRED_BASE    = f"/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-20/run{RUN_NUMBER}/out"
PRED_DIR     = f"/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-20/run{RUN_NUMBER}/predictions"
TRAIN_SCORE_BASE = f"/srv/scratch/z5423210/StanleyThesis2026/prev_runs/sleepedf-20/run{RUN_NUMBER}/training_scores"

# Output directory depends on mode
if REFINEMENT_MODE == "none":
    OUT_DIR = PRED_DIR                          # write to base predictions dir
elif REFINEMENT_MODE == "hmm":
    OUT_DIR = os.path.join(PRED_DIR, "hmmRefined")
elif REFINEMENT_MODE == "hsmm":
    OUT_DIR = os.path.join(PRED_DIR, "hsmmRefined")
else:
    raise ValueError(f"Unknown REFINEMENT_MODE: '{REFINEMENT_MODE}'. Choose 'none', 'hmm', or 'hsmm'.")

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
seq_len  = 200
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

def forward_pass(y_pred, B, A, pi):
    """
    Forward algorithm — computes alpha[t, k] = P(y_1...y_t, X_t=k | theta)

    y_pred : (T,)   — discrete observations (model argmax predictions, values in 0..K-1)
    B      : (K, K) — emission matrix, B[i,j] = P(observe j | state i)
    A      : (K, K) — transition matrix, A[i,j] = P(state j | state i)
    pi     : (K,)   — initial state distribution

    Returns:
        alpha     : (T, K) — forward probabilities (scaled)
        log_scale : (T,)   — log scaling factors for numerical stability (preventing probabilities to drop to 0 quickly in some cases)
    """
    T = len(y_pred)
    K = len(pi)
    alpha = np.zeros((T, K))
    log_scale = np.zeros(T)

    # Initialisation: alpha[0, i] = pi[i] * B[i, y_0]
    alpha[0] = pi * B[:, y_pred[0]]
    scale = alpha[0].sum()
    alpha[0] /= scale
    log_scale[0] = np.log(scale + 1e-300)

    # Recursion: alpha[t, j] = B[j, y_t] * sum_i( alpha[t-1, i] * A[i, j] )
    for t in range(1, T):
        alpha[t] = B[:, y_pred[t]] * (alpha[t-1] @ A)
        scale = alpha[t].sum()
        alpha[t] /= scale
        log_scale[t] = np.log(scale + 1e-300)

    return alpha, log_scale

def backward_pass(y_pred, B, A, log_scale):
    """
    Backward algorithm — computes beta[t, i] = P(y_{t+1}...y_T | X_t=i, theta)

    y_pred    : (T,)   — discrete observations (model argmax predictions)
    B         : (K, K) — emission matrix, B[i,j] = P(observe j | state i)
    A         : (K, K) — transition matrix
    log_scale : (T,)   — scaling factors from forward pass, reused for consistency

    Returns:
        beta : (T, K) — backward probabilities (scaled)
    """
    T = len(y_pred)
    K = B.shape[0]
    beta = np.zeros((T, K))

    # Initialisation: beta[T-1, i] = 1 for all i
    beta[T-1] = 1.0
    beta[T-1] /= np.exp(log_scale[T-1])

    # Recursion: beta[t, i] = sum_j( A[i,j] * B[j, y_{t+1}] * beta[t+1, j] )
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, y_pred[t+1]] * beta[t+1])
        beta[t] /= np.exp(log_scale[t])

    return beta

def compute_gamma_xi(y_pred, B, A, alpha, beta):
    """
    E-step: compute gamma and xi from forward-backward results. (Mainly used for updating state transition matrix A)

    gamma[t, i]    = P(X_t = i | Y, theta)
                   — posterior probability of being in state i at time t

    xi[t, i, j]   = P(X_t = i, X_{t+1} = j | Y, theta)
                   — posterior probability of transitioning i -> j at time t

    y_pred : (T,)   — discrete observations
    B      : (K, K) — emission matrix
    A      : (K, K) — transition matrix
    alpha  : (T, K) — from forward_pass
    beta   : (T, K) — from backward_pass

    Returns:
        gamma : (T, K)
        xi    : (T-1, K, K)
    """
    T = len(y_pred)
    K = B.shape[0]

    # gamma[t, i] = alpha[t,i] * beta[t,i] / sum_i( alpha[t,i] * beta[t,i] )
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    # xi[t, i, j] = alpha[t,i] * A[i,j] * B[j, y_{t+1}] * beta[t+1, j]
    #               normalised over all i, j
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        xi[t] = (alpha[t][:, None]
                 * A
                 * B[:, y_pred[t+1]][None, :]
                 * beta[t+1][None, :])
        xi[t] /= xi[t].sum() + 1e-300

    return gamma, xi

def baum_welch(obs_probs_list, y_true_list, n_iter=20, tol=1e-4):
    """
    Baum-Welch EM algorithm — learns A, B, and pi from model softmax sequences.

    obs_probs_list : list of (T_i, K) softmax arrays — one per training night
    y_true_list    : list of (T_i,)   GT label arrays — one per training night
                     used to initialise A, B, pi from empirical counts
    n_iter         : maximum EM iterations
    tol            : convergence threshold on log-likelihood

    Returns:
        A  : (K, K) learned transition matrix
        B  : (K, K) learned emission matrix
        pi : (K,)   learned initial state distribution
    """
    K = nstage

    # Precompute discrete observations from softmax
    y_pred_list = [obs.argmax(axis=1) for obs in obs_probs_list]

    # --- Initialise from GT empirical counts ---

    # pi — frequency of first stage across all training sequences
    pi_counts = np.zeros(K)
    for y_true in y_true_list:
        pi_counts[y_true[0]] += 1
    pi = pi_counts / pi_counts.sum()

    # A — empirical transition counts from GT label sequences
    A_counts = np.zeros((K, K))
    for y_true in y_true_list:
        for t in range(len(y_true) - 1):
            A_counts[y_true[t], y_true[t+1]] += 1
    A = A_counts / A_counts.sum(axis=1, keepdims=True)

    # B — empirical confusion matrix: GT state i, model predicted j
    B_counts = np.zeros((K, K))
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        for t in range(len(y_true)):
            B_counts[y_true[t], y_pred[t]] += 1
    B = B_counts / B_counts.sum(axis=1, keepdims=True)

    # Print the initialised parameters (before training)
    print("\n" + "="*70)
    print("INITIAL HMM PARAMETERS (before Baum-Welch)")
    print("="*70)

    print("\nTransition Matrix A")
    print("A[i,j] = P(next state j | current state i)\n")
    print(pd.DataFrame(
        np.round(A, 4),
        index=[STAGE_NAMES[s] for s in STAGES],
        columns=[STAGE_NAMES[s] for s in STAGES]
    ))

    print("\nEmission Matrix B")
    print("B[i,j] = P(model predicts j | true state i)\n")
    print(pd.DataFrame(
        np.round(B, 4),
        index=[STAGE_NAMES[s] for s in STAGES],
        columns=[STAGE_NAMES[s] for s in STAGES]
    ))

    print("\nDiagonal of B (correct prediction probability per state):")
    for i, val in enumerate(np.diag(B)):
        print(f"  {STAGE_NAMES[i+1]} : {val:.4f}")

    print("\nInitial State Distribution pi")
    for i in range(K):
        print(f"  {STAGE_NAMES[i+1]} : {pi[i]:.4f}")

    print("="*70 + "\n")
    
    prev_log_likelihood = -np.inf

    for iteration in range(n_iter):
        # Accumulators for M-step
        pi_sum   = np.zeros(K)
        xi_sum   = np.zeros((K, K))       # numerator for A update
        gamma_sum = np.zeros((K, K))      # B[i,j] numerator: gamma at timesteps where y_t = j
        gamma_total = np.zeros(K)         # B denominator: total gamma per state
        log_likelihood = 0.0

        # E-step
        for y_pred, obs_probs in zip(y_pred_list, obs_probs_list):
            alpha, log_scale = forward_pass(y_pred, B, A, pi)
            beta             = backward_pass(y_pred, B, A, log_scale)
            gamma, xi        = compute_gamma_xi(y_pred, B, A, alpha, beta)

            pi_sum      += gamma[0]
            xi_sum      += xi.sum(axis=0)
            gamma_total += gamma.sum(axis=0)

            # B numerator: for each symbol j, sum gamma[t, i] where y_t == j
            for j in range(K):
                mask = (y_pred == j)               # timesteps where model predicted j
                gamma_sum[:, j] += gamma[mask].sum(axis=0)

            log_likelihood += log_scale.sum()

        # M-step
        pi = pi_sum / pi_sum.sum()
        A  = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        B  = gamma_sum / gamma_total[:, None]

        print(f"  Iter {iteration+1:2d} | log-likelihood: {log_likelihood:.4f}")

        if abs(log_likelihood - prev_log_likelihood) < tol:
            print(f"  Converged at iteration {iteration+1}")
            break
        prev_log_likelihood = log_likelihood

    # Print learned parameters
    print("\n" + "="*70)
    print("LEARNED HMM PARAMETERS (Baum-Welch)")
    print("="*70)

    print("\nTransition Matrix A")
    print("A[i,j] = P(next state j | current state i)\n")
    A_df = pd.DataFrame(
        np.round(A, 4),
        index=[STAGE_NAMES[s] for s in STAGES],
        columns=[STAGE_NAMES[s] for s in STAGES]
    )
    print(A_df)

    print("\nEmission Matrix B")
    print("B[i,j] = P(model predicts j | true state i)\n")
    B_df = pd.DataFrame(
        np.round(B, 4),
        index=[STAGE_NAMES[s] for s in STAGES],
        columns=[STAGE_NAMES[s] for s in STAGES]
    )
    print(B_df)

    print("\nDiagonal of B (correct prediction probability per state):")
    for i, val in enumerate(np.diag(B)):
        print(f"  {STAGE_NAMES[i+1]} : {val:.4f}")

    print("\nInitial State Distribution pi")
    for i in range(K):
        print(f"  {STAGE_NAMES[i+1]} : {pi[i]:.4f}")

    print("="*70 + "\n")

    return A, B, pi

def estimate_hsmm_durations(train_ytrue_files, max_duration=200):
    """
    Estimate per-stage duration distributions from run lengths in training data.
    Returns:
        duration_probs: dict {stage_0idx: np.array of shape (max_duration,)}
                        where duration_probs[s][d] = P(duration = d+1 | stage = s)
    """
    # Collect run lengths per stage
    stage_durations = defaultdict(list)

    for fpath in train_ytrue_files:
        y = np.load(fpath).astype(int) - 1      # 0-indexed
        runs = get_run_lengths(y)
        for (stage, length) in runs:
            stage_durations[stage].append(length)

    duration_probs = {}
    for s in range(nstage):
        counts = np.ones(max_duration)           # Laplace smoothing
        for length in stage_durations[s]:
            idx = min(length, max_duration) - 1  # clip to max_duration
            counts[idx] += 1
        duration_probs[s] = counts / counts.sum()

    return duration_probs

# ============================================================
# VITERBI DECODING
# ============================================================
def viterbi_hmm(obs_probs, log_A, log_B, log_pi):
    T = len(obs_probs)
    K = nstage

    # Weights to transition and emission matrices (multiplied to matrices for log scale)
    alpha = 1
    beta = 1

    # Sleep stages predicted by the model
    y_obs = np.argmax(obs_probs, axis=1)

    viterbi = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)

    # Initialisation: pi[s] * B[s, obs_0] (becomes addition in log scale)
    viterbi[0] = log_pi + beta*log_B[:, y_obs[0]]

    for t in range(1, T):
        for k in range(K):
            scores = viterbi[t-1] + alpha*log_A[:, k]
            best = np.argmax(scores)
            viterbi[t, k] = scores[best] + beta* log_B[k, y_obs[t]]
            backptr[t, k] = best

    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(viterbi[T-1])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]

    return path

def viterbi_hsmm(obs_probs, log_A, log_pi, duration_probs, max_duration=200):
    """
    HSMM Viterbi with explicit duration modelling.
    obs_probs      : (T, 5) — softmax probability of each stage at each epoch
    log_A          : (5, 5) — log transition matrix (no self-transitions used)
    log_pi         : (5,)   — log initial distribution
    duration_probs : dict {stage: array of shape (max_duration,)}
    Returns predicted label sequence (0-indexed).
    """
    T = len(obs_probs)
    K = nstage

    log_obs = np.log(obs_probs + 1e-10)         # (T, 5)
    log_dur = {s: np.log(duration_probs[s] + 1e-10) for s in range(K)}

    # D[t, k] = best log-prob of sequence ending at t with state k finishing here
    D       = np.full((T, K), -np.inf)
    # backptr stores (prev_end_time, prev_state) for backtracking
    bp_time = np.full((T, K), -1, dtype=int)
    bp_state= np.full((T, K), -1, dtype=int)

    # Initialise: segments starting at t=0
    for k in range(K):
        for d in range(1, min(max_duration, T) + 1):
            end = d - 1                          # segment covers [0, end]
            if end >= T:
                break
            dur_score = log_dur[k][d-1]
            obs_score = log_obs[0:end+1, k].sum()
            score     = log_pi[k] + dur_score + obs_score
            if score > D[end, k]:
                D[end, k]        = score
                bp_time[end, k]  = -1            # segment starts at 0
                bp_state[end, k] = -1

    # Fill DP
    for t in range(1, T):
        for k in range(K):
            for d in range(1, min(max_duration, t+1) + 1):
                start = t - d + 1
                if start <= 0:
                    break
                prev_end = start - 1
                dur_score = log_dur[k][d-1]
                obs_score = log_obs[start:t+1, k].sum()

                for j in range(K):
                    if j == k:
                        continue               # HSMM: no self-transitions
                    if D[prev_end, j] == -np.inf:
                        continue
                    score = D[prev_end, j] + log_A[j, k] + dur_score + obs_score
                    if score > D[t, k]:
                        D[t, k]        = score
                        bp_time[t, k]  = prev_end
                        bp_state[t, k] = j

    # Backtrack from T-1
    path = np.zeros(T, dtype=int)
    cur_state = np.argmax(D[T-1])
    cur_time  = T - 1

    while cur_time >= 0:
        prev_time  = bp_time[cur_time, cur_state]
        prev_state = bp_state[cur_time, cur_state]

        if prev_time == -1:
            start = 0
        else:
            start = prev_time + 1

        path[start:cur_time+1] = cur_state

        if prev_time == -1:
            break
        cur_time  = prev_time
        cur_state = prev_state

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
    score      = mat['score']                        # (N, 200, 5)
    score      = np.transpose(score, (1, 0, 2))      # → (200, N, 5)

    patient_id = int(patient[1:])
    eeg_pattern= os.path.join(BASE, f"mat_30min/n{patient_id:02d}_*_eeg.mat")
    test_files = sorted(glob.glob(eeg_pattern))

    # HMM / HSMM: estimate parameters from all OTHER subjects
    if REFINEMENT_MODE == "hmm":
        obs_probs_list = []
        train_ytrue_list = []

        # UPDATE: Included new inference script to extract softmax probabilities for the training patients as well
        train_mat_path  = os.path.join(TRAIN_SCORE_BASE, patient, "test_ret.mat")
        train_mat       = hdf5storage.loadmat(train_mat_path)
        train_score     = np.transpose(train_mat['score'], (1, 0, 2))  # (200, N, 5)

        # Load ground truth labels for all training subjects of this fold
        train_list_path = os.path.join(BASE, f"file_list_20sub/eeg/train_list_n{patient[1:]}.txt")
        with open(train_list_path, "r") as f:
            train_files = [os.path.join(BASE, "network/lseqsleepnet", line.strip().split('\t')[0]) for line in f if line.strip()]

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

        print(f"  Running Baum-Welch on {len(obs_probs_list)} training nights...")
        A_learned, B_learned, pi_learned = baum_welch(obs_probs_list, train_ytrue_list)
        log_A  = np.log(A_learned + 1e-300)
        log_B  = np.log(B_learned + 1e-300)
        log_pi = np.log(pi_learned + 1e-300)
    # ----------------------------------------------------------
    # HSMM: estimate duration distributions from training subjects
    # ----------------------------------------------------------
    elif REFINEMENT_MODE == "hsmm":
        train_ytrue_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_ytrue.npy")))
        train_ytrue_files = [
            f for f in train_ytrue_files
            if not os.path.basename(f).startswith(patient)
        ]
        log_A, log_B, log_pi = estimate_hmm_parameters(train_ytrue_files)
        duration_probs = estimate_hsmm_durations(train_ytrue_files)
        print(f"  Duration distributions estimated.")
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
        # NO REFINEMENT
        # ----------------------------------------------------------

        if REFINEMENT_MODE == "none":
            y_pred_final = y_pred_raw

        # ----------------------------------------------------------
        # HMM REFINEMENT
        # ----------------------------------------------------------

        elif REFINEMENT_MODE == "hmm":
            obs_probs = aggregate_probs(score_night)

            path = viterbi_hmm(
                obs_probs,
                log_A,
                log_B,
                log_pi
            )

            y_pred_final = path + 1

            # ------------------------------------------------------
            # Compare raw vs HMM
            # ------------------------------------------------------

            num_changed = np.sum(y_pred_raw != y_pred_final)

            percentage_changed = (
                100 * num_changed / len(y_pred_raw)
            )

            print(
                f"  HMM changed epochs: "
                f"{num_changed}/{len(y_pred_raw)} "
                f"({percentage_changed:.2f}%)"
            )

        # ----------------------------------------------------------
        # HSMM REFINEMENT
        # ----------------------------------------------------------

        elif REFINEMENT_MODE == "hsmm":
            obs_probs = aggregate_probs(score_night)

            path = viterbi_hsmm(
                obs_probs,
                log_A,
                log_pi,
                duration_probs
            )

            y_pred_final = path + 1

            # ------------------------------------------------------
            # Compare raw vs HSMM
            # ------------------------------------------------------

            num_changed = np.sum(y_pred_raw != y_pred_final)

            percentage_changed = (
                100 * num_changed / len(y_pred_raw)
            )

            print(
                f"  HSMM changed epochs: "
                f"{num_changed}/{len(y_pred_raw)} "
                f"({percentage_changed:.2f}%)"
            )

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

        np.save(
            os.path.join(
                OUT_DIR,
                f"{patient}_night{i+1}_ypred.npy"
            ),
            y_pred_final
        )

        np.save(
            os.path.join(
                OUT_DIR,
                f"{patient}_night{i+1}_ytrue.npy"
            ),
            y_true
        )

        sum_size += valid_len

    print(f"  Done {patient}\n")

print("All patients processed.")
print(f"Results saved to: {OUT_DIR}")
