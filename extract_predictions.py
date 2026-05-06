import numpy as np
import hdf5storage
import os
import glob
from collections import defaultdict

# ============================================================
# CONFIGURATION — EDIT THESE BEFORE EACH RUN
# ============================================================
RUN_NUMBER = 2

# Mode: "none", "hmm", or "hsmm"
REFINEMENT_MODE = "hsmm"

# ============================================================
# PATHS
# ============================================================
BASE         = "/srv/scratch/z5423210/StanleyThesis2026/lseqsleep/L-SeqSleepNet/sleepedf-20"
PRED_BASE    = f"/srv/scratch/z5423210/StanleyThesis2026/prev_runs/run{RUN_NUMBER}/out"
PRED_DIR     = f"/srv/scratch/z5423210/StanleyThesis2026/prev_runs/run{RUN_NUMBER}/predictions"

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

def estimate_hmm_parameters(train_ytrue_files):
    """
    Estimate HMM parameters from training subjects' ground truth files.
    Returns:
        log_A : (5, 5) log transition matrix (rows = from, cols = to), 0-indexed
        log_B : (5, 5) log emission matrix  (rows = hidden stage, cols = observed stage)
        log_pi: (5,)   log initial state distribution
    All in 0-indexed form internally (stage-1 for indexing).
    """
    # Accumulators
    trans_counts  = np.ones((nstage, nstage))   # Laplace smoothing
    emit_counts   = np.ones((nstage, nstage))
    init_counts   = np.ones(nstage)

    for fpath in train_ytrue_files:
        y = np.load(fpath).astype(int) - 1      # shift to 0-indexed

        # Initial state
        init_counts[y[0]] += 1

        # Transitions
        for t in range(len(y) - 1):
            trans_counts[y[t], y[t+1]] += 1

        # Emissions: use argmax of model prediction as observed label.
        # We approximate B from training data by treating ytrue as both
        # hidden and observed (diagonal dominance expected).
        # Actual emission scoring at inference uses the softmax probs directly.
        for t in range(len(y)):
            emit_counts[y[t], y[t]] += 1        # self-emission as prior

    # Normalise rows
    A  = trans_counts  / trans_counts.sum(axis=1, keepdims=True)
    B  = emit_counts   / emit_counts.sum(axis=1, keepdims=True)
    pi = init_counts   / init_counts.sum()

    return np.log(A), np.log(B), np.log(pi)

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
def viterbi_hmm(obs_probs, log_A, log_pi):
    """
    Standard HMM Viterbi.
    obs_probs : (T, 5) — softmax probability of each stage at each epoch
    log_A     : (5, 5) — log transition matrix
    log_pi    : (5,)   — log initial distribution
    Returns predicted label sequence (0-indexed).
    """
    T = len(obs_probs)
    K = nstage

    log_obs = np.log(obs_probs + 1e-10)         # (T, 5)

    viterbi  = np.full((T, K), -np.inf)
    backptr  = np.zeros((T, K), dtype=int)

    viterbi[0] = log_pi + log_obs[0]

    for t in range(1, T):
        for k in range(K):
            scores = viterbi[t-1] + log_A[:, k]
            best   = np.argmax(scores)
            viterbi[t, k]  = scores[best] + log_obs[t, k]
            backptr[t, k]  = best

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(viterbi[T-1])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]

    return path   # 0-indexed

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

    # ----------------------------------------------------------
    # HMM / HSMM: estimate parameters from all OTHER subjects
    # ----------------------------------------------------------
    if REFINEMENT_MODE in ("hmm", "hsmm"):
        train_ytrue_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_ytrue.npy")))
        # Exclude this patient's own nights
        train_ytrue_files = [
            f for f in train_ytrue_files
            if not os.path.basename(f).startswith(patient)
        ]
        print(f"  Estimating parameters from {len(train_ytrue_files)} training night files ...")

        log_A, log_B, log_pi = estimate_hmm_parameters(train_ytrue_files)

        if REFINEMENT_MODE == "hsmm":
            duration_probs = estimate_hsmm_durations(train_ytrue_files)
            print(f"  Duration distributions estimated.")

    # ----------------------------------------------------------
    # Per-night prediction
    # ----------------------------------------------------------
    sum_size = 0

    for i, fpath in enumerate(test_files):
        data  = hdf5storage.loadmat(file_name=fpath)
        label = np.array(data['label']).squeeze()

        n         = len(label)
        valid_len = n - (seq_len - 1)

        score_night = score[:, sum_size:sum_size + valid_len, :]

        if REFINEMENT_MODE == "none":
            y_pred = aggregate_mul(score_night)

        elif REFINEMENT_MODE == "hmm":
            obs_probs = aggregate_probs(score_night)          # (T, 5)
            path      = viterbi_hmm(obs_probs, log_A, log_pi) # 0-indexed
            y_pred    = path + 1                               # back to 1-indexed

        elif REFINEMENT_MODE == "hsmm":
            obs_probs = aggregate_probs(score_night)
            path      = viterbi_hsmm(obs_probs, log_A, log_pi, duration_probs)
            y_pred    = path + 1

        y_true = label

        print(f"  Night {i+1}: pred shape={y_pred.shape}, true shape={y_true.shape}")
        print(f"  Unique pred: {np.unique(y_pred)}, Unique true: {np.unique(y_true)}")

        np.save(os.path.join(OUT_DIR, f"{patient}_night{i+1}_ypred.npy"), y_pred)
        np.save(os.path.join(OUT_DIR, f"{patient}_night{i+1}_ytrue.npy"), y_true)

        sum_size += valid_len

    print(f"  Done {patient}\n")

print("All patients processed.")
print(f"Results saved to: {OUT_DIR}")
