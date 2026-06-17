# Code for training the HMM using k-nearest path and Viterbi decoding

import torch
import numpy as np

def calc_viterbi_k_best(A, P, Pi, alpha, k_best, K, T, device="cuda"):
    """
    :param A: Transition matrix of the HMM model
    :param P: P-matrix created by the base model
    :param Pi: Initial distribution of the HMM model; if not given, it is set to uniform
    :param alpha: Parameter alpha for weighing the transition matrix and the P-matrix
    :param k_best: Number of the best paths returned by the Viterbi algorithm
    :param K: cardinality of the state space - will be A.shape[0]
    :param T: length of the sequence - will be len(P)
    :param device: torch tensor device
    """

    # Initialize the tracking tables
    T1 = torch.full((k_best, K, T), fill_value=float('-inf'), dtype=torch.float64, device=device)
    T2 = torch.full((k_best, K, T, 2), fill_value=-1, dtype=torch.int, device=device)

    # only the first-best path can be given for initialization; It is initialized with the initial distribution
    T1[0, :, 0] = Pi + P[0]
    T2[0, :, 0] = 0

    # Iterate through the observations updating the tracking tables
    for i in range(1, T):
        temp = T1[:, :, i - 1] + alpha * A.T[:, None]
        flattened_vals, flattened_idx = torch.topk(temp.flatten(start_dim=1, end_dim=2), k_best)
        T1[:, :, i] = flattened_vals.T + P[i]
        unraveled = np.array(np.unravel_index(flattened_idx.cpu().numpy(), shape=(k_best, K))).T
        T2[:, :, i, :] = torch.tensor(unraveled, device=device)

    final_vals, final_idx = torch.topk((T1[:, :, -1]).flatten(), k_best)
    last_valid = torch.where(final_vals == float('-inf'))[0]
    if len(last_valid) > 0:  # check if there are less than k_best valid paths
        last_valid = last_valid[0]
        final_idx = final_idx[0:last_valid]
        x = torch.empty((last_valid, T), dtype=torch.int64, device=device)
    else:
        x = torch.empty((k_best, T), dtype=torch.int64, device=device)

    final_state = final_idx % K  # find the last state of the k-best paths
    k_prev = final_idx // K
    x[:, -1] = final_state

    # iterate through T to find the previous states
    for i in reversed(range(1, T)):
        x[:, i - 1] = T2[k_prev, x[:, i], i, 1]
        k_prev = T2[k_prev, x[:, i], i, 0]

    return x, T1, T2, None, None

def calc_MMI_loss(A, P, Pi, alpha, k_best, K, T, labels, device="cuda"):
    """
    :param A: Transition matrix of the HMM model
    :param P: P-matrix created by the base model
    :param Pi: Initial distribution of the HMM model; if not given, it is set to uniform
    :param alpha: Parameter alpha for weighing the transition matrix and the P-matrix
    :param k_best: Number of the best paths returned by the Viterbi algorithm
    :param K: cardinality of the state space - will be A.shape[0]
    :param T: length of the sequence - will be len(P)
    :param labels: correct labels
    :param device: torch tensor device
    """
    k_best += 1  # if the correct path is in one of the k-best paths, then we need to have one more path

    if not isinstance(A, torch.Tensor):
        A = torch.from_numpy(A)
        P = torch.from_numpy(P)
        if Pi is not None:
            Pi = torch.from_numpy(Pi)

    else:
        A_optimizable = torch.clone(A)
        if torch.is_tensor(alpha):
            alpha_optimizable = torch.clone(alpha)
            alpha.detach()
        else:
            alpha_optimizable = alpha
        A.detach()

    best_paths, _, _, _, _ = calc_viterbi_k_best(A, P, Pi, alpha, k_best, K, T, device=device)  # only the k_best paths are of interest

    exclude_path = len(best_paths)  # exclude the path that equals the correct one, or exclude the last path
    for i in range(len(best_paths) - 1):
        if torch.equal(labels, best_paths[i]):
            exclude_path = i
            break

    try:
        num = Pi[labels[0]] + torch.sum(P[torch.arange(len(labels)), labels])
    except:
        raise RuntimeError
    transitions = torch.zeros((K, K), device=device)

    for i in range(len(labels) - 1):
        transitions[labels[i], labels[i + 1]] += 1
    num = num + alpha_optimizable * (transitions * A_optimizable).sum()

    den = 0
    transitions = torch.zeros((K, K), device=device)

    for i in range(len(best_paths)):
        if i == exclude_path:
            continue
        den_temp = Pi[best_paths[i][0]] + torch.sum(P[torch.arange(len(best_paths[i])), best_paths[i]])
        transitions = transitions * 0
        for j in range(len(best_paths[i]) - 1):
            transitions[best_paths[i][j], best_paths[i][j + 1]] += 1
        den_temp = den_temp + alpha_optimizable * (transitions * A_optimizable).sum()
        den = den + den_temp.exp()
    den = torch.log(den)

    return best_paths[0], -(num - den)

def train_hmm_mmi(obs_probs_list, train_ytrue_list, A_init, pi_init,
                  k_best=20, n_epochs=10, lr=1e-5, device="cuda"):
    """
    Train A and alpha jointly using MMI loss.
    obs_probs_list : list of (T_i, K) float arrays — aggregate_probs() output
    train_ytrue_list : list of (T_i,) int arrays — 0-indexed ground truth
    A_init, pi_init  : initialised from estimate_hmm_parameters_from_gt()
    """
    K = A_init.shape[0]

    # Trainable parameters — warm start from empirical estimates
    A_raw = torch.nn.Parameter(
        torch.tensor(np.log(A_init + 1e-10), dtype=torch.float64)
    )
    alpha = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))  # initial alpha obtained from alpha sweep prior to training
    pi = torch.tensor(np.log(pi_init + 1e-10), dtype=torch.float64).to(device)

    optimiser = torch.optim.Adam([A_raw, alpha], lr=lr)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for obs_probs, y_true in zip(obs_probs_list, train_ytrue_list):
            # Normalise A_raw → valid log-prob transition matrix each forward pass
            A_log = A_raw - torch.logsumexp(A_raw, dim=1, keepdim=True)  # (K,K) log-softmax rows
            A_prob = A_log.exp()

            T = len(y_true)
            P = torch.tensor(
                np.log(obs_probs + 1e-10), dtype=torch.float64
            )  # (T, K) log-emissions
            labels = torch.tensor(y_true, dtype=torch.long)

            _, loss = calc_MMI_loss(
                A_prob, P, pi, alpha,
                k_best=k_best, K=K, T=T,
                labels=labels, device=device
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        print(f"  [MMI epoch {epoch+1}/{n_epochs}] loss={epoch_loss:.4f}  alpha={alpha.item():.4f}")

    # Return numpy versions for your existing viterbi_hmm_softmax
    A_final  = A_log.exp().detach().numpy()
    pi_final = pi_init  # pi not trained, consistent with paper
    alpha_final = alpha.item()
    print(f"\n  Training complete. Final alpha={alpha_final:.4f}")
    return A_final, pi_final, alpha_final
