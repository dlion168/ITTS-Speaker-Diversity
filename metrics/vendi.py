"""Vendi Score with a cosine kernel. Higher = more diverse.

    K_ij = cos(x_i, x_j),   Vendi = exp(-sum_k lambda_k log lambda_k),  lambda = eig(K / n)

It is the effective number of distinct embeddings and grows with n, so only compare sets of
equal size (n = 10,000 in §4.1, 3,000 in §4.2 and N = 1,500 subsets in §5).
"""
import numpy as np

from mpcd import cosine_similarity_matrix


def vendi(X: np.ndarray) -> float:
    """Vendi score of the rows of X (n, d)."""
    S = cosine_similarity_matrix(X)
    eigvals = np.linalg.eigvalsh(S / S.shape[0])
    eigvals = eigvals[eigvals > 1e-10]
    return float(np.exp(-np.sum(eigvals * np.log(eigvals))))


def vendi_from_gram(G, N):
    """Batched torch form. G: (B, d, d) = X_S^T X_S for B subsets of N unit rows -> (B,).

    The non-zero eigenvalues of X X^T / N (N x N) equal those of X^T X / N (d x d)."""
    import torch
    lam = torch.linalg.eigvalsh(G / N).clamp_min(0)
    return torch.exp(-(lam * torch.log(lam.clamp_min(1e-12))).sum(-1))
