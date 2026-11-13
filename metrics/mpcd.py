"""Mean pairwise cosine distance (MPCD). Higher = more diverse.

    MPCD = 1 - mean_{i != j} cos(x_i, x_j)
"""
import numpy as np


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    Xn = X / norms
    return Xn @ Xn.T


def mpcd(X: np.ndarray) -> float:
    """MPCD of the rows of X (n, d), over all n(n-1) ordered pairs i != j."""
    S = cosine_similarity_matrix(X)
    n = S.shape[0]
    if n < 2:
        return 0.0
    return 1 - float(S[~np.eye(n, dtype=bool)].mean())


def mpcd_from_sum(s, N):
    """Batched form for unit-norm rows. s: (B, d) = sum of the N rows of each of B subsets -> (B,).

    sum_{i != j} cos = |sum_i x_i|^2 - N, so no N x N matrix is needed."""
    return 1 - ((s * s).sum(-1) - N) / (N * (N - 1))
