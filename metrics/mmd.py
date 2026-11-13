"""Squared maximum mean discrepancy between two embedding sets (encoder validation only).

Gaussian kernel k(a, b) = exp(-||a - b||^2 / (2 sigma^2)), sigma = median pairwise distance over
X ∪ Y (median heuristic), biased estimator (diagonal terms included):

    MMD^2 = mean k(X, X) + mean k(Y, Y) - 2 mean k(X, Y)

As in the paper's appendix.
"""
import numpy as np
from scipy.spatial.distance import cdist


def median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    XY = np.vstack([X, Y])
    D2 = cdist(XY, XY, metric="sqeuclidean")
    sigma = float(np.median(D2[D2 > 0]) ** 0.5)
    return sigma if sigma >= 1e-8 else 1.0


def _mean_kernel(X, Y, sigma):
    return float(np.exp(-cdist(X, Y, metric="sqeuclidean") / (2 * sigma ** 2)).mean())


def mmd2(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    if sigma is None:
        sigma = median_bandwidth(X, Y)
    return _mean_kernel(X, X, sigma) + _mean_kernel(Y, Y, sigma) - 2 * _mean_kernel(X, Y, sigma)
