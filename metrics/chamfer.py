"""Chamfer distance between two embedding sets (encoder validation only, paper §3.2).

    Chamfer(X, Y) = mean_x min_y ||x - y|| + mean_y min_x ||x - y||

On raw (un-normalised) embeddings, as in the paper's appendix.
"""
import numpy as np
from scipy.spatial.distance import cdist


def chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    D = cdist(X, Y, metric="euclidean")
    return float(D.min(axis=1).mean() + D.min(axis=0).mean())
