"""Analytical expectations under the independent-edge approximation.

Overview
--------
This module computes closed-form expectations for links, wedges,
triangles, and node-degree distributions from a probability matrix
without Monte Carlo simulation.

Key conventions
---------------
- `p` is treated as a symmetric probability matrix with zero diagonal.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import math
from itertools import combinations

__all__ = ["expectations", "degree_distribution", "z_test"]

def expectations(p: NDArray[np.float_]):
    """Compute expected links, wedges, and triangles from ``p``.

    Parameters
    ----------
    p : ndarray float, shape (N, N)
        Symmetric edge-probability matrix.

    Returns
    -------
    tuple of float
        Expected counts ``(L, W, T)``.

    Raises
    ------
    ValueError
        If ``p`` is not square.

    Notes
    -----
    Wedges are computed exactly by summing over nodes.
    Triangles are computed as ``trace(p @ p @ p) / 6``.

    Complexity
    ----------
    ``O(N^3)``.

    Examples
    --------
    >>> p = np.zeros((2, 2))
    >>> expectations(p)
    (0.0, 0.0, 0.0)
    """
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("p must be square")
    n = p.shape[0]
    iu = np.triu_indices(n, 1)
    L = float(p[iu].sum())
    deg = p.sum(axis=1)
    W = 0.0
    for i in range(n):
        probs = p[i].copy()
        probs[i] = 0.0
        W += 0.5 * (probs.sum() ** 2 - (probs ** 2).sum())

    T = float(np.trace(p @ p @ p) / 6.0)
    return L, W, T

def degree_distribution(p: NDArray[np.float_], max_k: int | None = None):
    """Compute the expected degree-distribution histogram.

    Parameters
    ----------
    p : ndarray float, shape (N, N)
        Edge-probability matrix.
    max_k : int, optional
        Maximum degree to include. Default is ``N - 1``.

    Returns
    -------
    ndarray float, shape (max_k+1,)
        Expected probability mass for each degree.

    Notes
    -----
    Uses a Poisson-binomial convolution for each node.

    Complexity
    ----------
    ``O(N^2 * max_k)``.

    Examples
    --------
    >>> p = np.zeros((2, 2))
    >>> degree_distribution(p)
    array([1., 0.])
    """
    n = p.shape[0]
    if max_k is None:
        max_k = n - 1
    hist = np.zeros(max_k + 1)
    for i in range(n):
        probs = p[i].copy()
        probs[i] = 0.0
        q = np.array([1.0])
        for pr in probs:
            q = np.concatenate([q * (1 - pr), q * 0])
            q[1:] += pr * q[:-1]
        hist[: len(q)] += q
    hist /= n
    return hist

def z_test(empirical: float, expected: float, var: float):
    """Compute a Z-score for an observed count.

    Parameters
    ----------
    empirical : float
        Observed value.
    expected : float
        Expected value under the null model.
    var : float
        Variance of the aggregated Bernoulli sum.

    Returns
    -------
    float
        Normalized Z-score. Returns ``nan`` if ``var`` is zero.

    Notes
    -----
    The function does not validate consistency between ``empirical`` and
    ``expected``.

    Complexity
    ----------
    ``O(1)``.

    Examples
    --------
    >>> z_test(5, 5, 1)
    0.0
    """
    if var == 0:
        return np.nan
    return (empirical - expected) / math.sqrt(var)
