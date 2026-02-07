"""Theoretical probabilities and motif expectations.

Overview
--------
This module computes link probabilities, expected motif counts, and
simple Z-scores for empirical deviations under the core-periphery
maximum-entropy model.

Key conventions
---------------
- Inputs are interpreted on undirected simple graphs.
- Probability matrices are symmetric with zero diagonal.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from math import comb

def prob_matrix(y: float, x: NDArray[np.float_], core_idx: list[int], n_nodes: int) -> NDArray[np.float_]:
    """Compute pairwise link probabilities.

    Parameters
    ----------
    y : float
        Global field.
    x : ndarray float, shape (m,)
        Core-node fields.
    core_idx : list[int]
        Core-node indices.
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    ndarray float, shape (N, N)
        Symmetric matrix ``p_ij`` with zero diagonal.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> prob_matrix(0.0, np.array([0.5]), [0], 2).shape
    (2, 2)
    """
    add = np.zeros(n_nodes, dtype=float)
    add[core_idx] = x
    logits = y + add[:, None] + add[None, :]
    p = expit(logits)
    np.fill_diagonal(p, 0.0)
    return p

def expected_counts(y: float, x: NDArray[np.float_], core_idx: list[int], n_nodes: int) -> dict[str, float]:
    """Estimate expected motif counts and approximate variances.

    Parameters
    ----------
    y : float
        Global field.
    x : ndarray float, shape (m,)
        Core-node fields.
    core_idx : list[int]
        Core-node indices.
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    dict
        Dictionary with ``L_exp``, ``var_L``, ``W_exp``, ``var_W``,
        ``T_exp``, ``var_T``, and ``k_exp``.

    Notes
    -----
    Variances for ``W`` and ``T`` are approximated or left undefined.

    Complexity
    ----------
    O(N^3) for triangle computation.

    Examples
    --------
    >>> expected_counts(0.0, np.array([]), [], 2)['L_exp']
    0.0
    """
    p = prob_matrix(y,x,core_idx,n_nodes)
    iu = np.triu_indices(n_nodes,1)
    pij = p[iu]
    L_exp = pij.sum()
    var_L = np.sum(pij*(1-pij))

    # Expected degrees.
    k_exp = p.sum(axis=1)
    # Expected wedges with a plug-in approximation on expected degrees.
    W_exp = np.sum( k_exp*(k_exp-1)/2 )
    # Variance approximation not implemented.
    var_W = np.nan

    # Expected triangles over all triplets i < j < k.
    T_exp = 0.0
    n = n_nodes
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                T_exp += p[i,j]*p[j,k]*p[k,i]
    var_T = np.nan
    return dict(L_exp=L_exp, var_L=var_L, W_exp=W_exp, var_W=var_W, T_exp=T_exp, var_T=var_T, k_exp=k_exp)

def z_score(obs: float, exp: float, var: float, eps: float=1e-12) -> float:
    """Compute the Z-score of an observed count.

    Parameters
    ----------
    obs : float
        Observed count.
    exp : float
        Expected count.
    var : float
        Count variance.
    eps : float, default 1e-12
        Numerical stabilization term.

    Returns
    -------
    float
        Z-score ``(obs - exp) / sqrt(var + eps)``.

    Notes
    -----
    Intended for motif-level diagnostics.

    Complexity
    ----------
    O(1).

    Examples
    --------
    >>> z_score(5, 5, 1)
    0.0
    """
    return (obs - exp)/np.sqrt(var+eps)
