"""Sample undirected networks from the core-periphery logistic model.

Overview
--------
The module provides Monte Carlo sampling for one or multiple synthetic
networks under independent Bernoulli edges with probabilities generated
from core-periphery fields.

Key conventions
---------------
- Outputs are canonical upper-triangular boolean adjacency matrices.
- Reproducibility uses `numpy.random.Generator` instances.

Public API
----------
- `sample`
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.special import expit  # sigmoid

# ------------------------------------------------------------------ #
# Logger                                                             #
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------ #
# Sampling function                                                  #
# ------------------------------------------------------------------ #
def sample(
    y: float,
    x_vec: NDArray[np.float_],
    core_idx: List[int],
    *,
    n_nodes: int,
    n_sim: int = 1,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    n_jobs: int = 1,
) -> NDArray[np.bool_]:
    """Sample undirected networks from the core-periphery model.

    Parameters
    ----------
    y : float
        Global field.
    x_vec : ndarray float, shape (m,)
        Core-node fields.
    core_idx : list[int]
        Core-node indices.
    n_nodes : int
        Total number of nodes.
    n_sim : int, default 1
        Number of Monte Carlo replications.
    rng : numpy.random.Generator, optional
        Random generator to use.
    seed : int, optional
        Fallback seed if ``rng`` is not provided.
    n_jobs : int, default 1
        Number of parallel processes when ``n_sim > 1``.

    Returns
    -------
    ndarray bool
        Matrix ``(N, N)`` if ``n_sim=1`` or ``(n_sim, N, N)`` otherwise.

    Raises
    ------
    ValueError
        If the input arguments are inconsistent (for example ``n_sim <= 0``).

    Notes
    -----
    If ``core_idx`` is empty, the model reduces to Erdos-Renyi.

    Complexity
    ----------
    O(n_sim * N^2).

    Examples
    --------
    >>> sample(-4.0, np.array([]), [], n_nodes=3).shape
    (3, 3)
    """
    # ---------------- input checks ------------------------------------
    if n_sim <= 0:
        raise ValueError("n_sim must be >= 1")
    if len(core_idx) != x_vec.size and len(core_idx):
        raise ValueError("x_vec and core_idx must have the same length")
    if core_idx and n_nodes <= max(core_idx):
        raise ValueError("n_nodes must be > max(core_idx)")

    if rng is None:
        rng = np.random.default_rng(seed)
    N = n_nodes
    iu = np.triu_indices(N, k=1)  # upper-triangle indices

    # =================================================================
    # CASE 1: empty core -> Erdos-Renyi with p = sigma(y)
    # =================================================================
    if len(core_idx) == 0:
        p = expit(y)
        M = iu[0].size
        p_ut = np.full(M, p, dtype=float)
    else:
        # =================================================================
        # CASE 2: non-empty core -> standard CP model
        # =================================================================
        add = np.zeros(N, dtype=float)
        add[core_idx] = x_vec
        logits = y + add[:, None] + add[None, :]
        p_full = expit(logits)
        np.fill_diagonal(p_full, 0.0)
        p_ut = p_full[iu]  # M = N(N-1)/2

    # ---------------- sampling ---------------------------------------
    if n_sim == 1:
        edges = rng.random(p_ut.shape) < p_ut
        A = np.zeros((N, N), dtype=bool)
        A[iu] = edges
        return A

    if n_jobs == 1:
        edges = rng.random((n_sim, p_ut.size)) < p_ut
        sims = np.zeros((n_sim, N, N), dtype=bool)
        sims[:, iu[0], iu[1]] = edges
        return sims

    def _one_sim(seed: int) -> NDArray[np.bool_]:
        r = np.random.default_rng(seed)
        e = r.random(p_ut.shape) < p_ut
        A = np.zeros((N, N), dtype=bool)
        A[iu] = e
        return A

    seeds = rng.integers(1 << 32, size=n_sim)
    sims = Parallel(n_jobs=n_jobs)(delayed(_one_sim)(s) for s in seeds)
    return np.stack(sims)


# ------------------------------------------------------------------ #
# CLI debug                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":  # pragma: no cover
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Simulate an undirected core-periphery network."
    )
    parser.add_argument("--N", type=int, default=164)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--n_sim", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    N = args.N
    core = list(range(args.k))
    y_demo = -4.0
    x_demo = np.full(args.k, 1.5)

    rng = np.random.default_rng(args.seed)
    nets = sample(
        y_demo,
        x_demo,
        core,
        n_nodes=N,
        n_sim=args.n_sim,
        rng=rng,
        n_jobs=args.n_jobs,
    )
    dens = nets.mean() if args.n_sim == 1 else nets.mean(axis=(1, 2))
    print("shape:", nets.shape, "density:", dens)
