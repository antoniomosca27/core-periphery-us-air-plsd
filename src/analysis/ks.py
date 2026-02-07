"""
----------------------------------------------------------------------
FILE: src/analysis/ks.py
----------------------------------------------------------------------

Purpose
-------
Deterministic KS testing on degree distributions using a fixed subset
of Monte Carlo simulations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp


def ks_degree_sampled(
    deg_emp: NDArray[np.int_],
    deg_sims: NDArray[np.int_],
    *,
    n_KS: int,
    seed_KS: int,
    period_pos: int,
) -> tuple[NDArray[np.float_], float, NDArray[np.int_]]:
    """Compute KS p-values on a deterministic subset of simulations."""
    if deg_sims.ndim != 2:
        raise ValueError("deg_sims must be a 2D array of simulated degrees.")
    n_sim = deg_sims.shape[0]
    if n_KS <= 0:
        raise ValueError("n_KS must be positive.")
    if n_KS > n_sim:
        raise ValueError("n_KS must be <= number of simulations.")

    rng = np.random.default_rng(np.random.SeedSequence([seed_KS, period_pos]))
    idx = rng.choice(n_sim, size=n_KS, replace=False)
    p_values = np.array(
        [ks_2samp(deg_emp, deg_sims[i]).pvalue for i in idx],
        dtype=float,
    )
    p_median = float(np.median(p_values)) if p_values.size else float("nan")
    return p_values, p_median, idx
