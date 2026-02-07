"""Ranking utilities for core selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.preprocessing.canonicalize import validate_upper_tri_bool


def rank_nodes_for_period(
    A_time: NDArray[np.bool_],
    period_pos: int,
    window_size: int,
    active_nodes_current: NDArray[np.int_] | list[int],
) -> tuple[list[int], NDArray[np.int_]]:
    """Rank active nodes from degree sums over the previous periods.

    Parameters
    ----------
    A_time : ndarray bool, shape (N, N, T)
        Canonical adjacency tensor.
    period_pos : int
        Current period position (1-based).
    window_size : int
        Number of previous periods used for ranking.
    active_nodes_current : array-like of int
        Node indices active in the current period.
    """
    if A_time.ndim != 3 or A_time.shape[0] != A_time.shape[1]:
        raise ValueError("A_time must have shape (N, N, T).")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    n_nodes, _, n_periods = A_time.shape
    if period_pos < 1 or period_pos > n_periods:
        raise ValueError("period_pos out of range.")
    if period_pos <= window_size:
        raise ValueError("Not enough previous periods for ranking.")

    active_nodes = np.asarray(active_nodes_current, dtype=int)
    if active_nodes.size == 0:
        return [], np.zeros(n_nodes, dtype=int)

    scores = np.zeros(n_nodes, dtype=int)
    start_period = period_pos - window_size
    end_period = period_pos - 1

    for s in range(start_period, end_period + 1):
        period_idx = s - 1
        A_period = A_time[:, :, period_idx]
        validate_upper_tri_bool(A_period)
        deg = A_period.sum(axis=0) + A_period.sum(axis=1)
        scores += deg.astype(int, copy=False)

    scores_active = scores[active_nodes]
    order = np.lexsort((active_nodes, -scores_active))
    ranking = active_nodes[order].tolist()
    return ranking, scores
