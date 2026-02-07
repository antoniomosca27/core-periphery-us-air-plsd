# -*- coding: utf-8 -*-
"""PLSD scoring and lambda calibration for core-periphery partitions.

Overview
--------
This module evaluates candidate core sets using PLSD, where PLSD stands
for Penalized Likelihood with Structural Discrepancies. It also provides
data-driven calibration helpers for motif discrepancy weights.

Key conventions
---------------
- Inputs use canonical upper-triangular boolean adjacency matrices.
- Motif moments are computed under an independent-edge approximation.

Public API
----------
- `compute_plsd_score`
- `estimate_lambda_motif`
- `estimate_lambda_from_scan`
- `estimate_lambda_from_scan_maha`
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation

from .inference_parameters import _prob_matrix, fit
from src.preprocessing.canonicalize import validate_upper_tri_bool


__all__ = [
    "compute_plsd_score",
    "estimate_lambda_motif",
    "estimate_lambda_from_scan",
    "estimate_lambda_from_scan_maha",
    "motif_empirical_counts",
    "motif_moments",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _triangles(A_sym: NDArray[np.bool_]) -> int:
    """Count undirected triangles in a symmetric adjacency matrix.

    Parameters
    ----------
    A_sym : ndarray bool, shape (N, N)
        Symmetric adjacency matrix.

    Returns
    -------
    int
        Number of triangles ``i < j < k``.

    Notes
    -----
    Uses ``A @ A @ A`` divided by 6.

    Complexity
    ----------
    ``O(N^3)``.

    Examples
    --------
    >>> A = np.eye(3, dtype=bool)
    >>> _triangles(A)
    0
    """
    A_int = A_sym.astype(int)
    return int((A_int @ A_int @ A_int).trace() // 6)


def _wedges(A_sym: NDArray[np.bool_]) -> int:
    """Count wedges (open and closed) in a symmetric adjacency matrix.

    Parameters
    ----------
    A_sym : ndarray bool, shape (N, N)
        Symmetric adjacency matrix.

    Returns
    -------
    int
        Total number of wedges.

    Notes
    -----
    Computed as ``deg * (deg - 1) / 2`` over node degrees.

    Complexity
    ----------
    ``O(N^2)``.

    Examples
    --------
    >>> A = np.eye(3, dtype=bool)
    >>> _wedges(A)
    0
    """
    deg = A_sym.sum(axis=1)
    return int(((deg * (deg - 1)) // 2).sum())


def motif_empirical_counts(A: NDArray[np.bool_]) -> tuple[int, int]:
    """Empirical wedge/triangle counts from a canonical upper-tri adjacency."""
    validate_upper_tri_bool(A)
    A_sym = np.logical_or(A, A.T)
    np.fill_diagonal(A_sym, 0)
    T_emp = _triangles(A_sym)
    W_emp = _wedges(A_sym)
    return W_emp, T_emp


def motif_moments(
    P: NDArray[np.float_],
) -> tuple[float, float, float, float, float]:
    """Vectorized moments under independent-edge approximation."""
    P = np.asarray(P, dtype=float)
    P = (P + P.T) * 0.5
    np.fill_diagonal(P, 0.0)

    P2 = P * P
    d = P.sum(axis=1)
    s2 = P2.sum(axis=1)

    mu_T = float(np.trace(P @ P @ P) / 6.0)
    mu_W = float(0.5 * np.sum(d * d - s2))

    var_T = float(mu_T - np.trace(P2 @ P2 @ P2) / 6.0)
    P4 = P2 * P2
    s4 = P4.sum(axis=1)
    sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
    var_W = float(mu_W - sum_q2)

    cov_TW = float(0.5 * (np.trace(P @ P @ P) - np.trace(P2 @ P @ P2)))
    return mu_T, var_T, mu_W, var_W, cov_TW


def _stabilize_sigma(
    var_T: float,
    var_W: float,
    cov_TW: float,
    *,
    eps_var: float,
    eps_sigma: float,
    det_min: float,
    max_bumps: int,
    rho_delta: float,
) -> tuple[NDArray[np.float_] | None, bool, float, float, float, float]:
    """Clamp correlation, ridge until PD, then return 2x2 inverse."""
    var_T = float(max(var_T, eps_var))
    var_W = float(max(var_W, eps_var))
    denom = float(np.sqrt(var_T * var_W))
    if not np.isfinite(denom) or denom == 0.0:
        return None, True, eps_sigma, var_T, var_W, float(cov_TW)

    rho = float(cov_TW) / denom
    rho = float(np.clip(rho, -(1.0 - rho_delta), (1.0 - rho_delta)))
    cov_TW = rho * denom

    eps = float(eps_sigma)
    for _ in range(max_bumps + 1):
        a = var_T + eps
        c = var_W + eps
        det = a * c - cov_TW * cov_TW
        if np.isfinite(det) and det > det_min:
            inv = (1.0 / det) * np.array([[c, -cov_TW], [-cov_TW, a]], dtype=float)
            return inv, False, eps, var_T, var_W, cov_TW
        eps *= 10.0

    return None, True, eps, var_T, var_W, cov_TW


# ------------------------------------------------------------------
# Motif expectations and variances under the independent-edge model
# ------------------------------------------------------------------


def _motif_expectations(
    p: NDArray[np.float_],
) -> tuple[float, float, float, float]:
    """Compute wedge and triangle moments from a probability matrix.

    Parameters
    ----------
    p : ndarray float, shape (N, N)
        Link probabilities from the fitted model.

    Returns
    -------
    tuple of float
        ``(E_W, Var_W, E_T, Var_T)``.

    Notes
    -----
    Uses matrix formulas under independent edges.

    Complexity
    ----------
    ``O(N^3)``.

    Examples
    --------
    >>> p = np.zeros((3, 3))
    >>> _motif_expectations(p)
    (0.0, 0.0, 0.0, 0.0)
    """
    p = np.asarray(p, dtype=float)
    p = 0.5 * (p + p.T)
    p = p.copy()
    np.fill_diagonal(p, 0.0)
    P2 = p * p
    d = p.sum(axis=1)
    s2 = P2.sum(axis=1)

    E_T = float(np.trace(p @ p @ p) / 6.0)
    E_W = float(0.5 * np.sum(d * d - s2))

    var_T = float(E_T - np.trace(P2 @ P2 @ P2) / 6.0)
    P4 = P2 * P2
    s4 = P4.sum(axis=1)
    sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
    var_W = float(E_W - sum_q2)

    return E_W, var_W, E_T, var_T


# ------------------------------------------------------------------
# Score computation
# ------------------------------------------------------------------


def compute_plsd_score(
    A: NDArray[np.bool_],
    core_idx: List[int],
    y_hat: float,
    x_hat: NDArray[np.float_],
    *,
    kind: str = "diag",
    lambda_T: float | None = None,
    lambda_W: float | None = None,
    lambda_M: float | None = None,
    freeze: bool = True,
    frozen_vars: tuple[float, float] | None = None,
    frozen_sigma: NDArray[np.float_] | None = None,
    eps_var: float = 1e-12,
    eps_sigma: float = 1e-12,
    det_min: float | None = None,
    rho_delta: float = 1e-6,
    max_sigma_bumps: int = 8,
    sigma_fail_value: float = 1e12,
    return_parts: bool = False,
) -> float | NDArray[np.float_]:
    """Compute the PLSD score for a core-periphery partition.

    Parameters
    ----------
    A : ndarray bool, shape (N, N)
        Canonical adjacency matrix.
    core_idx : list of int
        Core-node indices.
    y_hat : float
        Estimated global field ``y``.
    x_hat : ndarray float, shape (|core_idx|,)
        Estimated core fields ``x``.
    kind : {'diag', 'maha'}, default 'diag'
        Type of PLSD penalty.
    lambda_T : float, optional
        Weight of the triangle discrepancy term (diag mode).
    lambda_W : float, optional
        Weight of the wedge discrepancy term (diag mode).
    lambda_M : float, optional
        Weight of the Mahalanobis term (maha mode).
    freeze : bool, default True
        If True, use frozen variances or covariance matrix when provided.
    frozen_vars : tuple(float, float), optional
        Frozen denominators ``(Var_T, Var_W)`` for diag mode.
    frozen_sigma : ndarray float, shape (2, 2), optional
        Frozen covariance matrix for Mahalanobis mode.
    eps_var : float, default 1e-12
        Ridge added to variance denominators.
    eps_sigma : float, default 1e-12
        Initial ridge for the covariance matrix.
    det_min : float, optional
        Minimum covariance determinant (default ``eps_var**2``).
    rho_delta : float, default 1e-6
        Margin used when clamping correlation.
    max_sigma_bumps : int, default 8
        Maximum number of ridge increases for covariance stabilization.
    sigma_fail_value : float, default 1e12
        Fallback value of ``D2`` when covariance stabilization fails.
    return_parts : bool, default=False
        If True, return ``[neg_ll, DeltaT2, DeltaW2]`` (diag) or
        ``[neg_ll, D2]`` (maha).

    Returns
    -------
    float or ndarray float
        Aggregated score or separated components.

    Notes
    -----
    Expectations are computed in vectorized form by :func:`motif_moments`.

    Complexity
    ----------
    ``O(N^3)``.

    Examples
    --------
    >>> A = np.eye(3, dtype=bool)
    >>> compute_plsd_score(A, [0], 0.0, np.array([0.0]))
    0.0
    """
    kind = kind.lower()
    if kind in {"plsd", "plsd_diag"}:
        kind = "diag"
    elif kind in {"plsd_maha", "maha"}:
        kind = "maha"
    if kind not in {"diag", "maha"}:
        raise ValueError("kind must be 'diag' or 'maha'.")

    lambda_T = 1.0 if lambda_T is None else float(lambda_T)
    lambda_W = 1.0 if lambda_W is None else float(lambda_W)
    lambda_M = 1.0 if lambda_M is None else float(lambda_M)

    validate_upper_tri_bool(A)
    n = A.shape[0]
    core_mask = np.zeros(n, dtype=bool)
    core_mask[core_idx] = True
    p = _prob_matrix(y_hat, x_hat, core_mask)

    iu = np.triu_indices(n, 1)
    A_ut = A[iu]
    p_ut = p[iu]
    nll = -float(
        np.sum(A_ut * np.log(p_ut + 1e-12) + (1 - A_ut) * np.log(1 - p_ut + 1e-12))
    )

    W_emp, T_emp = motif_empirical_counts(A)
    mu_T, var_T, mu_W, var_W, cov_TW = motif_moments(p)

    r_T = float(T_emp) - mu_T
    r_W = float(W_emp) - mu_W

    if kind == "diag":
        if freeze and frozen_vars is not None:
            var_T, var_W = frozen_vars
        var_T = max(float(var_T), eps_var)
        var_W = max(float(var_W), eps_var)
        delta_T2 = (r_T * r_T) / (var_T + eps_var)
        delta_W2 = (r_W * r_W) / (var_W + eps_var)
        parts = np.array([nll, delta_T2, delta_W2], dtype=float)
        if return_parts:
            return parts
        return parts[0] + lambda_T * parts[1] + lambda_W * parts[2]

    if freeze and frozen_sigma is not None:
        var_T = float(frozen_sigma[0, 0])
        var_W = float(frozen_sigma[1, 1])
        cov_TW = float(frozen_sigma[0, 1])

    if det_min is None:
        det_min = eps_var * eps_var
    inv_sigma, sigma_failed, _, _, _, _ = _stabilize_sigma(
        var_T,
        var_W,
        cov_TW,
        eps_var=eps_var,
        eps_sigma=eps_sigma,
        det_min=det_min,
        max_bumps=max_sigma_bumps,
        rho_delta=rho_delta,
    )
    if sigma_failed or inv_sigma is None or not np.isfinite(inv_sigma).all():
        D2 = float(sigma_fail_value)
    else:
        r = np.array([r_T, r_W], dtype=float)
        v = inv_sigma @ r
        D2 = float(r @ v)

    parts = np.array([nll, D2], dtype=float)
    if return_parts:
        return parts
    return nll + lambda_M * D2


# ------------------------------------------------------------------
# Lambda estimation via robust scale weighting
# ------------------------------------------------------------------


def estimate_lambda_from_scan(
    nll_values: Iterable[float],
    delta_t2_values: Iterable[float],
    delta_w2_values: Iterable[float],
) -> NDArray[np.float_]:
    """Estimate lambda_T and lambda_W from scan vectors."""
    nlls = np.asarray(list(nll_values), dtype=float)
    deltaT2 = np.asarray(list(delta_t2_values), dtype=float)
    deltaW2 = np.asarray(list(delta_w2_values), dtype=float)

    mask = np.isfinite(nlls) & np.isfinite(deltaT2) & np.isfinite(deltaW2)
    nlls = nlls[mask]
    deltaT2 = deltaT2[mask]
    deltaW2 = deltaW2[mask]

    if deltaT2.size == 0 or deltaW2.size == 0 or nlls.size == 0:
        return np.array([1.0, 1.0], dtype=float)

    mad_T = float(median_abs_deviation(deltaT2, scale=1.0))
    sigma_T = 1.4826 * mad_T if mad_T > 0 else float(np.std(deltaT2, ddof=1))
    mad_W = float(median_abs_deviation(deltaW2, scale=1.0))
    sigma_W = 1.4826 * mad_W if mad_W > 0 else float(np.std(deltaW2, ddof=1))
    mean_nll = float(np.mean(nlls))

    lambda_T = mean_nll / max(sigma_T, 1e-9)
    lambda_W = mean_nll / max(sigma_W, 1e-9)
    return np.array([lambda_T, lambda_W], dtype=float)


def estimate_lambda_from_scan_maha(
    nll_values: Iterable[float],
    d2_values: Iterable[float],
) -> float:
    """Estimate lambda_M from scan vectors (Mahalanobis)."""
    nlls = np.asarray(list(nll_values), dtype=float)
    d2s = np.asarray(list(d2_values), dtype=float)

    mask = np.isfinite(nlls) & np.isfinite(d2s)
    nlls = nlls[mask]
    d2s = d2s[mask]

    if d2s.size == 0 or nlls.size == 0:
        return 1.0

    mad = float(median_abs_deviation(d2s, scale=1.0))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(d2s, ddof=1))
    mean_nll = float(np.mean(nlls))
    return float(mean_nll / max(sigma, 1e-9))


def estimate_lambda_motif(
    A: NDArray[np.bool_],
    candidate_sizes: Iterable[int],
    *,
    n_samples: int | None = None,
    seed: int = 42,
    tol: float = 1e-8,
) -> NDArray[np.float_]:
    """Estimate robust motif weights ``lambda_T`` and ``lambda_W``.

    Parameters
    ----------
    A : ndarray bool, shape (N, N)
        Canonical adjacency matrix.
    candidate_sizes : iterable of int
        Candidate core sizes.
    n_samples : int, optional
        Numero massimo di dimensioni da valutare.
    seed : int, default=42
        Random generator seed.
    tol : float, default=1e-8
        Solver tolerance for parameter fitting.

    Returns
    -------
    ndarray float, shape (2,)
        Values ``[lambda_T, lambda_W]``.

    Notes
    -----
    ``lambda_T`` and ``lambda_W`` use robust MAD scaling with standard
    deviation fallback.

    Complexity
    ----------
    Depends on ``|candidate_sizes|`` PLSD evaluations and the cost of
    ``fit``.

    Examples
    --------
    >>> A = np.eye(3, dtype=bool)
    >>> estimate_lambda_motif(A, [1])
    array([1., 1.])
    """
    validate_upper_tri_bool(A)
    rng = np.random.default_rng(seed)
    A_sym = np.logical_or(A, A.T)
    ranking = np.argsort(A_sym.sum(axis=1))[::-1]
    k_list = list(candidate_sizes)
    if n_samples is not None and n_samples < len(k_list):
        k_list = list(rng.choice(k_list, size=n_samples, replace=False))

    deltaT2, deltaW2, nlls = [], [], []

    for k in k_list:
        core = ranking[:k].tolist()
        y_hat, x_hat, _ = fit(A, core, tol=tol, max_iter=500)
        parts = compute_plsd_score(
            A, core, y_hat, x_hat, lambda_T=1.0, lambda_W=1.0, return_parts=True
        )
        nlls.append(parts[0])
        deltaT2.append(parts[1])            # DeltaT2
        deltaW2.append(parts[2])            # DeltaW2
    return estimate_lambda_from_scan(nlls, deltaT2, deltaW2)
