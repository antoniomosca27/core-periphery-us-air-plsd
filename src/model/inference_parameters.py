"""Parameter inference for the core-periphery model.

Overview
--------
This module estimates model parameters ``(y, x)`` from canonical
upper-triangular adjacency matrices using L-BFGS-B. It supports pure
negative log-likelihood fitting and PLSD-based objectives.

Key conventions
---------------
- Input networks are undirected simple graphs encoded in canonical
  upper-triangular boolean form.
- Core parameters can be constrained to ``x_i >= 0``.

Public API
----------
- `_prob_matrix`
- `fit`
"""

from __future__ import annotations

import logging
from typing import Tuple, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from src.preprocessing.canonicalize import validate_upper_tri_bool

# ------------------------------------------------------------------ #
# Module logger                                                   #
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False   # Avoid duplicate output if root logger has handlers.
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------ #
# ---------- 0. Utility: power-law sampling -------------------------- #
# ------------------------------------------------------------------ #
def _sample_powerlaw(
    size: int,
    *,
    alpha: float = 2.5,
    x_min: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float_]:
    """Sample a Pareto distribution.

    Parameters
    ----------
    size : int
        Number of samples to generate.
    alpha : float, default 2.5
        Pareto tail exponent (``>1``).
    x_min : float, default 1.0
        Minimum support value.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray float, shape (size,)
        Samples drawn from the distribution.

    Raises
    ------
    ValueError
        If ``alpha`` <= 1.

    Notes
    -----
    Uses the inverse Pareto CDF.

    Complexity
    ----------
    O(size).

    Examples
    --------
    >>> s = _sample_powerlaw(3, alpha=2)
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1 for a finite mean")
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(size)  # U ~ Uniform(0, 1)
    return x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def _init_x(
    n_core: int,
    *,
    method: Literal["zeros", "powerlaw"] = "zeros",
    alpha: float = 2.5,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float_]:
    """Initialize ``x`` values for core nodes.

    Parameters
    ----------
    n_core : int
        Number of core nodes.
    method : {'zeros', 'powerlaw'}, default 'zeros'
        Initialization strategy.
    alpha : float, default 2.5
        Pareto exponent for ``method='powerlaw'``.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray float, shape (n_core,)
        Initial values in log-space.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    Notes
    -----
    With ``method='powerlaw'``, samples are normalized to mean 1
    before taking logs.

    Complexity
    ----------
    O(n_core).

    Examples
    --------
    >>> _init_x(2).shape
    (2,)
    """
    if n_core == 0:
        return np.zeros(0, dtype=float)
    if method == "zeros":
        x_raw = np.ones(n_core, dtype=float)
    elif method == "powerlaw":
        x_raw = _sample_powerlaw(n_core, alpha=alpha, rng=rng)
        x_raw /= x_raw.mean()  # Normalize mean to 1 for stable y initialization.
    else:
        raise ValueError(f"unknown x_init method: {method}")

    return np.log(x_raw)  # log-space (always >0 in the original scale)


# ------------------------------------------------------------------ #
# ---------- 1. Shared function: probability matrix ------------------ #
# ------------------------------------------------------------------ #
def _prob_matrix(
    y: float,
    x: NDArray[np.float_],
    core_mask: NDArray[np.bool_],
) -> NDArray[np.float_]:
    """Link-probability matrix.

    Parameters
    ----------
    y : float
        Global field.
    x : ndarray float, shape (m,)
        Core-node fields.
    core_mask : ndarray bool, shape (N,)
        Mask identifying core nodes.

    Returns
    -------
    ndarray float, shape (N, N)
        Symmetric matrix ``p_ij`` with zero diagonal.

    Notes
    -----
    Periphery nodes have zero field.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> _prob_matrix(0.0, np.array([0.5]), np.array([True, False])).shape
    (2, 2)
    """
    N = core_mask.size
    add = np.zeros(N, dtype=float)
    add[core_mask] = x
    logits = y + add[:, None] + add[None, :]
    p = expit(logits)
    np.fill_diagonal(p, 0.0)
    return p


# ------------------------------------------------------------------ #
# ---------- 2. Objective (NLL/PLSD) and gradients ------------------- #
# ------------------------------------------------------------------ #
def _neg_ll_and_grad(
    params: NDArray[np.float_],
    A: NDArray[np.bool_],
    core_mask: NDArray[np.bool_],
) -> Tuple[float, NDArray[np.float_]]:
    """Negative log-likelihood and gradient.

    Parameters
    ----------
    params : ndarray float, shape (1+m,)
        First element is ``y`` and remaining elements are ``x_i``.
    A : ndarray bool, shape (N, N)
        Observed adjacency matrix.
    core_mask : ndarray bool, shape (N,)
        Core-node mask.

    Returns
    -------
    float
        Negative log-likelihood.
    ndarray float, shape (1+m,)
        Gradient with respect to ``(y, x)``.

    Notes
    -----
    Used as the objective for L-BFGS-B.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.zeros((2, 2), dtype=bool)
    >>> _neg_ll_and_grad(np.zeros(1), A, np.array([False, False]))
    (0.0, array([0.]))
    """
    y = params[0]
    x = params[1:]
    p = _prob_matrix(y, x, core_mask)

    iu = np.triu_indices_from(A, 1)
    p_ut = p[iu]
    A_ut = A[iu]
    eps = 1e-12
    nll = -float(
        np.sum(A_ut * np.log(p_ut + eps) + (1 - A_ut) * np.log(1 - p_ut + eps))
    )

    diff_ut = np.triu(p - A, 1)
    grad_y = float(diff_ut.sum())
    deg_diff = diff_ut.sum(axis=0) + diff_ut.sum(axis=1)
    grad_x = deg_diff[core_mask]
    grad = np.concatenate(([grad_y], grad_x))
    return nll, grad


def _edge_weight_sums(
    Q: NDArray[np.float_],
    W: NDArray[np.float_],
    core_mask: NDArray[np.bool_],
) -> Tuple[float, NDArray[np.float_]]:
    """Sum edge weights for y and x gradients over symmetric matrices."""
    weighted = Q * W
    d_y = float(np.triu(weighted, 1).sum())
    d_x = weighted.sum(axis=1)[core_mask].astype(float, copy=False)
    return d_y, d_x


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


def _objective_and_grad(
    params: NDArray[np.float_],
    A: NDArray[np.bool_],
    core_mask: NDArray[np.bool_],
    obj_cfg: dict[str, object],
) -> Tuple[float, NDArray[np.float_]]:
    """Objective (NLL or PLSD) and gradient for L-BFGS-B."""
    obj_kind = str(obj_cfg.get("obj_kind", "nll"))

    y = float(params[0])
    x = params[1:].astype(float, copy=False)
    P = _prob_matrix(y, x, core_mask)

    iu = np.triu_indices_from(A, 1)
    p_ut = P[iu]
    A_ut = A[iu]
    eps_nll = float(obj_cfg.get("eps_nll", 1e-12))
    nll = -float(
        np.sum(A_ut * np.log(p_ut + eps_nll) + (1 - A_ut) * np.log(1 - p_ut + eps_nll))
    )

    diff_ut = np.triu(P - A, 1)
    grad_y = float(diff_ut.sum())
    deg_diff = diff_ut.sum(axis=0) + diff_ut.sum(axis=1)
    grad_x = deg_diff[core_mask].astype(float, copy=False)
    grad = np.concatenate(([grad_y], grad_x))

    if obj_kind == "nll":
        return nll, grad

    T_emp = float(obj_cfg["T_emp"])
    W_emp = float(obj_cfg["W_emp"])
    eps_var = float(obj_cfg.get("eps_var", 1e-12))
    rho_delta = float(obj_cfg.get("rho_delta", 1e-6))
    eps_sigma = float(obj_cfg.get("eps_sigma", 1e-12))
    det_min = float(obj_cfg.get("det_min", eps_var * eps_var))
    max_sigma_bumps = int(obj_cfg.get("max_sigma_bumps", 8))
    freeze = bool(obj_cfg.get("freeze", True))
    sigma_fail_value = float(obj_cfg.get("sigma_fail_value", 1e12))

    # Common motif quantities
    Q = P * (1.0 - P)
    np.fill_diagonal(Q, 0.0)
    d = P.sum(axis=1)
    P2 = P * P
    s2 = P2.sum(axis=1)
    M = P @ P
    mu_T = float(np.trace(M @ P) / 6.0)
    mu_W = float(0.5 * np.sum(d * d - s2))

    S = d[:, None] + d[None, :] - 2.0 * P
    dmu_T_y, dmu_T_x = _edge_weight_sums(Q, M, core_mask)
    dmu_W_y, dmu_W_x = _edge_weight_sums(Q, S, core_mask)

    r_T = T_emp - mu_T
    r_W = W_emp - mu_W

    if obj_kind == "plsd_diag":
        lambda_T = float(obj_cfg.get("lambda_T", 1.0))
        lambda_W = float(obj_cfg.get("lambda_W", 1.0))

        if freeze and obj_cfg.get("frozen_var_T") is not None:
            var_T = float(obj_cfg["frozen_var_T"])
        else:
            var_T = float(mu_T - np.trace(P2 @ P2 @ P2) / 6.0)
        if freeze and obj_cfg.get("frozen_var_W") is not None:
            var_W = float(obj_cfg["frozen_var_W"])
        else:
            P4 = P2 * P2
            s4 = P4.sum(axis=1)
            sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
            var_W = float(mu_W - sum_q2)

        var_T = max(var_T, eps_var)
        var_W = max(var_W, eps_var)
        v_T = var_T + eps_var
        v_W = var_W + eps_var

        if freeze:
            dvar_T_y = 0.0
            dvar_T_x = np.zeros_like(dmu_T_x)
            dvar_W_y = 0.0
            dvar_W_x = np.zeros_like(dmu_W_x)
        else:
            B = P2 @ P2
            D_muT_sq = 2.0 * P * B
            dmuT_sq_y, dmuT_sq_x = _edge_weight_sums(Q, D_muT_sq, core_mask)
            dvar_T_y = dmu_T_y - dmuT_sq_y
            dvar_T_x = dmu_T_x - dmuT_sq_x

            P4 = P2 * P2
            s4 = P4.sum(axis=1)
            sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
            D_sum_q2 = 2.0 * P * (s2[:, None] + s2[None, :] - 2.0 * P2)
            dsum_q2_y, dsum_q2_x = _edge_weight_sums(Q, D_sum_q2, core_mask)
            dvar_W_y = dmu_W_y - dsum_q2_y
            dvar_W_x = dmu_W_x - dsum_q2_x

        delta_T2 = (r_T * r_T) / v_T
        delta_W2 = (r_W * r_W) / v_W

        d_delta_T_y = (-2.0 * r_T / v_T) * dmu_T_y - (r_T * r_T / (v_T * v_T)) * dvar_T_y
        d_delta_W_y = (-2.0 * r_W / v_W) * dmu_W_y - (r_W * r_W / (v_W * v_W)) * dvar_W_y
        d_delta_T_x = (-2.0 * r_T / v_T) * dmu_T_x - (r_T * r_T / (v_T * v_T)) * dvar_T_x
        d_delta_W_x = (-2.0 * r_W / v_W) * dmu_W_x - (r_W * r_W / (v_W * v_W)) * dvar_W_x

        obj = nll + lambda_T * delta_T2 + lambda_W * delta_W2
        grad_pen = np.concatenate((
            [lambda_T * d_delta_T_y + lambda_W * d_delta_W_y],
            lambda_T * d_delta_T_x + lambda_W * d_delta_W_x,
        ))
        return obj, grad + grad_pen

    if obj_kind != "plsd_maha":
        raise ValueError(f"unknown objective: {obj_kind}")

    lambda_M = float(obj_cfg.get("lambda_M", 1.0))
    sigma_failed = bool(obj_cfg.get("sigma_failed", False))

    if freeze and obj_cfg.get("frozen_inv_sigma") is not None:
        inv_sigma = np.asarray(obj_cfg["frozen_inv_sigma"], dtype=float)
    else:
        var_T = float(mu_T - np.trace(P2 @ P2 @ P2) / 6.0)
        P4 = P2 * P2
        s4 = P4.sum(axis=1)
        sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
        var_W = float(mu_W - sum_q2)
        cov_TW = float(0.5 * (np.trace(M @ P) - np.trace(P2 @ P @ P2)))
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
        obj = nll + lambda_M * sigma_fail_value
        return obj, grad

    r = np.array([r_T, r_W], dtype=float)
    v = inv_sigma @ r
    D2 = float(r @ v)

    if freeze:
        dD2_y = -2.0 * (dmu_T_y * v[0] + dmu_W_y * v[1])
        dD2_x = -2.0 * (dmu_T_x * v[0] + dmu_W_x * v[1])
    else:
        B = P2 @ P2
        D_muT_sq = 2.0 * P * B
        dmuT_sq_y, dmuT_sq_x = _edge_weight_sums(Q, D_muT_sq, core_mask)
        dvar_T_y = dmu_T_y - dmuT_sq_y
        dvar_T_x = dmu_T_x - dmuT_sq_x

        P4 = P2 * P2
        s4 = P4.sum(axis=1)
        sum_q2 = 0.5 * np.sum(s2 * s2 - s4)
        D_sum_q2 = 2.0 * P * (s2[:, None] + s2[None, :] - 2.0 * P2)
        dsum_q2_y, dsum_q2_x = _edge_weight_sums(Q, D_sum_q2, core_mask)
        dvar_W_y = dmu_W_y - dsum_q2_y
        dvar_W_x = dmu_W_x - dsum_q2_x

        A = P2
        D_f = (A @ A) + 2.0 * P * (P @ A + A @ P)
        G_cov = 3.0 * (M) - D_f
        dCov_y, dCov_x = _edge_weight_sums(Q, G_cov, core_mask)

        dD2_y = -2.0 * (dmu_T_y * v[0] + dmu_W_y * v[1])
        dD2_y -= (v[0] ** 2) * dvar_T_y + (v[1] ** 2) * dvar_W_y + 2.0 * v[0] * v[1] * dCov_y
        dD2_x = -2.0 * (dmu_T_x * v[0] + dmu_W_x * v[1])
        dD2_x -= (v[0] ** 2) * dvar_T_x + (v[1] ** 2) * dvar_W_x + 2.0 * v[0] * v[1] * dCov_x

    obj = nll + lambda_M * D2
    grad_pen = np.concatenate(([lambda_M * dD2_y], lambda_M * dD2_x))
    return obj, grad + grad_pen

def _kkt_residuals(
    y: float,
    x: NDArray[np.float_],
    L_obs: float,
    k_obs: NDArray[np.float_],
    core_idx: list[int],
) -> Tuple[float, float, float]:
    """KKT residuals for the ``x_i >= 0`` constraint."""
    N = k_obs.shape[0]
    core_mask = np.zeros(N, dtype=bool)
    core_mask[core_idx] = True
    m = len(x)
    p = N - m

    p_pp = float(expit(y))
    logits_iP = y + x
    p_iP = expit(logits_iP)
    logits_cc = y + x[:, None] + x[None, :]
    P_cc = expit(logits_cc)
    np.fill_diagonal(P_cc, 0.0)

    # float, not //, for numerical consistency
    E_pp = p * (p - 1) / 2.0
    L = E_pp * p_pp + p * float(p_iP.sum()) + float(np.triu(P_cc, 1).sum())
    k_i = p * p_iP + P_cc.sum(axis=1)
    g_x = k_i - k_obs[core_mask]

    # Active constraints consistent with KKT: x_i=0 and gradient points lower
    active = (x <= 1e-10) & (g_x > 0)
    free = ~active
    gnorm_free = float(np.max(np.abs(g_x[free]))) if free.any() else 0.0
    min_x_active = float(x[active].min()) if active.any() else 0.0
    min_grad_active = float(g_x[active].min()) if active.any() else 0.0
    return gnorm_free, min_x_active, min_grad_active


# ------------------------------------------------------------------ #
#  L-BFGS-B estimation: returns (y_hat, x_hat, obj_hat)                #
# ------------------------------------------------------------------ #
def _fit_bfgs(
    A: NDArray[np.bool_],
    core_mask: NDArray[np.bool_],
    *,
    tol: float,
    max_iter: int,
    x_init_method: str,
    alpha_pl: float,
    rng: np.random.Generator,
    start_params: Optional[Tuple[float, NDArray[np.float_]]] = None,
    bounded: bool = True,
    objective: str = "nll",
    obj_cfg: Optional[dict[str, object]] = None,
) -> Tuple[float, NDArray[np.float_], float]:
    """Estimate ``(y, x)`` with L-BFGS-B.

    Core parameters ``x_i`` are estimated under the constraint ``x_i >= 0``
    when ``bounded=True``.

    Parameters
    ----------
    A : ndarray bool, shape (N, N)
        Adjacency matrix.
    core_mask : ndarray bool, shape (N,)
        Core-node mask.
    tol : float
        Gradient tolerance.
    max_iter : int
        Maximum optimizer iterations.
    x_init_method : str
        Initialization method for ``x``.
    alpha_pl : float
        Pareto exponent for ``x_init_method='powerlaw'``.
    rng : numpy.random.Generator
        Random generator.
    start_params : tuple(float, ndarray), optional
        Initial values ``(y0, x0)``.
    bounded : bool, default True
        Enforce ``x_i >= 0`` when ``True``.
    objective : str, default "nll"
        Objective to optimize (``nll`` or PLSD).
    obj_cfg : dict, optional
        PLSD configuration (empirical counts, lambdas, freeze, etc.).

    Returns
    -------
    float
        Estimate of ``y``.
    ndarray float, shape (m,)
        Estimates of ``x``.
    float
        Optimized objective value (NLL or PLSD).

    Notes
    -----
    On error, returns triples of ``nan``.

    Complexity
    ----------
    O(N^2 * I), where ``I`` is the number of iterations.

    Examples
    --------
    >>> A = np.zeros((2, 2), dtype=bool)
    >>> _fit_bfgs(A, np.array([False, False]), tol=1e-6, max_iter=1,
    ...           x_init_method='zeros', alpha_pl=2.5,
    ...           rng=np.random.default_rng(), objective="nll")
    (0.0, array([], dtype=float64), 0.0)
    """
    # ---------- safe defaults -----------------------------------------
    n_core = core_mask.sum()
    fallback = (np.nan, np.full(n_core, np.nan), np.nan)

    # ---------- parameter initialization ------------------------------
    if start_params is None:
        N       = A.shape[0]
        density = A[np.triu_indices(N, 1)].mean()
        y0      = np.log(density / (1 - density)) if 0 < density < 1 else 0.0
        x0      = _init_x(n_core, method=x_init_method,
                          alpha=alpha_pl, rng=rng)
        params0 = np.concatenate(([y0], x0))
    else:
        y0, x0 = start_params
        params0 = np.concatenate(([float(y0)], np.asarray(x0, dtype=float)))

    try:
        bounds = None
        if bounded:
            bounds = [(None, None)] + [(0.0, None)] * n_core
        if obj_cfg is None:
            obj_cfg = {"obj_kind": "nll"}
        else:
            obj_cfg = dict(obj_cfg)
        obj_cfg["obj_kind"] = objective

        res = minimize(
            fun=lambda th: _objective_and_grad(th, A, core_mask, obj_cfg),
            x0=params0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options=dict(maxiter=max_iter, ftol=tol, gtol=tol),
        )

        # ---------- optimization outcome ----------------------------------
        y_hat = float(res.x[0])
        x_hat = res.x[1:].astype(float, copy=True)
        obj_hat = float(res.fun)

        if objective == "nll":
            _, grad_full = _neg_ll_and_grad(res.x, A, core_mask)
            grad_x = grad_full[1:]
            active = x_hat <= 1e-10
            free = ~active
            ginf = float(np.max(np.abs(grad_x[free]))) if free.any() else 0.0
            min_x_act = float(x_hat[active].min()) if active.any() else float("nan")
            min_g_act = float(grad_x[active].min()) if active.any() else float("nan")
            logger.info(
                "BFGS KKT: |A|=%d |F|=%d ||grad_F||_inf=%.3e min_x_A=%.3e min_grad_A=%.3e",
                int(active.sum()),
                int(free.sum()),
                ginf,
                min_x_act,
                min_g_act,
            )

        return y_hat, x_hat, obj_hat

    except Exception as err:
        # If SciPy fails: log and return fallback.
        import logging
        logging.getLogger(__name__).warning("BFGS failed: %s", err)
        return fallback


# ------------------------------------------------------------------ #
# ---------- 3. Public wrapper: fit -------------------------------- #
# ------------------------------------------------------------------ #
def fit(
    A_period: NDArray[np.bool_],
    core_idx: list[int],
    *,
    objective: str = "nll",
    tol: float = 1e-6,
    max_iter: int = 1_000,
    x_init: Literal["zeros", "powerlaw"] = "zeros",
    alpha_pl: float = 2.5,
    seed: Optional[int] = None,
    start_params: Optional[Tuple[float, NDArray[np.float_]]] = None,
    bounds_bfgs: bool = True,
    obj_cfg: Optional[dict[str, object]] = None,
) -> Tuple[float, NDArray[np.float_], float]:
    """Estimate core-periphery model parameters.

    Core parameters ``x_i`` are estimated under ``x_i >= 0``.

    Parameters
    ----------
    A_period : ndarray bool, shape (N, N)
        Canonical adjacency matrix for one period.
    core_idx : list[int]
        Core-node indices.
    objective : {'nll', 'plsd_diag', 'plsd_maha'}, default 'nll'
        Objective function to optimize.
    tol : float, default 1e-6
        Gradient tolerance.
    max_iter : int, default 1000
        Maximum optimizer iterations.
    x_init : {'zeros', 'powerlaw'}, default 'zeros'
        Initialization strategy for ``x``.
    alpha_pl : float, default 2.5
        Pareto exponent for ``x_init='powerlaw'``.
    seed : int, optional
        Random generator seed.
    start_params : tuple(float, ndarray), optional
        Initial values ``(y0, x0)``.
    bounds_bfgs : bool, default True
        Enforce ``x_i >= 0`` in the L-BFGS-B solver.
    obj_cfg : dict, optional
        Configuration for PLSD (empirical counts, lambdas, freeze, etc.).

    Returns
    -------
    float
        Estimate of ``y``.
    ndarray float, shape (m,)
        Estimates of ``x``.
    float
        Value of the optimized objective (NLL or PLSD).

    Raises
    ------
    ValueError
        If ``A_period`` is not square or ``objective`` is unknown.

    Notes
    -----
    Assumes ``A_period`` is canonical (upper-tri bool).

    Complexity
    ----------
    ``O(N^2 * I)`` for NLL; ``O(N^3 * I)`` when the objective includes PLSD.

    Examples
    --------
    >>> A = np.zeros((2, 2), dtype=bool)
    >>> fit(A, [], objective='nll')
    (0.0, array([], dtype=float64), 0.0)
    """
    # -------------------------------------------------------------- #
    # Preliminary checks                                             #
    # -------------------------------------------------------------- #
    if A_period.shape[0] != A_period.shape[1]:
        raise ValueError("A_period must be square.")
    objective = objective.lower()
    if objective == "plsd":
        objective = "plsd_diag"
    if objective not in {"nll", "plsd_diag", "plsd_maha"}:
        raise ValueError("objective must be 'nll', 'plsd_diag', or 'plsd_maha'.")

    validate_upper_tri_bool(A_period)
    A_sym = np.logical_or(A_period, A_period.T)

    rng = np.random.default_rng(seed)

    if objective != "nll" and obj_cfg is None:
        raise ValueError("obj_cfg is required for PLSD.")
    if objective != "nll" and not bool(obj_cfg.get("freeze", True)):
        logger.warning("PLSD freeze=False uses approximate gradients and may be unstable.")

    core_mask = np.zeros(A_period.shape[0], dtype=bool)
    core_mask[core_idx] = True
    y_hat, x_hat, obj_hat = _fit_bfgs(
        A_period,
        core_mask,
        tol=tol,
        max_iter=max_iter,
        x_init_method=x_init,
        alpha_pl=alpha_pl,
        rng=rng,
        start_params=start_params,
        bounded=bounds_bfgs,
        objective=objective,
        obj_cfg=obj_cfg,
    )

    iu = np.triu_indices(A_sym.shape[0], 1)
    L_obs = float(A_period[iu].sum())
    k_obs = A_sym.sum(axis=1).astype(float)
    if objective == "nll":
        g_free, min_x_act, min_grad_act = _kkt_residuals(
            y_hat, x_hat, L_obs, k_obs, core_idx
        )
    p_PP = float(expit(y_hat))
    p_iP = expit(y_hat + x_hat) if x_hat.size else np.zeros(0)
    p_CC = (
        expit(y_hat + x_hat[:, None] + x_hat[None, :]) if x_hat.size else np.zeros((0, 0))
    )
    min_p_iP = float(np.min(p_iP)) if p_iP.size else float("nan")
    min_p_CC = float(np.min(p_CC)) if p_CC.size else float("nan")
    if p_iP.size and min_p_iP < p_PP - 1e-10:
        logger.warning("p_iP < p_PP: monotonicity violated")
    if p_CC.size and min_p_CC < p_PP - 1e-10:
        logger.warning("p_CC < p_PP: monotonicity violated")
    if p_CC.size and p_iP.size and min_p_CC < min_p_iP - 1e-10:
        logger.warning("p_CC < p_iP: monotonicity violated")
    logger.info(
        "Probs: p_PP=%.3f min_p_iP=%.3f min_p_CC=%.3f",
        p_PP,
        min_p_iP,
        min_p_CC,
    )
    if objective == "nll":
        logger.info(
            "KKT: g_free=%.3e min_x_A=%.3e min_grad_A=%.3e",
            g_free,
            min_x_act,
            min_grad_act,
        )
    return y_hat, x_hat, obj_hat


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 50
    m = 5
    core_idx = list(range(m))
    p_pp, p_cp, p_cc = 0.05, 0.1, 0.2
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if i < m and j < m:
                prob = p_cc
            elif i < m or j < m:
                prob = p_cp
            else:
                prob = p_pp
            if rng.random() < prob:
                A[i, j] = A[j, i] = True

    iu = np.triu_indices(N, 1)
    L_obs = float(A[iu].sum())
    k_obs = A.sum(axis=1).astype(float)

    y, x, nll = fit(A, core_idx, objective="nll", seed=0, bounds_bfgs=True)
    assert np.all(x >= -1e-12)
    p_PP = float(expit(y))
    p_iP = expit(y + x) if x.size else np.zeros(0)
    p_CC = expit(y + x[:, None] + x[None, :]) if x.size else np.zeros((0, 0))
    if p_iP.size:
        assert p_iP.min() >= p_PP - 1e-10
    if p_CC.size:
        assert p_CC.min() >= p_iP.min() - 1e-10
    g_free, _, min_grad = _kkt_residuals(y, x, L_obs, k_obs, core_idx)
    assert g_free < 1e-2
    active = x <= 1e-10
    free = ~active
    print(f"NLL={nll:.2f} |A|={active.sum()} |F|={free.sum()}")
