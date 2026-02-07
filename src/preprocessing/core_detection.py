"""Scan candidate core sizes on the current period and select m."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from src.model.inference_parameters import _prob_matrix, fit
from src.model.plsd import (
    _stabilize_sigma,
    estimate_lambda_from_scan,
    estimate_lambda_from_scan_maha,
    motif_empirical_counts,
    motif_moments,
)
from src.preprocessing.canonicalize import validate_upper_tri_bool

# ------------------------------------------------------------------ #
# Logger                                                             #
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def scan_core_sizes_current_period(
    A_period: NDArray[np.bool_],
    ranking: list[int],
    candidate_sizes: Iterable[int],
    *,
    criterion: str = "nll",
    complexity_penalty: str | None = None,
    tol: float = 1e-8,
    max_iter: int = 1_000,
    max_iter_plsd: int = 80,
    freeze_plsd_variances: bool = True,
) -> dict[str, object]:
    """Scan core sizes on the current period and select by criterion."""
    validate_upper_tri_bool(A_period)
    crit_raw = criterion.strip().lower().replace("-", "_")
    if crit_raw == "plsd":
        crit_raw = "plsd_diag"
    if crit_raw == "diagonal_plsd":
        crit_raw = "plsd_diag"
    if crit_raw == "mahalanobis_plsd":
        crit_raw = "plsd_maha"
    if crit_raw not in {"nll", "plsd_diag", "plsd_maha"}:
        raise ValueError("criterion must be one of: nll, plsd_diag, plsd_maha.")

    penalty_raw = (
        "none"
        if complexity_penalty is None
        else str(complexity_penalty).strip().lower().replace("-", "_")
    )
    if penalty_raw in {"", "none", "null"}:
        penalty_raw = "none"
    if penalty_raw not in {"none", "aic", "bic"}:
        raise ValueError("complexity_penalty must be one of: None, 'none', 'aic', 'bic'.")

    n_nodes = A_period.shape[0]
    n_obs = n_nodes * (n_nodes - 1) // 2
    log_n_obs = float(np.log(max(n_obs, 1)))

    sizes = sorted(set(int(k) for k in candidate_sizes))
    sizes = [k for k in sizes if 0 <= k <= len(ranking)]
    if 0 not in sizes:
        sizes = [0] + sizes

    eps_var = 1e-12
    eps_sigma = 1e-12
    det_min = eps_var * eps_var
    rho_delta = 1e-6
    max_sigma_bumps = 8
    sigma_fail_value = 1e12

    W_emp, T_emp = motif_empirical_counts(A_period)

    nlls: list[float] = []
    aics: list[float] = []
    bics: list[float] = []
    delta_t2: list[float] = []
    delta_w2: list[float] = []
    d2s: list[float] = []
    y_mle_list: list[float] = []
    x_mle_list: list[NDArray[np.float_]] = []
    core_list: list[list[int]] = []
    var_T_mle: list[float] = []
    var_W_mle: list[float] = []
    inv_sigma_mle: list[NDArray[np.float_] | None] = []
    sigma_failed_mle: list[bool] = []

    for m in sizes:
        core_nodes = ranking[:m]
        y_hat, x_hat, nll_hat = fit(
            A_period,
            core_nodes,
            objective="nll",
            tol=tol,
            max_iter=max_iter,
            x_init="powerlaw",
            alpha_pl=2.5,
            start_params=None,
        )
        core_mask = np.zeros(n_nodes, dtype=bool)
        core_mask[core_nodes] = True
        P = _prob_matrix(y_hat, x_hat, core_mask)
        mu_T, var_T, mu_W, var_W, cov_TW = motif_moments(P)
        var_T_clamped = max(float(var_T), eps_var)
        var_W_clamped = max(float(var_W), eps_var)
        delta_t2.append(float((T_emp - mu_T) ** 2 / (var_T_clamped + eps_var)))
        delta_w2.append(float((W_emp - mu_W) ** 2 / (var_W_clamped + eps_var)))

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
            d2s.append(float(sigma_fail_value))
            inv_sigma_mle.append(None)
            sigma_failed_mle.append(True)
        else:
            r = np.array([float(T_emp - mu_T), float(W_emp - mu_W)], dtype=float)
            v = inv_sigma @ r
            d2s.append(float(r @ v))
            inv_sigma_mle.append(inv_sigma)
            sigma_failed_mle.append(False)

        k_params = 1 + m
        nlls.append(float(nll_hat))
        aics.append(float(nll_hat + 2 * k_params))
        bics.append(float(nll_hat + k_params * log_n_obs))
        y_mle_list.append(float(y_hat))
        x_mle_list.append(np.asarray(x_hat, dtype=float))
        core_list.append(list(core_nodes))
        var_T_mle.append(var_T_clamped)
        var_W_mle.append(var_W_clamped)

    lambda_T, lambda_W = estimate_lambda_from_scan(nlls, delta_t2, delta_w2)
    lambda_M = estimate_lambda_from_scan_maha(nlls, d2s)

    plsd_mle_diag = [
        float(nll + lambda_T * dT + lambda_W * dW)
        for nll, dT, dW in zip(nlls, delta_t2, delta_w2)
    ]
    plsd_mle_maha = [
        float(nll + lambda_M * d2) for nll, d2 in zip(nlls, d2s)
    ]

    plsd_kind = "maha" if crit_raw == "plsd_maha" else "diag"
    plsd_mle = plsd_mle_maha if plsd_kind == "maha" else plsd_mle_diag
    lambda_T_out = float(lambda_T) if plsd_kind == "diag" else float("nan")
    lambda_W_out = float(lambda_W) if plsd_kind == "diag" else float("nan")
    lambda_M_out = float(lambda_M) if plsd_kind == "maha" else float("nan")

    plsd_pen = [float("nan")] * len(sizes)
    y_pen_list = [float("nan")] * len(sizes)
    x_pen_list: list[NDArray[np.float_]] = [
        np.array([], dtype=float) for _ in sizes
    ]

    if crit_raw in {"plsd_diag", "plsd_maha"}:
        for i, m in enumerate(sizes):
            core_nodes = core_list[i]
            start_params = (y_mle_list[i], x_mle_list[i])
            obj_cfg = {
                "T_emp": float(T_emp),
                "W_emp": float(W_emp),
                "freeze": bool(freeze_plsd_variances),
                "eps_var": eps_var,
                "eps_sigma": eps_sigma,
                "det_min": det_min,
                "rho_delta": rho_delta,
                "max_sigma_bumps": max_sigma_bumps,
                "sigma_fail_value": sigma_fail_value,
            }
            if crit_raw == "plsd_diag":
                obj_cfg.update({
                    "lambda_T": float(lambda_T),
                    "lambda_W": float(lambda_W),
                })
                if freeze_plsd_variances:
                    obj_cfg.update({
                        "frozen_var_T": float(var_T_mle[i]),
                        "frozen_var_W": float(var_W_mle[i]),
                    })
            else:
                obj_cfg.update({"lambda_M": float(lambda_M)})
                if freeze_plsd_variances:
                    obj_cfg.update({
                        "frozen_inv_sigma": inv_sigma_mle[i],
                        "sigma_failed": sigma_failed_mle[i],
                    })

            y_pen, x_pen, obj_pen = fit(
                A_period,
                core_nodes,
                objective=crit_raw,
                tol=tol,
                max_iter=max_iter_plsd,
                x_init="powerlaw",
                alpha_pl=2.5,
                start_params=start_params,
                obj_cfg=obj_cfg,
            )
            if (
                not np.isfinite(obj_pen)
                or not np.isfinite(y_pen)
                or (np.asarray(x_pen).size and not np.isfinite(x_pen).all())
            ):
                plsd_pen[i] = float("inf")
                y_pen_list[i] = y_mle_list[i]
                x_pen_list[i] = x_mle_list[i]
            else:
                plsd_pen[i] = float(obj_pen)
                y_pen_list[i] = float(y_pen)
                x_pen_list[i] = np.asarray(x_pen, dtype=float)

    if crit_raw == "nll":
        base_values = nlls
        y_list = y_mle_list
        x_list = x_mle_list
        plsd_selected = plsd_mle
    else:
        base_values = plsd_pen
        use_fallback = not np.isfinite(np.asarray(plsd_pen, dtype=float)).any()
        if use_fallback:
            logger.warning("PLSD penalized fit failed for all m; using MLE PLSD.")
            base_values = plsd_mle
            y_list = y_mle_list
            x_list = x_mle_list
            plsd_selected = plsd_mle
        else:
            y_list = y_pen_list
            x_list = x_pen_list
            plsd_selected = plsd_pen

    penalties = []
    for m in sizes:
        k_params = 1 + int(m)
        if penalty_raw == "aic":
            penalty = 2 * k_params
        elif penalty_raw == "bic":
            penalty = k_params * log_n_obs
        else:
            penalty = 0.0
        penalties.append(float(penalty))

    nll_penalized = [
        float(nll + pen) for nll, pen in zip(nlls, penalties)
    ]
    plsd_penalized = [
        float(plsd + pen) for plsd, pen in zip(plsd_selected, penalties)
    ]
    crit_values = [
        float(base + pen) for base, pen in zip(base_values, penalties)
    ]

    crit_arr = np.asarray(crit_values, dtype=float)
    crit_arr[~np.isfinite(crit_arr)] = np.inf
    best_idx = int(np.argmin(crit_arr)) if crit_arr.size else 0
    logger.info(
        "scan_core_sizes_current_period - criterion=%s, penalty=%s, m*=%d",
        crit_raw,
        penalty_raw,
        sizes[best_idx] if sizes else 0,
    )

    return {
        "m_values": sizes,
        "core_nodes": core_list,
        "y_hat": y_list,
        "x_hat": x_list,
        "nll": nlls,
        "aic": aics,
        "bic": bics,
        "plsd_mle": plsd_mle,
        "plsd_pen": plsd_pen,
        "plsd_selected": plsd_selected,
        "nll_penalized": nll_penalized,
        "plsd_penalized": plsd_penalized,
        "penalty": penalties,
        "delta_t2": delta_t2,
        "delta_w2": delta_w2,
        "d2": d2s,
        "lambda_T": lambda_T_out,
        "lambda_W": lambda_W_out,
        "lambda_M": lambda_M_out,
        "criterion": crit_raw,
        "complexity_penalty": penalty_raw,
        "plsd_kind": plsd_kind,
        "freeze_plsd": bool(freeze_plsd_variances),
        "selected_index": best_idx,
    }
