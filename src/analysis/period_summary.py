"""Run the full inference and diagnostics pipeline for one period."""

from __future__ import annotations

import json
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.analysis.ks import ks_degree_sampled
from src.model.metrics import (
    assortativity_degree,
    batch_compute,
    compute_metrics,
    modularity_core_periphery,
    path_metrics,
)
from src.model.simulate_network import sample as sample_network
from src.preprocessing.canonicalize import validate_upper_tri_bool
from src.preprocessing.core_detection import scan_core_sizes_current_period
from src.preprocessing.ranking import rank_nodes_for_period

# ------------------------------------------------------------------ #
# Logger                                                             #
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _degree_from_upper(A: NDArray[np.bool_]) -> NDArray[np.int_]:
    if A.ndim == 2:
        return (A.sum(axis=0) + A.sum(axis=1)).astype(int, copy=False)
    if A.ndim == 3:
        return (A.sum(axis=2) + A.sum(axis=1)).astype(int, copy=False)
    raise ValueError("A must be 2D or 3D upper-tri adjacency.")


def _spectral_radius(A: NDArray[np.bool_]) -> float:
    if A.size == 0:
        return float("nan")
    A_sym = np.logical_or(A, A.T).astype(float, copy=False)
    return float(np.linalg.eigvalsh(A_sym).max()) if A_sym.size else float("nan")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")
    return mean, std


def _z_score(emp: float, mean: float, std: float) -> float:
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return (float(emp) - float(mean)) / float(std)


def summarize_period(
    A_time: NDArray[np.bool_],
    period_pos: int,
    *,
    batch_size: int = 20,
    k_range: Iterable[int],
    criterion: str = "nll",
    complexity_penalty: str | None = None,
    tol: float = 1e-8,
    max_iter: int = 1_000,
    max_iter_plsd: int = 80,
    freeze_plsd_variances: bool = True,
    R: int = 1_000,
    seed_MC: int = 0,
    n_KS: int = 200,
    seed_KS: int = 1,
    period_map: Optional[dict[int, tuple[pd.Timestamp, pd.Timestamp]]] = None,
    period_ids: Optional[dict[int, object]] = None,
    period_labels: Optional[dict[int, str]] = None,
) -> dict[str, object]:
    """Run the complete pipeline on one period and return a table row."""
    if A_time.ndim != 3 or A_time.shape[0] != A_time.shape[1]:
        raise ValueError("A_time must have shape (N, N, T).")
    n_nodes = A_time.shape[0]
    n_periods = A_time.shape[2]
    if period_pos < 1 or period_pos > n_periods:
        raise ValueError("period_pos out of range.")
    if period_pos <= batch_size:
        raise ValueError("period_pos must be > batch_size for ranking.")
    if R <= 0:
        raise ValueError("R must be positive.")
    if n_KS <= 0:
        raise ValueError("n_KS must be positive.")

    if period_map is None:
        period_map = {
            idx + 1: (pd.NaT, pd.NaT) for idx in range(n_periods)
        }

    def _fmt_date(value: pd.Timestamp) -> str:
        if value is None or pd.isna(value):
            return ""
        return pd.Timestamp(value).date().isoformat()

    period_idx = period_pos - 1
    A_t = A_time[:, :, period_idx]
    validate_upper_tri_bool(A_t)

    # Active nodes from current-period degrees.
    deg_full = _degree_from_upper(A_t)
    active_idx = np.flatnonzero(deg_full > 0)
    n_active = int(active_idx.size)

    if n_active == 0:
        start_dt, end_dt = period_map[period_pos]
        penalty_label = (
            "none"
            if complexity_penalty is None
            else str(complexity_penalty).strip().lower()
        )
        period_id = period_ids.get(period_pos) if period_ids else None
        period_label = period_labels.get(period_pos) if period_labels else None
        return {
            "period_pos": int(period_pos),
            "period_start_date": _fmt_date(start_dt),
            "period_end_date": _fmt_date(end_dt),
            "period_id": period_id,
            "period_label": period_label,
            "n_nodes_total": int(n_nodes),
            "n_nodes_active": 0,
            "top_rank_node_id": np.nan,
            "top_rank_score": np.nan,
            "m_selected": 0,
            "core_nodes": json.dumps([]),
            "x_hat": json.dumps([]),
            "y_hat": np.nan,
            "x_hat_mean": np.nan,
            "x_hat_std": np.nan,
            "x_hat_min": np.nan,
            "x_hat_max": np.nan,
            "nll_selected": np.nan,
            "aic_selected": np.nan,
            "bic_selected": np.nan,
            "plsd_selected": np.nan,
            "plsd_mle": np.nan,
            "plsd_pen": np.nan,
            "penalty_selected": np.nan,
            "nll_penalized_selected": np.nan,
            "plsd_penalized_selected": np.nan,
            "lambda_T": np.nan,
            "lambda_W": np.nan,
            "lambda_M": np.nan,
            "criterion": str(criterion),
            "complexity_penalty": penalty_label,
            "plsd_kind": np.nan,
            "freeze_plsd": bool(freeze_plsd_variances),
            "ks_p_median": np.nan,
            "z_T": np.nan,
            "z_T_ccc": np.nan,
            "z_T_ccp": np.nan,
            "z_T_cpp": np.nan,
            "z_T_ppp": np.nan,
        }

    # Ranking on the previous window, restricted to active nodes.
    ranking, D_scores = rank_nodes_for_period(
        A_time,
        period_pos,
        batch_size,
        active_idx,
    )
    top_rank_node_id = float(ranking[0]) if ranking else np.nan
    top_rank_score = float(D_scores[int(ranking[0])]) if ranking else np.nan

    # Induced subgraph on active nodes in canonical form.
    A_active = A_t[np.ix_(active_idx, active_idx)]
    validate_upper_tri_bool(A_active)

    # Map ranking to active-node positions.
    active_pos = {node: pos for pos, node in enumerate(active_idx.tolist())}
    ranking_sub = [active_pos[node] for node in ranking if node in active_pos]

    # Candidate sizes include m=0 and are truncated by active nodes.
    sizes = [int(k) for k in k_range if int(k) <= n_active]
    if 0 not in sizes:
        sizes = [0] + sizes

    scan = scan_core_sizes_current_period(
        A_active,
        ranking_sub,
        sizes,
        criterion=criterion,
        complexity_penalty=complexity_penalty,
        tol=tol,
        max_iter=max_iter,
        max_iter_plsd=max_iter_plsd,
        freeze_plsd_variances=freeze_plsd_variances,
    )
    sel_idx = scan["selected_index"]
    m_selected = scan["m_values"][sel_idx]
    core_sub = scan["core_nodes"][sel_idx]
    y_hat = scan["y_hat"][sel_idx]
    x_hat = scan["x_hat"][sel_idx]
    nll_selected = scan["nll"][sel_idx]
    aic_selected = scan["aic"][sel_idx]
    bic_selected = scan["bic"][sel_idx]
    plsd_selected = scan["plsd_selected"][sel_idx]
    plsd_mle = scan["plsd_mle"][sel_idx]
    plsd_pen = scan["plsd_pen"][sel_idx]
    penalty_selected = scan["penalty"][sel_idx]
    nll_penalized_selected = scan["nll_penalized"][sel_idx]
    plsd_penalized_selected = scan["plsd_penalized"][sel_idx]
    lambda_T = scan["lambda_T"]
    lambda_W = scan["lambda_W"]
    lambda_M = scan["lambda_M"]
    plsd_kind = scan["plsd_kind"]
    freeze_plsd = scan["freeze_plsd"]
    criterion_used = scan["criterion"]
    penalty_used = scan["complexity_penalty"]

    core_nodes_global = [int(active_idx[i]) for i in core_sub]

    # Monte Carlo simulations, seeded per replicate.
    sims = np.zeros((R, n_active, n_active), dtype=bool)
    for r in range(R):
        rng = np.random.default_rng(np.random.SeedSequence([seed_MC, period_pos, r]))
        sims[r] = sample_network(
            y_hat,
            x_hat,
            core_sub,
            n_nodes=n_active,
            n_sim=1,
            rng=rng,
        )

    # Empirical metrics.
    emp_metrics = compute_metrics(A_active, core_idx=core_sub)
    Q_emp = modularity_core_periphery(A_active, core_sub)["Q"]
    r_emp = assortativity_degree(A_active)["r"]
    path_emp = path_metrics(A_active)
    spectral_emp = _spectral_radius(A_active)

    # Simulated metrics (counts).
    sim_metrics = batch_compute(sims, core_idx=core_sub)
    L_sim_mean, L_sim_std = _mean_std(sim_metrics["L"])
    W_sim_mean, W_sim_std = _mean_std(sim_metrics["W"])
    T_sim_mean, T_sim_std = _mean_std(sim_metrics["T"])

    # Simulated assortativity, modularity, path metrics, and spectral radius.
    r_sims = []
    Q_sims = []
    ASPL_sims = []
    Diam_sims = []
    spectral_sims = []
    for S in sims:
        r_sims.append(assortativity_degree(S)["r"])
        Q_sims.append(modularity_core_periphery(S, core_sub)["Q"])
        pm = path_metrics(S)
        ASPL_sims.append(pm["aspl_gc"])
        Diam_sims.append(pm["diam_gc"])
        spectral_sims.append(_spectral_radius(S))

    r_sim_mean, r_sim_std = _mean_std(r_sims)
    Q_sim_mean, Q_sim_std = _mean_std(Q_sims)
    ASPL_sim_mean, ASPL_sim_std = _mean_std(ASPL_sims)
    Diam_sim_mean, Diam_sim_std = _mean_std(Diam_sims)
    spectral_sim_mean, spectral_sim_std = _mean_std(spectral_sims)

    # KS on degrees with deterministic subset selection.
    deg_emp = _degree_from_upper(A_active)
    deg_sims = _degree_from_upper(sims)
    _, ks_p_median, _ = ks_degree_sampled(
        deg_emp,
        deg_sims,
        n_KS=n_KS,
        seed_KS=seed_KS,
        period_pos=period_pos,
    )

    # Output row.
    start_dt, end_dt = period_map[period_pos]
    period_id = period_ids.get(period_pos) if period_ids else None
    period_label = period_labels.get(period_pos) if period_labels else None
    row: dict[str, object] = {
        "period_pos": int(period_pos),
        "period_start_date": _fmt_date(start_dt),
        "period_end_date": _fmt_date(end_dt),
        "period_id": period_id,
        "period_label": period_label,
        "n_nodes_total": int(n_nodes),
        "n_nodes_active": int(n_active),
        "top_rank_node_id": top_rank_node_id,
        "top_rank_score": top_rank_score,
        "m_selected": int(m_selected),
        "core_nodes": json.dumps(core_nodes_global),
        "x_hat": json.dumps([float(v) for v in x_hat]),
        "y_hat": float(y_hat),
        "x_hat_mean": float(np.mean(x_hat)) if len(x_hat) else np.nan,
        "x_hat_std": float(np.std(x_hat, ddof=1)) if len(x_hat) > 1 else np.nan,
        "x_hat_min": float(np.min(x_hat)) if len(x_hat) else np.nan,
        "x_hat_max": float(np.max(x_hat)) if len(x_hat) else np.nan,
        "nll_selected": float(nll_selected),
        "aic_selected": float(aic_selected),
        "bic_selected": float(bic_selected),
        "plsd_selected": float(plsd_selected),
        "plsd_mle": float(plsd_mle),
        "plsd_pen": float(plsd_pen),
        "penalty_selected": float(penalty_selected),
        "nll_penalized_selected": float(nll_penalized_selected),
        "plsd_penalized_selected": float(plsd_penalized_selected),
        "lambda_T": float(lambda_T),
        "lambda_W": float(lambda_W),
        "lambda_M": float(lambda_M),
        "criterion": str(criterion_used),
        "complexity_penalty": str(penalty_used),
        "plsd_kind": str(plsd_kind),
        "freeze_plsd": bool(freeze_plsd),
        "L_emp": float(emp_metrics["L"]),
        "W_emp": float(emp_metrics["W"]),
        "T_emp": float(emp_metrics["T"]),
        "L_cc_emp": float(emp_metrics.get("L_cc", np.nan)),
        "L_cp_emp": float(emp_metrics.get("L_cp", np.nan)),
        "L_pp_emp": float(emp_metrics.get("L_pp", np.nan)),
        "W_ccc_emp": float(emp_metrics.get("W_ccc", np.nan)),
        "W_ccp_emp": float(emp_metrics.get("W_ccp", np.nan)),
        "W_cpc_emp": float(emp_metrics.get("W_cpc", np.nan)),
        "W_cpp_emp": float(emp_metrics.get("W_cpp", np.nan)),
        "W_pcp_emp": float(emp_metrics.get("W_pcp", np.nan)),
        "W_ppp_emp": float(emp_metrics.get("W_ppp", np.nan)),
        "T_ccc_emp": float(emp_metrics.get("T_ccc", np.nan)),
        "T_ccp_emp": float(emp_metrics.get("T_ccp", np.nan)),
        "T_cpp_emp": float(emp_metrics.get("T_cpp", np.nan)),
        "T_ppp_emp": float(emp_metrics.get("T_ppp", np.nan)),
        "C_emp": float(emp_metrics.get("clustering", np.nan)),
        "r_emp": float(r_emp),
        "Q_emp": float(Q_emp),
        "ASPL_emp": float(path_emp["aspl_gc"]),
        "Diameter_emp": float(path_emp["diam_gc"]),
        "spectral_radius_emp": float(spectral_emp),
        "L_sim_mean": L_sim_mean,
        "L_sim_std": L_sim_std,
        "W_sim_mean": W_sim_mean,
        "W_sim_std": W_sim_std,
        "T_sim_mean": T_sim_mean,
        "T_sim_std": T_sim_std,
        "C_sim_mean": float(np.nanmean(sim_metrics["clustering"])),
        "C_sim_std": float(np.nanstd(sim_metrics["clustering"], ddof=1))
        if sim_metrics["clustering"].size > 1
        else np.nan,
        "r_sim_mean": r_sim_mean,
        "r_sim_std": r_sim_std,
        "Q_sim_mean": Q_sim_mean,
        "Q_sim_std": Q_sim_std,
        "ASPL_sim_mean": ASPL_sim_mean,
        "ASPL_sim_std": ASPL_sim_std,
        "Diameter_sim_mean": Diam_sim_mean,
        "Diameter_sim_std": Diam_sim_std,
        "spectral_radius_sim_mean": spectral_sim_mean,
        "spectral_radius_sim_std": spectral_sim_std,
        "ks_p_median": float(ks_p_median),
    }

    # Simulated class metrics
    for key in [
        "L_cc",
        "L_cp",
        "L_pp",
        "W_ccc",
        "W_ccp",
        "W_cpc",
        "W_cpp",
        "W_pcp",
        "W_ppp",
        "T_ccc",
        "T_ccp",
        "T_cpp",
        "T_ppp",
    ]:
        sim_key = sim_metrics.get(key)
        if sim_key is None:
            continue
        mean, std = _mean_std(sim_key)
        row[f"{key}_sim_mean"] = mean
        row[f"{key}_sim_std"] = std

    row["z_T"] = _z_score(row["T_emp"], row["T_sim_mean"], row["T_sim_std"])
    row["z_T_ccc"] = _z_score(
        row.get("T_ccc_emp", np.nan),
        row.get("T_ccc_sim_mean", np.nan),
        row.get("T_ccc_sim_std", np.nan),
    )
    row["z_T_ccp"] = _z_score(
        row.get("T_ccp_emp", np.nan),
        row.get("T_ccp_sim_mean", np.nan),
        row.get("T_ccp_sim_std", np.nan),
    )
    row["z_T_cpp"] = _z_score(
        row.get("T_cpp_emp", np.nan),
        row.get("T_cpp_sim_mean", np.nan),
        row.get("T_cpp_sim_std", np.nan),
    )
    row["z_T_ppp"] = _z_score(
        row.get("T_ppp_emp", np.nan),
        row.get("T_ppp_sim_mean", np.nan),
        row.get("T_ppp_sim_std", np.nan),
    )

    return row
