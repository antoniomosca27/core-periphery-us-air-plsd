"""Plot time-series diagnostics from period-level outputs."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import hashlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from src.analysis.core_stability import jaccard
from src.model.simulate_network import sample as sample_network
from src.preprocessing.canonicalize import validate_upper_tri_bool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = _PROJECT_ROOT / "figures"
_PATH_LIMIT = 240
_LONGEST_SUBDIR = len("mesoscopic metrics and distances")

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _save(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def _series_plot(
    df: pd.DataFrame,
    x: str,
    emp: str,
    sim_mean: str,
    sim_std: str,
    name: str,
    ylab: str,
    out_dir: Path,
) -> None:
    if not {emp, sim_mean, sim_std}.issubset(df.columns):
        return
    fig, ax = plt.subplots()
    ax.plot(df[x], df[emp], label="empirical", color="tab:blue")
    ax.plot(df[x], df[sim_mean], label="simulated", color="tab:orange", ls="--")
    ax.fill_between(
        df[x],
        df[sim_mean] - df[sim_std],
        df[sim_mean] + df[sim_std],
        color="orange",
        alpha=0.3,
    )
    ax.set_ylabel(ylab)
    ax.set_xlabel(x)
    ax.legend()
    _save(fig, name, out_dir)


def _line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str,
    ylab: str,
    out_dir: Path,
    *,
    marker: bool = False,
    hline: float | None = None,
) -> None:
    if not {x, y}.issubset(df.columns):
        return
    fig, ax = plt.subplots()
    style = "-o" if marker else "-"
    ax.plot(df[x], df[y], style, color="tab:blue")
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_ylabel(ylab)
    ax.set_xlabel(x)
    _save(fig, name, out_dir)


def _parse_core_nodes(value) -> list[int]:
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    return []


def _parse_x_hat(value) -> list[float]:
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    return []


def _degree_from_upper(A: np.ndarray) -> np.ndarray:
    if A.ndim == 2:
        return (A.sum(axis=0) + A.sum(axis=1)).astype(int, copy=False)
    if A.ndim == 3:
        return (A.sum(axis=2) + A.sum(axis=1)).astype(int, copy=False)
    raise ValueError("A must be 2D or 3D upper-tri adjacency.")


def _select_indices(n_rows: int, n_select: int) -> np.ndarray:
    if n_rows <= 0 or n_select <= 0:
        return np.array([], dtype=int)
    n_select = min(n_select, n_rows)
    return np.unique(np.linspace(0, n_rows - 1, n_select, dtype=int))


def _safe_run_id(run_id: str) -> str:
    base_len = len(str(FIGURES_DIR / run_id))
    if base_len + _LONGEST_SUBDIR + 2 <= _PATH_LIMIT:
        return run_id
    h = hashlib.sha1(run_id.encode()).hexdigest()[:10]
    safe = f"run_{h}"
    logger.warning("run_id is too long for Windows paths; using '%s' for figures.", safe)
    return safe


def _resolve_run_id(df: pd.DataFrame, run_id: str | None) -> str:
    if run_id:
        return _safe_run_id(run_id)
    if isinstance(df, pd.DataFrame):
        attr = df.attrs.get("run_id")
        if isinstance(attr, str) and attr:
            return _safe_run_id(attr)
    raise ValueError("run_id is required to save plots under figures/<run_id>.")


def plot_degree_distributions(
    A_time: np.ndarray,
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    n_periods: int = 10,
    n_sim: int = 1000,
    seed_MC: int = 0,
) -> None:
    """Plot empirical and simulated degree distributions for selected periods."""
    if df.empty:
        return
    if A_time.ndim != 3 or A_time.shape[0] != A_time.shape[1]:
        raise ValueError("A_time must have shape (N, N, T).")
    run_id = _resolve_run_id(df, run_id)
    if "period_pos" not in df.columns:
        return
    if not {"core_nodes", "x_hat", "y_hat"}.issubset(df.columns):
        return

    out_dir = FIGURES_DIR / run_id / "degree distributions"
    df_sorted = df.sort_values("period_pos").reset_index(drop=True)
    indices = _select_indices(len(df_sorted), n_periods)
    n_sim = max(1, int(n_sim))

    for idx in indices:
        row = df_sorted.iloc[idx]
        period_pos = int(row["period_pos"])
        if period_pos < 1 or period_pos > A_time.shape[2]:
            continue

        A_t = A_time[:, :, period_pos - 1]
        validate_upper_tri_bool(A_t)
        deg_full = _degree_from_upper(A_t)
        active_idx = np.flatnonzero(deg_full > 0)
        if active_idx.size == 0:
            continue

        A_active = A_t[np.ix_(active_idx, active_idx)]
        validate_upper_tri_bool(A_active)

        core_nodes = _parse_core_nodes(row["core_nodes"])
        x_hat = _parse_x_hat(row["x_hat"])
        y_hat = float(row["y_hat"]) if pd.notna(row["y_hat"]) else float("nan")
        if not np.isfinite(y_hat):
            continue

        active_pos = {node: pos for pos, node in enumerate(active_idx.tolist())}
        filtered = [(node, x) for node, x in zip(core_nodes, x_hat) if node in active_pos]
        core_sub = [active_pos[node] for node, _ in filtered]
        x_vec = np.array([x for _, x in filtered], dtype=float)
        if len(core_sub) != x_vec.size:
            continue

        n_active = int(active_idx.size)
        rng = np.random.default_rng(np.random.SeedSequence([seed_MC, period_pos, 991]))
        sims = sample_network(
            y_hat,
            x_vec,
            core_sub,
            n_nodes=n_active,
            n_sim=n_sim,
            rng=rng,
        )

        deg_emp = _degree_from_upper(A_active)
        if sims.ndim == 2:
            deg_sims = _degree_from_upper(sims)[None, :]
        else:
            deg_sims = _degree_from_upper(sims)

        fig, ax = plt.subplots()
        max_k = int(max(deg_emp.max(), deg_sims.max()))
        bins = np.arange(max_k + 2) - 0.5
        ax.hist(
            deg_emp,
            bins=bins,
            density=True,
            color="tab:blue",
            alpha=0.7,
            label="empirical",
        )
        ax.hist(
            deg_sims.ravel(),
            bins=bins,
            density=True,
            color="tab:orange",
            alpha=0.6,
            label="simulated mean",
        )
        ax.set_xlabel("Degree")
        ax.set_ylabel("P(k)")
        ax.set_title(f"Degree distribution - period {period_pos}")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlim(-0.5, max_k + 0.5)
        ax.legend()
        _save(fig, f"degree_dist_period_{period_pos}", out_dir)


def plot_series(df: pd.DataFrame, *, run_id: str | None = None) -> None:
    """Generate standard time-series plots from big_table data."""
    if df.empty:
        return
    run_id = _resolve_run_id(df, run_id)
    if "period_pos" not in df.columns:
        return
    x = "period_pos"
    base_dir = FIGURES_DIR / run_id
    model_dir = base_dir / "model metrics"
    motifs_dir = base_dir / "motifs"
    meso_dir = base_dir / "mesoscopic metrics and distances"
    risk_dir = base_dir / "risk metrics"

    # Core size
    _line_plot(df, x, "m_selected", "core_size", "Core size", model_dir, marker=True)

    # Network size (active nodes)
    _line_plot(
        df,
        x,
        "n_nodes_active",
        "network_size",
        "Active nodes",
        model_dir,
        marker=True,
    )

    # Core stability (Jaccard) using consecutive core sets
    if {"core_nodes", x}.issubset(df.columns) and len(df) > 1:
        core_lists = [_parse_core_nodes(v) for v in df["core_nodes"]]
        jac = [jaccard(core_lists[i], core_lists[i + 1]) for i in range(len(core_lists) - 1)]
        fig, ax = plt.subplots()
        ax.plot(df[x].iloc[1:], jac, "-o", color="tab:blue")
        ax.set_ylabel("Jaccard core")
        ax.set_xlabel(x)
        _save(fig, "core_jaccard", model_dir)

    # y_hat and x_hat summary
    _line_plot(df, x, "y_hat", "y_series", "y", model_dir)

    if {"x_hat_mean", "x_hat_std", x}.issubset(df.columns):
        fig, ax = plt.subplots()
        ax.plot(df[x], df["x_hat_mean"], color="tab:blue")
        ax.fill_between(
            df[x],
            df["x_hat_mean"] - df["x_hat_std"],
            df["x_hat_mean"] + df["x_hat_std"],
            color="lightblue",
            alpha=0.4,
        )
        ax.set_ylabel("x mean")
        ax.set_xlabel(x)
        _save(fig, "x_mean", model_dir)

    # PLSD lambdas
    if {"lambda_M", x}.issubset(df.columns) and df["lambda_M"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(df[x], df["lambda_M"], label="lambda_M", color="tab:purple")
        ax.set_ylabel("PLSD lambda")
        ax.set_xlabel(x)
        ax.legend()
        _save(fig, "plsd_lambda_M", model_dir)
    elif {"lambda_T", "lambda_W", x}.issubset(df.columns):
        if df["lambda_T"].notna().any() or df["lambda_W"].notna().any():
            fig, ax = plt.subplots()
            ax.plot(df[x], df["lambda_T"], label="lambda_T", color="tab:blue")
            ax.plot(df[x], df["lambda_W"], label="lambda_W", color="tab:orange")
            ax.set_ylabel("PLSD lambda")
            ax.set_xlabel(x)
            ax.legend()
            _save(fig, "plsd_lambdas", model_dir)

    # Selection criteria
    if {x, "nll_selected"}.issubset(df.columns):
        crit_raw = "nll"
        if "criterion" in df.columns:
            crit_series = df["criterion"].dropna()
            if not crit_series.empty:
                crit_raw = str(crit_series.iloc[0])
        crit_raw = crit_raw.strip().lower().replace("-", "_")
        if crit_raw == "plsd":
            crit_raw = "plsd_diag"
        if crit_raw == "diagonal_plsd":
            crit_raw = "plsd_diag"
        if crit_raw == "mahalanobis_plsd":
            crit_raw = "plsd_maha"

        penalty_raw = "none"
        if "complexity_penalty" in df.columns:
            penalty_series = df["complexity_penalty"].dropna()
            if not penalty_series.empty:
                penalty_raw = str(penalty_series.iloc[0])
        penalty_raw = penalty_raw.strip().lower().replace("-", "_")
        if penalty_raw in {"", "none", "null"}:
            penalty_raw = "none"

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plotted = False
        pen_color = {"aic": "tab:orange", "bic": "tab:green"}

        if df["nll_selected"].notna().any():
            ax.plot(df[x], df["nll_selected"], label="NLL", color="tab:blue")
            plotted = True

        if crit_raw == "nll" and penalty_raw in pen_color:
            if "nll_penalized_selected" in df.columns and df["nll_penalized_selected"].notna().any():
                label = f"NLL+{penalty_raw.upper()}"
                ax.plot(
                    df[x],
                    df["nll_penalized_selected"],
                    label=label,
                    color=pen_color[penalty_raw],
                )
                plotted = True

        if crit_raw in {"plsd_diag", "plsd_maha"}:
            if "plsd_selected" in df.columns and df["plsd_selected"].notna().any():
                ax.plot(df[x], df["plsd_selected"], label="PLSD", color="tab:red")
                plotted = True
            if penalty_raw in pen_color:
                if "plsd_penalized_selected" in df.columns and df["plsd_penalized_selected"].notna().any():
                    label = f"PLSD+{penalty_raw.upper()}"
                    ax.plot(
                        df[x],
                        df["plsd_penalized_selected"],
                        label=label,
                        color=pen_color[penalty_raw],
                    )
                    plotted = True

        if plotted:
            ax.set_ylabel("Criterion")
            ax.set_xlabel(x)
            ax.legend()
            _save(fig, "criteria", model_dir)
        else:
            plt.close(fig)

    # L, W, T
    _series_plot(df, x, "L_emp", "L_sim_mean", "L_sim_std", "L_series", "L", motifs_dir)
    _series_plot(df, x, "W_emp", "W_sim_mean", "W_sim_std", "W_series", "W", motifs_dir)
    _series_plot(df, x, "T_emp", "T_sim_mean", "T_sim_std", "T_series", "T", motifs_dir)

    # L by class
    _series_plot(
        df,
        x,
        "L_cc_emp",
        "L_cc_sim_mean",
        "L_cc_sim_std",
        "L_cc_series",
        "L_cc",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "L_cp_emp",
        "L_cp_sim_mean",
        "L_cp_sim_std",
        "L_cp_series",
        "L_cp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "L_pp_emp",
        "L_pp_sim_mean",
        "L_pp_sim_std",
        "L_pp_series",
        "L_pp",
        motifs_dir,
    )

    # W by class
    _series_plot(
        df,
        x,
        "W_ccc_emp",
        "W_ccc_sim_mean",
        "W_ccc_sim_std",
        "W_ccc_series",
        "W_ccc",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "W_ccp_emp",
        "W_ccp_sim_mean",
        "W_ccp_sim_std",
        "W_ccp_series",
        "W_ccp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "W_cpc_emp",
        "W_cpc_sim_mean",
        "W_cpc_sim_std",
        "W_cpc_series",
        "W_cpc",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "W_cpp_emp",
        "W_cpp_sim_mean",
        "W_cpp_sim_std",
        "W_cpp_series",
        "W_cpp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "W_pcp_emp",
        "W_pcp_sim_mean",
        "W_pcp_sim_std",
        "W_pcp_series",
        "W_pcp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "W_ppp_emp",
        "W_ppp_sim_mean",
        "W_ppp_sim_std",
        "W_ppp_series",
        "W_ppp",
        motifs_dir,
    )

    # T by class
    _series_plot(
        df,
        x,
        "T_ccc_emp",
        "T_ccc_sim_mean",
        "T_ccc_sim_std",
        "T_ccc_series",
        "T_ccc",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "T_ccp_emp",
        "T_ccp_sim_mean",
        "T_ccp_sim_std",
        "T_ccp_series",
        "T_ccp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "T_cpp_emp",
        "T_cpp_sim_mean",
        "T_cpp_sim_std",
        "T_cpp_series",
        "T_cpp",
        motifs_dir,
    )
    _series_plot(
        df,
        x,
        "T_ppp_emp",
        "T_ppp_sim_mean",
        "T_ppp_sim_std",
        "T_ppp_series",
        "T_ppp",
        motifs_dir,
    )

    # C, r, Q
    _series_plot(df, x, "C_emp", "C_sim_mean", "C_sim_std", "C_series", "C", meso_dir)
    _series_plot(df, x, "r_emp", "r_sim_mean", "r_sim_std", "r_series", "r", meso_dir)
    _series_plot(df, x, "Q_emp", "Q_sim_mean", "Q_sim_std", "Q_series", "Q", meso_dir)

    # Distances
    _series_plot(
        df,
        x,
        "ASPL_emp",
        "ASPL_sim_mean",
        "ASPL_sim_std",
        "ASPL_series",
        "ASPL",
        meso_dir,
    )
    _series_plot(
        df,
        x,
        "Diameter_emp",
        "Diameter_sim_mean",
        "Diameter_sim_std",
        "Diameter_series",
        "Diameter",
        meso_dir,
    )

    # Spectral radius
    _series_plot(
        df,
        x,
        "spectral_radius_emp",
        "spectral_radius_sim_mean",
        "spectral_radius_sim_std",
        "spectral_radius_series",
        "Spectral radius",
        risk_dir,
    )

    # Triangle z-scores
    _line_plot(df, x, "z_T", "z_T_series", "z_T", risk_dir, hline=0.0)
    _line_plot(df, x, "z_T_ccc", "z_T_ccc_series", "z_T_ccc", risk_dir, hline=0.0)
    _line_plot(df, x, "z_T_ccp", "z_T_ccp_series", "z_T_ccp", risk_dir, hline=0.0)
    _line_plot(df, x, "z_T_cpp", "z_T_cpp_series", "z_T_cpp", risk_dir, hline=0.0)
    _line_plot(df, x, "z_T_ppp", "z_T_ppp_series", "z_T_ppp", risk_dir, hline=0.0)

    # KS p-value
    if {"ks_p_median", x}.issubset(df.columns):
        fig, ax = plt.subplots()
        ax.plot(df[x], df["ks_p_median"], "-o", color="tab:blue")
        ax.axhline(0.05, color="red", ls="--")
        ax.set_ylabel("KS p-value (median)")
        ax.set_xlabel(x)
        _save(fig, "ks_p_median", model_dir)
