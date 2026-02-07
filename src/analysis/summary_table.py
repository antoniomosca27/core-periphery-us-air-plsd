"""
----------------------------------------------------------------------
FILE: src/analysis/summary_table.py
----------------------------------------------------------------------

Purpose
-------
Compute the one-row small_table summary from the big_table results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nrmse_relative(emp: pd.Series, sim: pd.Series) -> float:
    mask = emp > 0
    if not mask.any():
        return float("nan")
    rel_err = (sim[mask] - emp[mask]) / emp[mask]
    return float(np.sqrt(np.mean(rel_err**2)))


def _mae(emp: pd.Series, sim: pd.Series) -> float:
    err = (sim - emp).abs()
    return float(err.mean())


def _bias(emp: pd.Series, sim: pd.Series) -> float:
    diff = sim - emp
    diff = diff[np.isfinite(diff)]
    if diff.empty:
        return float("nan")
    return float(diff.mean())


def build_small_table(big_table: pd.DataFrame) -> pd.DataFrame:
    """Build the one-row summary table from period results."""
    if big_table.empty:
        raise ValueError("big_table must not be empty.")

    out = {
        "nRMSE_L": _nrmse_relative(big_table["L_emp"], big_table["L_sim_mean"]),
        "nRMSE_W": _nrmse_relative(big_table["W_emp"], big_table["W_sim_mean"]),
        "nRMSE_T": _nrmse_relative(big_table["T_emp"], big_table["T_sim_mean"]),
        "nMAE_C": _mae(big_table["C_emp"], big_table["C_sim_mean"]),
        "nMAE_r": _mae(big_table["r_emp"], big_table["r_sim_mean"]),
        "nMAE_Q": _mae(big_table["Q_emp"], big_table["Q_sim_mean"]),
        "bias_ASPL": _bias(big_table["ASPL_emp"], big_table["ASPL_sim_mean"]),
        "bias_Diameter": _bias(
            big_table["Diameter_emp"], big_table["Diameter_sim_mean"]
        ),
        "share_pKS_median_gt_0_05": float(
            np.nanmean(big_table["ks_p_median"] > 0.05)
        ),
    }
    return pd.DataFrame([out])
