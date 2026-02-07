import numpy as np
import pandas as pd
import pytest

from src.analysis.ks import ks_degree_sampled
from src.analysis.period_summary import summarize_period
from src.io.load_us_air import build_period_index
from src.model.inference_parameters import _neg_ll_and_grad
from src.preprocessing.canonicalize import validate_upper_tri_bool
from src.preprocessing.ranking import rank_nodes_for_period


def test_period_index_monthly_boundaries():
    index_df = pd.DataFrame(
        {
            "t": [1, 2],
            "year": [2020, 2020],
            "month": [1, 2],
            "quarter": [1, 1],
            "start_date": ["2020-01-01", "2020-02-01"],
            "end_date": ["2020-01-31", "2020-02-29"],
            "n_nodes": [3, 3],
        }
    )
    period_df, t_to_pos = build_period_index(index_df, "M")
    assert period_df.loc[0, "period_pos"] == 1
    assert period_df.loc[1, "period_pos"] == 2
    assert str(period_df.loc[0, "start_date"].date()) == "2020-01-01"
    assert str(period_df.loc[1, "end_date"].date()) == "2020-02-29"
    assert t_to_pos == {1: 1, 2: 2}


def test_validate_upper_tri_bool():
    A = np.zeros((3, 3), dtype=bool)
    A[0, 2] = True
    validate_upper_tri_bool(A)
    A_bad = A.copy()
    A_bad[2, 0] = True
    with pytest.raises(ValueError):
        validate_upper_tri_bool(A_bad)


def test_nll_upper_triangle_only():
    A = np.zeros((3, 3), dtype=bool)
    A[0, 2] = True
    params = np.array([0.0])
    core_mask = np.array([False, False, False])
    nll, _ = _neg_ll_and_grad(params, A, core_mask)
    expected = 3 * np.log(2.0)
    assert np.isclose(nll, expected, atol=1e-8)


def test_ranking_previous_window():
    A_time = np.zeros((3, 3, 4), dtype=bool)
    A_time[0, 1, 1] = True  # period 2
    A_time[1, 2, 2] = True  # period 3
    A_time[0, 2, 3] = True  # period 4 (should not count)
    active_nodes = np.array([0, 1, 2])
    ranking, scores = rank_nodes_for_period(
        A_time,
        period_pos=4,
        window_size=2,
        active_nodes_current=active_nodes,
    )
    assert ranking == [1, 0, 2]
    assert scores.tolist() == [1, 2, 1]


def test_seed_separation_mc_vs_ks():
    A_time = np.zeros((4, 4, 3), dtype=bool)
    A_time[0, 1, 0] = True
    A_time[1, 2, 1] = True
    A_time[0, 2, 2] = True

    row_1 = summarize_period(
        A_time,
        2,
        batch_size=1,
        k_range=range(0, 3),
        criterion="nll",
        tol=1e-6,
        max_iter=50,
        R=6,
        seed_MC=123,
        n_KS=2,
        seed_KS=1,
    )
    row_2 = summarize_period(
        A_time,
        2,
        batch_size=1,
        k_range=range(0, 3),
        criterion="nll",
        tol=1e-6,
        max_iter=50,
        R=6,
        seed_MC=123,
        n_KS=2,
        seed_KS=999,
    )
    assert row_1["period_pos"] == 2
    assert row_1["L_sim_mean"] == pytest.approx(row_2["L_sim_mean"])


def test_ks_subset_seed_controls_indices():
    deg_emp = np.array([0, 0, 0, 0])
    deg_sims = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 1, 0, 1],
        ],
        dtype=int,
    )
    _, _, idx = ks_degree_sampled(deg_emp, deg_sims, n_KS=2, seed_KS=42, period_pos=3)
    rng = np.random.default_rng(np.random.SeedSequence([42, 3]))
    expected = rng.choice(deg_sims.shape[0], size=2, replace=False)
    assert np.array_equal(idx, expected)
