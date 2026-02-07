"""Test purpose:
Validate `sample` in serial and parallel scenarios.

Success criteria:
- Simulated matrices are canonical upper-triangular with expected shapes.
- Repeated calls with the same seed produce identical networks, including
  when `n_jobs > 1`.

Synthetic data:
- Compact parameter fixtures with deterministic RNG streams.
"""

import numpy as np
import pytest
import os
import joblib

from src.model.simulate_network import sample


@pytest.fixture
def params_small():
    """Small valid parameter set for network sampling."""
    return dict(
        y=-2.0,
        x_vec=np.array([1.0, 1.0]),
        core_idx=[0, 1],
        n_nodes=4,
    )


def test_single_core(params_small):
    rng = np.random.default_rng(0)
    sims = sample(
        params_small["y"],
        params_small["x_vec"],
        params_small["core_idx"],
        n_nodes=params_small["n_nodes"],
        n_sim=2,
        n_jobs=1,
        rng=rng,
    )
    assert isinstance(sims, np.ndarray)
    assert len(sims) == 2
    assert sims.shape == (2, params_small["n_nodes"], params_small["n_nodes"])
    # Upper-triangular with zero diagonal and lower triangle
    assert not sims[:, np.arange(params_small["n_nodes"]), np.arange(params_small["n_nodes"])].any()
    lower = np.tril_indices(params_small["n_nodes"], k=-1)
    assert not sims[:, lower[0], lower[1]].any()


@pytest.mark.skipif(
    os.name == "nt" and "loky" not in joblib.parallel.BACKENDS,
    reason="loky backend unavailable",
)
def test_multi_core(params_small):
    rng1 = np.random.default_rng(123)
    sims1 = sample(
        params_small["y"],
        params_small["x_vec"],
        params_small["core_idx"],
        n_nodes=params_small["n_nodes"],
        n_sim=4,
        n_jobs=2,
        rng=rng1,
    )
    rng2 = np.random.default_rng(123)
    sims2 = sample(
        params_small["y"],
        params_small["x_vec"],
        params_small["core_idx"],
        n_nodes=params_small["n_nodes"],
        n_sim=4,
        n_jobs=2,
        rng=rng2,
    )
    assert isinstance(sims1, np.ndarray)
    assert len(sims1) == 4
    assert sims1.shape == (4, params_small["n_nodes"], params_small["n_nodes"])
    set1 = {tuple(g.flatten()) for g in sims1}  # compare configurations
    set2 = {tuple(g.flatten()) for g in sims2}
    assert set1 == set2
