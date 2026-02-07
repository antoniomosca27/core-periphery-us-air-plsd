"""Test purpose:
Verify baseline behavior of L-BFGS-B fitting and core scanning.

Success criteria:
- NLL fitting returns finite parameters for empty and non-empty cores.
- `scan_core_sizes_current_period` emits no warnings and returns coherent
  output.
"""

import logging
import numpy as np

from src.model.inference_parameters import fit
from src.preprocessing.core_detection import scan_core_sizes_current_period


def make_graph(rng, n, p):
    A = rng.random((n, n)) < p
    A = np.triu(A, 1)
    return A.astype(bool)


def test_fit_nll_empty_core():
    rng = np.random.default_rng(0)
    A = make_graph(rng, 20, 0.2)
    y_hat, x_hat, nll_hat = fit(A, [], objective="nll", max_iter=200, tol=1e-10, seed=0)
    assert np.isfinite(y_hat)
    assert x_hat.size == 0
    assert np.isfinite(nll_hat)


def test_fit_nll_nonempty_core():
    rng = np.random.default_rng(1)
    A = make_graph(rng, 20, 0.2)
    core = [0, 1, 2]
    y_hat, x_hat, nll_hat = fit(A, core, objective="nll", max_iter=200, tol=1e-10, seed=0)
    assert np.isfinite(y_hat)
    assert x_hat.size == len(core)
    assert np.all(np.isfinite(x_hat))
    assert np.isfinite(nll_hat)


def test_core_scan_no_warning(caplog):
    rng = np.random.default_rng(2)
    n = 8
    A = make_graph(rng, n, 0.2)
    ranking = np.argsort(-(A.sum(axis=0) + A.sum(axis=1))).tolist()
    with caplog.at_level(logging.WARNING):
        scan = scan_core_sizes_current_period(
            A,
            ranking,
            candidate_sizes=[0, 1, 2],
            criterion="nll",
            complexity_penalty="bic",
            tol=1e-8,
            max_iter=100,
        )
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert scan["selected_index"] in range(len(scan["m_values"]))
