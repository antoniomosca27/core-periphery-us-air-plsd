"""Test purpose:
Verify PLSD scoring and core-size scan behavior on synthetic core graphs.

Success criteria:
- PLSD score decreases when the correct core is used.
- Core optimization returns sizes between 10 and 70.

Synthetic data:
- Graphs with elevated within-core probabilities and fixed RNG seeds.
"""

import numpy as np
from src.preprocessing.core_detection import scan_core_sizes_current_period
from src.model.plsd import compute_plsd_score
from src.model.inference_parameters import fit


def _make_toy_graph(rng, n=50, core_size=25):
    p = np.full((n, n), 0.05)
    p[:core_size, :core_size] = 0.9  # high probability in the core
    p[:core_size, core_size:] = 0.2
    p[core_size:, :core_size] = 0.2
    A = rng.random((n, n)) < p
    A = np.triu(A, 1)  # Use only the strict upper triangle.
    return A.astype(bool)


def test_score_decreases_with_correct_core():
    rng = np.random.default_rng(0)  # fixed seed
    A = _make_toy_graph(rng)
    core_small = list(range(5))
    core_true = list(range(25))
    y_s, x_s, _ = fit(A, core_small, tol=1e-6, max_iter=200)
    y_t, x_t, _ = fit(A, core_true, tol=1e-6, max_iter=200)
    s_small = compute_plsd_score(A, core_small, y_s, x_s, lambda_T=1.0, lambda_W=1.0)
    s_true = compute_plsd_score(A, core_true, y_t, x_t, lambda_T=1.0, lambda_W=1.0)
    assert s_true < s_small


def _random_graph(rng, n=80, core_size=40):
    p = np.full((n, n), 0.05)
    p[:core_size, :core_size] = 0.8  # higher density in the core
    p[:core_size, core_size:] = 0.2
    p[core_size:, :core_size] = 0.2
    A = rng.random((n, n)) < p
    A = np.triu(A, 1)
    return A.astype(bool)


def test_optimized_core_reasonable_size():
    rng = np.random.default_rng(1)  # fixed seed
    sizes = []
    for _ in range(20):
        A = _random_graph(rng)
        ranking = np.argsort(-(A.sum(axis=0) + A.sum(axis=1))).tolist()
        scan = scan_core_sizes_current_period(
            A,
            ranking,
            candidate_sizes=list(range(10, 80, 10)),
            criterion="plsd_diag",
            tol=1e-6,
        )
        idx = scan["selected_index"]
        sizes.append(scan["m_values"][idx])
    assert all(10 <= s <= 70 for s in sizes)


def test_plsd_maha_scan_runs():
    rng = np.random.default_rng(2)
    A = _random_graph(rng, n=40, core_size=20)
    ranking = np.argsort(-(A.sum(axis=0) + A.sum(axis=1))).tolist()
    scan = scan_core_sizes_current_period(
        A,
        ranking,
        candidate_sizes=[0, 10, 20, 30],
        criterion="plsd_maha",
        tol=1e-6,
        max_iter=100,
        max_iter_plsd=30,
    )
    assert len(scan["m_values"]) == len(scan["plsd_selected"])
