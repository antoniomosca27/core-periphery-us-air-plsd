"""Test purpose:
Verify core-periphery modularity on simple controlled scenarios.

Success criteria:
- An empty graph yields zero modularity and zero counts.
- A complete graph returns the analytically expected value.
- The routine accepts core indices passed as a Python list.

Synthetic data:
- Hand-constructed adjacency matrices: empty, complete, and core-only edge.
"""

import numpy as np

from src.model.metrics import modularity_core_periphery

def test_modularity_empty():
    A = np.zeros((4, 4), dtype=bool)
    core = np.array([True, False, True, False])
    out = modularity_core_periphery(A, core)
    assert out["Q"] == 0.0
    assert out["m"] == 0.0
    assert out["e_cc"] == 0.0 and out["e_cp"] == 0.0 and out["e_pp"] == 0.0


def test_modularity_complete():
    n = 4
    A_full = np.ones((n, n), dtype=bool)
    np.fill_diagonal(A_full, 0)
    A = np.triu(A_full, 1).astype(bool)
    core = np.array([True, True, False, False])
    out = modularity_core_periphery(A, core)
    expected_Q = -38 / 36  # computed analytically
    assert np.isclose(out["Q"], expected_Q)
    assert out["n_c"] == 2 and out["n_p"] == 2


def test_modularity_core_only_edges():
    n = 5
    A = np.zeros((n, n), dtype=bool)
    A[0, 1] = True
    core = [0, 1]  # accepts plain Python lists
    out = modularity_core_periphery(A, core)
    assert out["Q"] == 0.0
    assert out["e_cc"] == 1.0 and out["e_cp"] == 0.0
    assert out["m"] == 1.0
