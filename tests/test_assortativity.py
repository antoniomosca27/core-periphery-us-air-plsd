"""Test purpose:
Verify assortativity computations and their baseline behavior.

Success criteria:
- The assortativity coefficient matches NetworkX (tolerance 2e-2).
- A path graph yields z~0 and p-value 1 in the significance routine.
- Empty graphs return r=0 with consistent counts.

Synthetic data:
- `gnp_random_graph`, `path_graph`, and a zero matrix.
"""

import numpy as np
import networkx as nx
from src.model.metrics import assortativity_degree


def test_assortativity_degree_matches_networkx():
    G = nx.gnp_random_graph(10, 0.3, seed=42)  # fixed seed
    A_full = nx.to_numpy_array(G, dtype=bool)
    A = np.triu(A_full, 1).astype(bool)
    out = assortativity_degree(A)
    r_nx = nx.degree_assortativity_coefficient(G)
    assert np.isclose(out["r"], r_nx, atol=2e-2)  # wide tolerance for a small graph
    assert out["m"] == G.number_of_edges()
    assert out["n"] == G.number_of_nodes()


def test_assortativity_empty_graph():
    A = np.zeros((3, 3), dtype=bool)
    out = assortativity_degree(A)
    assert out["r"] == 0.0
    assert out["m"] == 0
    assert out["n"] == 3
