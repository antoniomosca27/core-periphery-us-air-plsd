"""Test purpose:
Evaluate metric functions on undirected graphs with optional
core-periphery partitioning.

Success criteria:
- `compute_metrics` returns expected keys and coherent counts.
- Triangle counts match NetworkX.
- Class-wise clustering helpers produce compatible values.
- `path_metrics` handles connected and degenerate cases.

Synthetic data:
- Deterministic random graphs and hand-built adjacency matrices.
"""

import numpy as np
import networkx as nx
import pytest

from src.model.metrics import (
    compute_metrics,
    clustering_by_class_from_counts,
    clustering_by_class,
    path_metrics,
)

@pytest.fixture
def random_graph():
    # small deterministic random graph with <=20 nodes
    n = 10
    p = 0.3
    return nx.gnp_random_graph(n, p, seed=42)


def test_shape(random_graph):
    A_full = nx.to_numpy_array(random_graph, dtype=bool)
    A = np.triu(A_full, 1).astype(bool)
    out = compute_metrics(A)
    expected_keys = {"L", "W", "T", "clustering", "deg"}
    assert expected_keys.issubset(out.keys())
    assert isinstance(out["deg"], np.ndarray)
    assert out["deg"].shape[0] == A.shape[0]


def test_triangles_vs_networkx(random_graph):
    A_full = nx.to_numpy_array(random_graph, dtype=bool)
    A = np.triu(A_full, 1).astype(bool)
    out = compute_metrics(A)
    nx_triangles = sum(nx.triangles(random_graph).values()) // 3  # NetworkX counts each triangle three times
    assert out["T"] == nx_triangles


def test_clustering_by_class_from_counts():
    counts = {
        "W": 6,
        "T": 1,
        "W_ccc": 3,
        "T_ccc": 1,
        "W_ccp": 2,
        "T_ccp": 0,
        "W_cpp": 1,
        "T_cpp": 0,
        "W_ppp": 0,
        "T_ppp": 0,
    }
    res = clustering_by_class_from_counts(counts)
    assert res["C"] == pytest.approx(0.5)
    assert res["C_ccc"] == pytest.approx(1.0)
    assert np.isnan(res["C_ppp"])


def test_clustering_by_class_wrapper(random_graph):
    A_full = nx.to_numpy_array(random_graph, dtype=bool)
    A = np.triu(A_full, 1).astype(bool)
    core_idx = list(range(3))  # first three nodes in the core
    m = compute_metrics(A, core_idx=core_idx)
    keys = [
        "W",
        "T",
        "W_ccc",
        "T_ccc",
        "W_ccp",
        "T_ccp",
        "W_cpp",
        "T_cpp",
        "W_ppp",
        "T_ppp",
    ]
    counts = {k: m.get(k, np.nan) for k in keys}
    res_counts = clustering_by_class_from_counts(counts)
    res_wrapper = clustering_by_class(A, core_idx)
    for k in res_counts:
        if np.isnan(res_counts[k]):
            assert np.isnan(res_wrapper[k])
        else:
            assert res_wrapper[k] == pytest.approx(res_counts[k])


def test_path_metrics_lcc():
    A = np.zeros((5, 5), dtype=bool)
    edges = [(0, 1), (1, 2), (3, 4)]  # two disjoint components
    for i, j in edges:
        A[min(i, j), max(i, j)] = True
    res = path_metrics(A)
    assert res["diam_gc"] == 2
    assert res["aspl_gc"] == pytest.approx(4 / 3)


def test_path_metrics_degenerate():
    A = np.zeros((3, 3), dtype=bool)
    res = path_metrics(A)
    assert np.isnan(res["aspl_gc"])
    assert np.isnan(res["diam_gc"])
