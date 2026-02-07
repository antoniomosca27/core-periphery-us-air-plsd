"""Network metrics for core-periphery diagnostics.

Overview
--------
This module computes motif counts (L, W, T), clustering summaries,
path-based statistics, modularity, and degree assortativity for
undirected simple graphs, with optional core/periphery class breakdowns.

Key conventions
---------------
- Inputs are canonical upper-triangular boolean adjacency matrices.
- Symmetric adjacency views are constructed explicitly when required.

Public API
----------
- `compute_metrics`
- `batch_compute`
- `path_metrics`
- `modularity_core_periphery`
- `assortativity_degree`
- `clustering_by_class_from_counts`
- `clustering_by_class`
"""
from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from src.preprocessing.canonicalize import validate_upper_tri_bool

__all__ = [
    "compute_metrics",
    "batch_compute",
    "path_metrics",
    "modularity_core_periphery",
    "assortativity_degree",
    "clustering_by_class_from_counts",
    "clustering_by_class",
]


# ------------------------------------------------------------------ #
# General helpers                                                    #
# ------------------------------------------------------------------ #
def _symmetrize(A: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Build the symmetric adjacency from canonical upper-tri input.

    Parameters
    ----------
    A : ndarray bool, shape (N, N)
        Canonical upper-triangular adjacency matrix.

    Returns
    -------
    ndarray bool, shape (N, N)
        Matrix ``A | A.T`` with zero diagonal.

    Notes
    -----
    Removes any self-loops.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0, 1], [0, 0]], dtype=bool)
    >>> _symmetrize(A)
    array([[False, True],
           [ True, False]])
    """
    validate_upper_tri_bool(A)
    A_sym = np.logical_or(A, A.T)
    np.fill_diagonal(A_sym, False)
    return A_sym


def _count_links(A_sym: NDArray[np.bool_]) -> int:
    """Count undirected links.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix with zero diagonal.

    Returns
    -------
    int
        Count of distinct edges ``i < j``.

    Notes
    -----
    Assumes ``A_sym`` is already symmetric and has no
    self-loops.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0, 1], [1, 0]], dtype=bool)
    >>> _count_links(A)
    1
    """
    return int(np.triu(A_sym, k=1).sum())


def _count_triangles(A_sym: NDArray[np.bool_]) -> int:
    """Count undirected triangles.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix with boolean values.

    Returns
    -------
    int
        Number of distinct triangles ``i < j < k``.

    Notes
    -----
    The matrix is cast to integers to use matrix multiplication
    when counting length-2 paths.

    Complexity
    ----------
    O(N^3) for matrix multiplication.

    Examples
    --------
    >>> A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=bool)
    >>> _count_triangles(A)
    1
    """
    A_int = A_sym.astype(int)
    A2 = A_int @ A_int
    return int((A2 * A_int).sum() // 6)


def _count_wedges(A_sym: NDArray[np.bool_]) -> int:
    """Count total wedges.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix.

    Returns
    -------
    int
        Sum of ``deg_v choose 2`` across all nodes.

    Notes
    -----
    Includes both open wedges and wedges closed by triangles (3 per
    triangle).

    Complexity
    ----------
    O(N^2) for degree computation.

    Examples
    --------
    >>> A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=bool)
    >>> _count_wedges(A)
    1
    """
    deg = A_sym.sum(axis=1)
    return int(((deg * (deg - 1)) // 2).sum())


# ------------------------------------------------------------------ #
# Core-periphery helpers                                             #
# ------------------------------------------------------------------ #
_CP_LINK_KEYS = ("L_cc", "L_cp", "L_pp")
_CP_WEDGE_KEYS = ("W_ccc", "W_ccp", "W_cpc", "W_cpp", "W_pcp", "W_ppp")
_CP_TRI_KEYS = ("T_ccc", "T_ccp", "T_cpp", "T_ppp")


def _is_core_mask(
    n: int, core_idx: list[int] | NDArray[np.int_]
) -> NDArray[np.bool_]:
    """Build a boolean mask for core nodes.

    Parameters
    ----------
    n : int
        Total number of nodes.
    core_idx : list[int] or ndarray[int]
        0-based indices of nodes in the core.

    Returns
    -------
    ndarray[bool]
        Array of length ``n`` with ``True`` on core nodes.

    Complexity
    ----------
    O(|core_idx|).

    Examples
    --------
    >>> _is_core_mask(5, [0, 2])
    array([ True, False,  True, False, False])
    """
    mask = np.zeros(n, dtype=bool)
    mask[core_idx] = True
    return mask


# ---------- link ----------------------------------------------------
def _count_links_by_class(
    A_sym: NDArray[np.bool_], core_idx: list[int]
) -> dict[str, int]:
    """Count links by core/periphery class.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix.
    core_idx : list[int]
        Indices of nodes in the core.

    Returns
    -------
    dict[str, int]
        Counts ``L_cc``, ``L_cp``, and ``L_pp``.

    Notes
    -----
    A core-periphery edge is counted once.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0,1],[1,0]], dtype=bool)
    >>> _count_links_by_class(A, [0])
    {'L_cc': 0, 'L_cp': 1, 'L_pp': 0}
    """
    mask = _is_core_mask(A_sym.shape[0], core_idx)
    core, peri = mask, ~mask

    L_cc = int(np.triu(A_sym[np.ix_(core, core)], k=1).sum())
    L_pp = int(np.triu(A_sym[np.ix_(peri, peri)], k=1).sum())
    L_cp = int(A_sym[np.ix_(core, peri)].sum())  # core-periphery

    return {"L_cc": L_cc, "L_cp": L_cp, "L_pp": L_pp}


# ---------- wedges (open + closed) ----------------------------------
def _count_wedges_by_class(
    A_sym: NDArray[np.bool_], core_idx: list[int]
) -> dict[str, int]:
    """Count wedges by core/periphery class combination.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix.
    core_idx : list[int]
        Core-node indices.

    Returns
    -------
    dict[str, int]
        Keys ``W_ccc``, ``W_ccp``, ``W_cpc``, ``W_cpp``, ``W_pcp``, and ``W_ppp``.

    Notes
    -----
    Each wedge is assigned by the class of its center node and
    endpoints; each triangle contributes three wedges, one per node.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
    >>> _count_wedges_by_class(A, [0])['W_cpp']
    1
    """
    n = A_sym.shape[0]
    mask = _is_core_mask(n, core_idx)

    deg_core = A_sym[:, mask].sum(axis=1)
    deg_peri = A_sym[:, ~mask].sum(axis=1)

    core_center = mask
    peri_center = ~mask

    W_ccc = ((deg_core[core_center] * (deg_core[core_center] - 1)) // 2).sum()
    W_ccp = (deg_core[core_center] * deg_peri[core_center]).sum()
    W_pcp = ((deg_peri[core_center] * (deg_peri[core_center] - 1)) // 2).sum()

    W_cpc = ((deg_core[peri_center] * (deg_core[peri_center] - 1)) // 2).sum()
    W_cpp = (deg_core[peri_center] * deg_peri[peri_center]).sum()
    W_ppp = ((deg_peri[peri_center] * (deg_peri[peri_center] - 1)) // 2).sum()

    return {
        "W_ccc": int(W_ccc),
        "W_ccp": int(W_ccp),
        "W_cpc": int(W_cpc),
        "W_cpp": int(W_cpp),
        "W_pcp": int(W_pcp),
        "W_ppp": int(W_ppp),
    }


# ---------- triangles -----------------------------------------------
def _count_triangles_by_class(
    A_sym: NDArray[np.bool_],
    core_idx: list[int],
    *,
    T_ref: int | None = None,
) -> dict[str, int]:
    """Count triangles by core/periphery class combination.

    Parameters
    ----------
    A_sym : (N, N) ndarray[bool]
        Symmetric adjacency matrix.
    core_idx : list[int]
        Core-node indices.
    T_ref : int, optional
        Global triangle count used to enforce consistency, by default ``None``.

    Returns
    -------
    dict[str, int]
        Keys ``T_ccc``, ``T_ccp``, ``T_cpp``, and ``T_ppp``.

    Notes
    -----
    If ``T_ref`` is provided, counts are scaled by an integer factor
    so that their sum matches ``T_ref``.

    Complexity
    ----------
    O(N^3) for matrix multiplications.

    Examples
    --------
    >>> A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=bool)
    >>> _count_triangles_by_class(A, [0])['T_cpp']
    1
    """
    n = A_sym.shape[0]
    mask = _is_core_mask(n, core_idx)
    out = dict.fromkeys(_CP_TRI_KEYS, 0)

    A_cc = A_sym[np.ix_(mask, mask)]
    A_pp = A_sym[np.ix_(~mask, ~mask)]
    A_cp = A_sym[np.ix_(mask, ~mask)]
    A_pc = A_sym[np.ix_(~mask, mask)]

    if A_cc.size:
        out["T_ccc"] = _count_triangles(A_cc)
    if A_pp.size:
        out["T_ppp"] = _count_triangles(A_pp)

    if mask.sum() >= 2 and (~mask).sum() > 0:
        common_peri = A_cp.astype(int) @ A_cp.astype(int).T
        out["T_ccp"] = int((np.triu(A_cc.astype(int) * common_peri, k=1)).sum())

    if (~mask).sum() >= 2 and mask.sum() > 0:
        common_core = A_pc.astype(int) @ A_pc.astype(int).T
        out["T_cpp"] = int((np.triu(A_pp.astype(int) * common_core, k=1)).sum())

    # 2) optional scaling for consistency with T_ref
    if T_ref is not None:
        total = sum(out.values())
        if total != 0 and total != T_ref and T_ref % total == 0:
            f = T_ref // total
            for k in out:
                out[k] *= f

    return out


def clustering_by_class_from_counts(counts: dict) -> dict:
    """Compute global and class-wise clustering.

    Parameters
    ----------
    counts : dict
        Dictionary containing global and/or class-wise ``W`` and ``T`` keys.

    Returns
    -------
    dict
        Global clustering ``C`` and class-specific values such as ``C_ccc``.

    Notes
    -----
    Returns ``nan`` when the wedge count is zero.

    Complexity
    ----------
    O(1).

    Examples
    --------
    >>> clustering_by_class_from_counts({'W':2,'T':1})['C']
    1.5
    """
    get = counts.get
    W = float(get("W", np.nan))
    T = float(get("T", np.nan))
    out = {
        "C": np.nan if (W == 0 or np.isnan(W)) else 3.0 * T / W,
    }
    for lab in ("ccc", "ccp", "cpp", "ppp"):
        Wk = float(get(f"W_{lab}", np.nan))
        Tk = float(get(f"T_{lab}", np.nan))
        out[f"C_{lab}"] = (
            np.nan if (Wk == 0 or np.isnan(Wk)) else 3.0 * Tk / Wk
        )
    return out


# ------------------------------------------------------------------ #
# Metrics for a single network                                      #
# ------------------------------------------------------------------ #
def compute_metrics(
    A: NDArray[np.bool_],
    *,
    core_idx: list[int] | None = None,
) -> dict[str, object]:
    """Compute global and core-periphery metrics for one network.

    Parameters
    ----------
    A : (N, N) ndarray[bool]
        Canonical adjacency matrix (upper-tri bool).
    core_idx : list[int] | None, optional
        Core-node indices. If omitted, only global
        metrics are returned.

    Returns
    -------
    dict
        Primary keys ``L``, ``W``, ``T``, ``clustering``, and ``deg``.
        If ``core_idx`` is provided, includes class-wise counts
        ``L_*``, ``W_*``, and ``T_*``.

    Raises
    ------
    ValueError
        If ``A`` is not a square matrix.

    Notes
    -----
    The matrix is symmetrized only for metric computation.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0,1],[0,0]], dtype=bool)
    >>> compute_metrics(A)['L']
    1
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    A_sym = _symmetrize(A)

    # --- global ------------------------------------------------------
    L = _count_links(A_sym)
    W = _count_wedges(A_sym)
    T = _count_triangles(A_sym)
    clustering = 0.0 if W == 0 else 3.0 * T / W
    deg = A_sym.sum(axis=1).astype(int)

    out: dict[str, object] = {
        "L": L,
        "W": W,
        "T": T,
        "clustering": clustering,
        "deg": deg,
    }

    # --- core-periphery ---------------------------------------------
    if core_idx:
        link_cp = _count_links_by_class(A_sym, core_idx)
        wedge_cp = _count_wedges_by_class(A_sym, core_idx)
        tri_cp = _count_triangles_by_class(A_sym, core_idx, T_ref=T)

        # Consistency checks (links and wedges already match by construction).
        assert sum(link_cp.values()) == L, "Sum L_cc+L_cp+L_pp != L"
        assert sum(wedge_cp.values()) == W, "Sum wedges != W"
        assert sum(tri_cp.values()) == T, "Triangle sum != T"

        out.update(link_cp)
        out.update(wedge_cp)
        out.update(tri_cp)

    return out


def clustering_by_class(A: np.ndarray, core_idx: np.ndarray) -> dict:
    """Compute global and class-wise clustering from adjacency and core indices.

    Parameters
    ----------
    A : (N, N) ndarray
        Boolean adjacency matrix.
    core_idx : ndarray[int]
        Core-node indices.

    Returns
    -------
    dict
        Global clustering ``C`` and class-specific values such as ``C_ccc``.

    Notes
    -----
    Convenience function combining ``compute_metrics`` and
    ``clustering_by_class_from_counts``.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0,1],[1,0]], dtype=bool)
    >>> clustering_by_class(A, np.array([0]))['C']
    0.0
    """
    counts = compute_metrics(A, core_idx=list(core_idx))
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
    counts_use = {k: counts.get(k, np.nan) for k in keys}
    return clustering_by_class_from_counts(counts_use)


# ------------------------------------------------------------------ #
# Metrics for a batch of networks                                      #
# ------------------------------------------------------------------ #
def batch_compute(
    A_batch: NDArray[np.bool_],
    *,
    core_idx: list[int] | None = None,
) -> dict[str, object]:
    """Compute metrics for a batch of networks.

    Parameters
    ----------
    A_batch : (S, N, N) ndarray[bool]
        Batch of ``S`` adjacency matrices.
    core_idx : list[int] | None, optional
        Core-node indices, by default ``None``.

    Returns
    -------
    dict[str, object]
        Each key maps to an array of length ``S``; ``deg`` is a list
        of degree arrays.

    Raises
    ------
    ValueError
        If ``A_batch`` is not three-dimensional.

    Notes
    -----
    Class-wise metrics are included only when ``core_idx`` is provided.

    Complexity
    ----------
    O(S * N^2).

    Examples
    --------
    >>> A = np.zeros((2,2,2), dtype=bool)
    >>> batch_compute(A)['L'].tolist()
    [0, 0]
    """
    if A_batch.ndim != 3:
        raise ValueError("A_batch must have shape (n_sim, N, N).")

    base_keys = ("L", "W", "T", "clustering")
    cp_keys = _CP_LINK_KEYS + _CP_WEDGE_KEYS + _CP_TRI_KEYS if core_idx else ()

    results: dict[str, list] = {k: [] for k in base_keys}
    results["deg"] = []
    for k in cp_keys:
        results[k] = []

    for A in A_batch:
        m = compute_metrics(A, core_idx=core_idx)

        for k in base_keys:
            results[k].append(m[k])
        if core_idx:
            for k in cp_keys:
                results[k].append(m[k])
        results["deg"].append(m["deg"])

    # Cast to ndarray where appropriate.
    for k in base_keys + cp_keys:
        results[k] = np.asarray(results[k])

    return results


# ------------------------------------------------------------------ #
# Path-based metrics                                          #
# ------------------------------------------------------------------ #
def path_metrics(A: np.ndarray) -> dict:
    """Compute average shortest-path length and diameter on the LCC.

    Parameters
    ----------
    A : (N, N) ndarray
        Adjacency matrix; nonzero elements indicate edges.

    Returns
    -------
    dict
        ``aspl_gc`` and ``diam_gc`` computed on the
        largest connected component; ``nan`` if undefined.

    Notes
    -----
    Edges are treated as undirected and BFS is run from each node
    in the LCC.

    Complexity
    ----------
    O(|LCC| * (|LCC|+L)).

    Examples
    --------
    >>> A = np.array([[0,1],[1,0]], dtype=int)
    >>> path_metrics(A)['diam_gc']
    1
    """
    validate_upper_tri_bool(A)
    A_sym = np.logical_or(A, A.T)
    n = A_sym.shape[0]
    if n == 0:
        return {"aspl_gc": np.nan, "diam_gc": np.nan}

    np.fill_diagonal(A_sym, False)
    neigh = [np.flatnonzero(row).tolist() for row in A_sym]

    visited = np.zeros(n, bool)
    comps: list[list[int]] = []
    for v in range(n):
        if not visited[v]:
            stack = [v]
            visited[v] = True
            comp: list[int] = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in neigh[u]:
                    if not visited[w]:
                        visited[w] = True
                        stack.append(w)
            comps.append(comp)

    if not comps:
        return {"aspl_gc": np.nan, "diam_gc": np.nan}

    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return {"aspl_gc": np.nan, "diam_gc": np.nan}

    idx = {v: i for i, v in enumerate(lcc)}
    neigh_lcc = [
        [idx[w] for w in neigh[v] if w in idx]
        for v in lcc
    ]

    n_lcc = len(lcc)
    total = 0.0
    count = 0
    diam = 0

    for s in range(n_lcc):
        dist = np.full(n_lcc, -1, int)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in neigh_lcc[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        for t in range(s + 1, n_lcc):
            d = dist[t]
            if d > 0:
                total += d
                count += 1
                if d > diam:
                    diam = d

    if count == 0:
        return {"aspl_gc": np.nan, "diam_gc": np.nan}

    aspl = total / count
    return {"aspl_gc": float(aspl), "diam_gc": float(diam)}


# ------------------------------------------------------------------ #
# Core/periphery modularity                                          #
# ------------------------------------------------------------------ #
def modularity_core_periphery(
    A: NDArray[np.int_], core_idx: NDArray[np.bool_] | NDArray[np.int_]
) -> dict[str, float]:
    """Compute modularity for the core/periphery partition.

    Parameters
    ----------
    A : (N, N) ndarray[int]
        Canonical adjacency matrix (upper-tri).
    core_idx : ndarray[bool] or ndarray[int]
        Boolean mask or list of core-node indices.

    Returns
    -------
    dict[str, float]
        ``Q`` and intermediate quantities ``e_*``, ``a_*``, ``m``, ``n_c``, and ``n_p``.

    Raises
    ------
    ValueError
        If ``A`` is not square, symmetric, or has a nonzero diagonal.

    Notes
    -----
    When there are no edges (``m = 0``), returns zeros only.

    Complexity
    ----------
    O(N^2).

    Examples
    --------
    >>> A = np.array([[0,1],[1,0]])
    >>> modularity_core_periphery(A, np.array([True, False]))['Q']
    -0.25
    """
    validate_upper_tri_bool(A)
    A_bin = (A != 0)
    A_sym = np.logical_or(A_bin, A_bin.T)
    m = A_sym.sum() / 2.0
    n = A_sym.shape[0]

    core_arr = np.asarray(core_idx)
    if core_arr.dtype == bool:
        if core_arr.size != n:
            raise ValueError("core_idx boolean mask must match A's size")
        C = core_arr.astype(bool)
    else:
        C = np.zeros(n, bool)
        C[np.asarray(core_arr, int)] = True
    P = ~C
    n_c = int(C.sum())
    n_p = int(P.sum())

    if m == 0:
        return {
            "Q": 0.0,
            "e_cc": 0.0,
            "e_pp": 0.0,
            "e_cp": 0.0,
            "a_c": 0.0,
            "a_p": 0.0,
            "m": 0.0,
            "n_c": n_c,
            "n_p": n_p,
        }

    L_cc = A_sym[np.ix_(C, C)].sum() / 2.0
    L_pp = A_sym[np.ix_(P, P)].sum() / 2.0
    L_cp = A_sym[np.ix_(C, P)].sum()

    e_cc = L_cc / m
    e_pp = L_pp / m
    e_cp = L_cp / m

    a_c = e_cc + e_cp
    a_p = e_pp + e_cp

    Q = (e_cc - a_c**2) + (e_pp - a_p**2)

    return {
        "Q": float(Q),
        "e_cc": float(e_cc),
        "e_pp": float(e_pp),
        "e_cp": float(e_cp),
        "a_c": float(a_c),
        "a_p": float(a_p),
        "m": float(m),
        "n_c": n_c,
        "n_p": n_p,
    }


# ------------------------------------------------------------------ #
# Degree assortativity                                              #
# ------------------------------------------------------------------ #
def assortativity_degree(A: np.ndarray) -> dict:
    """Compute undirected degree assortativity.

    Parameters
    ----------
    A : (N, N) ndarray
        Canonical adjacency matrix (upper-tri).

    Returns
    -------
    dict
        ``r`` Pearson coefficient, ``m`` number of edges, and ``n`` nodes.

    Raises
    ------
    ValueError
        If ``A`` is not symmetric or contains self-loops.

    Notes
    -----
    The computation uses upper-triangular edges ``i < j``.

    Complexity
    ----------
    O(L) where ``L`` is the number of edges.

    Examples
    --------
    >>> A = np.array([[0,1],[1,0]])
    >>> assortativity_degree(A)['r']
    0.0
    """
    validate_upper_tri_bool(A)
    A_bin = A != 0
    A_sym = np.logical_or(A_bin, A_bin.T)
    n = A_sym.shape[0]
    i, j = np.where(np.triu(A_sym, k=1))
    m = i.size
    if m == 0:
        return {"r": 0.0, "m": 0, "n": int(n)}
    deg = A_sym.sum(axis=1).astype(float)
    x = deg[i]
    y = deg[j]
    sx = x.std()
    sy = y.std()
    r = 0.0 if (sx == 0.0 or sy == 0.0) else float(np.corrcoef(x, y)[0, 1])
    return {"r": r, "m": int(m), "n": int(n)}
