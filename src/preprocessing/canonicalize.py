"""
----------------------------------------------------------------------
FILE: src/preprocessing/canonicalize.py
----------------------------------------------------------------------

Purpose
-------
Canonical adjacency helpers for the pipeline:
- Symmetrize directed inputs with OR
- Keep strictly upper-triangular boolean adjacency (diag=0, lower=0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def symmetrize_or(A: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return an undirected adjacency using OR symmetrization."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    A_bool = A.astype(np.bool_, copy=False)
    A_sym = np.logical_or(A_bool, A_bool.T)
    np.fill_diagonal(A_sym, False)
    return A_sym


def to_upper_tri_bool(A_sym: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Keep only the strictly upper triangle as boolean adjacency."""
    if A_sym.ndim != 2 or A_sym.shape[0] != A_sym.shape[1]:
        raise ValueError("A_sym must be a square matrix.")
    A_bool = A_sym.astype(np.bool_, copy=False)
    return np.triu(A_bool, k=1).astype(np.bool_, copy=False)


def validate_upper_tri_bool(A: NDArray[np.bool_]) -> None:
    """Assert that A is boolean, strictly upper-triangular, diag=0."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if A.dtype != np.bool_:
        raise ValueError("A must have boolean dtype.")
    if np.any(np.diag(A)):
        raise ValueError("A must have zero diagonal.")
    if np.any(np.tril(A, k=-1)):
        raise ValueError("A must have zero lower triangle.")
