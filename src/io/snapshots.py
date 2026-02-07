"""
----------------------------------------------------------------------
FILE: src/io/snapshots.py
----------------------------------------------------------------------

Purpose
-------
Provide a lightweight time-series container for sparse adjacency
snapshots that behaves like a 3D array on demand (N, N, T).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp


@dataclass
class SparseTimeSeries:
    """Lazy 3D-like container over sparse (N, N) snapshots."""

    matrices: Sequence[sp.spmatrix]
    n_nodes: int

    def __post_init__(self) -> None:
        self.shape = (int(self.n_nodes), int(self.n_nodes), len(self.matrices))
        self.ndim = 3

    def __len__(self) -> int:
        return len(self.matrices)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            row_sel, col_sel, t_sel = key
            if row_sel == slice(None) and col_sel == slice(None) and isinstance(
                t_sel, (int, np.integer)
            ):
                return self._dense_snapshot(int(t_sel))
        raise IndexError("Use A[:, :, t] with t as an int.")

    def _dense_snapshot(self, pos: int) -> np.ndarray:
        mat = self.matrices[pos]
        if sp.isspmatrix(mat):
            dense = mat.toarray()
        else:
            dense = np.asarray(mat)
        return dense.astype(np.bool_, copy=False)

    def get_sparse(self, pos: int) -> sp.spmatrix:
        """Return the sparse snapshot at position pos (0-based)."""
        return self.matrices[pos]
