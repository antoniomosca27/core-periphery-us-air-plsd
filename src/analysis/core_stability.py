"""Core-set stability metrics.

Overview
--------
This module provides set-based stability measures for core memberships
across time. The current implementation exposes the Jaccard index.
"""
from typing import List


def jaccard(a: List[int], b: List[int]) -> float:
    """Compute the Jaccard index between two node sets.

    Parameters
    ----------
    a, b : list[int]
        Node identifier lists. Duplicates are ignored.

    Returns
    -------
    float
        Value in ``[0, 1]`` equal to ``|a intersection b| / |a union b|``.
        Returns ``0`` when the union is empty.

    Notes
    -----
    The operation is order-invariant.

    Complexity
    ----------
    O(|a| + |b|) for set construction and intersection.

    Examples
    --------
    >>> jaccard([1, 2], [2, 3])
    0.3333333333333333
    """
    a_set, b_set = set(a), set(b)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return 0.0 if union == 0 else inter / union
