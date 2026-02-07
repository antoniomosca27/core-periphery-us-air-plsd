"""Core-periphery model components.

Overview
--------
The model package contains simulation, parameter inference, motif
statistics, PLSD scoring, and analytical expectation routines.

Key conventions
---------------
- Adjacency inputs are canonical upper-triangular boolean matrices with
  zero diagonal.
- Functions that require undirected operations explicitly construct
  `A | A.T`.

Public API
----------
This module is a namespace initializer and does not define a stable API.
"""
