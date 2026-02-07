"""Test purpose:
Measure runtime stability of the core-periphery sampler.

Success criteria:
- `sample` completes 100 simulations without exceptions.

Synthetic data:
- A graph with 200 nodes, 20 core nodes, and deterministic parameters.
"""
from __future__ import annotations

import numpy as np
from src.model.simulate_network import sample


def main() -> None:
    N = 200
    k = 20
    core = list(range(k))
    y = -4.0
    x = np.full(k, 1.5)

    sample(
        y,
        x,
        core,
        n_nodes=N,
        n_sim=100,
        n_jobs=1,
        seed=42,  # Fixed seed for reproducibility.
    )


if __name__ == "__main__":
    main()
