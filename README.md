# Core-Periphery Network Model on US Air Traffic

[![CI](https://github.com/antoniomosca27/core-periphery-us-air-plsd/actions/workflows/ci.yml/badge.svg)](https://github.com/antoniomosca27/core-periphery-us-air-plsd/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

This repository provides the reference implementation for the experiments presented in the article **“Penalized Likelihood with Structural Discrepancies for Core-Periphery Networks”**, by **Antonio Mosca** and **Piero Mazzarisi**.

The code implements a maximum entropy core–periphery network model and associated inference, diagnostics, and validation procedures on the public US air traffic network. The repository is designed to support full reproducibility of the empirical results reported in the paper.

---

## Scope

- Dataset scope is limited to the public US air traffic network.
- Temporal indexing uses `period_pos` (1-based) consistently across configuration, outputs, and plots.
- Supported aggregation levels are monthly (`M`), quarterly (`Q`), and annual (`A`).
- Adjacency matrices are canonical: boolean, strictly upper-triangular, zero diagonal.

---

## Installation

Create and activate a virtual environment, then install runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

The analysis is notebook-driven.

```bash
jupyter notebook notebooks/build_us_air_monthly_edges.ipynb
jupyter notebook notebooks/core_periphery_analysis.ipynb
```

Both notebooks are designed to be executed end-to-end.

## Repository Layout

- `src/`: Python source code
- `analysis/`: diagnostics, summaries, and plotting utilities
- `model/`: core-periphery model, inference, and simulation
- `preprocessing/`: node ranking and core-size selection
- `io/`: US air traffic data loading and aggregation
- `notebooks/build_us_air_monthly_edges.ipynb`: preprocessing from raw CSVs
- `notebooks/core_periphery_analysis.ipynb`: full empirical pipeline
- `tests/`: automated test suite
- `data/`: local data directories (partially ignored by Git)
- `results/`, `figures/`: run outputs (ignored by Git)

## Data

Expected data layout:

- `data/raw/US_air_traffic_nodes.csv`
- `data/raw/US_air_traffic_gprops.csv`
- `data/processed/us_air/edges_ijt_monthly.parquet`
- `data/processed/us_air/index_monthly.csv`
- `data/processed/us_air/nodes.csv`
- `data/processed/us_air/gprops.csv`

The file `data/raw/US_air_traffic_edges.csv` is large and not tracked in version control. Place raw files locally before running the build notebook.

## Pipeline Overview

1. Build monthly edge lists from raw US air traffic data.
2. Load processed monthly data and aggregate to `M`, `Q`, or `A` periods.
3. Rank active nodes using a rolling window over previous periods (`batch_size`).
4. Select core size using `nll`, `plsd_diag`, or `plsd_maha`, optionally penalized by `aic` or `bic`.
5. Fit model parameters, simulate networks via Monte Carlo, and run KS diagnostics.
6. Export period-level tables and diagnostic figures.

## Reproducibility

- Randomness is controlled via explicit seeds in the analysis notebook.
- Monte Carlo simulations, KS tests, and ranking procedures are fully deterministic given the same configuration.
- All resolved configuration parameters are exported with the results.

## Testing

```bash
pytest -q
python -m compileall src tests
```

## Authors

- Antonio Mosca
- Piero Mazzarisi

## License

This project is released under the BSD 3-Clause License. See the LICENSE file for details.
