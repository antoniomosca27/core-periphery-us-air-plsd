# Core-Periphery Network Model for US Air Traffic

This repository implements and evaluates a maximum-entropy core-periphery model on the public US air traffic network.

## Scope
- Dataset scope is limited to US air traffic snapshots.
- Temporal indexing uses `period_pos` (1-based) across configuration, outputs, and plots.
- Supported aggregation levels are monthly (`M`), quarterly (`Q`), and annual (`A`) via `src/io/load_us_air.py`.
- Adjacency matrices are canonical: boolean, strictly upper-triangular, zero diagonal.

## Pipeline
1. Build monthly edge list data from local raw CSV files.
2. Load monthly data from `data/processed/us_air/` and aggregate to `M/Q/A` periods.
3. Rank active nodes using a rolling window over previous periods (`batch_size`).
4. Select core size with `nll`, `plsd_diag`, or `plsd_maha` with optional `aic`/`bic` complexity penalty.
5. Fit parameters `(y, x)`, simulate Monte Carlo networks, run KS diagnostics, and compute metrics.
6. Export period-level results and figures.

## Notebooks
- `notebooks/build_us_air_monthly_edges.ipynb`: build the processed monthly edge list from raw CSVs.
- `notebooks/core_periphery_analysis.ipynb`: run end-to-end inference, diagnostics, and exports.

## Data layout
- `data/raw/US_air_traffic_nodes.csv`
- `data/raw/US_air_traffic_gprops.csv`
- `data/processed/us_air/edges_ijt_monthly.parquet` (or `edges_ijt_monthly.csv.gz`)
- `data/processed/us_air/index_monthly.csv`
- `data/processed/us_air/nodes.csv`
- `data/processed/us_air/gprops.csv`

`data/raw/US_air_traffic_edges.csv` is large and not tracked in Git. Place raw files locally before running the build notebook.

## Outputs
- `results/<run_id>/tables/big_table.csv`: one row per period.
- `results/<run_id>/tables/small_table.csv`: summary row over all evaluated periods.
- `results/<run_id>/config_resolved.json`: resolved run configuration.
- `figures/<run_id>/...`: diagnostic plots.

## Configuration highlights
Key settings in `notebooks/core_periphery_analysis.ipynb`:
- `processed_dir`
- `time_agg` in `{M, Q, A}`
- `time_start`, `time_end` (1-based, inclusive, on period positions)
- `criterion` in `{nll, plsd_diag, plsd_maha}`
- `complexity_penalty` in `{none, aic, bic}`
- `batch_size` (rolling ranking window)
- `tol`, `max_iter`, `max_iter_plsd`
- `freeze_plsd_variances`
- `R`, `seed_MC`, `n_KS`, `seed_KS`

## Testing
```bash
pytest -q
python -m compileall src tests
```
