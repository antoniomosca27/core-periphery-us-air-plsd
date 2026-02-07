"""
----------------------------------------------------------------------
FILE: src/io/load_us_air.py
----------------------------------------------------------------------

Purpose
-------
Load preprocessed US air traffic data stored as a temporal edge list
(i, j, t) and build sparse adjacency snapshots for M/Q/A aggregation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.io.snapshots import SparseTimeSeries

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s|%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _select_edges_path(processed_dir: Path) -> Path:
    parquet_path = processed_dir / "edges_ijt_monthly.parquet"
    csv_path = processed_dir / "edges_ijt_monthly.csv.gz"
    if parquet_path.is_file():
        try:
            import pyarrow  # noqa: F401
        except Exception:
            if csv_path.is_file():
                logger.warning("pyarrow not available; using csv.gz fallback.")
                return csv_path
        return parquet_path
    if csv_path.is_file():
        return csv_path
    raise FileNotFoundError(
        "No edges_ijt_monthly file found (parquet or csv.gz) in "
        f"{processed_dir}"
    )


def load_index_monthly(index_path: Path) -> pd.DataFrame:
    if not index_path.is_file():
        raise FileNotFoundError(f"index_monthly.csv not found: {index_path}")
    df = pd.read_csv(index_path)
    required = {"t", "year", "month", "quarter", "start_date", "end_date", "n_nodes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"index_monthly missing columns: {sorted(missing)}")
    df = df.sort_values(["year", "month", "t"]).reset_index(drop=True)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def build_period_index(
    index_df: pd.DataFrame, agg: str
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Return period index and a t->period_pos mapping (1-based)."""
    agg = str(agg).strip().upper()
    idx = index_df.copy()
    idx["start_date"] = pd.to_datetime(idx["start_date"])
    idx["end_date"] = pd.to_datetime(idx["end_date"])

    if agg == "M":
        period_df = idx[
            ["t", "year", "quarter", "month", "start_date", "end_date"]
        ].copy()
        period_df["period_id"] = period_df["t"].astype(int)
        period_df["period_label"] = period_df["t"].astype(str)
        period_df = period_df.sort_values(["year", "month", "t"]).reset_index(drop=True)
    elif agg == "Q":
        idx["period_id"] = idx["year"] * 10 + idx["quarter"]
        idx["period_label"] = idx["year"].astype(str) + "Q" + idx["quarter"].astype(str)
        group_cols = ["period_id", "period_label", "year", "quarter"]
        period_df = (
            idx.groupby(group_cols, sort=True)
            .agg(start_date=("start_date", "min"), end_date=("end_date", "max"))
            .reset_index()
            .sort_values(["year", "quarter"])
            .reset_index(drop=True)
        )
        period_df["month"] = np.nan
    elif agg == "A":
        idx["period_id"] = idx["year"].astype(int)
        idx["period_label"] = idx["year"].astype(str)
        group_cols = ["period_id", "period_label", "year"]
        period_df = (
            idx.groupby(group_cols, sort=True)
            .agg(start_date=("start_date", "min"), end_date=("end_date", "max"))
            .reset_index()
            .sort_values(["year"])
            .reset_index(drop=True)
        )
        period_df["quarter"] = np.nan
        period_df["month"] = np.nan
    else:
        raise ValueError("agg must be one of: 'M', 'Q', 'A'.")

    period_df["period_pos"] = np.arange(1, len(period_df) + 1, dtype=int)

    if agg == "M":
        t_to_pos = dict(zip(period_df["t"].astype(int), period_df["period_pos"]))
    else:
        tmp = idx[["t", "period_id"]].drop_duplicates()
        tmp = tmp.merge(period_df[["period_id", "period_pos"]], on="period_id", how="left")
        t_to_pos = dict(zip(tmp["t"].astype(int), tmp["period_pos"].astype(int)))

    return period_df, t_to_pos


def build_period_maps(
    period_index: pd.DataFrame,
) -> Tuple[Dict[int, Tuple[pd.Timestamp, pd.Timestamp]], Dict[int, object], Dict[int, str]]:
    period_map: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    period_ids: Dict[int, object] = {}
    period_labels: Dict[int, str] = {}
    for _, row in period_index.iterrows():
        pos = int(row["period_pos"])
        period_map[pos] = (row["start_date"], row["end_date"])
        period_ids[pos] = row["period_id"]
        period_labels[pos] = str(row["period_label"])
    return period_map, period_ids, period_labels


def _iter_edges_chunks(
    edges_path: Path, *, chunksize: int
) -> Iterable[pd.DataFrame]:
    if edges_path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            logger.warning("pyarrow unavailable (%s); reading parquet at once.", exc)
            yield pd.read_parquet(edges_path, columns=["i", "j", "t"])
            return

        pq_file = pq.ParquetFile(edges_path)
        for rg in range(pq_file.num_row_groups):
            table = pq_file.read_row_group(rg, columns=["i", "j", "t"])
            yield table.to_pandas()
    else:
        for chunk in pd.read_csv(
            edges_path,
            usecols=["i", "j", "t"],
            chunksize=chunksize,
            dtype="int32",
        ):
            yield chunk


def load_us_air_snapshots(
    processed_dir: str | Path,
    *,
    agg: str = "M",
    chunksize: int = 2_000_000,
) -> Tuple[SparseTimeSeries, pd.DataFrame]:
    """Load sparse snapshots aggregated by M/Q/A without dense tensors."""
    processed_dir = Path(processed_dir)
    index_df = load_index_monthly(processed_dir / "index_monthly.csv")
    if index_df["n_nodes"].nunique() != 1:
        raise ValueError("index_monthly has inconsistent n_nodes values.")
    n_nodes = int(index_df["n_nodes"].iloc[0])
    period_index, t_to_pos = build_period_index(index_df, agg)
    edges_path = _select_edges_path(processed_dir)

    n_periods = len(period_index)
    accum: list[sp.spmatrix | None] = [None] * n_periods

    logger.info("Loading edges from %s (agg=%s)", edges_path.name, agg)
    for chunk in _iter_edges_chunks(edges_path, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk = chunk.astype({"i": "int32", "j": "int32", "t": "int32"}, copy=False)
        chunk = chunk[chunk["i"] != chunk["j"]]
        chunk["period_pos"] = chunk["t"].map(t_to_pos)
        missing = chunk["period_pos"].isna()
        if missing.any():
            missing_t = chunk.loc[missing, "t"].unique()
            raise ValueError(f"Missing period mapping for t values: {missing_t[:10]}")
        chunk["period_pos"] = chunk["period_pos"].astype(int)
        chunk = chunk.drop_duplicates(["period_pos", "i", "j"])

        for pos, sub in chunk.groupby("period_pos", sort=False):
            i = sub["i"].to_numpy(dtype=np.int32, copy=False)
            j = sub["j"].to_numpy(dtype=np.int32, copy=False)
            data = np.ones(len(sub), dtype=np.uint8)
            mat = sp.coo_matrix((data, (i, j)), shape=(n_nodes, n_nodes))
            idx = int(pos) - 1
            if accum[idx] is None:
                accum[idx] = mat
            else:
                accum[idx] = accum[idx] + mat

    matrices: list[sp.spmatrix] = []
    for mat in accum:
        if mat is None:
            matrices.append(sp.csr_matrix((n_nodes, n_nodes), dtype=bool))
            continue
        A = mat.tocsr()
        if A.nnz:
            A.data = np.ones_like(A.data, dtype=np.uint8)
        A_sym = (A + A.T).astype(bool)
        A_sym.setdiag(0)
        A_upper = sp.triu(A_sym, k=1, format="csr").astype(bool)
        if A_upper.nnz:
            A_upper.data = np.ones_like(A_upper.data, dtype=np.uint8)
        matrices.append(A_upper)

    series = SparseTimeSeries(matrices, n_nodes)
    logger.info("Snapshots ready: shape=%s", series.shape)
    return series, period_index
