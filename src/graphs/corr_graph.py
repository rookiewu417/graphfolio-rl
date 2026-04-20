"""Dynamic rolling-correlation graph with signed edges, rebuilt daily."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _corr_to_edge_index(
    corr: np.ndarray,
    threshold: float = 0.3,
    topk: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert correlation matrix to sparse edge_index + SIGNED edge_weight.

    Edges are kept when |corr| >= threshold AND within top-k per row.
    edge_weight retains the sign (positive = co-movement, negative = contrarian).
    """
    n = corr.shape[0]
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    mask = abs_corr >= threshold

    topk_mask = np.zeros_like(abs_corr, dtype=bool)
    for i in range(n):
        row = abs_corr[i]
        k = min(topk, int((row > 0).sum()))
        if k > 0:
            idx = np.argpartition(row, -k)[-k:]
            topk_mask[i, idx] = True

    combined = mask & topk_mask
    src, dst = np.where(combined)
    weights = corr[src, dst].astype(np.float32)  # signed

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_corr_graph(
    returns: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.3,
    topk: int = 5,
) -> Data:
    """
    returns: DataFrame (T, N) of log returns.
    Uses the last `window` rows.  Returns PyG Data with signed edge_weight.
    """
    min_obs = max(5, window // 4)
    if len(returns) < min_obs:
        n = returns.shape[1]
        return Data(
            num_nodes=n,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float32),
        )

    corr = returns.tail(window).corr().values.astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0)
    edge_index, edge_weight = _corr_to_edge_index(corr, threshold, topk)
    return Data(
        num_nodes=returns.shape[1],
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def build_corr_graph_series(
    log_returns: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.3,
    topk: int = 5,
    rebuild_freq: str = "B",
) -> dict[pd.Timestamp, Data]:
    """
    Build {date: Data} graphs rebuilt at `rebuild_freq` frequency.
    Default "B" = every business day (was "W-FRI").
    Each graph uses only the `window` days of returns up to that date (no look-ahead).
    """
    rebuild_dates = log_returns.resample(rebuild_freq).last().index
    graphs: dict[pd.Timestamp, Data] = {}
    for rd in rebuild_dates:
        hist = log_returns.loc[:rd]
        graphs[rd] = build_corr_graph(hist, window, threshold, topk)
    return graphs


def get_corr_graph_for_date(
    date: pd.Timestamp,
    graphs: dict[pd.Timestamp, Data],
) -> Data:
    """Return the most recent correlation graph built on or before `date`."""
    valid = [d for d in graphs if d <= date]
    if not valid:
        g = next(iter(graphs.values()))
        return Data(
            num_nodes=g.num_nodes,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float32),
        )
    return graphs[max(valid)]
