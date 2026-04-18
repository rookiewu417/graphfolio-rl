"""Dynamic rolling-correlation graph, rebuilt weekly."""
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
    """Convert correlation matrix to sparse edge_index + edge_weight via threshold + top-k."""
    n = corr.shape[0]
    np.fill_diagonal(corr, 0.0)

    # threshold mask
    mask = corr >= threshold

    # top-k per row (keep strongest connections)
    topk_mask = np.zeros_like(corr, dtype=bool)
    for i in range(n):
        row = corr[i]
        k = min(topk, (row > 0).sum())
        if k > 0:
            idx = np.argpartition(row, -k)[-k:]
            topk_mask[i, idx] = True

    combined = mask & topk_mask
    src, dst = np.where(combined)
    weights = corr[src, dst]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_corr_graph(
    returns: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.3,
    topk: int = 5,
) -> Data:
    """
    returns: DataFrame of shape (window, N) — log returns for the lookback window.
    Returns PyG Data with edge_index and edge_weight.
    """
    if len(returns) < window:
        n = returns.shape[1]
        return Data(num_nodes=n, edge_index=torch.zeros(2, 0, dtype=torch.long))

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
    window: int = 60,
    threshold: float = 0.3,
    topk: int = 5,
    rebuild_freq: str = "W-FRI",
) -> dict[pd.Timestamp, Data]:
    """
    Build a dict of {date: Data} correlation graphs, rebuilt at rebuild_freq.
    log_returns: DataFrame (dates × symbols).
    Uses only historical data up to each rebuild date (no look-ahead).
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
    """Return the most recent correlation graph built before or on `date`."""
    valid = [d for d in graphs if d <= date]
    if not valid:
        g = list(graphs.values())[0]
        return Data(num_nodes=g.num_nodes, edge_index=torch.zeros(2, 0, dtype=torch.long))
    return graphs[max(valid)]
