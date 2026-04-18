"""Multi-view graph container: industry (static) + correlation (dynamic)."""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
from torch_geometric.data import Data

from src.graphs.industry_graph import build_industry_graph
from src.graphs.corr_graph import build_corr_graph_series, get_corr_graph_for_date


@dataclass
class MultiViewGraphs:
    symbols: list[str]
    industry_graph: Data
    corr_graphs: dict[pd.Timestamp, Data]  # date → PyG Data

    def get(self, date: pd.Timestamp) -> tuple[Data, Data]:
        """Return (industry_graph, corr_graph) for given date."""
        corr = get_corr_graph_for_date(date, self.corr_graphs)
        return self.industry_graph, corr


def build_multi_view_graphs(
    symbols: list[str],
    log_returns: pd.DataFrame,
    cfg,
) -> MultiViewGraphs:
    """
    One-time construction of all static + dynamic graphs.
    Call this once before training; pass the result to the environment.
    """
    g_cfg = cfg.graph
    ind_graph = build_industry_graph(symbols)
    corr_graphs = build_corr_graph_series(
        log_returns=log_returns,
        window=g_cfg.corr_graph.window,
        threshold=g_cfg.corr_graph.threshold,
        topk=g_cfg.corr_graph.topk,
        rebuild_freq="W-FRI",
    )
    return MultiViewGraphs(
        symbols=symbols,
        industry_graph=ind_graph,
        corr_graphs=corr_graphs,
    )
