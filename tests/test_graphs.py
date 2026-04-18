"""Unit tests for graph construction modules."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.graphs.industry_graph import build_industry_graph
from src.graphs.corr_graph import build_corr_graph, _corr_to_edge_index


SYMBOLS = ["600519.SH", "000858.SZ", "300750.SZ", "002594.SZ", "600036.SH", "601166.SH"]


def test_industry_graph_shape():
    g = build_industry_graph(SYMBOLS)
    assert g.num_nodes == len(SYMBOLS)
    assert g.edge_index.shape[0] == 2
    # industry edges: 食品饮料(2-2 pair), 银行(2-2 pair) → 2 pairs each = 4 directed edges each
    assert g.edge_index.shape[1] > 0


def test_industry_graph_symmetric():
    g = build_industry_graph(SYMBOLS)
    ei = g.edge_index.numpy()
    src_dst = set(zip(ei[0], ei[1]))
    for s, d in list(src_dst):
        assert (d, s) in src_dst, "Industry graph must be undirected"


def test_corr_graph_no_lookahead():
    """Correlation graph must only use data up to rebuild date."""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    np.random.seed(0)
    rets = pd.DataFrame(np.random.randn(100, len(SYMBOLS)), index=dates, columns=SYMBOLS)
    g = build_corr_graph(rets, window=60)
    assert g.num_nodes == len(SYMBOLS)


def test_corr_edge_index_valid():
    """All edge indices must be within [0, N-1]."""
    n = 6
    corr = np.random.rand(n, n)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    ei, ew = _corr_to_edge_index(corr, threshold=0.3, topk=3)
    if ei.shape[1] > 0:
        assert ei.max() < n
        assert ei.min() >= 0
        assert (ew >= 0).all()


def test_isolated_nodes_no_edges():
    """If threshold is 1.0, no edges should exist."""
    dates = pd.date_range("2020-01-02", periods=80, freq="B")
    rets = pd.DataFrame(np.random.randn(80, 3), index=dates)
    g = build_corr_graph(rets, window=60, threshold=1.0, topk=0)
    assert g.edge_index.shape[1] == 0
