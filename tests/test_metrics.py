"""Unit tests for metrics computation."""
import numpy as np
import pytest

from src.eval.metrics import compute_metrics


def test_metrics_keys():
    rets = np.random.randn(252) * 0.01
    m = compute_metrics(rets)
    required = {"annual_return", "annual_volatility", "max_drawdown", "sharpe_ratio",
                "sortino_ratio", "calmar_ratio", "win_rate", "cumulative_return"}
    assert required.issubset(m.keys())


def test_positive_return_positive_sharpe():
    rets = np.full(252, 0.001)  # constant positive daily return
    m = compute_metrics(rets, risk_free_rate=0.0)
    assert m["sharpe_ratio"] > 0
    assert m["annual_return"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-3)


def test_all_negative_returns():
    rets = np.full(252, -0.001)
    m = compute_metrics(rets)
    assert m["annual_return"] < 0
    assert m["max_drawdown"] < 0


def test_win_rate_bounds():
    rets = np.random.randn(500) * 0.01
    m = compute_metrics(rets)
    assert 0.0 <= m["win_rate"] <= 1.0
