"""Portfolio performance metrics."""
from __future__ import annotations

import numpy as np


def compute_metrics(log_returns: np.ndarray, risk_free_rate: float = 0.02) -> dict[str, float]:
    """
    log_returns: 1-D array of daily log portfolio returns.
    risk_free_rate: annual risk-free rate.
    """
    daily_rf = risk_free_rate / 252
    r = log_returns
    cum_returns = np.exp(np.cumsum(r))

    annual_return = float(np.exp(r.mean() * 252) - 1)
    annual_vol = float(r.std() * np.sqrt(252))
    sharpe = float((r.mean() - daily_rf) / (r.std() + 1e-8) * np.sqrt(252))

    # Sortino: downside deviation
    downside = r[r < daily_rf]
    sortino = float((r.mean() - daily_rf) / (downside.std() + 1e-8) * np.sqrt(252)) if len(downside) else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_dd = float(drawdown.min())

    calmar = float(annual_return / (abs(max_dd) + 1e-8))
    win_rate = float((r > 0).mean())

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "cumulative_return": float(cum_returns[-1] - 1),
    }
