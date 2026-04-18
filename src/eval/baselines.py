"""Baseline portfolio strategies: 1/N, Markowitz, Buy-and-Hold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.eval.metrics import compute_metrics


def equal_weight(returns: pd.DataFrame) -> dict[str, float]:
    """1/N equal-weight, daily rebalanced."""
    w = np.ones(returns.shape[1]) / returns.shape[1]
    log_rets = (returns @ w).values
    return compute_metrics(log_rets)


def buy_and_hold(returns: pd.DataFrame) -> dict[str, float]:
    """Buy-and-hold equal weight (no rebalancing)."""
    n = returns.shape[1]
    w = np.ones(n) / n
    cum_rets = np.exp(returns.cumsum())
    port_value = (cum_rets * w).sum(axis=1)
    log_port_rets = np.log(port_value / port_value.shift(1).fillna(port_value.iloc[0])).values
    return compute_metrics(log_port_rets[1:])


def markowitz(
    returns: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_freq: str = "ME",
    risk_free_rate: float = 0.02,
    cov_shrinkage: bool = True,
) -> dict[str, float]:
    """Rolling Markowitz max-Sharpe portfolio with Ledoit-Wolf shrinkage."""
    from sklearn.covariance import LedoitWolf

    rebalance_dates = returns.resample(rebalance_freq).last().index
    weights_hist: dict[pd.Timestamp, np.ndarray] = {}
    n = returns.shape[1]

    for rd in rebalance_dates:
        hist = returns.loc[:rd].tail(estimation_window)
        if len(hist) < 30:
            weights_hist[rd] = np.ones(n) / n
            continue
        mu = hist.mean().values * 252
        if cov_shrinkage:
            lw = LedoitWolf().fit(hist.values)
            sigma = lw.covariance_ * 252
        else:
            sigma = hist.cov().values * 252

        def neg_sharpe(w):
            p_ret = w @ mu
            p_vol = np.sqrt(w @ sigma @ w)
            return -(p_ret - risk_free_rate) / (p_vol + 1e-8)

        cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        weights_hist[rd] = res.x if res.success else w0

    # Simulate portfolio
    log_rets = []
    current_w = np.ones(n) / n
    for i, date in enumerate(returns.index):
        past_rd = [rd for rd in rebalance_dates if rd <= date]
        if past_rd:
            current_w = weights_hist[max(past_rd)]
        log_rets.append(float(returns.iloc[i].values @ current_w))

    return compute_metrics(np.array(log_rets))


def csi300_benchmark(start_date: str, end_date: str) -> dict[str, float]:
    """Download CSI 300 index returns via tushare and compute metrics."""
    import os, tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    pro = ts.pro_api()
    df = pro.index_daily(
        ts_code="000300.SH",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        fields="trade_date,close",
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    log_rets = np.log(df["close"] / df["close"].shift(1)).dropna().values
    return compute_metrics(log_rets)


def run_all_baselines(log_returns: pd.DataFrame, cfg) -> dict[str, dict[str, float]]:
    """Run all baseline strategies and return results dict."""
    results = {}
    results["equal_weight"] = equal_weight(log_returns)
    results["buy_and_hold"] = buy_and_hold(log_returns)
    results["markowitz"] = markowitz(log_returns, risk_free_rate=cfg.eval.risk_free_rate)
    try:
        idx = log_returns.index
        results["csi300"] = csi300_benchmark(str(idx.min().date()), str(idx.max().date()))
    except Exception as e:
        import logging; logging.getLogger(__name__).warning(f"CSI300 download failed: {e}")
    return results
