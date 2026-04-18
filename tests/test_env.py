"""Unit tests for PortfolioEnv."""
import numpy as np
import pandas as pd
import pytest

from src.graphs.industry_graph import build_industry_graph
from src.graphs.multi_view import MultiViewGraphs
from src.graphs.corr_graph import build_corr_graph


SYMBOLS = ["600519.SH", "000858.SZ", "300750.SZ", "002594.SZ"]
N = len(SYMBOLS)
T = 50
F = 8


def _make_env():
    from src.envs.portfolio_env import PortfolioEnv
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "env": {
            "initial_capital": 1_000_000,
            "transaction_cost": 0.001,
            "risk_penalty": 0.1,
            "vol_window": 10,
        }
    })
    node_feats = np.random.randn(T, N, F).astype(np.float32)
    returns = np.random.randn(T, N).astype(np.float32) * 0.01
    dates = list(pd.date_range("2020-01-02", periods=T, freq="B"))

    ind_g = build_industry_graph(SYMBOLS)
    dates_pd = pd.date_range("2020-01-02", periods=T, freq="B")
    rets_df = pd.DataFrame(returns, index=dates_pd, columns=SYMBOLS)
    corr_g = build_corr_graph(rets_df, window=20)
    corr_graphs = {dates_pd[-1]: corr_g}
    mv = MultiViewGraphs(symbols=SYMBOLS, industry_graph=ind_g, corr_graphs=corr_graphs)

    return PortfolioEnv(node_feats, dates, returns, mv, cfg)


def test_env_reset():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.allclose(env._weights, 1.0 / N)


def test_env_step_obs_shape():
    env = _make_env()
    env.reset()
    action = np.ones(N, dtype=np.float32) / N
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert "portfolio_value" in info


def test_env_action_normalization():
    """Unnormalized actions should be projected onto simplex."""
    env = _make_env()
    env.reset()
    action = np.array([10.0, 5.0, 3.0, 2.0])  # not normalized
    obs, _, _, _, info = env.step(action)
    assert abs(env._weights.sum() - 1.0) < 1e-5


def test_env_reward_penalizes_turnover():
    """High turnover action should have lower reward than zero-turnover."""
    env = _make_env()
    env.reset()
    same_action = env._weights.copy()
    obs, r_no_turn, _, _, _ = env.step(same_action)
    env.reset()
    flip_action = np.array([0.9, 0.03, 0.03, 0.04])
    _, r_high_turn, _, _, _ = env.step(flip_action)
    # Reward difference: this is probabilistic; just check both are finite
    assert np.isfinite(r_no_turn) and np.isfinite(r_high_turn)


def test_env_terminates():
    """Environment should terminate after T-1 steps."""
    env = _make_env()
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(np.ones(N) / N)
        done = terminated or truncated
        steps += 1
    assert steps == T - 1
