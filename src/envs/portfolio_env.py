"""
Custom Gymnasium environment for portfolio optimization with graph state.
Observation: dict with node features, graph handles, and portfolio context.
Action: weight vector in simplex (N,).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data

from src.graphs.multi_view import MultiViewGraphs


class PortfolioEnv(gym.Env):
    """
    Episode = one contiguous time segment of daily data.
    At each step t, the agent observes today's node features + graphs,
    outputs target weights, and receives reward based on t+1 returns.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        node_features: np.ndarray,    # (T, N, F) — already aligned to dates
        dates: list[pd.Timestamp],
        returns: np.ndarray,           # (T, N) — log returns per step
        mv_graphs: MultiViewGraphs,
        cfg,
    ):
        super().__init__()
        self.node_features = node_features  # (T, N, F)
        self.dates = dates
        self.returns = returns
        self.mv_graphs = mv_graphs
        self.cfg = cfg

        T, N, F = node_features.shape
        self.T, self.N, self.F = T, N, F

        self.transaction_cost = cfg.env.transaction_cost
        self.risk_penalty = cfg.env.risk_penalty
        self.vol_window = cfg.env.vol_window
        self.initial_capital = cfg.env.initial_capital

        # Observation: flat array for compatibility; actual tensors unpacked in training loop
        obs_dim = N * F + N + 2  # node features (flat) + weights + cash + cum_ret
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)

        self._t = 0
        self._weights = np.ones(N, dtype=np.float32) / N
        self._portfolio_value = float(self.initial_capital)
        self._return_history: list[float] = []

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        feats = self.node_features[self._t].flatten()
        cum_ret = np.array([self._cum_return()], dtype=np.float32)
        cash = np.array([0.0], dtype=np.float32)  # fully invested; cash_ratio always 0
        return np.concatenate([feats, self._weights, cash, cum_ret])

    def _cum_return(self) -> float:
        if not self._return_history:
            return 0.0
        return float(np.exp(np.sum(self._return_history)) - 1)

    def _rolling_vol(self) -> float:
        if len(self._return_history) < 2:
            return 0.0
        window = self._return_history[-self.vol_window:]
        return float(np.std(window))

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._weights = np.ones(self.N, dtype=np.float32) / self.N
        self._portfolio_value = float(self.initial_capital)
        self._return_history = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 1e-6, None)
        new_weights = action / action.sum()

        turnover = np.abs(new_weights - self._weights).sum()
        tc = self.transaction_cost * turnover

        r_assets = self.returns[self._t]                          # log returns (N,)
        port_log_ret = float(np.dot(new_weights, r_assets))       # portfolio log return
        risk_pen = self.risk_penalty * self._rolling_vol()
        reward = port_log_ret - risk_pen - tc

        self._return_history.append(port_log_ret)
        self._portfolio_value *= np.exp(port_log_ret - tc)
        self._weights = new_weights
        self._t += 1

        terminated = self._t >= self.T - 1
        truncated = False
        info = {
            "portfolio_value": self._portfolio_value,
            "port_log_ret": port_log_ret,
            "turnover": turnover,
            "date": self.dates[self._t] if not terminated else self.dates[-1],
        }
        return self._obs(), reward, terminated, truncated, info

    def get_graph_data(self) -> tuple[Data, Data]:
        """Return (ind_graph, corr_graph) for current timestep."""
        return self.mv_graphs.get(self.dates[self._t])

    def get_node_features_tensor(self):
        """Return raw node feature matrix for current step as numpy array (N, F)."""
        return self.node_features[self._t]
