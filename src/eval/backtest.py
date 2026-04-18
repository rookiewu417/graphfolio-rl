"""Unified backtest interface for all strategies."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config, get_device
from src.data.downloader import load_data, split_data
from src.data.features import build_feature_matrix, make_lookback_tensor, get_raw_log_returns
from src.graphs.multi_view import build_multi_view_graphs
from src.envs.portfolio_env import PortfolioEnv
from src.models.actor_critic import GNNActorCritic
from src.eval.metrics import compute_metrics
from src.eval.baselines import run_all_baselines

logger = logging.getLogger(__name__)


def backtest_model(
    model: GNNActorCritic,
    node_feats: np.ndarray,
    dates: list,
    returns_arr: np.ndarray,
    mv_graphs,
    cfg,
    device,
) -> tuple[dict, list[float]]:
    env = PortfolioEnv(node_feats, dates, returns_arr, mv_graphs, cfg)
    model.eval()
    port_returns = []
    obs, _ = env.reset()
    done = False
    while not done:
        x = torch.tensor(env.get_node_features_tensor(), dtype=torch.float32).to(device)
        ig, cg = env.get_graph_data()
        ig, cg = ig.to(device), cg.to(device)
        w = torch.tensor(env._weights, dtype=torch.float32).to(device)
        cash = torch.zeros(1, device=device)
        cr = torch.tensor([env._cum_return()], dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(
                x.unsqueeze(0), ig, cg, w.unsqueeze(0), cash.unsqueeze(0), cr.unsqueeze(0),
                deterministic=True,
            )
        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        port_returns.append(info["port_log_ret"])
        done = terminated or truncated
    model.train()
    return compute_metrics(np.array(port_returns), cfg.eval.risk_free_rate), port_returns


def print_results_table(results: dict[str, dict]):
    metrics = ["annual_return", "annual_volatility", "max_drawdown",
               "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate"]
    header = f"{'Strategy':<25}" + "".join(f"{m:>18}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<25}" + "".join(f"{m.get(k, float('nan')):>18.4f}" for k in metrics)
        print(row)


def main(config_path: str = "configs/default.yaml", checkpoint: str = None):
    cfg = load_config(config_path)
    device = get_device(cfg)
    symbols = list(cfg.data.universe)

    data = load_data(cfg)
    _, _, test_data = split_data(data, cfg)
    features = build_feature_matrix(data)
    log_returns = get_raw_log_returns(data, symbols)

    test_dates = sorted(set(test_data.index.get_level_values("date")))
    node_feats, valid_dates = make_lookback_tensor(
        features, pd.DatetimeIndex(test_dates), symbols, cfg.features.lookback_window
    )
    returns_arr = log_returns.reindex(valid_dates).values.astype(np.float32)

    mv_graphs = build_multi_view_graphs(symbols, log_returns, cfg)

    # --- Baseline results ---
    test_log_returns = log_returns.reindex(valid_dates)
    results = run_all_baselines(test_log_returns, cfg)

    # --- Model result ---
    ckpt = checkpoint or Path(cfg.training.checkpoint_dir) / "best_model.pt"
    if Path(ckpt).exists():
        node_feat_dim = len(features.columns)
        model = GNNActorCritic(
            n_assets=len(symbols),
            node_feat_dim=node_feat_dim,
            gnn_hidden=cfg.model.gnn.hidden_dim,
            gnn_out=cfg.model.gnn.hidden_dim,
            gnn_heads=cfg.model.gnn.heads,
            actor_dims=list(cfg.model.actor.hidden_dims),
            critic_dims=list(cfg.model.critic.hidden_dims),
            dropout=cfg.model.gnn.dropout,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model_metrics, _ = backtest_model(model, node_feats, valid_dates, returns_arr, mv_graphs, cfg, device)
        results["graphfolio_rl"] = model_metrics
    else:
        logger.warning(f"Checkpoint not found: {ckpt}")

    print_results_table(results)

    # Save CSV
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).T.to_csv("results/backtest_results.csv")
    logger.info("Saved results/backtest_results.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
