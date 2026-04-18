"""PPO training loop with multi-view GNN encoder."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import trange

from src.utils.config import load_config, get_device
from src.data.downloader import load_data, split_data
from src.data.features import (
    build_feature_matrix,
    make_lookback_tensor,
    get_raw_log_returns,
)
from src.graphs.multi_view import build_multi_view_graphs
from src.envs.portfolio_env import PortfolioEnv
from src.models.actor_critic import GNNActorCritic, FlatActorCritic
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rollout(env: PortfolioEnv, model: GNNActorCritic, n_steps: int, device):
    """Collect n_steps of experience."""
    obs_list, act_list, logp_list, val_list, rew_list, done_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    node_feats_list, ind_graphs, corr_graphs, weights_list = [], [], [], []

    obs, _ = env.reset()
    for _ in range(n_steps):
        x = torch.tensor(env.get_node_features_tensor(), dtype=torch.float32).to(device)
        ind_g, corr_g = env.get_graph_data()
        ind_g = ind_g.to(device)
        corr_g = corr_g.to(device)
        w = torch.tensor(env._weights, dtype=torch.float32).to(device)
        cash = torch.zeros(1, device=device)
        cum_ret = torch.tensor([env._cum_return()], dtype=torch.float32).to(device)

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(
                x.unsqueeze(0),
                ind_g,
                corr_g,
                w.unsqueeze(0),
                cash.unsqueeze(0),
                cum_ret.unsqueeze(0),
            )

        act_np = action.squeeze(0).cpu().numpy()
        obs_next, reward, terminated, truncated, _ = env.step(act_np)

        obs_list.append(obs)
        act_list.append(act_np)
        logp_list.append(log_prob.item())
        val_list.append(value.item())
        rew_list.append(reward)
        done_list.append(float(terminated or truncated))
        node_feats_list.append(x.cpu())
        ind_graphs.append(ind_g.cpu())
        corr_graphs.append(corr_g.cpu())
        weights_list.append(w.cpu())

        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset()

    return dict(
        node_feats=node_feats_list,
        ind_graphs=ind_graphs,
        corr_graphs=corr_graphs,
        weights=weights_list,
        actions=np.array(act_list),
        log_probs=np.array(logp_list),
        values=np.array(val_list),
        rewards=np.array(rew_list),
        dones=np.array(done_list),
    )


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, rollout, cfg, device):
    ppo_cfg = cfg.ppo
    n = len(rollout["actions"])
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"],
        ppo_cfg.gamma,
        ppo_cfg.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_pg_loss = total_vf_loss = total_ent = 0.0
    idx = np.arange(n)

    for _ in range(ppo_cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, ppo_cfg.batch_size):
            mb = idx[start : start + ppo_cfg.batch_size]
            mb_adv = torch.tensor(advantages[mb], dtype=torch.float32, device=device)
            mb_ret = torch.tensor(returns[mb], dtype=torch.float32, device=device)
            mb_old_logp = torch.tensor(
                rollout["log_probs"][mb], dtype=torch.float32, device=device
            )
            mb_act = torch.tensor(
                rollout["actions"][mb], dtype=torch.float32, device=device
            )

            logits_list, value_list, logp_list, ent_list = [], [], [], []
            for i in mb:
                x = rollout["node_feats"][i].to(device)
                ig = rollout["ind_graphs"][i].to(device)
                cg = rollout["corr_graphs"][i].to(device)
                w = rollout["weights"][i].to(device)
                cash = torch.zeros(1, device=device)
                cum_r = torch.zeros(1, device=device)

                _, lp, ent, val = model.get_action_and_value(
                    x.unsqueeze(0),
                    ig,
                    cg,
                    w.unsqueeze(0),
                    cash.unsqueeze(0),
                    cum_r.unsqueeze(0),
                )
                logp_list.append(lp)
                ent_list.append(ent)
                value_list.append(val)

            new_logp = torch.stack(logp_list)
            ent = torch.stack(ent_list).mean()
            new_val = torch.stack(value_list)

            ratio = torch.exp(new_logp - mb_old_logp)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(
                ratio, 1 - ppo_cfg.clip_range, 1 + ppo_cfg.clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            vf_loss = ((new_val - mb_ret) ** 2).mean()
            loss = pg_loss + ppo_cfg.vf_coef * vf_loss - ppo_cfg.ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent += ent.item()

    n_updates = ppo_cfg.n_epochs * max(1, n // ppo_cfg.batch_size)
    return {
        "pg_loss": total_pg_loss / n_updates,
        "vf_loss": total_vf_loss / n_updates,
        "entropy": total_ent / n_updates,
    }


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.project.seed)
    device = get_device(cfg)
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU name: {torch.cuda.get_device_name(device)}")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        tags=list(cfg.wandb.tags),
        config=dict(cfg),
        mode="online" if cfg.wandb.entity else "offline",
    )

    # --- Data pipeline ---
    data = load_data(cfg)
    symbols = list(cfg.data.universe)
    train_data, val_data, _ = split_data(data, cfg)

    features = build_feature_matrix(data)
    log_returns = get_raw_log_returns(
        data, symbols
    )  # raw returns for env reward & graph corr

    train_dates = sorted(set(train_data.index.get_level_values("date")))
    node_feats, valid_dates = make_lookback_tensor(
        features, pd.DatetimeIndex(train_dates), symbols, cfg.features.lookback_window
    )
    returns_arr = np.nan_to_num(
        log_returns.reindex(valid_dates).values.astype(np.float32), nan=0.0
    )

    mv_graphs = build_multi_view_graphs(
        symbols, log_returns.loc[: cfg.data.train_end], cfg
    )

    # --- Environment ---
    env = PortfolioEnv(
        node_features=node_feats,
        dates=valid_dates,
        returns=returns_arr,
        mv_graphs=mv_graphs,
        cfg=cfg,
    )

    # --- Model ---
    node_feat_dim = len(features.columns)
    ablation_variant = getattr(getattr(cfg, "ablation", None), "variant", "full_model")
    use_gnn = ablation_variant != "no_graph"

    if use_gnn:
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
    else:
        model = FlatActorCritic(
            n_assets=len(symbols),
            node_feat_dim=node_feat_dim,
            actor_dims=list(cfg.model.actor.hidden_dims),
            critic_dims=list(cfg.model.critic.hidden_dims),
            dropout=cfg.model.gnn.dropout,
        ).to(device)
    logger.info(
        f"Model: {'GNNActorCritic' if use_gnn else 'FlatActorCritic (no_graph)'}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)

    # --- Training loop ---
    n_rollouts = cfg.ppo.total_timesteps // cfg.ppo.n_steps
    best_sharpe = -np.inf
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for rollout_idx in trange(n_rollouts, desc="Training"):
        rollout = collect_rollout(env, model, cfg.ppo.n_steps, device)
        losses = ppo_update(model, optimizer, rollout, cfg, device)
        wandb.log({"rollout": rollout_idx, **losses})

        if rollout_idx % cfg.training.val_freq == 0:
            val_metrics = evaluate(
                model, val_data, features, log_returns, mv_graphs, symbols, cfg, device
            )
            wandb.log({"val/" + k: v for k, v in val_metrics.items()})
            logger.info(f"Rollout {rollout_idx}: {val_metrics}")

            if val_metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
                best_sharpe = val_metrics["sharpe_ratio"]
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    wandb.finish()
    logger.info("Training complete. Best val Sharpe: %.4f", best_sharpe)


def evaluate(model, val_data, features, log_returns, mv_graphs, symbols, cfg, device):
    val_dates = sorted(set(val_data.index.get_level_values("date")))
    node_feats, valid_dates = make_lookback_tensor(
        features, pd.DatetimeIndex(val_dates), symbols, cfg.features.lookback_window
    )
    returns_arr = np.nan_to_num(
        log_returns.reindex(valid_dates).values.astype(np.float32), nan=0.0
    )
    val_env = PortfolioEnv(node_feats, valid_dates, returns_arr, mv_graphs, cfg)
    model.eval()
    portfolio_returns = []
    obs, _ = val_env.reset()
    done = False
    while not done:
        x = torch.tensor(val_env.get_node_features_tensor(), dtype=torch.float32).to(
            device
        )
        ig, cg = val_env.get_graph_data()
        ig, cg = ig.to(device), cg.to(device)
        w = torch.tensor(val_env._weights, dtype=torch.float32).to(device)
        cash = torch.zeros(1, device=device)
        cr = torch.tensor([val_env._cum_return()], dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(
                x.unsqueeze(0),
                ig,
                cg,
                w.unsqueeze(0),
                cash.unsqueeze(0),
                cr.unsqueeze(0),
                deterministic=True,
            )
        obs, _, terminated, truncated, info = val_env.step(
            action.squeeze(0).cpu().numpy()
        )
        portfolio_returns.append(info["port_log_ret"])
        done = terminated or truncated
    model.train()
    return compute_metrics(np.array(portfolio_returns), cfg.eval.risk_free_rate)


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
