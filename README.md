# graphfolio-rl

Multi-view GNN + PPO for A-share portfolio optimization.

## Overview

Combines a heterogeneous graph neural network (industry + correlation views) with Proximal Policy Optimization to learn daily rebalancing policies on 20 A-share blue-chip stocks (2018–2024).

**Architecture:**
- **Graph encoder**: GAT (2 layers, 64 hidden, 4 heads) with gated fusion of two views
  - *Industry graph*: static, Shenwan L1 sector edges
  - *Correlation graph*: dynamic, 20-day rolling Pearson correlation (threshold 0.3, top-5 neighbors)
- **Actor-Critic**: Dirichlet policy head + value head (256→128 MLP)
- **Training**: PPO with GAE (γ=0.99, λ=0.95, clip=0.2)
- **Reward**: daily log return − λ·rolling volatility penalty

**Data:** 20 CSI 300 constituents, daily OHLCV, 7 technical features, z-score normalized.  
**Split:** Train 2018–2022 | Val 2023 | Test 2024.

## Setup

```bash
# Install pixi (https://pixi.sh) then:
pixi install

# Add Tushare token to .env
echo "TUSHARE_TOKEN=your_token_here" > .env
```

## Usage

```bash
# Download & process data
pixi run download

# Run baselines (equal-weight, momentum, min-variance)
pixi run baselines

# Train GNN+PPO (full model)
pixi run train

# Train no-GNN ablation (FlatActorCritic)
pixi run train-ablation

# Run full experiment pipeline
pixi run bash run_experiments.sh

# Run tests
pixi run test
```

## Project Structure

```
configs/          # default.yaml + ablation.yaml
src/
  data/           # Tushare downloader, feature engineering
  graphs/         # industry_graph, corr_graph, multi_view fusion
  models/         # GNNActorCritic, FlatActorCritic, GAT encoder
  agents/         # PPO training loop
  envs/           # PortfolioEnv (Gymnasium interface)
  eval/           # metrics, backtesting, visualization
tests/            # unit tests for env, graphs, metrics
run_experiments.py / run_experiments.sh  # end-to-end pipeline
```

## Ablation

Two main variants compared:

| Variant | Model | Description |
|---------|-------|-------------|
| `no_graph` | FlatActorCritic | PPO only, no GNN |
| `full_model` | GNNActorCritic | GAT + gated fusion + PPO |

Config in `configs/ablation.yaml`. Run via `pixi run train-ablation`.
