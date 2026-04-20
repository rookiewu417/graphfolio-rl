#!/bin/bash
# Run all experiments sequentially and save results
# Usage: pixi run bash run_experiments.sh

set -e  # exit on error

CKPT_DIR="results/checkpoints"
LOG_DIR="results/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "=========================================="
echo " graphfolio-rl: Full Experiment Pipeline"
echo "=========================================="

# ── 1. Baselines ──────────────────────────────
echo ""
echo "[1/3] Running baselines..."
pixi run baselines 2>&1 | tee "$LOG_DIR/baselines.log"
echo "✓ Baselines done"

# ── 2. W3: no_graph (FlatActorCritic) ─────────
echo ""
echo "[2/3] Training W3: no_graph PPO (FlatActorCritic)..."
pixi run train-ablation 2>&1 | tee "$LOG_DIR/train_no_graph.log"
cp "$CKPT_DIR/best_model.pt" "$CKPT_DIR/best_model_no_graph.pt"
echo "✓ W3 done → saved to best_model_no_graph.pt"

# ── 3. W4-W5: full_model (GNN + PPO) ──────────
echo ""
echo "[3/3] Training W4-W5: full_model (GNN + PPO)..."
pixi run train 2>&1 | tee "$LOG_DIR/train_full_model.log"
cp "$CKPT_DIR/best_model.pt" "$CKPT_DIR/best_model_full.pt"
echo "✓ W4-W5 done → saved to best_model_full.pt"

# ── 4. Final evaluation ────────────────────────
echo ""
echo "[4/4] Evaluating all models on 2025 test set..."
pixi run python - << 'PYEOF'
import torch, numpy as np, pandas as pd
from src.utils.config import load_config
from src.data.downloader import load_data, split_data
from src.data.features import build_feature_matrix, make_lookback_tensor, get_raw_log_returns
from src.graphs.multi_view import build_multi_view_graphs
from src.envs.portfolio_env import PortfolioEnv
from src.models.actor_critic import FlatActorCritic, GNNActorCritic
from src.eval.metrics import compute_metrics

def eval_model(model, cfg, node_feats, valid_dates, returns_arr, mv_graphs, device):
    env = PortfolioEnv(node_feats, valid_dates, returns_arr, mv_graphs, cfg)
    model.eval()
    rets = []
    obs, _ = env.reset()
    done = False
    while not done:
        x = torch.tensor(env.get_node_features_tensor(), dtype=torch.float32).unsqueeze(0).to(device)
        ig, cg = env.get_graph_data()
        ig, cg = ig.to(device), cg.to(device)
        w = torch.tensor(env._weights, dtype=torch.float32).unsqueeze(0).to(device)
        cash = torch.zeros(1, 1, device=device)
        cr = torch.tensor([[env._cum_return()]], dtype=torch.float32, device=device)
        with torch.no_grad():
            action, *_ = model.get_action_and_value(x, ig, cg, w, cash, cr, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        rets.append(info['port_log_ret'])
        done = terminated or truncated
    return compute_metrics(np.array(rets), cfg.eval.risk_free_rate)

device = torch.device('cuda')
cfg_ablation = load_config('configs/ablation.yaml')
cfg_full = load_config('configs/default.yaml')
data = load_data(cfg_full)
symbols = list(cfg_full.data.universe)
_, _, test_data = split_data(data, cfg_full)
features = build_feature_matrix(data)
log_returns = get_raw_log_returns(data, symbols)
mv_graphs = build_multi_view_graphs(symbols, log_returns.loc[:cfg_full.data.train_end], cfg_full)
test_dates = sorted(set(test_data.index.get_level_values('date')))
node_feats, valid_dates = make_lookback_tensor(features, pd.DatetimeIndex(test_dates), symbols, cfg_full.features.lookback_window)
returns_arr = np.nan_to_num(log_returns.reindex(valid_dates).values.astype(np.float32), nan=0.0)
node_feat_dim = len(features.columns)

results = {}

# no_graph
m = FlatActorCritic(n_assets=20, node_feat_dim=node_feat_dim, actor_dims=[256,128], critic_dims=[256,128]).to(device)
m.load_state_dict(torch.load('results/checkpoints/best_model_no_graph.pt', map_location=device, weights_only=True))
results['PPO (no_graph)'] = eval_model(m, cfg_ablation, node_feats, valid_dates, returns_arr, mv_graphs, device)

# full_model
m = GNNActorCritic(n_assets=20, node_feat_dim=node_feat_dim, gnn_hidden=64, gnn_out=64, gnn_heads=4, actor_dims=[256,128], critic_dims=[256,128]).to(device)
m.load_state_dict(torch.load('results/checkpoints/best_model_full.pt', map_location=device, weights_only=True))
results['PPO (GNN full)'] = eval_model(m, cfg_full, node_feats, valid_dates, returns_arr, mv_graphs, device)

# Print comparison table
print("\n=== 2025 Test Set Results ===")
metrics = ['annual_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']
header = f"{'Strategy':<22}" + "".join(f"{m:>14}" for m in metrics)
print(header)
print("-" * len(header))
for name, r in results.items():
    row = f"{name:<22}" + "".join(f"{r[m]:>14.4f}" for m in metrics)
    print(row)

# Save to CSV
rows = [{'strategy': k, **v} for k, v in results.items()]
df = pd.DataFrame(rows).set_index('strategy')
df.to_csv('results/ppo_results_2025.csv')
print("\nSaved to results/ppo_results_2025.csv")
PYEOF

echo ""
echo "=========================================="
echo " All experiments complete!"
echo " Logs: results/logs/"
echo " Checkpoints: results/checkpoints/"
echo " Results: results/ppo_results_2025.csv"
echo "=========================================="
