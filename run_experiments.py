"""Full experiment pipeline: baselines → W3 → full_model → eval."""
import subprocess, sys, shutil, os
from pathlib import Path

CKPT = Path("results/checkpoints")
LOGS = Path("results/logs")
CKPT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

def run(cmd, log_file):
    print(f"\n>>> {cmd}")
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()
        if proc.returncode != 0:
            print(f"[ERROR] command failed with code {proc.returncode}")
            sys.exit(proc.returncode)

print("=" * 50)
print("  graphfolio-rl: Full Experiment Pipeline")
print("=" * 50)

# 1. Baselines
print("\n[1/4] Running baselines...")
run("pixi run baselines", LOGS / "baselines.log")
print("✓ Baselines done")

# 2. W3: no_graph
print("\n[2/4] Training W3: no_graph PPO...")
run("pixi run train-ablation", LOGS / "train_no_graph.log")
shutil.copy(CKPT / "best_model.pt", CKPT / "best_model_no_graph.pt")
print("✓ W3 done → best_model_no_graph.pt")

# 3. full_model
print("\n[3/4] Training full_model GNN+PPO...")
run("pixi run train", LOGS / "train_full_model.log")
shutil.copy(CKPT / "best_model.pt", CKPT / "best_model_full.pt")
print("✓ full_model done → best_model_full.pt")

# 4. Evaluation
print("\n[4/4] Evaluating on 2024 test set...")
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
        rets.append(info["port_log_ret"])
        done = terminated or truncated
    return compute_metrics(np.array(rets), cfg.eval.risk_free_rate)

device = torch.device("cuda")
cfg_abl = load_config("configs/ablation.yaml")
cfg_full = load_config("configs/default.yaml")
data = load_data(cfg_full)
symbols = list(cfg_full.data.universe)
_, _, test_data = split_data(data, cfg_full)
features = build_feature_matrix(data)
log_returns = get_raw_log_returns(data, symbols)
mv_graphs = build_multi_view_graphs(symbols, log_returns, cfg_full)  # full returns: test graphs reflect 2024 conditions
test_dates = sorted(set(test_data.index.get_level_values("date")))
node_feats, valid_dates = make_lookback_tensor(
    features, pd.DatetimeIndex(test_dates), symbols, cfg_full.features.lookback_window
)
returns_arr = np.nan_to_num(
    log_returns.reindex(valid_dates).values.astype(np.float32), nan=0.0
)
node_feat_dim = len(features.columns)

results = {}

m = FlatActorCritic(n_assets=20, node_feat_dim=node_feat_dim,
                    actor_dims=[256, 128], critic_dims=[256, 128]).to(device)
m.load_state_dict(torch.load(CKPT / "best_model_no_graph.pt", map_location=device, weights_only=True))
results["PPO (no_graph)"] = eval_model(m, cfg_abl, node_feats, valid_dates, returns_arr, mv_graphs, device)

m = GNNActorCritic(n_assets=20, node_feat_dim=node_feat_dim, gnn_hidden=64, gnn_out=64,
                   gnn_heads=4, actor_dims=[256, 128], critic_dims=[256, 128]).to(device)
m.load_state_dict(torch.load(CKPT / "best_model_full.pt", map_location=device, weights_only=True))
results["PPO (GNN full)"] = eval_model(m, cfg_full, node_feats, valid_dates, returns_arr, mv_graphs, device)

metrics = ["annual_return", "max_drawdown", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate"]
print("\n=== 2024 Test Set Results ===")
header = f"{'Strategy':<22}" + "".join(f"{k:>15}" for k in metrics)
print(header)
print("-" * len(header))
for name, r in results.items():
    print(f"{name:<22}" + "".join(f"{r[k]:>15.4f}" for k in metrics))

df = pd.DataFrame([{"strategy": k, **v} for k, v in results.items()]).set_index("strategy")
df.to_csv("results/ppo_results_2024.csv")
print("\nSaved → results/ppo_results_2024.csv")
print("\n" + "=" * 50)
print("  All experiments complete!")
print("=" * 50)
