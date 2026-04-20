"""Generate all visualizations for the graphfolio-rl experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.downloader import load_data, split_data
from src.data.features import build_feature_matrix, make_lookback_tensor, get_raw_log_returns
from src.graphs.multi_view import build_multi_view_graphs
from src.envs.portfolio_env import PortfolioEnv
from src.models.actor_critic import FlatActorCritic, GNNActorCritic

FIGURES_DIR = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "PPO (GNN)":       "#2563EB",
    "PPO (no_graph)":  "#7C3AED",
    "Equal Weight":    "#CA8A04",
    "Markowitz":       "#DC2626",
    "CSI300":          "#6B7280",
}

STYLE = {
    "PPO (GNN)":       dict(lw=2.5, zorder=5),
    "PPO (no_graph)":  dict(lw=2.0, zorder=4, ls="--"),
    "Equal Weight":    dict(lw=1.5, zorder=3, ls="-."),
    "Markowitz":       dict(lw=1.2, zorder=2, ls=":"),
    "CSI300":          dict(lw=1.2, zorder=2, ls=":"),
}

STOCK_NAMES = {
    "600519.SH": "茅台", "000858.SZ": "五粮液", "300750.SZ": "宁德时代",
    "002594.SZ": "比亚迪", "600036.SH": "招商银行", "601166.SH": "兴业银行",
    "600276.SH": "恒瑞医药", "000661.SZ": "长春高新", "600031.SH": "三一重工",
    "601899.SH": "紫金矿业", "000725.SZ": "京东方A", "002241.SZ": "歌尔股份",
    "600309.SH": "万华化学", "002304.SZ": "洋河股份", "601628.SH": "中国人寿",
    "600690.SH": "海尔智家", "000651.SZ": "格力电器", "601318.SH": "中国平安",
    "600585.SH": "海螺水泥", "000333.SZ": "美的集团",
}

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    cfg_abl = load_config(str(ROOT / "configs" / "ablation.yaml"))
    data = load_data(cfg)
    symbols = list(cfg.data.universe)
    _, _, test_data = split_data(data, cfg)
    features = build_feature_matrix(data)
    log_returns = get_raw_log_returns(data, symbols)
    mv_graphs = build_multi_view_graphs(
        symbols, log_returns, cfg  # full returns so test graphs reflect actual 2024 market
    )
    test_dates = sorted(set(test_data.index.get_level_values("date")))
    node_feats, valid_dates = make_lookback_tensor(
        features, pd.DatetimeIndex(test_dates), symbols, cfg.features.lookback_window
    )
    returns_arr = np.nan_to_num(
        log_returns.reindex(valid_dates).values.astype(np.float32), nan=0.0
    )
    # log_returns on test set only (for baselines)
    lr_test = log_returns.reindex(valid_dates).fillna(0.0)
    node_feat_dim = len(features.columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (cfg, cfg_abl, symbols, node_feats, valid_dates, returns_arr,
            mv_graphs, lr_test, node_feat_dim, device)


# ── Model rollout ─────────────────────────────────────────────────────────────

def rollout(model, cfg, node_feats, valid_dates, returns_arr, mv_graphs, device):
    env = PortfolioEnv(node_feats, valid_dates, returns_arr, mv_graphs, cfg)
    model.eval()
    obs, _ = env.reset()
    dates, rets, weights, turnovers = [], [], [], []
    done = False
    while not done:
        t = env._t
        x = torch.tensor(env.get_node_features_tensor(), dtype=torch.float32).unsqueeze(0).to(device)
        ig, cg = env.get_graph_data()
        ig, cg = ig.to(device), cg.to(device)
        w = torch.tensor(env._weights, dtype=torch.float32).unsqueeze(0).to(device)
        cash = torch.zeros(1, 1, device=device)
        cr = torch.tensor([[env._cum_return()]], dtype=torch.float32, device=device)
        with torch.no_grad():
            action, *_ = model.get_action_and_value(x, ig, cg, w, cash, cr, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        dates.append(valid_dates[t])
        rets.append(info["port_log_ret"])
        weights.append(env._weights.copy())
        turnovers.append(info["turnover"])
        done = terminated or truncated
    return (pd.DatetimeIndex(dates), np.array(rets),
            np.array(weights), np.array(turnovers))


# ── Baseline return series ────────────────────────────────────────────────────

def baseline_return_series(lr_test: pd.DataFrame, cfg) -> dict[str, pd.Series]:
    """Returns dict of strategy -> pd.Series of daily log returns."""
    results = {}
    n = lr_test.shape[1]

    # Equal weight
    ew = (lr_test @ np.ones(n) / n)
    results["Equal Weight"] = ew

    # Markowitz (rolling, monthly rebalance)
    rebalance_dates = lr_test.resample("ME").last().index
    weights_hist: dict = {}
    for rd in rebalance_dates:
        hist = lr_test.loc[:rd].tail(252)
        if len(hist) < 30:
            weights_hist[rd] = np.ones(n) / n
            continue
        mu = hist.mean().values * 252
        lw = LedoitWolf().fit(hist.values)
        sigma = lw.covariance_ * 252
        def neg_sharpe(w):
            return -(w @ mu - cfg.eval.risk_free_rate) / (np.sqrt(w @ sigma @ w) + 1e-8)
        res = minimize(neg_sharpe, np.ones(n)/n, method="SLSQP",
                       bounds=[(0,1)]*n, constraints={"type":"eq","fun": lambda w: w.sum()-1})
        weights_hist[rd] = res.x if res.success else np.ones(n)/n

    mz_rets = []
    cur_w = np.ones(n) / n
    for date in lr_test.index:
        past = [rd for rd in rebalance_dates if rd <= date]
        if past:
            cur_w = weights_hist[max(past)]
        mz_rets.append(float(lr_test.loc[date].values @ cur_w))
    results["Markowitz"] = pd.Series(mz_rets, index=lr_test.index)

    # CSI300
    try:
        import tushare as ts
        ts.set_token(os.environ["TUSHARE_TOKEN"])
        pro = ts.pro_api()
        start = str(lr_test.index[0].date())
        end = str(lr_test.index[-1].date())
        df = pro.index_daily(ts_code="000300.SH",
                             start_date=start.replace("-",""),
                             end_date=end.replace("-",""),
                             fields="trade_date,close")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").set_index("trade_date")
        csi = np.log(df["close"] / df["close"].shift(1)).dropna()
        csi = csi.reindex(lr_test.index).fillna(0.0)
        results["CSI300"] = csi
    except Exception as e:
        print(f"  [skip] CSI300 download failed: {e}")

    return results


def cum_ret_series(log_rets):
    return pd.Series(np.exp(np.cumsum(log_rets)) - 1, index=log_rets.index)


def drawdown_series(log_rets):
    vals = np.asarray(log_rets)
    cum = np.exp(np.cumsum(vals))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    if isinstance(log_rets, pd.Series):
        return pd.Series(dd, index=log_rets.index)
    return dd


def rolling_sharpe(log_rets, window=30, rf=0.02):
    daily_rf = rf / 252
    r = pd.Series(log_rets)
    mean = r.rolling(window).mean()
    std = r.rolling(window).std()
    return ((mean - daily_rf) / (std + 1e-8) * np.sqrt(252)).values


def hhi(weights):
    return (weights ** 2).sum(axis=1)


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_equity_curves(all_series: dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, s in all_series.items():
        c = PALETTE.get(name, "#888")
        st = STYLE.get(name, dict(lw=1.5))
        ax.plot(s.index, s.values * 100, color=c, label=name, **st)
    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("2024年测试集 — 各策略累计收益率", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益率")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_equity_curves.png")
    plt.close(fig)
    print("  [done]01_equity_curves.png")


def plot_drawdown(all_series: dict[str, pd.Series], all_log: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, s in all_series.items():
        c = PALETTE.get(name, "#888")
        st = STYLE.get(name, dict(lw=1.5))
        dd = drawdown_series(pd.Series(np.log(s.values + 1), index=s.index))
        ax.fill_between(s.index, dd.values * 100, 0,
                        alpha=0.15, color=c)
        ax.plot(s.index, dd.values * 100, color=c, label=name, **st)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("2024年测试集 — 各策略回撤", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_drawdown.png")
    plt.close(fig)
    print("  [done]02_drawdown.png")


def plot_rolling_sharpe(ppo_dates, ppo_gnn_rets, ppo_ng_rets, baselines_log):
    fig, ax = plt.subplots(figsize=(12, 4))
    show = ["Equal Weight"]
    for name in show:
        if name not in baselines_log:
            continue
        rs = rolling_sharpe(baselines_log[name].values)
        ax.plot(baselines_log[name].index, rs, color=PALETTE[name],
                label=name, **STYLE[name])
    rs_ng = rolling_sharpe(ppo_ng_rets)
    ax.plot(ppo_dates, rs_ng, color=PALETTE["PPO (no_graph)"],
            label="PPO (no_graph)", **STYLE["PPO (no_graph)"])
    rs_gnn = rolling_sharpe(ppo_gnn_rets)
    ax.plot(ppo_dates, rs_gnn, color=PALETTE["PPO (GNN)"],
            label="PPO (GNN)", **STYLE["PPO (GNN)"])
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_ylim(-4, 6)
    ax.set_title("2024年测试集 — 滚动30日夏普率", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("夏普率（年化）")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_rolling_sharpe.png")
    plt.close(fig)
    print("  [done]03_rolling_sharpe.png")


def plot_return_distribution(ppo_gnn_rets, ppo_ng_rets, baselines_log):
    fig, ax = plt.subplots(figsize=(10, 5))
    show = {
        "PPO (GNN)": ppo_gnn_rets,
        "PPO (no_graph)": ppo_ng_rets,
        "Equal Weight": baselines_log.get("Equal Weight", pd.Series(dtype=float)).values,
    }
    for name, rets in show.items():
        if len(rets) < 10:
            continue
        rets = rets[~np.isnan(rets)]
        kde = gaussian_kde(rets, bw_method=0.5)
        x = np.linspace(rets.min() - 0.005, rets.max() + 0.005, 400)
        ax.plot(x, kde(x), color=PALETTE[name], label=name, **STYLE[name])
        ax.axvline(rets.mean(), color=PALETTE[name], lw=1, ls=":", alpha=0.7)
    ax.set_title("2024年测试集 — 日收益率分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("日对数收益率")
    ax.set_ylabel("概率密度")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_return_distribution.png")
    plt.close(fig)
    print("  [done]04_return_distribution.png")


def plot_weight_heatmap(dates, weights, symbols, title_suffix, filename):
    labels = [STOCK_NAMES.get(s, s) for s in symbols]
    df = pd.DataFrame(weights, index=dates, columns=labels)
    # Downsample to ~monthly for readability
    df_monthly = df.resample("W").mean()

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = LinearSegmentedColormap.from_list("wt", ["#F0F9FF", "#1E40AF"])
    sns.heatmap(df_monthly.T, ax=ax, cmap=cmap, vmin=0, vmax=0.15,
                linewidths=0.3, linecolor="#E5E7EB",
                cbar_kws={"label": "权重", "shrink": 0.6},
                xticklabels=4)
    ax.set_title(f"组合权重热力图 — {title_suffix}", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期（每周）")
    ax.set_ylabel("股票")
    ax.set_xticklabels([str(d)[:10] for d in df_monthly.index[::4]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)
    print(f"  [done] {filename}")


def plot_monthly_returns_heatmap(dates, log_rets, title, filename):
    s = pd.Series(log_rets, index=dates)
    monthly = (s.resample("ME").sum()).rename("log_ret")
    monthly_pct = (np.exp(monthly) - 1) * 100
    df = monthly_pct.to_frame()
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="month", columns="year", values="log_ret")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_names[i-1] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1]*2.5), 5))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.1)
    cmap = LinearSegmentedColormap.from_list("rg", ["#DC2626","#F9FAFB","#16A34A"])
    sns.heatmap(pivot, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt=".1f", annot_kws={"size": 10},
                linewidths=0.5, linecolor="#E5E7EB",
                cbar_kws={"label": "月收益率 (%)","shrink":0.8})
    ax.set_title(f"月度收益热力图 — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("年份")
    ax.set_ylabel("月份")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)
    print(f"  [done] {filename}")


def plot_turnover(gnn_dates, gnn_tv, ng_dates, ng_tv):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    for ax, dates, tv, name, color in [
        (ax1, gnn_dates, gnn_tv, "PPO (GNN)", PALETTE["PPO (GNN)"]),
        (ax2, ng_dates, ng_tv, "PPO (no_graph)", PALETTE["PPO (no_graph)"]),
    ]:
        rolling_tv = pd.Series(tv, index=dates).rolling(10).mean()
        ax.bar(dates, tv, color=color, alpha=0.3, width=1)
        ax.plot(rolling_tv.index, rolling_tv.values, color=color, lw=1.8, label="10日均值")
        ax.set_ylabel("换手率")
        ax.set_title(f"{name} — 日换手率", fontsize=12)
        ax.legend()
        mean_tv = np.mean(tv)
        ax.axhline(mean_tv, color="black", lw=1, ls="--", alpha=0.5,
                   label=f"均值 {mean_tv:.3f}")
    fig.suptitle("2024年测试集 — 组合换手率", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_turnover.png")
    plt.close(fig)
    print("  [done]07_turnover.png")


def plot_concentration(gnn_dates, gnn_weights, ng_dates, ng_weights):
    gnn_hhi = hhi(gnn_weights)
    ng_hhi = hhi(ng_weights)
    equal_hhi = 1.0 / gnn_weights.shape[1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(gnn_dates, gnn_hhi, color=PALETTE["PPO (GNN)"], lw=2,
            label="PPO (GNN)")
    ax.plot(ng_dates, ng_hhi, color=PALETTE["PPO (no_graph)"], lw=2,
            ls="--", label="PPO (no_graph)")
    ax.axhline(equal_hhi, color=PALETTE["Equal Weight"], lw=1.5,
               ls="-.", label=f"等权重 (1/N = {equal_hhi:.3f})")
    ax.set_title("2024年测试集 — 赫芬达尔集中度指数 (HHI)", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("HHI")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_concentration_hhi.png")
    plt.close(fig)
    print("  [done]08_concentration_hhi.png")


def plot_metrics_bar(all_metrics: dict[str, dict]):
    metrics = ["annual_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
               "max_drawdown", "win_rate"]
    labels = ["年化收益", "夏普率", "Sortino", "Calmar", "最大回撤", "胜率"]
    n_metrics = len(metrics)
    n_strats = len(all_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_strats

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, m) in enumerate(all_metrics.items()):
        vals = [m.get(k, 0) for k in metrics]
        bars = ax.bar(x + i * width - (n_strats - 1) * width / 2,
                      vals, width, label=name,
                      color=PALETTE.get(name, "#888"), alpha=0.85)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("2024年测试集 — 各策略指标对比", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_metrics_bar.png")
    plt.close(fig)
    print("  [done]09_metrics_bar.png")


def plot_radar(all_metrics: dict[str, dict]):
    from matplotlib.patches import FancyArrowPatch

    metrics = ["annual_return", "sharpe_ratio", "calmar_ratio",
               "win_rate", "annual_volatility", "max_drawdown"]
    labels = ["年化收益", "夏普率", "Calmar", "胜率", "年化波动↓", "最大回撤↓"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize: for each metric, scale 0-1 (higher=better)
    all_vals = {m: [d.get(m, 0) for d in all_metrics.values()] for m in metrics}
    norm_range = {}
    for m in metrics:
        lo, hi = min(all_vals[m]), max(all_vals[m])
        norm_range[m] = (lo, hi if hi > lo else lo + 1e-8)

    def normalize(val, m):
        lo, hi = norm_range[m]
        v = (val - lo) / (hi - lo)
        # Invert for "lower is better" metrics
        if m in ("annual_volatility", "max_drawdown"):
            v = 1 - v
        return np.clip(v, 0, 1)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, m_dict in all_metrics.items():
        vals = [normalize(m_dict.get(m, 0), m) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=PALETTE.get(name, "#888"),
                lw=2, label=name)
        ax.fill(angles, vals, color=PALETTE.get(name, "#888"), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, alpha=0.5)
    ax.set_title("各策略雷达图对比\n（2024年测试集）", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_radar.png")
    plt.close(fig)
    print("  [done]10_radar.png")


def plot_dashboard(all_cum: dict[str, pd.Series], gnn_dates, gnn_rets,
                   gnn_weights, ppo_ng_rets, ng_dates, symbols):
    """4-panel summary dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Equity curves ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name, s in all_cum.items():
        c = PALETTE.get(name, "#888")
        st = {k: v for k, v in STYLE.get(name, {}).items() if k != "zorder"}
        ax1.plot(s.index, s.values * 100, color=c, label=name, **st)
    ax1.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.set_title("累计收益率", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name, s in all_cum.items():
        if name not in ("PPO (GNN)", "PPO (no_graph)", "Equal Weight"):
            continue
        c = PALETTE.get(name, "#888")
        st = {k: v for k, v in STYLE.get(name, {}).items() if k != "zorder"}
        dd = drawdown_series(pd.Series(np.log(s.values + 1), index=s.index))
        ax2.fill_between(s.index, dd.values * 100, 0, alpha=0.12, color=c)
        ax2.plot(s.index, dd.values * 100, color=c, label=name, **st)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.set_title("回撤", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)

    # ── Panel 3: GNN weight heatmap (top stocks by avg weight) ────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    labels = [STOCK_NAMES.get(s, s) for s in symbols]
    df_w = pd.DataFrame(gnn_weights, index=gnn_dates, columns=labels)
    avg_w = df_w.mean()
    top8 = avg_w.nlargest(8).index.tolist()
    df_top = df_w[top8].resample("W").mean()
    cmap = LinearSegmentedColormap.from_list("wt", ["#F0F9FF", "#1E40AF"])
    sns.heatmap(df_top.T, ax=ax3, cmap=cmap, vmin=0, vmax=0.15,
                linewidths=0.5, cbar_kws={"label": "权重", "shrink": 0.7},
                xticklabels=4)
    ax3.set_xticklabels([str(d)[:7] for d in df_top.index[::4]], rotation=45, ha="right", fontsize=7)
    ax3.set_title("GNN策略 — Top-8持仓权重（周均）", fontsize=12, fontweight="bold")
    ax3.set_xlabel("")

    # ── Panel 4: Rolling Sharpe ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for name, rets, dates in [
        ("PPO (GNN)", gnn_rets, gnn_dates),
        ("PPO (no_graph)", ppo_ng_rets, ng_dates),
    ]:
        rs = rolling_sharpe(rets)
        ax4.plot(dates, rs, color=PALETTE[name], label=name,
                 **{k:v for k,v in STYLE[name].items() if k!="zorder"})
    ax4.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax4.set_ylim(-4, 6)
    ax4.set_title("滚动30日夏普率 (PPO策略)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("graphfolio-rl — 2024年测试集综合看板", fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(FIGURES_DIR / "00_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done]00_dashboard.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data and configs...")
    (cfg, cfg_abl, symbols, node_feats, valid_dates, returns_arr,
     mv_graphs, lr_test, node_feat_dim, device) = load_all()

    print(f"Test set: {valid_dates[0].date()} → {valid_dates[-1].date()}  ({len(valid_dates)} days)")

    print("\nRunning PPO (GNN full) rollout...")
    gnn_model = GNNActorCritic(
        n_assets=20, node_feat_dim=node_feat_dim,
        gnn_hidden=64, gnn_out=64, gnn_heads=4,
        actor_dims=[256, 128], critic_dims=[256, 128],
    ).to(device)
    gnn_model.load_state_dict(torch.load(
        CKPT_DIR / "best_model_full.pt", map_location=device, weights_only=True))
    gnn_dates, gnn_rets, gnn_weights, gnn_tv = rollout(
        gnn_model, cfg, node_feats, valid_dates, returns_arr, mv_graphs, device)

    print("Running PPO (no_graph) rollout...")
    ng_model = FlatActorCritic(
        n_assets=20, node_feat_dim=node_feat_dim,
        actor_dims=[256, 128], critic_dims=[256, 128],
    ).to(device)
    ng_model.load_state_dict(torch.load(
        CKPT_DIR / "best_model_no_graph.pt", map_location=device, weights_only=True))
    ng_dates, ng_rets, ng_weights, ng_tv = rollout(
        ng_model, cfg_abl, node_feats, valid_dates, returns_arr, mv_graphs, device)

    print("Computing baseline return series...")
    bl_series = baseline_return_series(lr_test, cfg)

    # Build cumulative return series for all strategies
    from src.eval.metrics import compute_metrics
    all_cum: dict[str, pd.Series] = {}
    all_metrics: dict[str, dict] = {}

    all_cum["PPO (GNN)"]      = cum_ret_series(pd.Series(gnn_rets, index=gnn_dates))
    all_cum["PPO (no_graph)"] = cum_ret_series(pd.Series(ng_rets, index=ng_dates))
    all_metrics["PPO (GNN)"]      = compute_metrics(gnn_rets, cfg.eval.risk_free_rate)
    all_metrics["PPO (no_graph)"] = compute_metrics(ng_rets, cfg.eval.risk_free_rate)

    bl_name_map = {
        "Equal Weight": "Equal Weight",
        "Markowitz":    "Markowitz",
        "CSI300":       "CSI300",
    }
    for k, s in bl_series.items():
        name = bl_name_map.get(k, k)
        all_cum[name]     = cum_ret_series(s)
        all_metrics[name] = compute_metrics(s.values, cfg.eval.risk_free_rate)

    print(f"\nGenerating {FIGURES_DIR}/ ...")

    plot_equity_curves(all_cum)
    plot_drawdown(all_cum, {})
    plot_rolling_sharpe(gnn_dates, gnn_rets, ng_rets, bl_series)
    plot_return_distribution(gnn_rets, ng_rets, bl_series)
    plot_weight_heatmap(gnn_dates, gnn_weights, symbols,
                        "PPO (GNN full)", "05_weights_gnn.png")
    plot_weight_heatmap(ng_dates, ng_weights, symbols,
                        "PPO (no_graph)", "06_weights_no_graph.png")
    plot_monthly_returns_heatmap(gnn_dates, gnn_rets,
                                 "PPO (GNN)", "07_monthly_returns_gnn.png")
    plot_turnover(gnn_dates, gnn_tv, ng_dates, ng_tv)
    plot_concentration(gnn_dates, gnn_weights, ng_dates, ng_weights)
    plot_metrics_bar(all_metrics)
    plot_radar(all_metrics)
    plot_dashboard(all_cum, gnn_dates, gnn_rets, gnn_weights,
                   ng_rets, ng_dates, symbols)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
