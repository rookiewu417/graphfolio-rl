# graphfolio-rl

Multi-view Graph Neural Network + PPO for A-Share Portfolio Optimization.

**课程**：统计学习与机器学习结课项目  
**方法**：将多视图 GNN（行业图 + 动态相关性图）嵌入 MDP 状态表示，PPO 算法优化 A 股投资组合策略。

---

## 快速开始（另一台机器）

### 1. 前置要求

- [pixi](https://pixi.sh) 包管理器（Windows 安装：`winget install prefix-dev.pixi`）
- NVIDIA GPU（RTX 4070 Laptop 或同等），CUDA 12.4 驱动
- Git

### 2. 克隆与安装

```bash
git clone https://github.com/rookiewu417/graphfolio-rl.git
cd graphfolio-rl
pixi install          # 自动安装 Python 3.12 + PyTorch CUDA 12.4 + PyG 等所有依赖
```

### 3. 配置 Tushare Token

在项目根目录创建 `.env` 文件（不会被 git 追踪）：

```
TUSHARE_TOKEN=b4856215597879a65316d18c0805bad5a33c4c6b8b4df5de76c83116
```

### 4. 下载数据

```bash
pixi run download-data    # 下载 A 股 20 只标的，2018-2024 日线，约 4 秒
```

### 5. 运行基线策略（验证环境）

```bash
pixi run baselines        # 1/N、Markowitz、买入持有、沪深300，输出 2024 测试结果
```

### 6. 训练主模型

```bash
pixi run train            # 多视图 GNN + PPO，训练约 500k 步，结果存 results/checkpoints/
```

### 7. 消融实验

```bash
pixi run train-ablation   # 无图/单视图等消融，配置见 configs/ablation.yaml
```

### 8. 回测与评估

```bash
pixi run backtest         # 加载最优模型，与所有基线对比，输出 results/backtest_results.csv
```

### 9. 测试

```bash
pixi run -e dev test      # 运行 14 个单元测试
```

---

## 项目结构

```
graphfolio-rl/
├── configs/
│   ├── default.yaml      # 主配置（资产池、超参、训练设置）
│   └── ablation.yaml     # 消融实验配置
├── src/
│   ├── data/
│   │   ├── downloader.py # tushare 数据下载 + 缓存
│   │   └── features.py   # 技术指标特征工程（ta 库）
│   ├── graphs/
│   │   ├── industry_graph.py  # 申万行业静态图
│   │   ├── corr_graph.py      # 动态滚动相关性图（周频重建）
│   │   └── multi_view.py      # 多视图图容器
│   ├── models/
│   │   ├── gnn_encoder.py     # 双路 GAT + 门控融合
│   │   └── actor_critic.py    # PPO Actor-Critic
│   ├── envs/
│   │   └── portfolio_env.py   # 自定义 Gymnasium 环境（图状态）
│   ├── agents/
│   │   └── ppo_train.py       # PPO 训练主循环
│   └── eval/
│       ├── metrics.py         # 夏普、回撤、Sortino 等指标
│       ├── baselines.py       # 1/N、Markowitz、买入持有、沪深300
│       └── backtest.py        # 统一回测接口
├── tests/                # pytest 单元测试（14 个）
├── data/                 # 自动生成，不追踪
│   ├── raw/
│   └── processed/        # tushare 缓存（parquet）
├── results/              # 训练产出
│   ├── checkpoints/      # 最优模型权重
│   └── figures/
├── experiments/          # wandb 日志
├── pixi.toml             # 依赖声明
└── pixi.lock             # 锁定版本
```

---

## MDP 建模

| 元素 | 定义 |
|---|---|
| **状态 S** | GNN 编码的多视图图嵌入 + 当前持仓权重 + 累计收益 |
| **动作 A** | N 维权重向量（单纯形），softmax 归一化 |
| **奖励 R** | `log_return · w − λ·rolling_vol − c·turnover` |
| **图视图 1** | 申万一级行业共属图（静态） |
| **图视图 2** | 60 日滑窗 Pearson 相关系数图（每周五重建） |
| **融合** | 门控注意力（Gated Multi-View Fusion） |

---

## 基线结果（2024 测试集）

| 策略 | 年化收益 | 最大回撤 | 夏普比 |
|---|---|---|---|
| 等权重 1/N | 16.3% | -13.3% | 0.60 |
| 买入持有（等权） | 21.5% | -13.0% | 0.81 |
| Markowitz (LW) | 11.2% | -18.8% | 0.37 |
| 沪深 300 ETF | 17.0% | -14.4% | 0.65 |
| **GNN-PPO（本文）** | 待训练 | — | — |

---

## 实验流程（里程碑）

| 周 | 任务 | 状态 |
|---|---|---|
| W1 | 数据管道 + 图构建 + 基线 | ✅ 完成 |
| W2 | 行业图 + 动态图验证 | ✅ 完成 |
| W3 | 纯 PPO（无图）baseline 训练 | ⬜ 待执行 |
| W4–5 | 多视图 GNN+PPO 端到端训练 | ⬜ |
| W6 | 消融矩阵 | ⬜ |
| W7 | 完整评估 + 可视化 | ⬜ |
| W8 | 缓冲 / 论文准备 | ⬜ |

---

## 当前进度说明（给接手机器的说明）

**已完成：**
- pixi 环境配置（Python 3.12 + PyTorch CUDA 12.4 + PyG）
- 20 只 A 股数据下载（2018-2024，已验证无泄露）
- 8 维技术指标特征工程（MACD、RSI、ATR、BBands、Momentum 等）
- 行业图 + 动态相关性图（238 个周频快照，avg 度 4.5）
- 四个基线策略计算完毕
- 14 个单元测试全部通过

**下一步（接手后立刻执行）：**

```bash
# 1. 先修复 actor_critic.py 里的维度 bug（squeeze 问题），然后：
pixi run train          # W3：纯 PPO 无图 baseline（在 configs/default.yaml 中临时禁用 GNN）
```

**已知待修复 bug：**  
`src/models/actor_critic.py` 中 `_encode_state` 的 `cash_ratio.unsqueeze(-1)` 会在 batch=1 时产生维度不匹配（`RuntimeError: Tensors must have same number of dimensions`）。  
修复方式：将第 81 行改为：
```python
ctx = torch.cat([weights, cash_ratio, cum_return], dim=-1)
```

---

## 注意事项

- `.env` 文件含 tushare token，**不提交 git**，需在每台机器手动创建
- `data/processed/` 不提交 git，首次运行 `pixi run download-data` 约 4 秒
- 训练时 wandb 默认 offline 模式（未配置 entity），日志存 `experiments/wandb/`
- 所有随机实验固定 seed=42，论文报告请用 3 个 seed 取均值
