"""Static industry membership graph (申万一级行业)."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

# 申万一级行业映射（人工维护，与 default.yaml universe 对应）
# key: tushare ts_code, value: SW L1 industry name
SW_INDUSTRY: dict[str, str] = {
    "600519.SH": "食品饮料",
    "000858.SZ": "食品饮料",
    "300750.SZ": "电力设备",
    "002594.SZ": "汽车",
    "600036.SH": "银行",
    "601166.SH": "银行",
    "600276.SH": "医药生物",
    "000661.SZ": "医药生物",
    "600031.SH": "机械设备",
    "601899.SH": "有色金属",
    "000725.SZ": "电子",
    "002241.SZ": "电子",
    "600309.SH": "基础化工",
    "002304.SZ": "食品饮料",
    "601628.SH": "非银金融",
    "600690.SH": "家用电器",
    "000651.SZ": "家用电器",
    "601318.SH": "非银金融",
    "600585.SH": "建筑材料",
    "000333.SZ": "家用电器",
}


def build_industry_graph(symbols: list[str]) -> Data:
    """
    Build undirected industry membership graph.
    Two stocks are connected if they share the same SW L1 industry.
    Returns PyG Data with edge_index and no node features (added later by GNN encoder).
    """
    n = len(symbols)
    industries = [SW_INDUSTRY.get(s, "其他") for s in symbols]

    edge_index_src, edge_index_dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if industries[i] == industries[j]:
                edge_index_src += [i, j]
                edge_index_dst += [j, i]

    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    # Uniform weight = 1.0 so SingleViewGAT always receives edge_attr without branching
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return Data(
        num_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        symbols=symbols,
        industry=industries,
    )
