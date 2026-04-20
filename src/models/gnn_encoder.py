"""Multi-view GAT encoder with gated fusion, edge-weight support, and residual skip."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class SingleViewGAT(nn.Module):
    """2-layer GATv2 that ingests signed edge weights as attention bias."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True, edge_dim=1
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False, edge_dim=1
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        # Linear fallback when there are no edges (replaces the old all-zeros bug)
        self.fallback = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index

        if edge_index.shape[1] == 0:
            return self.fallback(x)

        edge_weight = getattr(graph, "edge_weight", None)
        # edge_attr: (E, 1) — signed correlation or uniform 1.0 for industry graph
        if edge_weight is not None:
            edge_attr = edge_weight.unsqueeze(-1)
        else:
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device)

        h = self.dropout(self.act(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return h


class GatedMultiViewFusion(nn.Module):
    """Gated fusion: per-node learned blend of two view embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, 2), nn.Softmax(dim=-1))

    def forward(self, h_ind: torch.Tensor, h_corr: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([h_ind, h_corr], dim=-1))  # (N, 2)
        return alpha[:, 0:1] * h_ind + alpha[:, 1:2] * h_corr


class MultiViewGNNEncoder(nn.Module):
    """
    Encodes N assets using industry + correlation graph views.

    Architecture:
      raw features (N, F)
        ├─ proj → relu → GAT_industry  ┐
        │                               ├─ gated fusion → fused (N, out_dim)
        └─ proj → relu → GAT_corr     ┘
      residual_proj(raw features) ──────────────────────────────┐
                                        LayerNorm(fused + skip) ← output
    The residual ensures per-asset raw signals always reach the actor,
    even when graph aggregation is unhelpful.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)  # skip connection from raw features
        self.gat_ind = SingleViewGAT(hidden_dim, hidden_dim, out_dim, heads, dropout)
        self.gat_corr = SingleViewGAT(hidden_dim, hidden_dim, out_dim, heads, dropout)
        self.fusion = GatedMultiViewFusion(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, ind_graph: Data, corr_graph: Data) -> torch.Tensor:
        h = torch.relu(self.proj(x))              # (N, hidden_dim)
        h_ind = self.gat_ind(h, ind_graph)        # (N, out_dim)
        h_corr = self.gat_corr(h, corr_graph)    # (N, out_dim)
        fused = self.fusion(h_ind, h_corr)        # (N, out_dim)
        skip = self.residual_proj(x)              # (N, out_dim)
        return self.norm(fused + skip)
