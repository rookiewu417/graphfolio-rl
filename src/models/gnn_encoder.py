"""Multi-view GAT encoder with gated fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class SingleViewGAT(nn.Module):
    """2-layer GAT for one graph view."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index
        edge_weight = getattr(graph, "edge_weight", None)

        if edge_index.shape[1] == 0:
            # isolated nodes: identity mapping
            return self.dropout(self.act(x @ torch.zeros(x.shape[-1], self.conv2.out_channels).to(x)))

        h = self.dropout(self.act(self.conv1(x, edge_index)))
        h = self.conv2(h, edge_index)
        return h


class GatedMultiViewFusion(nn.Module):
    """Gated attention fusion for two graph views."""

    def __init__(self, dim: int):
        super().__init__()
        # gate: learned scalar weight per view, conditioned on both embeddings
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, h_ind: torch.Tensor, h_corr: torch.Tensor) -> torch.Tensor:
        # h_ind, h_corr: (N, dim)
        alpha = self.gate(torch.cat([h_ind, h_corr], dim=-1))  # (N, 2)
        fused = alpha[:, 0:1] * h_ind + alpha[:, 1:2] * h_corr
        return fused


class MultiViewGNNEncoder(nn.Module):
    """
    Encodes asset nodes using two graph views (industry + correlation).
    Input: node feature matrix X (N, F_in), two PyG Data graphs.
    Output: fused node embeddings (N, out_dim).
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
        self.proj = nn.Linear(in_dim, hidden_dim)  # shared input projection
        self.gat_ind = SingleViewGAT(hidden_dim, hidden_dim, out_dim, heads, dropout)
        self.gat_corr = SingleViewGAT(hidden_dim, hidden_dim, out_dim, heads, dropout)
        self.fusion = GatedMultiViewFusion(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        ind_graph: Data,
        corr_graph: Data,
    ) -> torch.Tensor:
        h = torch.relu(self.proj(x))      # (N, hidden_dim)
        h_ind = self.gat_ind(h, ind_graph)   # (N, out_dim)
        h_corr = self.gat_corr(h, corr_graph)  # (N, out_dim)
        fused = self.fusion(h_ind, h_corr)   # (N, out_dim)
        return self.norm(fused)
