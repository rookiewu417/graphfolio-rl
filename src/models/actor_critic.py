"""PPO Actor-Critic network built on top of MultiViewGNNEncoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.gnn_encoder import MultiViewGNNEncoder


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers[:-1])  # no dropout after last layer


class GNNActorCritic(nn.Module):
    """
    Shared GNN encoder + separate Actor / Critic heads.
    State = (node_features, ind_graph, corr_graph, portfolio_context).
    portfolio_context: [current_weights (N), cash_ratio (1), cumulative_return (1)]
    """

    def __init__(
        self,
        n_assets: int,
        node_feat_dim: int,
        gnn_hidden: int = 64,
        gnn_out: int = 64,
        gnn_heads: int = 4,
        actor_dims: list[int] = (256, 128),
        critic_dims: list[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets

        self.gnn = MultiViewGNNEncoder(
            in_dim=node_feat_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            dropout=dropout,
        )

        # global state: flattened GNN output + portfolio context
        global_dim = n_assets * gnn_out + n_assets + 2  # embeddings + weights + cash + cum_ret

        actor_in = [global_dim] + list(actor_dims)
        critic_in = [global_dim] + list(critic_dims)

        self.actor_mlp = _mlp(actor_in, dropout)
        self.actor_head = nn.Linear(actor_dims[-1], n_assets)

        self.critic_mlp = _mlp(critic_in, dropout)
        self.critic_head = nn.Linear(critic_dims[-1], 1)

    def _encode_state(
        self,
        node_feats: torch.Tensor,
        ind_graph: Data,
        corr_graph: Data,
        weights: torch.Tensor,
        cash_ratio: torch.Tensor,
        cum_return: torch.Tensor,
    ) -> torch.Tensor:
        B = node_feats.shape[0] if node_feats.dim() == 3 else 1
        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0)

        h_list = []
        for b in range(B):
            h = self.gnn(node_feats[b], ind_graph, corr_graph)  # (N, gnn_out)
            h_list.append(h.flatten())
        h_flat = torch.stack(h_list)  # (B, N*gnn_out)

        ctx = torch.cat([weights, cash_ratio, cum_return], dim=-1)
        return torch.cat([h_flat, ctx], dim=-1)

    def forward(
        self,
        node_feats: torch.Tensor,
        ind_graph: Data,
        corr_graph: Data,
        weights: torch.Tensor,
        cash_ratio: torch.Tensor,
        cum_return: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._encode_state(node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return)
        logits = self.actor_head(self.actor_mlp(s))
        value = self.critic_head(self.critic_mlp(s)).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return)
        # Use Dirichlet-parameterized via softmax + Gaussian noise for continuous action
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Dirichlet(probs * 10 + 1e-6)
        if deterministic:
            action = probs
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class FlatActorCritic(nn.Module):
    """No-GNN baseline: flatten node features + portfolio context → MLP → actor/critic.
    Accepts the same call signature as GNNActorCritic so the training loop is unchanged.
    """

    def __init__(
        self,
        n_assets: int,
        node_feat_dim: int,
        actor_dims: list[int] = (256, 128),
        critic_dims: list[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        # input: flattened node features + weights + cash + cum_ret
        in_dim = n_assets * node_feat_dim + n_assets + 2

        actor_in = [in_dim] + list(actor_dims)
        critic_in = [in_dim] + list(critic_dims)

        self.actor_mlp = _mlp(actor_in, dropout)
        self.actor_head = nn.Linear(actor_dims[-1], n_assets)
        self.critic_mlp = _mlp(critic_in, dropout)
        self.critic_head = nn.Linear(critic_dims[-1], 1)

    def forward(self, node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return):
        # node_feats: (B, N, F) or (N, F)
        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0)
        B = node_feats.shape[0]
        flat = node_feats.reshape(B, -1)
        ctx = torch.cat([weights, cash_ratio, cum_return], dim=-1)
        s = torch.cat([flat, ctx], dim=-1)
        logits = self.actor_head(self.actor_mlp(s))
        value = self.critic_head(self.critic_mlp(s)).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(node_feats, ind_graph, corr_graph, weights, cash_ratio, cum_return)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Dirichlet(probs * 10 + 1e-6)
        action = probs if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
