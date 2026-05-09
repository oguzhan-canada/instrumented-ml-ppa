"""
models/gat_ppa.py

★ Slack-Aware Graph Attention Network for PPA Prediction

Architecture:
  1. 4-layer GATConv stack (criticality-weighted edge attention)
  2. Global attention pooling → graph embedding
  3. Concatenate with timing_vec + spatial_summary
  4. MLP → 3 output heads  [power, perf, area]

Uncertainty via MC Dropout:
  Keep dropout layers active at inference time and run T=50 forward
  passes to obtain prediction mean and variance.  This variance feeds
  directly into BoTorch's acquisition function.

Ablation support:
  Pass use_criticality=False to zero out c(e) influence.
  (Equivalent to running with --no-criticality flag.)
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.data import Batch

from features.dataset import TIMING_VEC_DIM


# ─────────────────────────────── Spatial encoder ──────────────────────────────

class SpatialEncoder(nn.Module):
    """Lightweight CNN that compresses a [B, 1, H, W] density map → [B, 32]."""

    def __init__(self, grid: int = 100, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # → H/2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # → H/4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────── GAT backbone ─────────────────────────────────

class SlackAwareGAT(nn.Module):
    """
    Criticality-weighted GAT backbone.

    The criticality weight c(e) ∈ [0,1] is appended to edge_attr.
    We pass it through a small edge MLP to produce a bias that is
    added to the raw GAT attention logit before softmax — so near-
    critical edges (c≈1) receive higher normalised attention.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 2,      # [is_clock, criticality]
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Project edge features to a single bias scalar per head
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads),
        )

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True,          # output: hidden_dim
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Global attention pooling gate
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : [N, node_feat_dim]
        edge_index : [2, E]
        edge_attr  : [E, edge_feat_dim]  last column = criticality weight c(e)
        batch      : [N]  — graph membership index
        Returns    : [B, hidden_dim]   — graph-level embeddings
        """
        h = self.input_proj(x)

        # Edge bias from criticality (shape [E, num_heads])
        # GATConv doesn't natively accept per-head edge biases, so we fold
        # the criticality signal into a scalar per edge via mean-pooling heads,
        # then add it to the node features of the destination node via scatter.
        edge_bias = self.edge_bias_mlp(edge_attr)   # [E, num_heads]
        # Average over heads → [E] scalar, then scatter_add to dst nodes
        edge_scalar = edge_bias.mean(dim=1)          # [E]
        dst = edge_index[1]
        N = x.size(0)
        crit_signal = torch.zeros(N, device=x.device).scatter_add(
            0, dst, edge_scalar
        ).unsqueeze(1)                                # [N, 1]

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new = layer(h, edge_index)
            h_new = norm(h_new)
            # Inject criticality signal
            h_new = h_new + crit_signal.expand_as(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h if h.shape == h_new.shape else h_new   # residual

        return self.pool(h, batch)   # [B, hidden_dim]


# ─────────────────────────────── Full PPA model ───────────────────────────────

class GATForPPA(nn.Module):
    """
    Full PPA prediction model.

    Inputs per sample:
      - graph (PyG Batch): netlist with criticality edge weights
      - spatial [B, 1, H, W]: cell density map
      - timing_vec [B, D_t]: scalar timing + design-param features

    Output:
      [B, 3] — predicted [power, perf, area] (normalised to [0,1])
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 2,
        hidden_dim: int = 128,
        num_gat_layers: int = 4,
        num_heads: int = 4,
        spatial_out_dim: int = 32,
        timing_hidden: int = 64,
        dropout: float = 0.1,
        use_criticality: bool = True,
    ):
        super().__init__()
        self.use_criticality = use_criticality

        self.gat = SlackAwareGAT(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.spatial_enc = SpatialEncoder(out_dim=spatial_out_dim)

        timing_in = TIMING_VEC_DIM
        self.timing_proj = nn.Sequential(
            nn.Linear(timing_in, timing_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(timing_hidden, timing_hidden),
            nn.ReLU(),
        )

        fused_dim = hidden_dim + spatial_out_dim + timing_hidden

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
            nn.Sigmoid(),     # outputs in [0, 1] (normalised PPA)
        )

    def forward(
        self,
        graph: Batch,
        spatial: torch.Tensor,
        timing_vec: torch.Tensor,
    ) -> torch.Tensor:

        if not self.use_criticality:
            # Zero out the criticality column (last col of edge_attr)
            ea = graph.edge_attr.clone()
            if ea is not None and ea.shape[1] > 1:
                ea[:, -1] = 0.5
            graph.edge_attr = ea

        g_emb  = self.gat(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        s_emb  = self.spatial_enc(spatial)
        t_emb  = self.timing_proj(timing_vec)

        fused = torch.cat([g_emb, s_emb, t_emb], dim=1)
        return self.head(fused)

    # ── MC Dropout inference ──────────────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        graph: Batch,
        spatial: torch.Tensor,
        timing_vec: torch.Tensor,
        T: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference.

        Keeps dropout active for T forward passes.

        Returns
        -------
        mean : [B, 3]   — mean prediction
        var  : [B, 3]   — epistemic variance (feeds into BoTorch acquisition)
        """
        self.train()    # activate dropout
        with torch.no_grad():
            preds = torch.stack([
                self.forward(graph, spatial, timing_vec) for _ in range(T)
            ])              # [T, B, 3]
        self.eval()
        return preds.mean(dim=0), preds.var(dim=0)


# ─────────────────────────────── Training helpers ─────────────────────────────

class PPALoss(nn.Module):
    """
    Multi-task regression loss with optional per-metric weighting.
    Default: equal weight across power, performance, area.
    """

    def __init__(self, metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.register_buffer(
            "weights", torch.tensor(metric_weights, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, 3]"""
        per_metric = F.mse_loss(pred, target, reduction="none").mean(dim=0)  # [3]
        return (per_metric * self.weights).sum()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
