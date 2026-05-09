"""
features/criticality.py

Computes per-edge criticality weights for the Slack-Aware GAT.

    c(e) = exp( -slack(e) / slack_target )

  - Edges on the critical path (slack ≈ 0) → c(e) ≈ 1.0  (maximum attention)
  - Edges with ample slack             → c(e) ≈ 0.0  (attenuated)

Pre-synthesis fallback:
  When timing reports are unavailable, criticality is estimated from
  topological path depth:
    c(e) = depth(dst_node) / max_depth

Usage:
  from features.criticality import compute_criticality_weights
  edge_criticality = compute_criticality_weights(graph, timing_report_path)

Adapted for real data: validates timing report format before parsing.
"""

import re
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def _parse_net_slacks_from_report(rpt_path: str) -> dict[str, float]:
    """
    Extract per-net (wire) slack values from an OpenROAD timing report.
    Returns {net_name: slack_value}.

    Supports both OpenROAD path report formats:
      - Standard: net_name/A (net) ... slack: -0.123
      - OpenSTA:  Slack (MET/VIOLATED) -0.123
    """
    net_slacks: dict[str, float] = {}
    if not rpt_path or (isinstance(rpt_path, float) and np.isnan(rpt_path)):
        return net_slacks
    if not Path(rpt_path).exists():
        return net_slacks

    with open(rpt_path) as f:
        content = f.read()

    # Validate: must contain at least one slack value
    if "slack" not in content.lower() and "Slack" not in content:
        return net_slacks

    # Pattern 1: net-level slack from path reports
    path_pattern = re.compile(
        r'(\S+)\s+\(net\).*?([+-]?[0-9]+\.[0-9]+)\s*$',
        re.MULTILINE
    )
    for m in path_pattern.finditer(content):
        net_name = m.group(1).split('/')[0]
        slack = float(m.group(2))
        if net_name not in net_slacks or slack < net_slacks[net_name]:
            net_slacks[net_name] = slack

    # Pattern 2: OpenSTA endpoint slack (fallback)
    if not net_slacks:
        endpoint_pattern = re.compile(
            r'(\S+)\s+.*?slack\s+\((?:MET|VIOLATED)\)\s+([+-]?[0-9]+\.?[0-9]*)',
            re.MULTILINE | re.IGNORECASE,
        )
        for m in endpoint_pattern.finditer(content):
            pin_name = m.group(1).split('/')[0]
            slack = float(m.group(2))
            if pin_name not in net_slacks or slack < net_slacks[pin_name]:
                net_slacks[pin_name] = slack

    return net_slacks


def compute_criticality_weights(
    graph: Data,
    timing_report_path: str | None = None,
    slack_target: float = 0.1,
    use_fallback: bool = True,
) -> torch.Tensor:
    """
    Compute a criticality weight tensor for every edge in `graph`.

    Parameters
    ----------
    graph               : PyG Data object
    timing_report_path  : path to OpenROAD timing .rpt
    slack_target        : normalisation constant (ns); default 0.1 ns
    use_fallback        : if True, fall back to depth-based proxy

    Returns
    -------
    criticality : FloatTensor [E]  — one weight per edge, range [0, 1]
    """
    E = graph.edge_index.shape[1]
    if E == 0:
        return torch.zeros(0)

    net_slacks = _parse_net_slacks_from_report(timing_report_path)

    # ── Strategy 1: slack-based weights from timing report ───────────────
    if net_slacks:
        cell_names = getattr(graph, "cell_names", None)
        criticality = torch.zeros(E)
        src_nodes = graph.edge_index[0].tolist()
        dst_nodes = graph.edge_index[1].tolist()

        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            if cell_names and src < len(cell_names):
                net_key = cell_names[src]
                slack = net_slacks.get(net_key, None)
                if slack is None:
                    for suffix in ["/Z", "/Y", "/Q", "/ZN", "/X", "/CO"]:
                        slack = net_slacks.get(net_key + suffix, None)
                        if slack is not None:
                            break
            else:
                slack = None

            if slack is not None:
                c = float(np.exp(-slack / slack_target))
                criticality[i] = min(c, 1.0)
            else:
                criticality[i] = 0.5

        return criticality

    # ── Strategy 2: topological depth fallback ────────────────────────────
    if use_fallback and graph.x is not None and graph.x.shape[1] > 0:
        depth_col = graph.x.shape[1] - 1
        dst_depths = graph.x[graph.edge_index[1], depth_col]
        return dst_depths.clamp(0.0, 1.0)

    return torch.ones(E) * 0.5


def add_criticality_to_graph(
    graph: Data,
    timing_report_path: str | None = None,
    slack_target: float = 0.1,
) -> Data:
    """Augment a PyG graph with criticality weights appended to edge_attr."""
    c = compute_criticality_weights(graph, timing_report_path, slack_target)

    if graph.edge_attr is None or graph.edge_attr.shape[0] == 0:
        graph.edge_attr = c.unsqueeze(1)
    else:
        graph.edge_attr = torch.cat([graph.edge_attr, c.unsqueeze(1)], dim=1)

    return graph
