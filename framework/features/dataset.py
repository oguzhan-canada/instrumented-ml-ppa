"""
features/dataset.py

Unified FeatureDataset for the Real-Data PPA Optimization Framework.

Key differences from synthetic version:
  - Leave-one-design-out cross-validation (5 folds, one per design)
  - Heterogeneous labels: some samples have delay-only (OpenABC-D),
    some have full PPA (OpenROAD). The 'has_delay', 'has_power' columns
    track availability. NaN labels are replaced with 0 and a validity
    mask is provided for the loss function.
  - Source tracking: each sample carries its 'source' tag

Each sample yields:
  - graph      : PyG Data object (netlist graph + criticality edge weights)
  - spatial    : FloatTensor [1, H, W]  (normalised cell-density map)
  - timing_vec : FloatTensor [D_t]      (scalar timing/design features)
  - label      : FloatTensor [3]        (normalised power, perf, area)
  - label_mask : BoolTensor  [3]        (True where label is valid)
  - design_id  : str
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from features.criticality import add_criticality_to_graph


# ─────────────────────────────── Label normalizer ─────────────────────────────

class LabelNormalizer:
    """
    Min-max normalises PPA labels to [0, 1] per metric.
    Fitted on training set (valid values only); applied to all sets.
    Handles NaN values gracefully — NaN inputs produce 0.0 output.
    """

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, labels: np.ndarray, mask: np.ndarray | None = None) -> "LabelNormalizer":
        """
        Fit on training labels.

        Parameters
        ----------
        labels : [N, 3] — raw PPA values (may contain NaN)
        mask   : [N, 3] — boolean, True where value is valid
        """
        if mask is None:
            mask = ~np.isnan(labels)

        masked = np.where(mask, labels, np.nan)
        self.min_ = np.nanmin(masked, axis=0)
        self.max_ = np.nanmax(masked, axis=0)

        # Avoid division by zero
        span = self.max_ - self.min_
        span[span < 1e-12] = 1.0
        self.max_ = self.min_ + span
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Normalise to [0, 1]. NaN values become 0.0."""
        normed = (labels - self.min_) / (self.max_ - self.min_)
        return np.nan_to_num(normed, nan=0.0).clip(0.0, 1.0).astype(np.float32)

    def inverse_transform(self, normed: np.ndarray) -> np.ndarray:
        return normed * (self.max_ - self.min_) + self.min_


# ─────────────────────────────── Feature columns ──────────────────────────────

TIMING_FEATURE_COLS = [
    "wns", "tns", "freq_mhz",
    "slack_p0", "slack_p10", "slack_p50", "slack_p90",
    "violation_count", "clock_period",
]

SPATIAL_SUMMARY_COLS = [
    "mean_density", "std_density", "peak_density",
    "hotspot_tile_count", "utilization_estimate",
]

DESIGN_PARAM_COLS = ["clock_target_ns"]

TIMING_VEC_DIM = len(TIMING_FEATURE_COLS) + len(SPATIAL_SUMMARY_COLS) + len(DESIGN_PARAM_COLS)


def _build_timing_vector(row: pd.Series) -> np.ndarray:
    """Extract scalar timing + spatial + design-param features from a metadata row."""
    cols = TIMING_FEATURE_COLS + SPATIAL_SUMMARY_COLS + DESIGN_PARAM_COLS
    vec = []
    for c in cols:
        val = row.get(c, 0.0)
        vec.append(float(val) if pd.notna(val) else 0.0)
    return np.array(vec, dtype=np.float32)


# ─────────────────────────────── Label columns ────────────────────────────────

LABEL_COLS = ["total_power", "freq_mhz", "cell_area"]
LABEL_NAMES = ["power", "perf", "area"]

# Validity flag columns in manifest
HAS_COLS = ["has_power", "has_delay", "has_power"]
# Maps: label index 0=power→has_power, 1=perf→has_delay, 2=area→has_power
# (power and area come together from OpenROAD; delay comes from all sources)


def _build_label_mask(row: pd.Series) -> np.ndarray:
    """Build a [3] boolean mask: True where the label is available."""
    has_power = bool(row.get("has_power", False))
    has_delay = bool(row.get("has_delay", False))
    return np.array([has_power, has_delay, has_power], dtype=bool)


# ─────────────────────────────── Main Dataset ─────────────────────────────────

class PPADataset(Dataset):
    """
    Parameters
    ----------
    manifest_csv      : CSV path with columns per configs/default.yaml schema
    hold_out_design   : design name to exclude (for leave-one-design-out CV)
                        If None, use all data. If set, the split param controls
                        whether this is the train set (everything except) or
                        test set (only this design).
    split             : "train" | "test" | "all"
                        "train" = all designs except hold_out_design
                        "test"  = only hold_out_design
                        "all"   = everything (no split)
    normalizer        : fitted LabelNormalizer (pass None to auto-fit on train)
    slack_target      : criticality c(e) normalisation constant (ns)
    use_criticality   : if False, skip edge criticality (ablation)
    timing_rpt_col    : column name for timing report path
    spatial_grid_size : expected H=W of density maps
    full_label_only   : if True, exclude samples without all 3 PPA labels
                        (use for GBDT which can't handle masked loss)
    seed              : random seed
    """

    def __init__(
        self,
        manifest_csv: str,
        hold_out_design: str | None = None,
        split: str = "train",
        normalizer: LabelNormalizer | None = None,
        slack_target: float = 0.1,
        use_criticality: bool = True,
        timing_rpt_col: str = "timing_rpt_path",
        spatial_grid_size: int = 100,
        full_label_only: bool = False,
        seed: int = 42,
    ):
        self.slack_target = slack_target
        self.use_criticality = use_criticality
        self.timing_rpt_col = timing_rpt_col
        self.grid = spatial_grid_size
        self.seed = seed
        self.hold_out_design = hold_out_design

        # ── Load manifest ──────────────────────────────────────────────────
        df = pd.read_csv(manifest_csv)

        # Filter by split (leave-one-design-out)
        if hold_out_design is not None and split != "all":
            if split == "test":
                df = df[df["design_name"] == hold_out_design].reset_index(drop=True)
            elif split == "train":
                df = df[df["design_name"] != hold_out_design].reset_index(drop=True)

        # Filter to full-label samples only (for GBDT)
        if full_label_only:
            if "has_power" in df.columns and "has_delay" in df.columns:
                df = df[
                    (df["has_power"].astype(bool)) & (df["has_delay"].astype(bool))
                ].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No samples found for split='{split}', hold_out='{hold_out_design}'"
            )

        self.df = df

        # ── PPA labels + mask ──────────────────────────────────────────────
        raw_labels = df[LABEL_COLS].values.astype(np.float32)

        # Build validity mask
        if "has_power" in df.columns:
            self.label_mask = torch.tensor(
                np.stack([_build_label_mask(row) for _, row in df.iterrows()]),
                dtype=torch.bool
            )
        else:
            # Assume all labels valid if no has_* columns
            self.label_mask = torch.ones(len(df), 3, dtype=torch.bool)

        # Fit normalizer on valid values only
        if normalizer is None:
            self.normalizer = LabelNormalizer().fit(
                raw_labels, self.label_mask.numpy()
            )
        else:
            self.normalizer = normalizer

        self.labels = torch.tensor(
            self.normalizer.transform(raw_labels), dtype=torch.float32
        )

        n_valid = self.label_mask.all(dim=1).sum().item()
        print(
            f"[PPADataset] split={split}, hold_out={hold_out_design}, "
            f"samples={len(df)}, full_label={n_valid}/{len(df)}, "
            f"criticality={'ON' if use_criticality else 'OFF'}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Graph ──────────────────────────────────────────────────────────
        graph_path = row.get("graph_path", "")
        if graph_path and not pd.isna(graph_path) and Path(str(graph_path)).exists():
            graph: Data = torch.load(graph_path, weights_only=False)
        else:
            # Minimal placeholder graph
            graph = Data(
                x=torch.randn(10, 8),
                edge_index=torch.randint(0, 10, (2, 20)),
            )

        if self.use_criticality:
            rpt = row.get(self.timing_rpt_col, None)
            if pd.isna(rpt) or rpt == "":
                rpt = None
            graph = add_criticality_to_graph(graph, rpt, self.slack_target)
        else:
            E = graph.edge_index.shape[1]
            crit = torch.full((E, 1), 0.5)
            if graph.edge_attr is None or graph.edge_attr.shape[0] == 0:
                graph.edge_attr = crit
            else:
                graph.edge_attr = torch.cat([graph.edge_attr, crit], dim=1)

        # ── Spatial density map ────────────────────────────────────────────
        spatial_path = row.get("spatial_path", "")
        if spatial_path and not pd.isna(spatial_path) and Path(spatial_path).exists():
            density = np.load(spatial_path).astype(np.float32)
            if density.shape != (self.grid, self.grid):
                from scipy.ndimage import zoom
                zy = self.grid / density.shape[0]
                zx = self.grid / density.shape[1]
                density = zoom(density, (zy, zx))
            spatial = torch.tensor(density).unsqueeze(0)
        else:
            spatial = torch.zeros(1, self.grid, self.grid)

        # ── Timing / design feature vector ────────────────────────────────
        timing_vec = torch.tensor(_build_timing_vector(row))

        # ── Label + mask ──────────────────────────────────────────────────
        label = self.labels[idx]
        label_mask = self.label_mask[idx]

        return {
            "graph": graph,
            "spatial": spatial,
            "timing_vec": timing_vec,
            "label": label,
            "label_mask": label_mask,
            "design_id": str(row.get("design_name", row.get("sample_id", idx))),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate for heterogeneous batch (graph + spatial + timing)."""
        from torch_geometric.data import Batch

        graphs = Batch.from_data_list([b["graph"] for b in batch])
        spatials = torch.stack([b["spatial"] for b in batch])
        timing_vecs = torch.stack([b["timing_vec"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        label_masks = torch.stack([b["label_mask"] for b in batch])
        design_ids = [b["design_id"] for b in batch]

        return {
            "graph": graphs,
            "spatial": spatials,
            "timing_vec": timing_vecs,
            "label": labels,
            "label_mask": label_masks,
            "design_ids": design_ids,
        }
