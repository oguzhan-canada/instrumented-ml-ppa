"""
optimize/gbdt_predictor.py

GBDT-backed PPA predictor for the RL environment.

Architecture rationale (May 2025 finding):
  The GAT surrogate uses graph + spatial + timing inputs. When deployed as an RL
  env surrogate, the graph and spatial branches were filled with random tensors
  (no real circuit data available at RL time), causing predictions to be
  dominated by noise and invariant to action knobs. This made the RL landscape
  completely flat (all actions yield identical reward).

  GBDT operates on tabular features, is deterministic, and directly maps
  design knobs to PPA predictions. It's the correct architectural choice for
  an action-sensitive RL environment surrogate.

Usage:
  from optimize.gbdt_predictor import make_gbdt_predictor
  predictor = make_gbdt_predictor("models/gbdt")
  env = ASICEnv(ppa_predictor=predictor)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from models.gbdt_ppa import GBDTPPAModel, ALL_FEATURE_COLS


# Map env knobs to GBDT feature positions
# Env knobs: [clock_target_ns, buffer_strength, utilization_target, pipeline_depth]
# GBDT features (21 total):
#   [0:6]  GRAPH_SUMMARY: max_fan_out, register_count, avg_path_depth, total_edges, node_count, edge_density
#   [6:15] TIMING: wns, tns, freq_mhz, slack_p0, slack_p10, slack_p50, slack_p90, violation_count, clock_period
#   [15:20] SPATIAL: mean_density, std_density, peak_density, hotspot_tile_count, utilization_estimate
#   [20]   PARAM: clock_target_ns

# Feature indices for knob-influenced features
IDX_CLOCK_TARGET = 20       # clock_target_ns → PARAM_COLS[0]
IDX_CLOCK_PERIOD = 14       # clock_period (= clock_target_ns in practice)
IDX_FREQ_MHZ = 8            # freq_mhz = 1000 / clock_target_ns
IDX_UTILIZATION = 19        # utilization_estimate ← utilization_target
IDX_REGISTER_COUNT = 1      # register_count ← proxy for buffer_strength
IDX_AVG_PATH_DEPTH = 2      # avg_path_depth ← proxy for pipeline_depth


def make_gbdt_predictor(
    model_dir: str,
    base_features: np.ndarray | None = None,
    design_context: dict | None = None,
) -> callable:
    """
    Create a GBDT-backed PPA predictor compatible with ASICEnv.

    Parameters
    ----------
    model_dir : str
        Path to directory containing gbdt_power.json, gbdt_perf.json,
        gbdt_area.json, and gbdt_normalizer.npz.
    base_features : np.ndarray [21], optional
        Base feature vector representing the design context. If None, uses
        sensible defaults derived from the training data distribution.
    design_context : dict, optional
        Named overrides for base features (e.g., {"node_count": 1500}).

    Returns
    -------
    predictor : callable(knobs: np.ndarray[N, 4]) → np.ndarray[N, 3]
        Deterministic PPA predictor suitable for ASICEnv.
    """
    model = GBDTPPAModel()
    model.load(model_dir)

    # Default base features (representative of a mid-range ASAP7 design)
    if base_features is None:
        base_features = np.array([
            # GRAPH_SUMMARY (6)
            12.0,       # max_fan_out
            500.0,      # register_count
            8.0,        # avg_path_depth
            3000.0,     # total_edges
            1200.0,     # node_count
            0.005,      # edge_density
            # TIMING (9)
            -0.1,       # wns (slight violation)
            -5.0,       # tns
            500.0,      # freq_mhz
            -0.2,       # slack_p0
            -0.05,      # slack_p10
            0.1,        # slack_p50
            0.3,        # slack_p90
            10.0,       # violation_count
            2.0,        # clock_period (ns)
            # SPATIAL (5)
            0.45,       # mean_density
            0.15,       # std_density
            0.85,       # peak_density
            5.0,        # hotspot_tile_count
            0.70,       # utilization_estimate
            # PARAM (1)
            2.0,        # clock_target_ns
        ], dtype=np.float32)

    # Apply design context overrides
    if design_context:
        for name, value in design_context.items():
            if name in ALL_FEATURE_COLS:
                idx = ALL_FEATURE_COLS.index(name)
                base_features[idx] = value

    base_features = base_features.copy()

    def predictor(knobs: np.ndarray) -> np.ndarray:
        """
        Map env action knobs to GBDT features and predict PPA.

        Parameters
        ----------
        knobs : np.ndarray [N, 4]
            [clock_target_ns, buffer_strength, utilization_target, pipeline_depth]

        Returns
        -------
        ppa : np.ndarray [N, 3]
            Predicted [power, perf, area] normalised to [0, 1].
        """
        batch_size = knobs.shape[0]

        # Start from base features for each sample
        X = np.tile(base_features, (batch_size, 1))

        # Map knobs to feature positions
        clock_target = knobs[:, 0]     # clock_target_ns
        buffer_str = knobs[:, 1]       # buffer_strength [0.5, 2.0]
        util_target = knobs[:, 2]      # utilization_target [0.6, 0.9]
        pipe_depth = knobs[:, 3]       # pipeline_depth [1, 4]

        # Direct mappings
        X[:, IDX_CLOCK_TARGET] = clock_target
        X[:, IDX_CLOCK_PERIOD] = clock_target
        X[:, IDX_FREQ_MHZ] = 1000.0 / np.clip(clock_target, 0.5, 10.0)

        # Utilization → utilization_estimate
        X[:, IDX_UTILIZATION] = util_target

        # Buffer strength → affects register count and WNS
        # More buffers → more registers, better timing
        X[:, IDX_REGISTER_COUNT] = base_features[IDX_REGISTER_COUNT] * buffer_str
        # Buffer insertion improves WNS (more buffers → less negative slack)
        X[:, 6] = base_features[6] + (buffer_str - 1.0) * 0.2  # wns
        X[:, 7] = base_features[7] + (buffer_str - 1.0) * 2.0  # tns

        # Pipeline depth → affects avg_path_depth and timing
        X[:, IDX_AVG_PATH_DEPTH] = base_features[IDX_AVG_PATH_DEPTH] / np.clip(pipe_depth, 1, 4)

        # Cross-effects: tighter clock → more violations, higher utilization → worse density
        X[:, 13] = np.clip(base_features[13] * (2.0 / np.clip(clock_target, 0.5, 10.0)), 0, 100)  # violation_count
        X[:, 15] = base_features[15] * (util_target / 0.7)  # mean_density scales with util
        X[:, 17] = np.clip(base_features[17] * (util_target / 0.7) ** 2, 0, 1)  # peak_density

        return model.predict(X)

    return predictor


def make_gbdt_predictor_from_manifest(
    model_dir: str,
    manifest_path: str,
    design_name: str | None = None,
) -> callable:
    """
    Create a GBDT predictor with base features derived from a real design
    in the manifest.

    Parameters
    ----------
    model_dir : str
        Path to GBDT model directory.
    manifest_path : str
        Path to manifest_real.csv.
    design_name : str, optional
        Design to use as context (e.g., "ibex"). If None, uses median across all.

    Returns
    -------
    predictor : callable
    """
    import pandas as pd
    from models.gbdt_ppa import build_feature_matrix

    df = pd.read_csv(manifest_path)

    if design_name and "design_name" in df.columns:
        mask = df["design_name"] == design_name
        if mask.sum() > 0:
            df = df[mask]

    X = build_feature_matrix(df)
    # Use median row as base features (robust to outliers)
    base_features = np.median(X, axis=0).astype(np.float32)

    return make_gbdt_predictor(model_dir, base_features=base_features)
