"""
models/gbdt_ppa.py

XGBoost baseline for PPA prediction.
Operates on the flat tabular feature vector (no graph structure).

Tabular feature vector columns (in order):
  Graph summaries : max_fan_out, register_count, avg_path_depth, total_edges,
                    node_count, edge_density
  Timing features : wns, tns, freq_mhz, slack_p0, slack_p10, slack_p50, slack_p90,
                    violation_count, clock_period
  Spatial stats   : mean_density, std_density, peak_density, hotspot_tile_count,
                    utilization_estimate
  Design params   : clock_target_ns

Trains three separate XGBoost regressors — one per PPA metric.
Multi-output is handled by iterating over targets.
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold


# ─────────────────────────────── Feature schema ───────────────────────────────

GRAPH_SUMMARY_COLS = [
    "max_fan_out", "register_count", "avg_path_depth",
    "total_edges", "node_count", "edge_density",
]

TIMING_COLS = [
    "wns", "tns", "freq_mhz",
    "slack_p0", "slack_p10", "slack_p50", "slack_p90",
    "violation_count", "clock_period",
]

SPATIAL_COLS = [
    "mean_density", "std_density", "peak_density",
    "hotspot_tile_count", "utilization_estimate",
]

PARAM_COLS = ["clock_target_ns"]

ALL_FEATURE_COLS = GRAPH_SUMMARY_COLS + TIMING_COLS + SPATIAL_COLS + PARAM_COLS

TARGET_COLS = ["total_power", "freq_mhz_label", "cell_area"]
TARGET_NAMES = ["power", "perf", "area"]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and return a [N, D] feature matrix from a manifest DataFrame.
    Missing columns are filled with 0.
    """
    matrix = np.zeros((len(df), len(ALL_FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(ALL_FEATURE_COLS):
        if col in df.columns:
            matrix[:, j] = df[col].fillna(0.0).values.astype(np.float32)
    return matrix


# ─────────────────────────────── Model wrapper ────────────────────────────────

class GBDTPPAModel:
    """
    Three XGBoost regressors — one per PPA output — with:
      - Hyperparameter defaults aligned with TIP Appendix D
      - 5-fold cross-validation for hyperparameter tuning
      - Feature importance export
    """

    # Default hyperparameters from TIP Appendix D
    DEFAULT_PARAMS = {
        "n_estimators":     300,
        "max_depth":        8,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "early_stopping_rounds": 20,
    }

    SEARCH_GRID = {
        "n_estimators": [100, 300, 500],
        "max_depth":    [6, 8, 10],
        "learning_rate":[0.05, 0.1, 0.2],
    }

    def __init__(self, params: dict | None = None):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.models = {
            name: xgb.XGBRegressor(**{k: v for k, v in p.items()
                                      if k != "early_stopping_rounds"},
                                   early_stopping_rounds=p["early_stopping_rounds"])
            for name in TARGET_NAMES
        }
        self.label_min: np.ndarray | None = None
        self.label_max: np.ndarray | None = None
        self._fitted = False

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _fit_normalizer(self, y: np.ndarray):
        self.label_min = y.min(axis=0)
        self.label_max = y.max(axis=0)
        self._range = self.label_max - self.label_min
        self._range[self._range == 0] = 1.0

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        return (y - self.label_min) / self._range

    def inverse_normalize(self, y: np.ndarray) -> np.ndarray:
        return y * self._range + self.label_min

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ):
        """
        y_train : [N, 3]  — [power, perf, area] in original units
        """
        self._fit_normalizer(y_train)
        yn_train = self._normalize(y_train)
        eval_set = [(X_val, self._normalize(y_val))] if X_val is not None else None

        for i, name in enumerate(TARGET_NAMES):
            if verbose:
                print(f"  Training GBDT for {name}…")
            fit_kwargs = dict(verbose=False)
            if eval_set:
                fit_kwargs["eval_set"] = [(X_val, self._normalize(y_val)[:, i])]
            else:
                # Disable early stopping when no eval set is provided
                self.models[name].set_params(early_stopping_rounds=None)
            self.models[name].fit(X_train, yn_train[:, i], **fit_kwargs)

        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, 3] in normalised [0,1] space."""
        assert self._fitted, "Call fit() first."
        return np.stack([
            self.models[name].predict(X).clip(0, 1)
            for name in TARGET_NAMES
        ], axis=1)

    def predict_original(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, 3] in original units."""
        return self.inverse_normalize(self.predict(X))

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """
        k-fold CV on the combined dataset.
        Returns per-metric {"mae", "rmse", "r2"} mean across folds.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        self._fit_normalizer(y)
        yn = self._normalize(y)

        results: dict[str, dict[str, list]] = {
            n: {"mae": [], "rmse": [], "r2": []} for n in TARGET_NAMES
        }

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            for i, name in enumerate(TARGET_NAMES):
                m = xgb.XGBRegressor(**{
                    k: v for k, v in self.DEFAULT_PARAMS.items()
                    if k != "early_stopping_rounds"
                })
                m.fit(X[tr_idx], yn[tr_idx, i], verbose=False)
                yp = m.predict(X[va_idx]).clip(0, 1)
                yt = yn[va_idx, i]
                results[name]["mae"].append(mean_absolute_error(yt, yp))
                results[name]["rmse"].append(mean_squared_error(yt, yp) ** 0.5)
                results[name]["r2"].append(r2_score(yt, yp))

        return {
            name: {
                "mae":  float(np.mean(v["mae"])),
                "rmse": float(np.mean(v["rmse"])),
                "r2":   float(np.mean(v["r2"])),
            }
            for name, v in results.items()
        }

    # ── Metrics ───────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
        """Evaluate on a held-out set. y in original units."""
        yn = self._normalize(y)
        yp = self.predict(X)
        out = {}
        for i, name in enumerate(TARGET_NAMES):
            out[name] = {
                "mae":  float(mean_absolute_error(yn[:, i], yp[:, i])),
                "rmse": float(mean_squared_error(yn[:, i], yp[:, i]) ** 0.5),
                "r2":   float(r2_score(yn[:, i], yp[:, i])),
            }
        return out

    def feature_importances(self) -> dict[str, pd.Series]:
        """Return per-metric feature importance Series (by gain)."""
        result = {}
        for name in TARGET_NAMES:
            scores = self.models[name].get_booster().get_score(importance_type="gain")
            series = pd.Series(scores).reindex(
                [f"f{i}" for i in range(len(ALL_FEATURE_COLS))]
            ).fillna(0)
            series.index = ALL_FEATURE_COLS[:len(series)]
            result[name] = series.sort_values(ascending=False)
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        for name in TARGET_NAMES:
            self.models[name].save_model(f"{directory}/gbdt_{name}.json")
        np.savez(
            f"{directory}/gbdt_normalizer.npz",
            min=self.label_min, max=self.label_max
        )
        print(f"Saved GBDT models → {directory}/")

    def load(self, directory: str):
        for name in TARGET_NAMES:
            self.models[name].load_model(f"{directory}/gbdt_{name}.json")
        d = np.load(f"{directory}/gbdt_normalizer.npz")
        self.label_min = d["min"]
        self.label_max = d["max"]
        self._range = self.label_max - self.label_min
        self._range[self._range == 0] = 1.0
        self._fitted = True
        return self
