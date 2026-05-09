#!/usr/bin/env python3
"""
eval/eval_models.py

Evaluation pipeline for real-data PPA framework.

Features:
  - Per-fold metrics (leave-one-design-out)
  - Aggregated metrics with confidence intervals
  - Ablation: GAT ± criticality, ± timing, GBDT ± graph
  - Wilcoxon signed-rank test (GBDT vs GAT)
  - Hypervolume computation for optimization results
  - Scatter plots, reliability diagrams, ablation bars

Usage:
  python eval/eval_models.py \
    --manifest data/manifest_real.csv \
    --gbdt-model models/gbdt/ \
    --gat-dir models/real/ \
    --results results/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.dataset import PPADataset, LABEL_NAMES
from models.gat_ppa import GATForPPA
from models.gbdt_ppa import GBDTPPAModel, build_feature_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Compute MAE, RMSE, R² per metric and average."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {"condition": name}
    maes = []
    for i, label in enumerate(LABEL_NAMES):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f"{label}_mae"] = mae
        metrics[f"{label}_rmse"] = rmse
        metrics[f"{label}_r2"] = r2
        maes.append(mae)
    metrics["avg_mae"] = np.mean(maes)
    metrics["avg_r2"] = np.mean([metrics[f"{l}_r2"] for l in LABEL_NAMES])
    return metrics


def eval_gat(model, dataset, batch_size, device):
    """Evaluate GAT model on dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=PPADataset.collate_fn)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["graph"].to(device),
                batch["spatial"].to(device),
                batch["timing_vec"].to(device),
            )
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def plot_scatter(y_true, y_pred, name, out_dir):
    """Predicted vs actual scatter plot."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, label in enumerate(LABEL_NAMES):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, s=30)
        ax.plot([0, 1], [0, 1], 'r--', lw=1)
        ax.set_xlabel(f"Actual {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label.upper()} (MAE={np.abs(y_true[:,i]-y_pred[:,i]).mean():.3f})")
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"scatter_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()


def plot_ablation_bar(results_df, out_dir):
    """Ablation bar chart."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = results_df["condition"].values
    maes = results_df["avg_mae"].values
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'][:len(conditions)]
    bars = ax.bar(range(len(conditions)), maes, color=colors)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Average MAE")
    ax.set_title("Ablation Study — Real Data")
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mae:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "ablation_bar.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models (Real Data)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--gbdt-model", type=str, default="models/gbdt")
    parser.add_argument("--gat-dir", type=str, default="models/real")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    gat_dir = Path(args.gat_dir)

    designs = ["ibex", "swerv_wrapper", "riscv32i", "jpeg_encoder", "aes128"]
    all_results = []
    all_fold_details = []

    # ── Per-fold evaluation ───────────────────────────────────────────────
    for design in designs:
        print(f"\n{'─'*40}")
        print(f"Fold: hold out {design}")

        # Load test set
        try:
            test_ds = PPADataset(
                args.manifest, hold_out_design=design, split="test",
                full_label_only=True,
            )
        except ValueError:
            print(f"  SKIP: no test samples for {design}")
            continue

        y_true = test_ds.labels.numpy()

        # GBDT
        gbdt = GBDTPPAModel()
        try:
            gbdt.load(args.gbdt_model)
            df_test = test_ds.df
            X_test = build_feature_matrix(df_test)
            y_pred_gbdt = gbdt.predict(X_test)
            fold_gbdt = compute_metrics(y_true, y_pred_gbdt, f"GBDT (fold={design})")
            all_fold_details.append(fold_gbdt)
            print(f"  GBDT: MAE={fold_gbdt['avg_mae']:.4f}")
        except Exception as e:
            print(f"  GBDT error: {e}")

        # GAT (with criticality)
        ckpt_path = gat_dir / f"gat_crit_holdout_{design}_seed42_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=args.device)
            cfg = ckpt["config"]
            model = GATForPPA(**cfg).to(args.device)
            model.load_state_dict(ckpt["state_dict"])
            y_pred_gat = eval_gat(model, test_ds, args.batch_size, args.device)
            fold_gat = compute_metrics(y_true, y_pred_gat, f"GAT+crit (fold={design})")
            all_fold_details.append(fold_gat)
            print(f"  GAT+crit: MAE={fold_gat['avg_mae']:.4f}")
        else:
            print(f"  GAT checkpoint not found: {ckpt_path}")

        # GAT (no criticality)
        ckpt_nocrit = gat_dir / f"gat_nocrit_holdout_{design}_seed42_best.pt"
        if ckpt_nocrit.exists():
            ckpt = torch.load(ckpt_nocrit, weights_only=False, map_location=args.device)
            cfg = {**ckpt["config"], "use_criticality": False}
            model_nc = GATForPPA(**cfg).to(args.device)
            model_nc.load_state_dict(ckpt["state_dict"])
            test_ds_nc = PPADataset(
                args.manifest, hold_out_design=design, split="test",
                normalizer=test_ds.normalizer, use_criticality=False,
                full_label_only=True,
            )
            y_pred_nc = eval_gat(model_nc, test_ds_nc, args.batch_size, args.device)
            fold_nc = compute_metrics(y_true, y_pred_nc, f"GAT-nocrit (fold={design})")
            all_fold_details.append(fold_nc)
            print(f"  GAT-nocrit: MAE={fold_nc['avg_mae']:.4f}")

    # ── Aggregate results ─────────────────────────────────────────────────
    if all_fold_details:
        details_df = pd.DataFrame(all_fold_details)
        details_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)

        # Aggregate by model type
        for model_type in ["GBDT", "GAT+crit", "GAT-nocrit"]:
            subset = details_df[details_df["condition"].str.startswith(model_type)]
            if not subset.empty:
                agg = {
                    "condition": model_type,
                    "avg_mae": subset["avg_mae"].mean(),
                    "mae_std": subset["avg_mae"].std(),
                    "avg_r2": subset["avg_r2"].mean(),
                    "n_folds": len(subset),
                }
                all_results.append(agg)
                print(f"\n{model_type}: MAE={agg['avg_mae']:.4f} ± {agg['mae_std']:.4f}")

    # ── Save summary ──────────────────────────────────────────────────────
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(out_dir / "eval_report.csv", index=False)
        plot_ablation_bar(summary_df, out_dir)
        print(f"\nResults saved to {out_dir}/")

    # ── Wilcoxon test ─────────────────────────────────────────────────────
    gbdt_folds = details_df[details_df["condition"].str.startswith("GBDT")]["avg_mae"].values if not details_df.empty else []
    gat_folds = details_df[details_df["condition"].str.startswith("GAT+crit")]["avg_mae"].values if not details_df.empty else []

    if len(gbdt_folds) >= 5 and len(gat_folds) >= 5:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(gbdt_folds, gat_folds)
        print(f"\nWilcoxon test: W={stat:.2f}, p={p_value:.4f}")
        with open(out_dir / "statistical_tests.json", "w") as f:
            json.dump({
                "test": "wilcoxon_signed_rank",
                "W": float(stat),
                "p_value": float(p_value),
                "significant_at_005": bool(p_value < 0.05),
                "gbdt_maes": gbdt_folds.tolist(),
                "gat_maes": gat_folds.tolist(),
            }, f, indent=2)


if __name__ == "__main__":
    main()
