#!/usr/bin/env python3
"""
optimize/rl_postmortem.py

Phase 1 post-mortem diagnostics to distinguish four hypotheses:
  H1: Surrogate noise (random graphs → no learnable signal)
  H2: Reward shape saturation
  H3: Insufficient exploration (ent_coef too low)
  H4: Surrogate adversarial point (action space pathology)

Three diagnostics:
  1. Action distribution at end of training (per seed)
  2. Surrogate reward landscape (1000 random actions)
  3. Policy generalization (same action regardless of context?)

Usage:
  python optimize/rl_postmortem.py \
    --phase1-dir results/rl_phase1 \
    --model models/real/gat_crit_holdout_ibex_seed42_best.pt \
    --manifest data/manifest_real.csv \
    --out results/rl_phase1/postmortem
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gat_ppa import GATForPPA
from optimize.rl_env import ASICEnv, KNOB_LOWER, KNOB_UPPER, ACTION_DELTAS
from optimize.rl_env import ALPHA, BETA, GAMMA, LAMBDA


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_surrogate(model_path: str, device: str = "cpu"):
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = GATForPPA(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_ppa_predictor(gat_model, device="cpu"):
    import torch_geometric.data as pyg_data
    from features.dataset import TIMING_VEC_DIM

    def predictor(knobs: np.ndarray) -> np.ndarray:
        batch_size = knobs.shape[0]
        num_nodes = 50
        num_edges = num_nodes * 4
        all_ei, all_nf, all_ea, all_b = [], [], [], []
        for i in range(batch_size):
            offset = i * num_nodes
            all_ei.append(torch.randint(0, num_nodes, (2, num_edges)) + offset)
            all_nf.append(torch.randn(num_nodes, 8))
            all_ea.append(torch.full((num_edges, 1), 0.5))
            all_b.append(torch.full((num_nodes,), i, dtype=torch.long))
        graph = pyg_data.Data(
            x=torch.cat(all_nf), edge_index=torch.cat(all_ei, dim=1),
            edge_attr=torch.cat(all_ea), batch=torch.cat(all_b),
        ).to(device)
        spatial = torch.randn(batch_size, 1, 100, 100).to(device)
        tv = torch.tensor(knobs[:, :3], dtype=torch.float32).to(device)
        if tv.shape[1] < TIMING_VEC_DIM:
            tv = torch.nn.functional.pad(tv, (0, TIMING_VEC_DIM - tv.shape[1]))
        with torch.no_grad():
            ppa = gat_model(graph, spatial, tv)
        return ppa.cpu().numpy()
    return predictor


def compute_reward(ppa: np.ndarray) -> float:
    """Compute single-step reward from PPA (same as rl_env.py)."""
    p, perf, a = float(ppa[0]), float(ppa[1]), float(ppa[2])
    wns = perf - 0.5  # WNS proxy from rl_env
    violation_pen = LAMBDA * max(0.0, -wns) ** 2
    return -ALPHA * p + BETA * perf - GAMMA * a - violation_pen


def diagnostic_1_action_distribution(phase1_dir: Path, out_dir: Path):
    """Check whether all seeds collapsed to the same or different actions."""
    print("\n=== Diagnostic 1: Action Distribution per Seed ===")

    from stable_baselines3 import PPO

    results = {}
    seed_dirs = sorted(phase1_dir.glob("seed_*"))

    for sd in seed_dirs:
        model_path = sd / "ppo_final.zip"
        if not model_path.exists():
            print(f"  {sd.name}: no ppo_final.zip, skipping")
            continue

        model = PPO.load(str(model_path))

        # Sample 500 random observations and get action distribution
        obs_samples = np.random.uniform(-0.5, 1.5, size=(500, 9)).astype(np.float32)
        actions = []
        action_probs_list = []
        for obs in obs_samples:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))

            # Get action probabilities
            obs_t = torch.tensor(obs).unsqueeze(0)
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().numpy().flatten()
            action_probs_list.append(probs)

        action_counts = np.bincount(actions, minlength=5)
        mean_probs = np.mean(action_probs_list, axis=0)

        seed_name = sd.name
        results[seed_name] = {
            "dominant_action": int(np.argmax(action_counts)),
            "action_counts": action_counts.tolist(),
            "action_pcts": (action_counts / len(actions) * 100).tolist(),
            "mean_probs": mean_probs.tolist(),
            "entropy": float(-np.sum(mean_probs * np.log(mean_probs + 1e-10))),
        }
        print(f"  {seed_name}: dominant={results[seed_name]['dominant_action']} "
              f"counts={action_counts.tolist()} "
              f"entropy={results[seed_name]['entropy']:.4f}")

    # Cross-seed analysis
    dominant_actions = [r["dominant_action"] for r in results.values()]
    all_same = len(set(dominant_actions)) == 1
    results["cross_seed"] = {
        "dominant_actions": dominant_actions,
        "all_collapsed_to_same": all_same,
        "interpretation": (
            "SAME action across seeds → reward landscape/adversarial issue (H2/H4)"
            if all_same else
            "DIFFERENT actions across seeds → surrogate is noise, arbitrary local min (H1)"
        ),
    }
    print(f"\n  Cross-seed: {dominant_actions}")
    print(f"  All same? {all_same} → {results['cross_seed']['interpretation']}")

    with open(out_dir / "diag1_action_distribution.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    return results


def diagnostic_2_reward_landscape(predictor, out_dir: Path, n_samples=1000):
    """Sample 1000 random knob configs, score with surrogate, plot reward histogram."""
    print("\n=== Diagnostic 2: Surrogate Reward Landscape ===")

    rng = np.random.default_rng(123)
    knobs_samples = rng.uniform(
        KNOB_LOWER, KNOB_UPPER, size=(n_samples, 4)
    ).astype(np.float32)
    # Quantize pipeline_depth
    knobs_samples[:, 3] = np.round(knobs_samples[:, 3])

    rewards = []
    ppas = []
    for i in range(n_samples):
        ppa = predictor(knobs_samples[i:i+1])[0]
        r = compute_reward(ppa)
        rewards.append(r)
        ppas.append(ppa.tolist())

    rewards = np.array(rewards)
    ppas = np.array(ppas)

    results = {
        "n_samples": n_samples,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_p10": float(np.percentile(rewards, 10)),
        "reward_p25": float(np.percentile(rewards, 25)),
        "reward_p50": float(np.percentile(rewards, 50)),
        "reward_p75": float(np.percentile(rewards, 75)),
        "reward_p90": float(np.percentile(rewards, 90)),
        "ppa_power_range": [float(ppas[:, 0].min()), float(ppas[:, 0].max())],
        "ppa_perf_range": [float(ppas[:, 1].min()), float(ppas[:, 1].max())],
        "ppa_area_range": [float(ppas[:, 2].min()), float(ppas[:, 2].max())],
    }

    # Check for saturation: reward range < 1.0 suggests flat landscape
    reward_range = results["reward_max"] - results["reward_min"]
    results["reward_range"] = float(reward_range)
    results["is_saturated"] = reward_range < 2.0
    results["interpretation"] = (
        "SATURATED: reward range < 2.0 → landscape is flat, PPO can't find gradient (H2)"
        if results["is_saturated"] else
        f"NOT saturated: reward range = {reward_range:.2f} → gradient exists, "
        "problem is likely exploration or surrogate noise (H1/H3)"
    )

    print(f"  Reward range: [{results['reward_min']:.3f}, {results['reward_max']:.3f}] "
          f"(range={reward_range:.3f})")
    print(f"  Reward mean±std: {results['reward_mean']:.3f} ± {results['reward_std']:.3f}")
    print(f"  Interpretation: {results['interpretation']}")

    # Generate histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(rewards, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(rewards), color="red", linestyle="--", label=f"Mean={np.mean(rewards):.2f}")
        axes[0].set_xlabel("Reward")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Reward Distribution (1000 random configs)")
        axes[0].legend()

        axes[1].scatter(ppas[:, 0], ppas[:, 1], c=rewards, cmap="RdYlGn", s=10, alpha=0.5)
        axes[1].set_xlabel("Power (norm)")
        axes[1].set_ylabel("Performance (norm)")
        axes[1].set_title("PPA Scatter (color=reward)")
        plt.colorbar(axes[1].collections[0], ax=axes[1], label="Reward")

        axes[2].scatter(ppas[:, 2], rewards, s=10, alpha=0.3)
        axes[2].set_xlabel("Area (norm)")
        axes[2].set_ylabel("Reward")
        axes[2].set_title("Area vs Reward")

        plt.tight_layout()
        plt.savefig(out_dir / "diag2_reward_landscape.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / 'diag2_reward_landscape.png'}")
    except ImportError:
        print("  matplotlib not available — skipping plot")

    with open(out_dir / "diag2_reward_landscape.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    return results


def diagnostic_3_policy_generalization(phase1_dir: Path, out_dir: Path):
    """Test if the policy outputs the same action regardless of observation context."""
    print("\n=== Diagnostic 3: Policy Generalization ===")

    from stable_baselines3 import PPO

    # Create 5 distinct "design contexts" (different observation regions)
    contexts = {
        "low_power":    np.array([0.1, 0.8, 0.2, 0.5, 0.3, 0.7, 0.4, 0.5, 0.6], dtype=np.float32),
        "high_power":   np.array([0.9, 0.3, 0.8, -0.3, -0.5, 0.8, -0.4, -0.3, -0.2], dtype=np.float32),
        "timing_viol":  np.array([0.5, 0.2, 0.5, -0.8, -1.0, 0.6, -0.9, -0.8, -0.7], dtype=np.float32),
        "balanced":     np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.7, -0.1, 0.0, 0.1], dtype=np.float32),
        "ideal":        np.array([0.2, 0.9, 0.3, 0.4, 0.3, 0.65, 0.3, 0.4, 0.5], dtype=np.float32),
    }

    results = {}
    seed_dirs = sorted(phase1_dir.glob("seed_*"))

    for sd in seed_dirs:
        model_path = sd / "ppo_final.zip"
        if not model_path.exists():
            continue

        model = PPO.load(str(model_path))
        seed_actions = {}
        for ctx_name, obs in contexts.items():
            action, _ = model.predict(obs, deterministic=True)
            seed_actions[ctx_name] = int(action)

        all_same_action = len(set(seed_actions.values())) == 1
        results[sd.name] = {
            "actions_per_context": seed_actions,
            "same_action_all_contexts": all_same_action,
        }
        print(f"  {sd.name}: {seed_actions} → {'SAME' if all_same_action else 'DIFFERENT'}")

    # Aggregate
    seeds_with_same = sum(1 for r in results.values() if r["same_action_all_contexts"])
    results["aggregate"] = {
        "seeds_with_same_action_all_contexts": seeds_with_same,
        "total_seeds": len(results),
        "interpretation": (
            "ALL seeds output same action regardless of context → "
            "surrogate is pure noise to the agent (H1) or adversarial point (H4)"
            if seeds_with_same == len(results) else
            f"{seeds_with_same}/{len(results)} seeds context-invariant → "
            "some signal exists but may be weak"
        ),
    }
    print(f"\n  {results['aggregate']['interpretation']}")

    with open(out_dir / "diag3_policy_generalization.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 RL post-mortem diagnostics")
    parser.add_argument("--phase1-dir", type=str, default="results/rl_phase1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/manifest_real.csv")
    parser.add_argument("--out", type=str, default="results/rl_phase1/postmortem")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gat_model = load_surrogate(args.model, args.device)
    predictor = make_ppa_predictor(gat_model, args.device)

    # Run all three diagnostics
    d1 = diagnostic_1_action_distribution(phase1_dir, out_dir)
    d2 = diagnostic_2_reward_landscape(predictor, out_dir)
    d3 = diagnostic_3_policy_generalization(phase1_dir, out_dir)

    # Summary
    summary = {
        "hypotheses": {
            "H1_surrogate_noise": "Random graph placeholders → no consistent PPA signal",
            "H2_reward_saturation": "Reward landscape flat/saturated → no gradient",
            "H3_exploration": "ent_coef too low → premature collapse",
            "H4_adversarial": "Agent found surrogate adversarial point",
        },
        "evidence": {
            "action_collapse_same_across_seeds": d1.get("cross_seed", {}).get("all_collapsed_to_same"),
            "reward_landscape_saturated": d2.get("is_saturated"),
            "policy_context_invariant": d3.get("aggregate", {}).get("seeds_with_same_action_all_contexts", 0) == len(list(phase1_dir.glob("seed_*"))),
        },
    }

    # Interpret
    same_action = summary["evidence"]["action_collapse_same_across_seeds"]
    saturated = summary["evidence"]["reward_landscape_saturated"]
    ctx_invariant = summary["evidence"]["policy_context_invariant"]

    if not same_action and ctx_invariant:
        summary["likely_hypothesis"] = "H1 (surrogate noise)"
        summary["recommendation"] = "Phase 2a (real graphs) is the right fix"
    elif same_action and saturated:
        summary["likely_hypothesis"] = "H2 (reward saturation)"
        summary["recommendation"] = "Revise reward function before Phase 2a"
    elif same_action and not saturated:
        summary["likely_hypothesis"] = "H4 (surrogate adversarial point)"
        summary["recommendation"] = "Add surrogate calibration before Phase 2a"
    else:
        summary["likely_hypothesis"] = "Mixed / inconclusive"
        summary["recommendation"] = "Try ent_coef ablation (0.01, 0.02, 0.05) first"

    print(f"\n{'='*60}")
    print(f"  POST-MORTEM SUMMARY")
    print(f"{'='*60}")
    print(f"  Likely hypothesis: {summary['likely_hypothesis']}")
    print(f"  Recommendation: {summary['recommendation']}")

    with open(out_dir / "postmortem_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  All diagnostics saved to {out_dir}/")


if __name__ == "__main__":
    main()
