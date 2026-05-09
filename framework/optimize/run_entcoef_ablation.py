#!/usr/bin/env python3
"""
optimize/run_entcoef_ablation.py

Pre-registered ent_coef ablation: 3 runs × 100K steps, seed 42.
Determines if entropy collapse is CAUSE or SYMPTOM of flat reward.

Grid (locked):
  ent_coef ∈ {0.0, 0.01, 0.05}
  All other hyperparams identical to Phase 1.
  Seed 42 (directly comparable to Phase 1 seed-42 trace).

Custom metrics logged every 10K steps:
  1. Action distribution histogram (trajectory, not just final)
  2. Per-action mean reward

Pre-registered success criteria:
  See CRITERIA dict below — interpretation is locked before seeing results.

Usage:
  python optimize/run_entcoef_ablation.py \
    --model models/real/gat_crit_holdout_ibex_seed42_best.pt \
    --manifest data/manifest_real.csv \
    --results results/entcoef_ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from models.gat_ppa import GATForPPA
from optimize.rl_env import ASICEnv


# ── Pre-registered criteria (locked before seeing results) ──────────────────

CRITERIA = {
    "A": {
        "observation": "All 3 runs collapse to action 1, entropy < -0.5",
        "interpretation": "Exploration is NOT the bottleneck. Reward landscape fully responsible.",
        "next_action": "Skip directly to reward redesign",
    },
    "B": {
        "observation": "ent_coef=0.05 maintains entropy > -0.3 AND visits >=3 actions, but reward stays flat",
        "interpretation": "Exploration was suppressed but landscape is still flat. Reward redesign needed; higher ent_coef alone insufficient.",
        "next_action": "Reward redesign, then re-test ent_coef on new reward",
    },
    "C": {
        "observation": "ent_coef=0.05 finds reward > -15 (any meaningful improvement over -17.5)",
        "interpretation": "Current reward HAS learnable signal that PPO failed to find with default exploration. Two-stage diagnosis needs revision.",
        "next_action": "Phase 2a moves up in priority; reward may still need work but surrogate signal exists",
    },
    "D": {
        "observation": "ent_coef=0.05 produces chaotic non-converging behavior (high entropy, oscillating reward)",
        "interpretation": "Over-exploration; action space too coarse or reward genuinely flat",
        "next_action": "Reward redesign is the right call",
    },
}

# ── Ablation grid (locked) ──────────────────────────────────────────────────

ABLATION_GRID = {
    "ent_coefs": [0.0, 0.01, 0.05],
    "steps": 100_000,
    "seed": 42,
    "n_envs": 8,
    "eval_freq": 10_000,
    # All below identical to Phase 1
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "target_kl": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Custom callback for action distribution + per-action reward ─────────────

class ActionRewardTracker(BaseCallback):
    """
    Logs two custom metrics every `log_freq` steps:
      1. Action distribution histogram (counts per action)
      2. Per-action mean reward
    """

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(list)
        self.snapshots = []  # [{timestep, action_counts, action_mean_rewards, entropy}]

    def _on_step(self) -> bool:
        # Collect actions and rewards from the rollout buffer
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")

        if actions is not None and rewards is not None:
            for a, r in zip(actions.flatten(), rewards.flatten()):
                self.action_counts[int(a)] += 1
                self.action_rewards[int(a)].append(float(r))

        # Snapshot every log_freq steps
        if self.num_timesteps >= (len(self.snapshots) + 1) * self.log_freq:
            self._take_snapshot()

        return True

    def _take_snapshot(self):
        counts = {i: self.action_counts.get(i, 0) for i in range(5)}
        total = sum(counts.values())
        probs = {i: counts[i] / max(total, 1) for i in range(5)}

        # Entropy of action distribution
        prob_arr = np.array(list(probs.values()), dtype=np.float64)
        prob_arr = np.clip(prob_arr, 1e-10, 1.0)
        entropy = float(-np.sum(prob_arr * np.log(prob_arr)))

        mean_rewards = {}
        for i in range(5):
            rs = self.action_rewards.get(i, [])
            mean_rewards[i] = float(np.mean(rs)) if rs else None

        snapshot = {
            "timestep": self.num_timesteps,
            "action_counts": counts,
            "action_pcts": {i: round(probs[i] * 100, 1) for i in range(5)},
            "action_mean_rewards": mean_rewards,
            "entropy": entropy,
        }
        self.snapshots.append(snapshot)

        if self.verbose:
            print(f"  [ActionTracker @ {self.num_timesteps}] "
                  f"counts={list(counts.values())} entropy={entropy:.3f} "
                  f"per-action R={[f'{mean_rewards[i]:.2f}' if mean_rewards[i] is not None else 'N/A' for i in range(5)]}")

    def _on_training_end(self):
        # Final snapshot
        self._take_snapshot()


# ── Model / env setup ───────────────────────────────────────────────────────

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
        num_nodes, num_edges = 50, 200
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


def make_env(predictor, seed):
    def _init():
        return ASICEnv(ppa_predictor=predictor, seed=seed)
    return _init


# ── Run single ablation ────────────────────────────────────────────────────

def run_ablation_point(predictor, ent_coef: float, out_dir: Path, device: str):
    """Train PPO at a single ent_coef, return results."""
    g = ABLATION_GRID
    label = f"entcoef_{ent_coef:.3f}"
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ent_coef={ent_coef} | {g['steps']} steps | seed {g['seed']}")
    print(f"{'='*60}")
    t0 = time.time()

    env_fns = [make_env(predictor, g["seed"] + i) for i in range(g["n_envs"])]
    vec_env = DummyVecEnv(env_fns)
    eval_env = ASICEnv(ppa_predictor=predictor, seed=g["seed"] + 100)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=g["seed"],
        tensorboard_log=str(run_dir / "tb_logs"),
        learning_rate=g["learning_rate"],
        n_steps=g["n_steps"],
        batch_size=g["batch_size"],
        n_epochs=g["n_epochs"],
        target_kl=g["target_kl"],
        ent_coef=ent_coef,
        gamma=g["gamma"],
        gae_lambda=g["gae_lambda"],
    )

    # Custom callback for action/reward tracking
    action_tracker = ActionRewardTracker(log_freq=10_000, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(g["eval_freq"] // g["n_envs"], 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=g["steps"],
        callback=[eval_callback, action_tracker],
    )

    model.save(str(run_dir / "ppo_final"))
    elapsed = time.time() - t0

    # Final deterministic evaluation (20 episodes)
    final_rewards = []
    final_actions = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        ep_reward, ep_actions = 0.0, []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            ep_reward += reward
            ep_actions.append(int(action))
            done = terminated or truncated
        final_rewards.append(ep_reward)
        final_actions.extend(ep_actions)

    # Final action distribution
    final_action_counts = np.bincount(final_actions, minlength=5)

    # Get final policy entropy from the model
    obs_samples = np.random.RandomState(42).uniform(-0.5, 1.5, size=(200, 9)).astype(np.float32)
    entropies = []
    for obs in obs_samples:
        obs_t = torch.tensor(obs).unsqueeze(0)
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.detach().numpy().flatten()
        probs = np.clip(probs, 1e-10, 1.0)
        entropies.append(float(-np.sum(probs * np.log(probs))))
    final_entropy = float(np.mean(entropies))

    # Dominant action
    dominant_action = int(np.argmax(final_action_counts))
    n_distinct_actions = int(np.sum(final_action_counts > 0))

    result = {
        "ent_coef": ent_coef,
        "elapsed_sec": elapsed,
        "final_mean_reward": float(np.mean(final_rewards)),
        "final_std_reward": float(np.std(final_rewards)),
        "final_min_reward": float(np.min(final_rewards)),
        "final_max_reward": float(np.max(final_rewards)),
        "final_entropy": final_entropy,
        "dominant_action": dominant_action,
        "n_distinct_actions": n_distinct_actions,
        "final_action_counts": final_action_counts.tolist(),
        "action_trajectory": action_tracker.snapshots,
    }

    with open(run_dir / "ablation_result.json", "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"\n  ent_coef={ent_coef}: reward={np.mean(final_rewards):.3f}±{np.std(final_rewards):.3f} "
          f"entropy={final_entropy:.3f} dominant={dominant_action} "
          f"distinct={n_distinct_actions} time={elapsed:.0f}s")

    vec_env.close()
    return result


# ── Interpret results against pre-registered criteria ───────────────────────

def interpret(results: dict) -> dict:
    """Match results to pre-registered criteria."""
    r0 = results.get(0.0, {})
    r01 = results.get(0.01, {})
    r05 = results.get(0.05, {})

    # Check criterion A: all collapse to action 1, entropy < -0.5
    all_collapse = all(
        r.get("dominant_action") == 1 and r.get("final_entropy", 0) < 0.5
        for r in [r0, r01, r05] if r
    )

    # Check criterion C: ent_coef=0.05 finds reward > -15
    improved = r05.get("final_mean_reward", -20) > -15 if r05 else False

    # Check criterion B: ent_coef=0.05 has entropy > 0.3 and >=3 actions but flat reward
    explored_but_flat = (
        r05.get("final_entropy", 0) > 0.3
        and r05.get("n_distinct_actions", 0) >= 3
        and r05.get("final_mean_reward", -20) <= -15
    ) if r05 else False

    # Check criterion D: chaotic (high entropy variation in trajectory)
    chaotic = False
    if r05 and r05.get("action_trajectory"):
        traj_entropies = [s["entropy"] for s in r05["action_trajectory"]]
        if len(traj_entropies) > 2:
            chaotic = np.std(traj_entropies) > 0.5 and traj_entropies[-1] > 1.2

    if all_collapse:
        matched = "A"
    elif improved:
        matched = "C"
    elif explored_but_flat:
        matched = "B"
    elif chaotic:
        matched = "D"
    else:
        matched = "inconclusive"

    return {
        "matched_criterion": matched,
        "criterion_details": CRITERIA.get(matched, {"interpretation": "No clear match", "next_action": "Review manually"}),
        "evidence": {
            "all_collapse_to_action1": all_collapse,
            "ent05_reward_improved": improved,
            "ent05_explored_but_flat": explored_but_flat,
            "ent05_chaotic": chaotic,
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ent_coef ablation (pre-registered)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/manifest_real.csv")
    parser.add_argument("--results", type=str, default="results/entcoef_ablation")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save grid + criteria before running (pre-registration)
    with open(out_dir / "pre_registration.json", "w") as f:
        json.dump({"grid": ABLATION_GRID, "criteria": CRITERIA}, f, indent=2)
    print("Pre-registration saved.")

    gat_model = load_surrogate(args.model, args.device)
    predictor = make_ppa_predictor(gat_model, args.device)

    results = {}
    total_t0 = time.time()

    for ent_coef in ABLATION_GRID["ent_coefs"]:
        r = run_ablation_point(predictor, ent_coef, out_dir, args.device)
        results[ent_coef] = r

    total_elapsed = time.time() - total_t0

    # Interpret against pre-registered criteria
    interpretation = interpret(results)

    # Summary
    summary = {
        "grid": ABLATION_GRID,
        "total_elapsed_sec": total_elapsed,
        "per_run": {
            f"ent_{ec}": {
                "mean_reward": r["final_mean_reward"],
                "std_reward": r["final_std_reward"],
                "entropy": r["final_entropy"],
                "dominant_action": r["dominant_action"],
                "n_distinct_actions": r["n_distinct_actions"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for ec, r in results.items()
        },
        "interpretation": interpretation,
    }

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Generate comparison plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ent_coefs = list(results.keys())
        labels = [f"ent={ec}" for ec in ent_coefs]

        # 1. Final reward comparison
        ax = axes[0, 0]
        means = [results[ec]["final_mean_reward"] for ec in ent_coefs]
        stds = [results[ec]["final_std_reward"] for ec in ent_coefs]
        ax.bar(labels, means, yerr=stds, capsize=5, color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_ylabel("Mean Reward")
        ax.set_title("Final Reward by ent_coef")
        ax.axhline(-15, color="gray", linestyle="--", alpha=0.5, label="Criterion C threshold (-15)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Final entropy comparison
        ax = axes[0, 1]
        entropies = [results[ec]["final_entropy"] for ec in ent_coefs]
        ax.bar(labels, entropies, color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_ylabel("Policy Entropy")
        ax.set_title("Final Policy Entropy by ent_coef")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Collapse threshold (0.5)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Action distribution (stacked bar)
        ax = axes[1, 0]
        action_names = ["inc_drive", "ins_buffer", "relax_clk", "swap_cell", "no_op"]
        bottoms = np.zeros(len(ent_coefs))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#95a5a6"]
        for a_idx in range(5):
            counts = [results[ec]["final_action_counts"][a_idx] for ec in ent_coefs]
            totals = [sum(results[ec]["final_action_counts"]) for ec in ent_coefs]
            pcts = [c / max(t, 1) * 100 for c, t in zip(counts, totals)]
            ax.bar(labels, pcts, bottom=bottoms, label=action_names[a_idx], color=colors[a_idx])
            bottoms += pcts
        ax.set_ylabel("Action %")
        ax.set_title("Action Distribution")
        ax.legend(loc="upper right", fontsize=8)

        # 4. Entropy trajectory over training
        ax = axes[1, 1]
        for ec in ent_coefs:
            traj = results[ec].get("action_trajectory", [])
            if traj:
                ts = [s["timestep"] for s in traj]
                ents = [s["entropy"] for s in traj]
                ax.plot(ts, ents, marker="o", markersize=3, label=f"ent={ec}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action Entropy")
        ax.set_title("Entropy Trajectory During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"ent_coef Ablation — Matched: {interpretation['matched_criterion']}", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "ablation_comparison.png", dpi=150)
        plt.close()
        print(f"\n  Saved: {out_dir / 'ablation_comparison.png'}")
    except ImportError:
        print("  matplotlib not available — skipping plot")

    print(f"\n{'='*60}")
    print(f"  ABLATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"  Matched criterion: {interpretation['matched_criterion']}")
    print(f"  Interpretation: {interpretation['criterion_details'].get('interpretation', 'N/A')}")
    print(f"  Next action: {interpretation['criterion_details'].get('next_action', 'N/A')}")
    for ec in ent_coefs:
        r = results[ec]
        print(f"  ent={ec}: reward={r['final_mean_reward']:.3f} entropy={r['final_entropy']:.3f} "
              f"dominant={r['dominant_action']} distinct={r['n_distinct_actions']}")
    print(f"\n  Results: {out_dir}/")


if __name__ == "__main__":
    main()
