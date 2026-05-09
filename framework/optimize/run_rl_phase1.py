#!/usr/bin/env python3
"""
optimize/run_rl_phase1.py

Phase 1: 500K RL × 5 seeds with SubprocVecEnv (n_envs=8).
Reports IQM (Interquartile Mean) using rliable for robust comparison.

Usage:
  python optimize/run_rl_phase1.py \
    --model models/real/gat_crit_holdout_ibex_seed42_best.pt \
    --manifest data/manifest_real.csv \
    --steps 500000 \
    --n-seeds 5 \
    --n-envs 8 \
    --results results/rl_phase1/ \
    --base-seed 42
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gat_ppa import GATForPPA
from optimize.rl_env import ASICEnv


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
    """Load trained GAT model as surrogate."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = GATForPPA(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_ppa_predictor(gat_model, device: str = "cpu"):
    """Wrap GAT as knobs → PPA predictor for ASICEnv."""
    import torch_geometric.data as pyg_data
    from features.dataset import TIMING_VEC_DIM

    def predictor(knobs: np.ndarray) -> np.ndarray:
        batch_size = knobs.shape[0]
        num_nodes = 50
        num_edges = num_nodes * 4

        all_edge_index, all_node_feat, all_edge_attr, all_batch = [], [], [], []
        for i in range(batch_size):
            offset = i * num_nodes
            all_edge_index.append(torch.randint(0, num_nodes, (2, num_edges)) + offset)
            all_node_feat.append(torch.randn(num_nodes, 8))
            all_edge_attr.append(torch.full((num_edges, 1), 0.5))
            all_batch.append(torch.full((num_nodes,), i, dtype=torch.long))

        graph = pyg_data.Data(
            x=torch.cat(all_node_feat, dim=0),
            edge_index=torch.cat(all_edge_index, dim=1),
            edge_attr=torch.cat(all_edge_attr, dim=0),
            batch=torch.cat(all_batch, dim=0),
        ).to(device)
        spatial = torch.randn(batch_size, 1, 100, 100).to(device)
        timing_vec = torch.tensor(knobs[:, :3], dtype=torch.float32).to(device)
        if timing_vec.shape[1] < TIMING_VEC_DIM:
            timing_vec = torch.nn.functional.pad(
                timing_vec, (0, TIMING_VEC_DIM - timing_vec.shape[1])
            )
        with torch.no_grad():
            ppa = gat_model(graph, spatial, timing_vec)
        return ppa.cpu().numpy()

    return predictor


def make_env(predictor, seed: int):
    """Factory for SubprocVecEnv."""
    def _init():
        env = ASICEnv(ppa_predictor=predictor, seed=seed)
        return env
    return _init


def evaluate_policy(model, env, n_episodes: int = 20) -> np.ndarray:
    """Evaluate model for n_episodes, return array of total rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.array(rewards)


def compute_iqm(scores: np.ndarray) -> float:
    """Interquartile Mean: mean of middle 50% of scores."""
    q25, q75 = np.percentile(scores, [25, 75])
    mask = (scores >= q25) & (scores <= q75)
    if mask.sum() == 0:
        return float(np.mean(scores))
    return float(np.mean(scores[mask]))


def run_single_seed(
    gat_model,
    predictor,
    seed: int,
    n_envs: int,
    total_steps: int,
    eval_freq: int,
    out_dir: Path,
    device: str,
):
    """Train PPO for one seed, return eval rewards."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed {seed} — {total_steps} steps, {n_envs} parallel envs")
    print(f"{'='*60}")
    t0 = time.time()

    # Create vectorized environment (DummyVecEnv — avoids pickling GAT model)
    env_fns = [make_env(predictor, seed + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Single eval env for deterministic evaluation
    eval_env = ASICEnv(ppa_predictor=predictor, seed=seed + 100)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(seed_dir / "tb_logs"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,  # larger batch for 8 envs
        n_epochs=10,
        target_kl=0.01,  # early stopping on KL divergence
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_dir),
        log_path=str(seed_dir),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
    )

    # Save final model
    model.save(str(seed_dir / "ppo_final"))

    # Final evaluation (20 episodes)
    final_rewards = evaluate_policy(model, eval_env, n_episodes=20)

    elapsed = time.time() - t0
    print(f"  Seed {seed} done in {elapsed/60:.1f} min")
    print(f"  Mean reward: {np.mean(final_rewards):.4f} ± {np.std(final_rewards):.4f}")

    # Save seed summary
    seed_summary = {
        "seed": seed,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "elapsed_sec": elapsed,
        "final_mean_reward": float(np.mean(final_rewards)),
        "final_std_reward": float(np.std(final_rewards)),
        "final_min_reward": float(np.min(final_rewards)),
        "final_max_reward": float(np.max(final_rewards)),
        "final_rewards": final_rewards.tolist(),
    }
    with open(seed_dir / "seed_summary.json", "w") as f:
        json.dump(seed_summary, f, indent=2, cls=NumpyEncoder)

    # Load evaluation curve
    eval_npz = seed_dir / "evaluations.npz"
    eval_curve = []
    if eval_npz.exists():
        data = np.load(str(eval_npz))
        for i, ts in enumerate(data["timesteps"]):
            eval_curve.append({
                "timestep": int(ts),
                "mean_reward": float(np.mean(data["results"][i])),
                "std_reward": float(np.std(data["results"][i])),
            })

    vec_env.close()
    return final_rewards, eval_curve


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 500K RL × 5 seeds")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--results", type=str, default="results/rl_phase1")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Phase 1: {args.steps} steps × {args.n_seeds} seeds × {args.n_envs} envs")
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device}")

    gat_model = load_surrogate(args.model, args.device)
    predictor = make_ppa_predictor(gat_model, args.device)

    all_final_rewards = []
    all_eval_curves = []
    total_t0 = time.time()

    for seed in seeds:
        rewards, curve = run_single_seed(
            gat_model=gat_model,
            predictor=predictor,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=args.steps,
            eval_freq=args.eval_freq,
            out_dir=out_dir,
            device=args.device,
        )
        all_final_rewards.append(rewards)
        all_eval_curves.append(curve)

    total_elapsed = time.time() - total_t0

    # Aggregate results
    all_rewards_flat = np.concatenate(all_final_rewards)
    per_seed_means = [float(np.mean(r)) for r in all_final_rewards]

    # Compute IQM using rliable if available, otherwise manual
    try:
        from rliable import metrics as rly_metrics
        # rliable expects shape [n_runs, n_tasks] or [n_runs]
        score_matrix = np.array(per_seed_means).reshape(-1, 1)
        iqm_val = float(rly_metrics.aggregate_iqm(score_matrix))
        iqm_source = "rliable"
    except (ImportError, Exception):
        iqm_val = compute_iqm(np.array(per_seed_means))
        iqm_source = "manual"

    # Summary
    summary = {
        "phase": "Phase 1",
        "total_steps_per_seed": args.steps,
        "n_seeds": args.n_seeds,
        "n_envs": args.n_envs,
        "seeds": seeds,
        "total_elapsed_sec": total_elapsed,
        "total_elapsed_min": total_elapsed / 60,
        "per_seed_mean_rewards": per_seed_means,
        "overall_mean_reward": float(np.mean(all_rewards_flat)),
        "overall_std_reward": float(np.std(all_rewards_flat)),
        "iqm_reward": iqm_val,
        "iqm_source": iqm_source,
        "median_reward": float(np.median(all_rewards_flat)),
        "model": args.model,
        "device": args.device,
    }

    with open(out_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Per-seed CSV
    with open(out_dir / "per_seed_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_reward", "std_reward", "min", "max", "iqm"])
        for i, seed in enumerate(seeds):
            r = all_final_rewards[i]
            writer.writerow([
                seed, f"{np.mean(r):.4f}", f"{np.std(r):.4f}",
                f"{np.min(r):.4f}", f"{np.max(r):.4f}",
                f"{compute_iqm(r):.4f}",
            ])

    # Generate aggregate reward curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (seed, curve) in enumerate(zip(seeds, all_eval_curves)):
            if curve:
                ts = [c["timestep"] for c in curve]
                means = [c["mean_reward"] for c in curve]
                ax.plot(ts, means, alpha=0.6, label=f"Seed {seed}")

        # Compute mean curve across seeds
        if all_eval_curves and all(len(c) > 0 for c in all_eval_curves):
            min_len = min(len(c) for c in all_eval_curves)
            mean_curve = np.mean(
                [[c[j]["mean_reward"] for j in range(min_len)] for c in all_eval_curves],
                axis=0,
            )
            std_curve = np.std(
                [[c[j]["mean_reward"] for j in range(min_len)] for c in all_eval_curves],
                axis=0,
            )
            ts = [all_eval_curves[0][j]["timestep"] for j in range(min_len)]
            ax.plot(ts, mean_curve, "k-", linewidth=2, label="Mean ± Std")
            ax.fill_between(ts, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.15, color="black")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Phase 1: PPO 500K × {args.n_seeds} seeds (IQM={iqm_val:.3f})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "phase1_reward_curves.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / 'phase1_reward_curves.png'}")
    except ImportError:
        print("  matplotlib not available — skipping plot")

    print(f"\n{'='*60}")
    print(f"  PHASE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"  Seeds: {seeds}")
    print(f"  Overall mean reward: {np.mean(all_rewards_flat):.4f} ± {np.std(all_rewards_flat):.4f}")
    print(f"  IQM reward ({iqm_source}): {iqm_val:.4f}")
    print(f"  Per-seed means: {[f'{m:.3f}' for m in per_seed_means]}")
    print(f"  Results: {out_dir}/")


if __name__ == "__main__":
    main()
