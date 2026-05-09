#!/usr/bin/env python3
"""
optimize/run_rl_gbdt.py

Phase 1 RL training with GBDT-backed predictor.

This replaces the broken GAT predictor (which used random graphs/spatial
tensors, causing flat reward) with the deterministic GBDT surrogate.

Architecture:
  - GBDT → RL environment (deterministic, action-sensitive, fast)
  - GAT  → BO surrogate (uncertainty-aware, used only for qNEHVI)

Usage:
  # Single-seed 100K validation
  python optimize/run_rl_gbdt.py \
    --gbdt-model models/gbdt/ \
    --manifest data/manifest_real.csv \
    --steps 100000 --seeds 42 \
    --results results/rl_gbdt_validation

  # Full 5-seed × 500K
  python optimize/run_rl_gbdt.py \
    --gbdt-model models/gbdt/ \
    --manifest data/manifest_real.csv \
    --steps 500000 --seeds 42 43 44 45 46 \
    --results results/rl_gbdt_phase1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from optimize.gbdt_predictor import make_gbdt_predictor_from_manifest
from optimize.rl_env import ASICEnv


# PPO hyperparameters (identical to Phase 1 for comparability)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "target_kl": 0.01,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}

N_ENVS = 8


def run_seed(predictor, seed: int, total_steps: int, out_dir: Path):
    """Train PPO for one seed, return results."""
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed {seed} | {total_steps:,} steps | {N_ENVS} envs")
    print(f"{'='*60}")

    t0 = time.time()

    # Create vectorized env
    env_fns = [lambda s=seed+i: ASICEnv(ppa_predictor=predictor, seed=s) for i in range(N_ENVS)]
    vec_env = DummyVecEnv(env_fns)
    eval_env = ASICEnv(ppa_predictor=predictor, seed=seed + 1000)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(seed_dir / "tb_logs"),
        **PPO_CONFIG,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_dir),
        log_path=str(seed_dir),
        eval_freq=max(10000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=total_steps, callback=eval_callback)
    model.save(str(seed_dir / "ppo_final"))

    elapsed = time.time() - t0

    # Final evaluation (50 episodes)
    final_rewards = []
    final_actions = []
    for _ in range(50):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(int(action))
            ep_reward += reward
            final_actions.append(int(action))
            done = term or trunc
        final_rewards.append(ep_reward)

    # Action distribution
    action_counts = np.bincount(final_actions, minlength=5)
    n_distinct = int(np.sum(action_counts > 0))

    result = {
        "seed": seed,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "min_reward": float(np.min(final_rewards)),
        "max_reward": float(np.max(final_rewards)),
        "action_counts": action_counts.tolist(),
        "n_distinct_actions": n_distinct,
        "dominant_action": int(np.argmax(action_counts)),
    }

    with open(seed_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Seed {seed}: reward={result['mean_reward']:.3f}±{result['std_reward']:.3f} "
          f"distinct={n_distinct} dominant={result['dominant_action']} "
          f"time={elapsed/60:.1f}min")

    vec_env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="RL Training with GBDT predictor")
    parser.add_argument("--gbdt-model", type=str, default="models/gbdt")
    parser.add_argument("--manifest", type=str, default="data/manifest_real.csv")
    parser.add_argument("--design", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--results", type=str, default="results/rl_gbdt")
    args = parser.parse_args()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RL TRAINING — GBDT PREDICTOR")
    print(f"  Steps: {args.steps:,} | Seeds: {args.seeds}")
    print(f"  Model: {args.gbdt_model}")
    print("=" * 60)

    # Create predictor
    predictor = make_gbdt_predictor_from_manifest(
        args.gbdt_model, args.manifest, design_name=args.design
    )

    # Run per-seed
    all_results = []
    total_t0 = time.time()

    for seed in args.seeds:
        r = run_seed(predictor, seed, args.steps, out_dir)
        all_results.append(r)

    total_elapsed = time.time() - total_t0

    # Aggregate
    rewards = [r["mean_reward"] for r in all_results]

    if len(args.seeds) > 1:
        # IQM (interquartile mean) for robust aggregation
        sorted_rewards = sorted(rewards)
        q1_idx = len(sorted_rewards) // 4
        q3_idx = 3 * len(sorted_rewards) // 4
        iqm = np.mean(sorted_rewards[q1_idx:q3_idx + 1])
    else:
        iqm = rewards[0]

    summary = {
        "config": {
            "steps": args.steps,
            "seeds": args.seeds,
            "n_envs": N_ENVS,
            "ppo": PPO_CONFIG,
            "predictor": "GBDT",
            "design": args.design,
        },
        "total_elapsed_sec": total_elapsed,
        "per_seed": all_results,
        "aggregate": {
            "iqm_reward": float(iqm),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "rewards": rewards,
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"  IQM reward: {iqm:.3f}")
    print(f"  Per-seed: {[f'{r:.3f}' for r in rewards]}")
    for r in all_results:
        print(f"    seed {r['seed']}: {r['mean_reward']:.3f} "
              f"(distinct={r['n_distinct_actions']}, dom={r['dominant_action']})")
    print(f"\n  Results: {out_dir}/")


if __name__ == "__main__":
    main()
