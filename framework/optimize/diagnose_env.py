#!/usr/bin/env python3
"""
optimize/diagnose_env.py

Three diagnostic tests to verify the GBDT-backed RL environment produces
action-sensitive, state-sensitive, varied predictions.

Pre-registered pass criteria:
  Test 1: Per-action reward variation from same state > 0.001
  Test 2: Same-action reward across 20 states: range > 0.01
  Test 3: Surrogate output range across 100 random obs: > 0.1

If all three pass, the GBDT predictor is valid for RL training.
If any fail, the predictor or env has a bug that must be fixed first.

Usage:
  python optimize/diagnose_env.py \
    --gbdt-model models/gbdt/ \
    --manifest data/manifest_real.csv
"""

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimize.gbdt_predictor import make_gbdt_predictor, make_gbdt_predictor_from_manifest
from optimize.rl_env import ASICEnv, KNOB_LOWER, KNOB_UPPER


def test_per_action_reward(env, n_states=10):
    """Test 1: Do different actions from the same state produce different rewards?"""
    print("\n" + "=" * 60)
    print("  TEST 1: Per-action reward variation (same state)")
    print("=" * 60)

    all_ranges = []
    for seed in range(n_states):
        obs, _ = env.reset(seed=seed)
        rewards = []
        for action in range(5):
            env_copy = deepcopy(env)
            _, reward, _, _, info = env_copy.step(action)
            rewards.append(reward)

        reward_range = max(rewards) - min(rewards)
        all_ranges.append(reward_range)
        print(f"  State {seed:2d}: actions → rewards = {[f'{r:.4f}' for r in rewards]} "
              f"range = {reward_range:.6f}")

    mean_range = np.mean(all_ranges)
    min_range = np.min(all_ranges)
    print(f"\n  Mean range: {mean_range:.6f}")
    print(f"  Min range:  {min_range:.6f}")
    passed = min_range > 0.001
    print(f"  PASS: {passed}" + (" ✅" if passed else " ❌ — actions don't differentiate rewards"))
    return passed, {"mean_range": mean_range, "min_range": min_range, "all_ranges": all_ranges}


def test_cross_state_reward(env, action=1, n_states=20):
    """Test 2: Does the same action from different states produce different rewards?"""
    print("\n" + "=" * 60)
    print(f"  TEST 2: Cross-state reward variation (action={action})")
    print("=" * 60)

    rewards = []
    for seed in range(n_states):
        obs, _ = env.reset(seed=seed)
        _, reward, _, _, info = env_copy.step(action) if False else (None, None, None, None, None)
        # Fresh env each time to test from initial state
        _, reward, _, _, info = env.step(action)
        rewards.append(reward)
        # Reset for next
        env.reset(seed=seed + 100)

    # Actually do it properly: reset and step once
    rewards = []
    for seed in range(n_states):
        obs, _ = env.reset(seed=seed * 7 + 13)
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)

    reward_range = max(rewards) - min(rewards)
    reward_std = np.std(rewards)
    print(f"  Rewards across {n_states} states: min={min(rewards):.4f} max={max(rewards):.4f}")
    print(f"  Range: {reward_range:.6f}, Std: {reward_std:.6f}")
    passed = reward_range > 0.01
    print(f"  PASS: {passed}" + (" ✅" if passed else " ❌ — state doesn't affect reward"))
    return passed, {"range": reward_range, "std": reward_std, "rewards": rewards}


def test_surrogate_output_range(predictor, n_samples=100):
    """Test 3: Does the surrogate produce varied outputs across the knob space?"""
    print("\n" + "=" * 60)
    print("  TEST 3: Surrogate output range (100 random configs)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    knobs = rng.uniform(KNOB_LOWER, KNOB_UPPER, size=(n_samples, 4)).astype(np.float32)
    # Pipeline depth is integer
    knobs[:, 3] = np.round(knobs[:, 3])

    ppa = predictor(knobs)

    for i, name in enumerate(["power", "perf", "area"]):
        vals = ppa[:, i]
        print(f"  {name:8s}: min={vals.min():.4f} max={vals.max():.4f} "
              f"range={vals.max()-vals.min():.4f} std={vals.std():.4f}")

    total_range = np.ptp(ppa, axis=0)
    min_dim_range = total_range.min()
    print(f"\n  Per-dim ranges: power={total_range[0]:.4f} perf={total_range[1]:.4f} area={total_range[2]:.4f}")
    print(f"  Min dimension range: {min_dim_range:.4f}")
    passed = min_dim_range > 0.1
    print(f"  PASS: {passed}" + (" ✅" if passed else " ❌ — surrogate output too compressed"))
    return passed, {"ranges": total_range.tolist(), "min_range": float(min_dim_range)}


def test_episode_trajectory(env, n_episodes=5):
    """Bonus: Run full episodes and check reward varies over time."""
    print("\n" + "=" * 60)
    print("  TEST 4 (bonus): Episode trajectory variation")
    print("=" * 60)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 100)
        rewards = []
        actions = []
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            if term or trunc:
                break
        ep_reward = sum(rewards)
        reward_var = np.std(rewards)
        print(f"  Episode {ep}: total_reward={ep_reward:.4f} step_std={reward_var:.6f} "
              f"steps={len(rewards)}")

    print("  (No pass/fail — informational only)")
    return True, {}


def main():
    parser = argparse.ArgumentParser(description="RL Environment Diagnostics")
    parser.add_argument("--gbdt-model", type=str, default="models/gbdt")
    parser.add_argument("--manifest", type=str, default="data/manifest_real.csv")
    parser.add_argument("--design", type=str, default=None,
                        help="Design name for context (e.g., 'ibex')")
    parser.add_argument("--output", type=str, default="results/env_diagnostic.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL ENVIRONMENT DIAGNOSTIC")
    print("  Predictor: GBDT (deterministic, tabular)")
    print("=" * 60)

    # Create predictor
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        predictor = make_gbdt_predictor_from_manifest(
            args.gbdt_model, str(manifest_path), design_name=args.design
        )
        print(f"  Base features from: {manifest_path} (design={args.design})")
    else:
        predictor = make_gbdt_predictor(args.gbdt_model)
        print(f"  Base features: defaults (no manifest found)")

    # Create env
    env = ASICEnv(ppa_predictor=predictor, seed=42)

    # Run diagnostics
    results = {}

    p1, d1 = test_per_action_reward(env)
    results["test1_per_action"] = {"passed": p1, **d1}

    p2, d2 = test_cross_state_reward(env)
    results["test2_cross_state"] = {"passed": p2, **d2}

    p3, d3 = test_surrogate_output_range(predictor)
    results["test3_output_range"] = {"passed": p3, **d3}

    p4, d4 = test_episode_trajectory(env)
    results["test4_trajectory"] = {"passed": p4, **d4}

    # Summary
    all_pass = p1 and p2 and p3
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'ALL PASS ✅' if all_pass else 'FAIL ❌'}")
    print("=" * 60)
    if all_pass:
        print("  → GBDT predictor is valid for RL training.")
        print("  → Proceed with 100K single-seed validation.")
    else:
        print("  → GBDT predictor has issues. Debug before training.")

    results["all_pass"] = all_pass

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
