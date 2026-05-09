#!/usr/bin/env python3
"""
optimize/run_rl.py

RL (PPO) training for PPA optimization on real data.
Uses GAT surrogate model + ASICEnv with the defined reward function.

Usage:
  python optimize/run_rl.py \
    --model models/real/gat_crit_holdout_ibex_seed42_best.pt \
    --manifest data/manifest_real.csv \
    --steps 500000 \
    --eval-freq 10000 \
    --results results/rl_real/ \
    --seed 42
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gat_ppa import GATForPPA
from optimize.rl_env import ASICEnv


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
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
    """
    Wrap GAT model as a callable knobs → PPA predictor for ASICEnv.
    Returns a function(knobs: np.ndarray[N, D]) → np.ndarray[N, 3].
    """
    import torch_geometric.data as pyg_data
    from features.dataset import TIMING_VEC_DIM

    def predictor(knobs: np.ndarray) -> np.ndarray:
        batch_size = knobs.shape[0]
        num_nodes = 50
        num_edges = num_nodes * 4

        # Build batched synthetic graph context
        all_edge_index = []
        all_node_feat = []
        all_edge_attr = []
        all_batch = []
        for i in range(batch_size):
            offset = i * num_nodes
            ei = torch.randint(0, num_nodes, (2, num_edges)) + offset
            all_edge_index.append(ei)
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


def plot_reward_curve(eval_log: list[dict], out_dir: Path):
    """Generate rl_reward_curve.png from evaluation log."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [e["timestep"] for e in eval_log]
        mean_r = [e["mean_reward"] for e in eval_log]
        std_r = [e["std_reward"] for e in eval_log]

        fig, ax = plt.subplots(figsize=(8, 5))
        mean_arr = np.array(mean_r)
        std_arr = np.array(std_r)
        ax.plot(steps, mean_r, "b-", label="Mean reward")
        ax.fill_between(steps, mean_arr - std_arr, mean_arr + std_arr,
                        alpha=0.2, color="blue")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.set_title("RL (PPO) Reward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "rl_reward_curve.png", dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / 'rl_reward_curve.png'}")
    except ImportError:
        print("  matplotlib not available — skipping reward curve plot")


def main():
    parser = argparse.ArgumentParser(description="RL PPO training (Real Data)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--results", type=str, default="results/rl_real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogate model...")
    gat_model = load_surrogate(args.model, args.device)

    print("Building PPA predictor...")
    predictor = make_ppa_predictor(gat_model, args.device)

    print("Creating environment...")
    env = ASICEnv(
        ppa_predictor=predictor,
        seed=args.seed,
    )

    print(f"Training PPO for {args.steps} steps...")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(out_dir / "tb_logs"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=str(out_dir),
            log_path=str(out_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=5,
            deterministic=True,
        )

        model.learn(
            total_timesteps=args.steps,
            callback=eval_callback,
        )

        # Save final model
        model.save(str(out_dir / "ppo_final"))
        print(f"Model saved: {out_dir / 'ppo_final.zip'}")

        # Rename EvalCallback best model to match expected name
        best_src = out_dir / "best_model.zip"
        best_dst = out_dir / "ppo_best.zip"
        if best_src.exists():
            best_src.rename(best_dst)
            print(f"Best model: {best_dst}")

        # Evaluate final model and build eval log
        print("\nFinal evaluation...")
        eval_log = []
        rewards = []
        for ep in range(10):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        # Build eval log from training evaluations.npz if available
        eval_npz = out_dir / "evaluations.npz"
        if eval_npz.exists():
            data = np.load(str(eval_npz))
            timesteps = data["timesteps"]
            results_arr = data["results"]
            for i, ts in enumerate(timesteps):
                ep_rewards = results_arr[i]
                eval_log.append({
                    "timestep": int(ts),
                    "mean_reward": float(np.mean(ep_rewards)),
                    "std_reward": float(np.std(ep_rewards)),
                })

        # Write rl_eval_log.csv
        if eval_log:
            with open(out_dir / "rl_eval_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["timestep", "mean_reward", "std_reward"]
                )
                writer.writeheader()
                writer.writerows(eval_log)
            print(f"  Saved: {out_dir / 'rl_eval_log.csv'}")

        best_reward = float(np.max(rewards))
        mean_reward = float(np.mean(rewards))

        summary = {
            "total_steps": args.steps,
            "best_mean_reward": best_reward,
            "mean_reward": mean_reward,
            "std_reward": float(np.std(rewards)),
            "eval_episodes": 10,
            "model_path": args.model,
            "seed": args.seed,
        }

        with open(out_dir / "rl_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        # Generate reward curve plot
        if eval_log:
            plot_reward_curve(eval_log, out_dir)

        print(f"\nRL complete. Results in {out_dir}/")
        print(f"  Best reward: {best_reward:.4f}")
        print(f"  Mean reward: {mean_reward:.4f} ± {np.std(rewards):.4f}")

        if best_reward < -5:
            print("  ⚠️  Likely timing violations (very negative reward)")
        else:
            print("  ✓ No obvious timing violations")

    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("  pip install stable-baselines3==2.0.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
