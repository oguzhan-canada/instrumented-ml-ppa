"""
optimize/bo_engine.py

Multi-objective Bayesian Optimization over ASIC design knobs,
using the GAT PPA predictor as a fast surrogate.

Search space (from TIP 4.2):
  - clock_target_ns    : float in [1.0, 4.0]    (target clock period)
  - buffer_strength    : float in [0.5, 2.0]    (drive strength multiplier)
  - utilization_target : float in [0.60, 0.90]  (placement density)
  - pipeline_depth     : int   in {1, 2, 3, 4}  (discretised)

Reward function (TIP Section 4.1 + Preamble):
  R(x) = -α·P_norm + β·Perf_norm - γ·A_norm - λ·max(0, -WNS)²
  with α=1.0, β=1.0, γ=0.5, λ=10.0

The BO loop:
  1. Initialise with SOBOL quasi-random points.
  2. At each iteration, use qLogNEHVI acquisition over the noisy Pareto front.
  3. Query the GAT surrogate (MC Dropout) for mean + variance.
  4. Every K=5 iterations, run the true EDA flow on top candidates
     to correct surrogate drift.
  5. Log Pareto front at each iteration.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

import botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples

import gpytorch


# ─────────────────────────────── Design space ─────────────────────────────────

# Continuous knobs: [clock_target_ns, buffer_strength, utilization_target]
# Pipeline depth treated as a 4th dimension discretised at eval time.
BOUNDS_CONTINUOUS = torch.tensor([
    [1.0, 0.5, 0.60],   # lower
    [4.0, 2.0, 0.90],   # upper
], dtype=torch.float64)

PIPELINE_DEPTHS = [1, 2, 3, 4]

# Reward weights (from TIP Preamble)
ALPHA = 1.0   # power weight
BETA  = 1.0   # performance weight
GAMMA = 0.5   # area weight
LAMBDA = 10.0 # timing violation penalty


# ─────────────────────────────── Reward helpers ───────────────────────────────

def compute_reward(ppa_norm: Tensor, wns: float = 0.0) -> float:
    """
    Scalar reward from normalised PPA vector [power_n, perf_n, area_n].
    Higher is better (we maximise).
    Timing penalty: large negative for timing violations.
    """
    p, perf, a = float(ppa_norm[0]), float(ppa_norm[1]), float(ppa_norm[2])
    violation_penalty = LAMBDA * max(0.0, -wns) ** 2
    return -ALPHA * p + BETA * perf - GAMMA * a - violation_penalty


def ppa_to_objectives(ppa_norm: Tensor) -> Tensor:
    """
    Convert normalised PPA [B, 3] to objective vector [B, 2] for EHVI.
    BoTorch maximises objectives, so:
      obj[0] = -power_norm   (minimise power → maximise negative)
      obj[1] =  perf_norm    (maximise frequency)
    Area handled via reward scalarisation; can extend to 3-obj EHVI.
    """
    return torch.stack([-ppa_norm[:, 0], ppa_norm[:, 1]], dim=1)


# ─────────────────────────────── Surrogate wrapper ────────────────────────────

class GATSurrogate:
    """
    Wraps the trained GATForPPA model to act as a BoTorch-compatible surrogate.
    Takes a raw design parameter vector x ∈ ℝ^3 and returns (mean, var) for
    each PPA metric.

    For full BO integration the caller is expected to have built a context
    that maps x → (graph, spatial, timing_vec) via a DesignContextBuilder.
    In the current prototype we use a GP over the raw parameter space,
    with the GNN predictions as observed outcomes.
    """

    def __init__(self, gat_model, context_builder, device="cpu"):
        self.gat = gat_model.to(device)
        self.gat.eval()
        self.context_builder = context_builder
        self.device = device

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x : [N, D_knobs]
        Returns means [N, 3] and variances [N, 3] in normalised PPA space.
        """
        graph, spatial, timing_vec = self.context_builder(x)
        graph   = graph.to(self.device)
        spatial = spatial.to(self.device)
        timing_vec = timing_vec.to(self.device)

        mean, var = self.gat.predict_with_uncertainty(graph, spatial, timing_vec)
        return mean.cpu().numpy(), var.cpu().numpy()


# ─────────────────────────────── BO Engine ────────────────────────────────────

class BOEngine:
    """
    Multi-objective BO loop.

    Parameters
    ----------
    surrogate       : GATSurrogate (or any callable returning mean, var)
    eda_oracle      : callable(x_np) → ppa_np  (runs true EDA; expensive)
    budget          : total surrogate evaluations
    eda_budget      : number of full EDA verification calls allowed
    eda_verify_every: run EDA every K surrogate steps
    log_dir         : directory to save Pareto front snapshots
    seed            : random seed
    """

    # Reference point for EHVI (slightly worse than worst expected)
    REF_POINT = torch.tensor([-1.05, -0.05], dtype=torch.float64)

    def __init__(
        self,
        surrogate: GATSurrogate,
        eda_oracle: Callable[[np.ndarray], np.ndarray],
        budget: int = 50,
        eda_budget: int = 10,
        eda_verify_every: int = 5,
        log_dir: str = "results/bo/",
        seed: int = 42,
    ):
        self.surrogate = surrogate
        self.eda_oracle = eda_oracle
        self.budget = budget
        self.eda_budget = eda_budget
        self.eda_verify_every = eda_verify_every
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        torch.manual_seed(seed)

        # Observations (surrogate-queried)
        self.X_obs: list[np.ndarray] = []     # design vectors
        self.Y_obs: list[np.ndarray] = []     # PPA [3]

        # Pareto front log
        self.pareto_log: list[dict] = []
        self.eda_calls = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise(self, n_init: int = 10):
        """Draw SOBOL quasi-random initial points and query the surrogate."""
        X_sobol = draw_sobol_samples(
            bounds=BOUNDS_CONTINUOUS.float(), n=n_init, q=1,
            seed=self.seed
        ).squeeze(1).numpy()    # [n_init, 3]

        print(f"[BO] Initialising with {n_init} SOBOL points…")
        means, _ = self.surrogate.predict(X_sobol)
        for x, y in zip(X_sobol, means):
            self.X_obs.append(x)
            self.Y_obs.append(y)

    # ── GP model ──────────────────────────────────────────────────────────────

    def _fit_gp(self) -> tuple[SingleTaskGP, gpytorch.mlls.ExactMarginalLogLikelihood]:
        """Fit a GP over the (X_obs, Y_obs) pairs."""
        X = torch.tensor(np.stack(self.X_obs), dtype=torch.float64)
        Y_ppa = torch.tensor(np.stack(self.Y_obs), dtype=torch.float64)
        Y_obj = ppa_to_objectives(Y_ppa.float()).double()

        gp = SingleTaskGP(
            X, Y_obj,
            input_transform=Normalize(d=X.shape[-1]),
            outcome_transform=Standardize(m=Y_obj.shape[-1]),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp, mll

    # ── Acquisition ───────────────────────────────────────────────────────────

    def _get_next_candidate(self, gp: SingleTaskGP) -> np.ndarray:
        """Optimise qLogNEHVI to get next query point.
        qLogNEHVI uses log-space formulation for better numerical stability
        and handles noisy surrogate observations by integrating over the
        unknown Pareto frontier."""
        X_obs = torch.tensor(np.stack(self.X_obs), dtype=torch.float64)

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=gp,
            ref_point=self.REF_POINT,
            X_baseline=X_obs,
            prune_baseline=True,
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=BOUNDS_CONTINUOUS.double(),
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"maxiter": 200},
        )
        return candidate.squeeze(0).detach().numpy()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, n_init: int = 10) -> dict:
        """
        Execute the full BO loop.

        Returns a dict with:
          best_x      : best design vector found
          best_ppa    : corresponding normalised PPA
          best_reward : scalar reward value
          pareto_log  : list of Pareto front snapshots per iteration
          eda_calls   : total full EDA invocations
        """
        self._initialise(n_init)
        best_reward = -float("inf")
        best_x, best_ppa = None, None

        for step in range(self.budget):
            # ── Fit GP & get next candidate ───────────────────────────────
            gp, _ = self._fit_gp()
            x_next = self._get_next_candidate(gp)

            # ── Query surrogate ───────────────────────────────────────────
            mean, var = self.surrogate.predict(x_next[None])
            ppa_pred = mean[0]
            self.X_obs.append(x_next)
            self.Y_obs.append(ppa_pred)

            reward = compute_reward(torch.tensor(ppa_pred))
            if reward > best_reward:
                best_reward = reward
                best_x  = x_next.copy()
                best_ppa = ppa_pred.copy()

            # ── EDA verification every K steps ────────────────────────────
            if (step + 1) % self.eda_verify_every == 0 and \
                    self.eda_calls < self.eda_budget:
                print(f"  [BO step {step+1}] Running EDA verification…")
                true_ppa = self.eda_oracle(x_next)
                # Replace surrogate observation with ground truth
                self.Y_obs[-1] = true_ppa
                self.eda_calls += 1
                err = np.abs(ppa_pred - true_ppa).mean()
                print(f"  Surrogate MAE vs EDA: {err:.4f}")

            # ── Pareto snapshot ───────────────────────────────────────────
            Y_all = torch.tensor(np.stack(self.Y_obs), dtype=torch.float32)
            Y_obj = ppa_to_objectives(Y_all)
            pareto_mask = is_non_dominated(Y_obj)
            snapshot = {
                "step": step + 1,
                "n_pareto": int(pareto_mask.sum()),
                "best_reward": float(best_reward),
                "eda_calls": self.eda_calls,
            }
            self.pareto_log.append(snapshot)

            if (step + 1) % 10 == 0:
                print(f"[BO step {step+1}/{self.budget}] "
                      f"Pareto size={snapshot['n_pareto']} | "
                      f"Best reward={best_reward:.4f} | "
                      f"EDA calls={self.eda_calls}")

        # ── Save results ──────────────────────────────────────────────────
        self._save_results(best_x, best_ppa, best_reward)

        return {
            "best_x":      best_x,
            "best_ppa":    best_ppa,
            "best_reward": best_reward,
            "pareto_log":  self.pareto_log,
            "eda_calls":   self.eda_calls,
        }

    def _save_results(self, best_x, best_ppa, best_reward):
        np.save(self.log_dir / "X_obs.npy", np.stack(self.X_obs))
        np.save(self.log_dir / "Y_obs.npy", np.stack(self.Y_obs))
        with open(self.log_dir / "pareto_log.json", "w") as f:
            json.dump(self.pareto_log, f, indent=2)
        summary = {
            "best_x":      best_x.tolist() if best_x is not None else None,
            "best_ppa":    best_ppa.tolist() if best_ppa is not None else None,
            "best_reward": float(best_reward),
            "eda_calls":   self.eda_calls,
        }
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[BO] Results saved → {self.log_dir}")
