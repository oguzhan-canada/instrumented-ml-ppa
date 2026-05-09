"""
optimize/rl_env.py

Gymnasium environment for PPO-based ASIC design optimisation.

State  (9-dim, from TIP 4.3):
  [power_norm, perf_norm, area_norm, wns_norm, tns_norm,
   utilization, slack_p10_norm, slack_p50_norm, slack_p90_norm]

Actions (discrete, 5):
  0 — increase_drive_strength  (buffer_strength += 0.1)
  1 — insert_buffer            (reduces congestion proxy; area += small Δ)
  2 — reduce_clock_target      (clock_target_ns += 0.1  → easier to meet)
  3 — swap_cell_type           (random perturbation of area/power)
  4 — no_op

Reward:
  R = -α·P_norm + β·Perf_norm - γ·A_norm - λ·max(0,-WNS_norm)²
  + step_bonus · ΔWNS  (reward shaping to encourage WNS improvement)

The environment uses the GATForPPA ML model as a fast simulator.
Full EDA is called only at episode end to obtain ground-truth R.

Reward weights (TIP Preamble):
  α=1.0 (power), β=1.0 (perf), γ=0.5 (area), λ=10.0 (timing penalty)
"""

from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Reward weights
ALPHA  = 1.0
BETA   = 1.0
GAMMA  = 0.5
LAMBDA = 10.0
STEP_BONUS = 0.1   # reward shaping coefficient for WNS improvement


# Action effects on the continuous design knob vector
# [clock_target_ns, buffer_strength, utilization_target, pipeline_depth]
ACTION_DELTAS = {
    0: np.array([ 0.00,  0.10,  0.00,  0.0], dtype=np.float32),   # increase drive
    1: np.array([ 0.00,  0.05,  -0.02, 0.0], dtype=np.float32),   # insert buffer
    2: np.array([ 0.10,  0.00,  0.00,  0.0], dtype=np.float32),   # relax clock
    3: np.array([ 0.00,  0.00,  0.00,  0.0], dtype=np.float32),   # cell swap (stochastic)
    4: np.array([ 0.00,  0.00,  0.00,  0.0], dtype=np.float32),   # no-op
}

KNOB_LOWER = np.array([1.0, 0.5, 0.60, 1.0], dtype=np.float32)
KNOB_UPPER = np.array([4.0, 2.0, 0.90, 4.0], dtype=np.float32)

MAX_EPISODE_STEPS = 50


class ASICEnv(gym.Env):
    """
    Parameters
    ----------
    ppa_predictor       : callable(knobs: np.ndarray) → ppa_norm: np.ndarray [3]
                          Should be GATForPPA.predict_with_uncertainty wrapped
                          to return mean PPA for the knob vector.
    initial_knobs       : starting design knob vector [4]; if None use midpoint
    eda_oracle          : optional callable(knobs) → true_ppa [3]
                          Called at episode end for ground-truth evaluation.
    seed                : RNG seed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ppa_predictor,
        initial_knobs: np.ndarray | None = None,
        eda_oracle=None,
        seed: int = 42,
    ):
        super().__init__()
        self.ppa_predictor = ppa_predictor
        self.eda_oracle = eda_oracle
        self.rng = np.random.default_rng(seed)

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # ── State ─────────────────────────────────────────────────────────
        self.initial_knobs = (
            initial_knobs if initial_knobs is not None
            else (KNOB_LOWER + KNOB_UPPER) / 2
        )
        self._knobs = self.initial_knobs.copy()
        self._ppa   = np.zeros(3, dtype=np.float32)
        self._step  = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._knobs = self.initial_knobs.copy()
        self._ppa   = self._query_predictor(self._knobs)
        self._step  = 0
        self._prev_wns = self._estimate_wns()

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # ── Apply action ─────────────────────────────────────────────────
        delta = ACTION_DELTAS[action].copy()

        # Action 3 (cell swap): add small random perturbation
        if action == 3:
            delta = self.rng.normal(0, 0.02, size=4).astype(np.float32)

        self._knobs = np.clip(self._knobs + delta, KNOB_LOWER, KNOB_UPPER)
        # Pipeline depth is integer
        self._knobs[3] = float(round(self._knobs[3]))

        # ── Query ML surrogate ───────────────────────────────────────────
        new_ppa = self._query_predictor(self._knobs)
        new_wns = self._estimate_wns()

        # ── Reward ───────────────────────────────────────────────────────
        p, perf, a = float(new_ppa[0]), float(new_ppa[1]), float(new_ppa[2])
        wns_norm = (new_wns + 1.0) / 2.0   # map [-1,1] → [0,1] heuristic
        violation_pen = LAMBDA * max(0.0, -new_wns) ** 2
        reward = (-ALPHA * p + BETA * perf - GAMMA * a - violation_pen)

        # Reward shaping: bonus for WNS improvement
        wns_improvement = new_wns - self._prev_wns
        reward += STEP_BONUS * max(0.0, wns_improvement)

        self._ppa  = new_ppa
        self._prev_wns = new_wns
        self._step += 1

        terminated = False
        truncated  = self._step >= MAX_EPISODE_STEPS

        info = {
            "knobs":   self._knobs.copy(),
            "ppa":     new_ppa.copy(),
            "wns":     new_wns,
            "step":    self._step,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── Episode end: EDA ground truth ─────────────────────────────────────────

    def evaluate_with_eda(self) -> dict[str, Any]:
        """
        Call the true EDA oracle on the current knob configuration.
        Returns {"true_ppa": array[3], "surrogate_ppa": array[3], "error": float}.
        Intended to be called after episode completion.
        """
        if self.eda_oracle is None:
            return {"error": "No EDA oracle configured."}

        true_ppa = self.eda_oracle(self._knobs)
        surrogate_ppa = self._ppa
        mae = float(np.abs(true_ppa - surrogate_ppa).mean())
        return {
            "knobs":         self._knobs.copy(),
            "true_ppa":      true_ppa,
            "surrogate_ppa": surrogate_ppa,
            "mae":           mae,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _query_predictor(self, knobs: np.ndarray) -> np.ndarray:
        """Query the ML surrogate. Returns normalised PPA [3]."""
        result = self.ppa_predictor(knobs[None])   # expects [N, D_knobs]
        if isinstance(result, tuple):
            return result[0][0]   # (mean, var) → mean[0]
        return np.asarray(result[0])

    def _estimate_wns(self) -> float:
        """
        Heuristic WNS proxy from surrogate PPA.
        Higher perf (closer to 1) and looser clock → better WNS.
        Returns a value in [-1, 1]; negative = timing violation.
        """
        perf = float(self._ppa[1])
        clock_ns = float(self._knobs[0])
        # Normalise: assume target freq = 1/clock_ns GHz
        # If predicted perf exceeds normalised threshold, no violation
        wns = perf - 0.5   # [0,1] perf centered at 0 → [-0.5, 0.5]
        return float(np.clip(wns, -1.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        """
        9-dim state vector (TIP 4.3):
        [power, perf, area, wns_norm, tns_proxy, utilization,
         slack_p10_proxy, slack_p50_proxy, slack_p90_proxy]
        """
        p, perf, a = self._ppa
        wns = self._estimate_wns()
        util = float(self._knobs[2])    # utilization_target as proxy

        # Slack percentile proxies: derived from WNS and a simple spread model
        spread = 0.1
        s10 = wns - spread
        s50 = wns
        s90 = wns + spread

        obs = np.array([p, perf, a, wns, wns * 2, util, s10, s50, s90],
                       dtype=np.float32)
        return obs
