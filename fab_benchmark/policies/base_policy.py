"""
Policy contract (v1) - HOTFIX.

This file restores the public API expected by runners/run_trial.py:
  - get_budget_stats()
and additionally provides:
  - set_rollout_count()
to support policies that update their rollout budget at runtime.

DO NOT rename public API.
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class Policy:
    name: str = "BasePolicy"

    def __init__(self, policy_cfg: Dict[str, Any], sim_params: Dict[str, Any]):
        self.policy_cfg = dict(policy_cfg or {})
        self.sim_params = dict(sim_params or {})
        pb = self.policy_cfg.get("planning_budget", {}) if isinstance(self.policy_cfg, dict) else {}
        self._budget_stats = {
            "planning_hz": float(pb.get("planning_hz", 2.0)),
            "candidates": int(pb.get("candidates", 32)),
            "rollouts": int(pb.get("rollout_count", 0)),
            "inference_ms_mean": 0.0,
            "planning_ms_mean": 0.0,
        }
        self._timing_n = 0
        self._planning_n = 0
        self._planning_ms_sum = 0.0
        self._inference_ms_sum = 0.0

    def reset(self, seed: int, scenario_spec: Dict[str, Any]) -> None:
        self.seed = int(seed)
        self.scenario_spec = scenario_spec
        self._timing_n = 0
        self._planning_n = 0
        self._planning_ms_sum = 0.0
        self._inference_ms_sum = 0.0
        # keep planning budget fields as-is

    def step(
        self,
        t: float,
        pose: Dict[str, float],
        measurement: Dict[str, float],
        belief_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # ----- timing stats -----
    def record_inference_ms(self, inference_ms: float) -> None:
        self._timing_n += 1
        self._inference_ms_sum += float(inference_ms)
        self._budget_stats["inference_ms_mean"] = self._inference_ms_sum / max(1, self._timing_n)

    def record_planning_ms(self, planning_ms: float) -> None:
        self._planning_n += 1
        self._planning_ms_sum += float(planning_ms)
        self._budget_stats["planning_ms_mean"] = self._planning_ms_sum / max(1, self._planning_n)

    # ----- budget knobs -----
    def set_candidate_count(self, n: int) -> None:
        self._budget_stats["candidates"] = int(n)

    def set_rollout_count(self, n: int) -> None:
        self._budget_stats["rollouts"] = int(n)

    # ----- runner API -----
    def get_budget_stats(self) -> Dict[str, Any]:
        """
        Return a dict with fixed keys expected by runners/run_trial.py.
        """
        # be defensive if keys are missing
        planning_hz = float(self._budget_stats.get("planning_hz", 2.0))
        candidates = int(self._budget_stats.get("candidates", 32))
        rollouts = int(self._budget_stats.get("rollouts", 0))
        inference_ms_mean = float(self._budget_stats.get("inference_ms_mean", 0.0))
        planning_ms_mean = float(self._budget_stats.get("planning_ms_mean", 0.0))
        return {
            "planning_hz": planning_hz,
            "candidates": candidates,
            "rollouts": rollouts,
            "inference_ms_mean": inference_ms_mean,
            "planning_ms_mean": planning_ms_mean,
        }
