
"""
Coverage policy (v1): deterministic waypoint tour of room centers.

This is a baseline for exploration; it ignores measurements.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import math
import time

from .base_policy import Policy
from fab_benchmark.scenarios.base_scenario import reconstruct_layout_from_spec


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a


class CoveragePolicy(Policy):
    name = "Coverage"

    def reset(self, seed: int, scenario_spec: Dict[str, Any]) -> None:
        super().reset(seed, scenario_spec)
        occ, rooms, doors, nav = reconstruct_layout_from_spec(scenario_spec)
        # Tour: sorted room centers then back to start
        self._waypoints = [tuple(r["center"]) for rid, r in sorted(rooms.items(), key=lambda kv: kv[0])]
        if not self._waypoints:
            self._waypoints = [(0.0, 0.0)]
        self._wp_idx = 0
        self._last_plan_t = -1e9
        self._last_action = {"type":"velocity","v":0.0,"w":0.0,"action_id":"coverage_init"}

    def step(self, t: float, pose: Dict[str, float], measurement: Dict[str, float], belief_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        plan_hz = float(self.policy_cfg.get("planning_budget", {}).get("planning_hz", 2.0))
        if t - self._last_plan_t < 1.0/max(1e-9, plan_hz):
            return self._last_action
        self._last_plan_t = float(t)
        t0 = time.perf_counter()

        x,y,yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        tx, ty = self._waypoints[self._wp_idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
            tx, ty = self._waypoints[self._wp_idx]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)

        desired = math.atan2(dy, dx)
        err = _wrap_pi(desired - yaw)
        w_max = float(self.sim_params["robot"]["w_max"])
        v_max = float(self.sim_params["robot"]["v_max"])
        w = max(-w_max, min(w_max, 1.2*err))
        v = v_max * max(0.0, math.cos(err))
        action = {"type":"velocity","v": float(v), "w": float(w), "action_id": f"coverage_wp_{self._wp_idx:02d}"}

        t1 = time.perf_counter()
        self.record_planning_ms((t1 - t0) * 1000.0)
        self._last_action = action
        return action
