
"""
Infotaxis policy (v1): select candidate with maximal predicted measurement variance (proxy for info gain).

This is a lightweight approximation of expected entropy reduction.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import time

import numpy as np

from .base_policy import Policy
from fab_benchmark.gas.gt_a import GasModelA


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a


class InfotaxisPolicy(Policy):
    name = "Infotaxis"

    def reset(self, seed: int, scenario_spec: Dict[str, Any]) -> None:
        super().reset(seed, scenario_spec)
        self._gas = GasModelA(mode_params=scenario_spec["hvac"]["mode_params"])
        self._theta = scenario_spec["hvac"]["theta_true"]
        self._q = 0.2
        self._start = float(scenario_spec["leak"]["start_time"])
        self._last_plan_t = -1e9
        self._last_action = {"type":"velocity","v":0.0,"w":0.0,"action_id":"infotaxis_init"}

    def step(self, t: float, pose: Dict[str, float], measurement: Dict[str, float], belief_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        plan_hz = float(self.policy_cfg.get("planning_budget", {}).get("planning_hz", 2.0))
        if t - self._last_plan_t < 1.0/max(1e-9, plan_hz):
            return self._last_action
        self._last_plan_t = float(t)
        t0 = time.perf_counter()

        x,y,yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        radius = float(self.policy_cfg.get("planning_budget", {}).get("local_perturb_radius", 1.0))
        candidates = int(self.policy_cfg.get("planning_budget", {}).get("candidates", 32))
        self.set_candidate_count(candidates)

        topk = []
        if belief_summary and isinstance(belief_summary.get("topk"), list):
            topk = belief_summary["topk"]
        if not topk:
            topk = [{"pos":[x,y,1.0],"p":1.0}]

        best_i = 0
        best_score = -1e9
        best_target = (x,y)
        for ci in range(candidates):
            ang = 2*math.pi*ci/max(1,candidates)
            tx = x + radius*math.cos(ang)
            ty = y + radius*math.sin(ang)
            ys = []
            ps = []
            for h in topk:
                pos = h["pos"]; p = float(h["p"])
                ys.append(self._gas.query(tx,ty,1.0,t,self._theta,(pos[0],pos[1],pos[2]),self._q,self._start))
                ps.append(p)
            ys = np.array(ys, dtype=np.float64)
            ps = np.array(ps, dtype=np.float64)
            ps = ps / max(1e-12, ps.sum())
            mu = float((ps*ys).sum())
            var = float((ps*(ys-mu)**2).sum())
            score = var  # proxy for IG
            if score > best_score:
                best_score = score
                best_i = ci
                best_target = (tx,ty)

        tx,ty = best_target
        dx,dy = tx-x, ty-y
        desired = math.atan2(dy, dx)
        err = _wrap_pi(desired - yaw)
        w_max = float(self.sim_params["robot"]["w_max"])
        v_max = float(self.sim_params["robot"]["v_max"])
        w = max(-w_max, min(w_max, 1.5*err))
        v = v_max * max(0.0, math.cos(err))
        action = {"type":"velocity","v": float(v), "w": float(w), "action_id": f"infotaxis_cand_{best_i}"}

        t1 = time.perf_counter()
        self.record_planning_ms((t1 - t0) * 1000.0)
        self._last_action = action
        return action
