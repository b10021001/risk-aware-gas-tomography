
"""
Lite backend (v1): fast 2D grid-world kinematics with collision against occupancy.

Dependency policy: stdlib + numpy only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math
import random

import numpy as np

from .base_backend import Backend
from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec, world_to_grid


class LiteBackend(Backend):
    def init(self, sim_params: Dict[str, Any]) -> None:
        self.sim_params = sim_params
        self._scene = None
        self._rng = random.Random(0)
        self._x = 0.0
        self._y = 0.0
        self._z = float(sim_params["robot"]["base_height_z"])
        self._yaw = 0.0
        self._v = 0.0
        self._w = 0.0
        self._collision = 0
        self._collision_count = 0

        # collision recovery state
        cr = sim_params["robot"]["collision_recovery"]
        self._cr_enabled = int(cr.get("enabled", 1)) == 1
        self._cr_backoff_time = float(cr.get("backoff_time", 1.0))
        self._cr_turn_time = float(cr.get("turn_in_place_time", 1.0))
        self._cr_backoff_v = float(cr.get("backoff_v", -0.2))
        self._cr_turn_w = float(cr.get("turn_w", 0.8))
        self._cr_phase = None  # None | ("backoff", t_remain) | ("turn", t_remain)

    def load_scene(self, scene_spec: Dict[str, Any]) -> Dict[str, Any]:
        scene = build_scene_dict_from_scenario_spec(scene_spec)
        self._scene = scene
        return scene

    def reset(self, seed: int, scene_spec: Dict[str, Any], init_pose: Dict[str, Any]) -> None:
        self._rng = random.Random(int(seed))
        if self._scene is None:
            self.load_scene(scene_spec)
        self._x = float(init_pose.get("x", 0.0))
        self._y = float(init_pose.get("y", 0.0))
        self._z = float(init_pose.get("z", self._z))
        self._yaw = float(init_pose.get("yaw", 0.0))
        self._v = 0.0
        self._w = 0.0
        self._collision = 0
        self._collision_count = 0
        self._cr_phase = None

    def apply_action(self, action: Dict[str, Any]) -> None:
        # action schema fixed
        if action.get("type") != "velocity":
            raise ValueError("LiteBackend only supports action.type == 'velocity'")
        self._v = float(action.get("v", 0.0))
        self._w = float(action.get("w", 0.0))

    def _is_collision(self, x: float, y: float) -> bool:
        occ = self._scene["occupancy"]
        res = float(self._scene["resolution"])
        origin = (float(self._scene["origin"][0]), float(self._scene["origin"][1]))
        i, j = world_to_grid(x, y, res, origin)
        H, W = occ.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return int(occ[i, j]) == 1

    def step(self, dt: float, substeps: int) -> None:
        dt = float(dt)
        substeps = int(substeps)
        if substeps < 1:
            substeps = 1

        # collision recovery override
        v_cmd = self._v
        w_cmd = self._w
        if self._cr_enabled and self._cr_phase is not None:
            phase, t_remain = self._cr_phase
            if phase == "backoff":
                v_cmd = self._cr_backoff_v
                w_cmd = 0.0
                t_remain -= dt
                if t_remain <= 0:
                    self._cr_phase = ("turn", self._cr_turn_time)
                else:
                    self._cr_phase = (phase, t_remain)
            elif phase == "turn":
                v_cmd = 0.0
                w_cmd = self._cr_turn_w
                t_remain -= dt
                if t_remain <= 0:
                    self._cr_phase = None
                else:
                    self._cr_phase = (phase, t_remain)

        # integrate
        self._collision = 0
        for _ in range(substeps):
            dts = dt / substeps
            x0, y0, yaw0 = self._x, self._y, self._yaw
            self._yaw = float(self._yaw + w_cmd * dts)
            # wrap yaw to [-pi, pi]
            if self._yaw > math.pi:
                self._yaw -= 2*math.pi
            if self._yaw < -math.pi:
                self._yaw += 2*math.pi
            self._x = float(self._x + v_cmd * dts * math.cos(self._yaw))
            self._y = float(self._y + v_cmd * dts * math.sin(self._yaw))
            if self._is_collision(self._x, self._y):
                # revert and register collision
                self._x, self._y, self._yaw = x0, y0, yaw0
                self._collision = 1
                self._collision_count += 1
                if self._cr_enabled and self._cr_phase is None:
                    self._cr_phase = ("backoff", self._cr_backoff_time)
                break

    def get_state(self) -> Dict[str, Any]:
        return {
            "x": float(self._x),
            "y": float(self._y),
            "z": float(self._z),
            "yaw": float(self._yaw),
            "v": float(self._v),
            "w": float(self._w),
            "collision": int(self._collision),
            "collision_count": int(self._collision_count),
        }

    def close(self) -> None:
        self._scene = None
