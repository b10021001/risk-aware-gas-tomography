
"""
Isaac3D backend (v1).

This backend assumes it is run within an Isaac Sim Python environment.
It builds a 3D scene with wall/door colliders and a Go2 visual asset (optional) and
steps physics headlessly as configured.

If Isaac packages are not available, initialization will raise ImportError.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math
import os
import time

import numpy as np

from .base_backend import Backend
from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec, world_to_grid
from fab_benchmark.isaac.headless import make_simulation_app
from fab_benchmark.isaac.scene_builder import build_scene
from fab_benchmark.isaac.go2_controller import create_go2_rigidbody, Go2BaseController
from fab_benchmark.isaac.utils import get_prim_pose, ensure_go2_collision_proxy
class IsaacBackend(Backend):
    def init(self, sim_params: Dict[str, Any]) -> None:
        self.sim_params = sim_params
        self._scene = None
        self._app = None
        self._world = None
        self._robot: Optional[Go2BaseController] = None
        self._collision_proxy_path: Optional[str] = None
        self._v = 0.0
        self._w = 0.0
        self._collision = 0
        self._collision_count = 0
        self._prev_collision = 0
        self._proxy_collision = 0
        self._prev_collision = 0
        self._proxy_collision = 0

        self._stage_path = None  # saved USD path

    def _ensure_started(self) -> None:
        if self._app is not None:
            return
        isaac = self.sim_params.get("isaac", {})
        headless = bool(int(isaac.get("headless", 1)) == 1)
        render = bool(int(isaac.get("render", 0)) == 1)
        enable_rtx = bool(int(isaac.get("enable_rtx", 0)) == 1)
        self._app = make_simulation_app(headless=headless, render=render, enable_rtx=enable_rtx)
        # Delayed imports after SimulationApp
        from omni.isaac.core import World
        self._world = World(stage_units_in_meters=1.0)

    def load_scene(self, scene_spec: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_started()
        scene = build_scene_dict_from_scenario_spec(scene_spec)
        self._scene = scene

        # Build actual USD stage geometry
        # Clear world first
        try:
            self._world.clear_instance()
        except Exception:
            pass
        try:
            self._world.reset()
        except Exception:
            pass
        backend_meta = build_scene(scene, parent_path="/World/Env", block_stride=4)
        # Save stage path later in runners (we expose via scene dict and backend_meta)
        scene["isaac_stage_path"] = None
        scene["backend_meta"] = backend_meta
        return scene

    def reset(self, seed: int, scene_spec: Dict[str, Any], init_pose: Dict[str, Any]) -> None:
        self._ensure_started()
        if self._scene is None:
            self.load_scene(scene_spec)
        self._collision = 0
        self._collision_count = 0

        # Create robot
        isaac_cfg = self.sim_params.get("isaac", {})
        go2_usd_path = isaac_cfg.get("go2_usd_path", None)
        x = float(init_pose.get("x", 0.0))
        y = float(init_pose.get("y", 0.0))
        z = float(init_pose.get("z", float(self.sim_params["robot"]["base_height_z"])))
        yaw = float(init_pose.get("yaw", 0.0))
        self._robot = create_go2_rigidbody(root_path="/World/Robot", go2_usd_path=go2_usd_path, init_pose=(x,y,z,yaw))
        # Create a simple collision proxy to avoid missing collision meshes in the Go2 asset
        try:
            self._collision_proxy_path = ensure_go2_collision_proxy("/World/Robot/Visual/Go2")
        except Exception:
            self._collision_proxy_path = None

        # Warmup
        try:
            self._world.reset()
        except Exception:
            pass
        for _ in range(5):
            try:
                self._world.step(render=False)
            except Exception:
                pass

    def apply_action(self, action: Dict[str, Any]) -> None:
        if action.get("type") != "velocity":
            raise ValueError("IsaacBackend only supports action.type == 'velocity'")
        self._v = float(action.get("v", 0.0))
        self._w = float(action.get("w", 0.0))
        if self._robot is not None:
            self._robot.set_cmd(self._v, self._w)

    def _occupancy_collision(self) -> bool:
        # Conservative collision check using occupancy grid at current pose
        if self._scene is None or self._robot is None:
            return False
        (x,y,z), yaw = get_prim_pose(self._robot.prim_path)
        occ = self._scene["occupancy"]
        res = float(self._scene["resolution"])
        origin = (float(self._scene["origin"][0]), float(self._scene["origin"][1]))
        i, j = world_to_grid(x, y, res, origin)
        H, W = occ.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return int(occ[i, j]) == 1


    def step(self, dt: float, substeps: int) -> None:
        """
        Step physics. Collision counting is EVENT-based:
        collision_count increments only on rising edge (0->1) of combined collision flag.
        """
        dt = float(dt)
        substeps = int(substeps)
        if substeps < 1:
            substeps = 1

        if self._robot is None:
            return

        # Reset per-step collision flags
        self._collision = 0
        self._proxy_collision = 0

        # Step physics
        for _ in range(substeps):
            try:
                self._robot.apply()
            except Exception:
                pass

            try:
                self._world.step(render=False)
            except Exception:
                # Conservative: if stepping fails, treat as collision
                self._collision = 1
                break

            if self._occupancy_collision():
                self._collision = 1
                break

        # Proxy collision (grid-based) using proxy prim pose, if available
        if self._collision_proxy_path and self._scene is not None:
            try:
                (px, py, pz), _ = get_prim_pose(self._collision_proxy_path)
                gx = int((px - float(self._scene["origin"][0])) / float(self._scene["resolution"]))
                gy = int((py - float(self._scene["origin"][1])) / float(self._scene["resolution"]))
                occ = self._scene["occupancy"]
                if 0 <= gy < occ.shape[0] and 0 <= gx < occ.shape[1]:
                    if int(occ[gy, gx]) == 1:
                        self._proxy_collision = 1
            except Exception:
                self._proxy_collision = 0

        # Combined collision for this step
        coll = 1 if (self._collision or self._proxy_collision) else 0

        # EVENT counting: increment only on rising edge
        if coll == 1 and self._prev_collision == 0:
            self._collision_count += 1
        self._prev_collision = coll


    def get_state(self) -> Dict[str, Any]:
        """
        Return state snapshot. MUST NOT mutate internal counters.
        """
        if self._robot is None:
            return {
                "x": 0.0,
                "y": 0.0,
                "z": float(self.sim_params["robot"]["base_height_z"]),
                "yaw": 0.0,
                "v": float(self._v),
                "w": float(self._w),
                "collision": int(self._prev_collision),
                "collision_count": int(self._collision_count),
            }

        (x, y, z), yaw = get_prim_pose(self._robot.prim_path)
        coll = 1 if (self._collision or self._proxy_collision) else 0

        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": float(yaw),
            "v": float(self._v),
            "w": float(self._w),
            "collision": int(coll),
            "collision_count": int(self._collision_count),
        }

    def save_stage(self, usd_path: str) -> None:
        """
        Optional helper (not part of base contract) to save current stage.
        Runners call this and record path into run_meta.json["paths"]["scene_usd"].
        """
        try:
            import omni.usd
            ctx = omni.usd.get_context()
            ctx.save_as_stage(usd_path)
            self._stage_path = usd_path
            if self._scene is not None:
                self._scene["isaac_stage_path"] = usd_path
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._world is not None:
                self._world.clear_instance()
        except Exception:
            pass
        try:
            if self._app is not None:
                self._app.close()
        except Exception:
            pass
        self._app = None
        self._world = None
        self._robot = None
        self._collision_proxy_path: Optional[str] = None
        self._scene = None
