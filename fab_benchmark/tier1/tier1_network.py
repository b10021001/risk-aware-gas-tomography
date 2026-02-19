"""
Tier-1 fixed sensor network (v1).

Goal in this benchmark:
- Provide coarse, *zone-level* alarm metadata (room / position / time) to emulate a fab TGMS.
- Do NOT reveal ground-truth leak position directly.
- Support "many sensors may alarm" as plume spreads.

Design notes:
- We place one Tier-1 sensor per room using a simple rule:
    Prefer near the primary doorway (midpoint of door segment), inset into the room.
    (Fallback: room center if geometry is missing.)
  This avoids trivially coinciding with the benchmark leak position (which is often at room center).
- Alarm logic uses debounce + optional latching (common in safety systems).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import math
import random

import numpy as np

from fab_benchmark.sensor.emulator import SensorEmulator


def _world_to_grid(x: float, y: float, resolution: float, origin: Tuple[float, float]) -> Tuple[int, int]:
    j = int(round((float(x) - float(origin[0])) / float(resolution)))
    i = int(round((float(y) - float(origin[1])) / float(resolution)))
    return i, j


class Tier1Network:
    def __init__(self, scene_dict: Dict[str, Any], scenario_spec: Dict[str, Any], sim_params: Dict[str, Any], seed: int = 0):
        self.scene = scene_dict
        self.scenario_spec = scenario_spec
        self.sim_params = sim_params

        # Config with safe defaults
        cfg = {}
        if isinstance(sim_params, dict):
            cfg = sim_params.get("tier1", {}) if isinstance(sim_params.get("tier1", {}), dict) else {}
        self.cfg = cfg

        self._rng = random.Random(int(seed) + 12345)

        # placement parameters
        self._inset_m = float(cfg.get("inset_m", 0.8))
        self._jitter_m = float(cfg.get("jitter_m", 0.15))
        self._placement = str(cfg.get("placement", "room_center"))  # room_center | door_inset
        self._mount = str(cfg.get("mount", "ceiling"))  # breathing|ceiling|floor
        self._mount_z = float(cfg.get("mount_z", -1.0))   # if >=0, overrides mount preset

        # alarm parameters
        self._alarm_thr = float(cfg.get("alarm_threshold", 0.15))
        self._reset_thr = float(cfg.get("reset_threshold", 0.10))
        self._debounce_s = float(cfg.get("debounce_s", 1.0))
        self._release_s = float(cfg.get("release_s", 1.0))
        self._latch = int(cfg.get("latch", 1)) == 1  # latch alarms by default

        # sensor emulator parameters (Tier-1 fixed sensors are usually less noisy / faster)
        sensor_spec = scenario_spec.get("sensor", {})
        tau_default = float(cfg.get("tau", sensor_spec.get("tau", 2.0) * 0.5))
        sigma_default = float(cfg.get("sigma", sensor_spec.get("sigma", 0.03) * 0.35))
        drift_default = cfg.get("drift", sensor_spec.get("drift", {"enabled": 0, "rate": 0.001, "bias0": 0.0}))

        self.sensors: List[Dict[str, Any]] = self._build_sensors()
        self._emu: List[SensorEmulator] = []
        for k in range(len(self.sensors)):
            self._emu.append(SensorEmulator(
                tau=float(tau_default),
                sigma=float(sigma_default),
                drift=dict(drift_default) if isinstance(drift_default, dict) else {"enabled": 0, "rate": 0.001, "bias0": 0.0},
                y_safe=0.0,  # Tier-1 hazard not used; alarms use thresholds above
                seed=int(seed) + 777 * (k + 1),
            ))

        # internal alarm bookkeeping
        self._alarm_state = [0 for _ in self.sensors]
        self._t_first_alarm: List[Optional[float]] = [None for _ in self.sensors]
        self._t_above = [0.0 for _ in self.sensors]
        self._t_below = [0.0 for _ in self.sensors]

    def reset(self, seed: int = 0) -> None:
        self._rng = random.Random(int(seed) + 12345)
        for k, emu in enumerate(self._emu):
            emu.reset(seed=int(seed) + 777 * (k + 1))
        self._alarm_state = [0 for _ in self.sensors]
        self._t_first_alarm = [None for _ in self.sensors]
        self._t_above = [0.0 for _ in self.sensors]
        self._t_below = [0.0 for _ in self.sensors]

    def _choose_mount_z(self) -> float:
        if self._mount_z >= 0.0:
            return float(self._mount_z)
        # Use existing z layers in the benchmark for consistency
        if self._mount == "ceiling":
            return 2.0
        if self._mount == "floor":
            return 0.2
        return 1.0  # breathing zone / mid-height

    def _is_free(self, x: float, y: float) -> bool:
        try:
            occ = self.scene["occupancy"]
            res = float(self.scene["resolution"])
            origin = tuple(float(v) for v in self.scene.get("origin", (0.0, 0.0)))
            i, j = _world_to_grid(x, y, res, origin)
            H, W = int(occ.shape[0]), int(occ.shape[1])
            if i < 0 or i >= H or j < 0 or j >= W:
                return False
            return int(occ[i, j]) == 0
        except Exception:
            return True

    def _build_sensors(self) -> List[Dict[str, Any]]:
        rooms = self.scene.get("rooms", {})
        doors = self.scene.get("doors", {})  # door_id -> {room_a, room_b, geom}
        mount_z = self._choose_mount_z()

        sensors: List[Dict[str, Any]] = []
        room_ids = sorted(list(rooms.keys()))

        # Pre-index doors by room id
        doors_by_room: Dict[str, List[Dict[str, Any]]] = {rid: [] for rid in room_ids}
        for did, d in doors.items():
            try:
                ra = str(d.get("room_a"))
                rb = str(d.get("room_b"))
                if ra in doors_by_room:
                    doors_by_room[ra].append({"door_id": str(did), "geom": d.get("geom", {})})
                if rb in doors_by_room:
                    doors_by_room[rb].append({"door_id": str(did), "geom": d.get("geom", {})})
            except Exception:
                continue

        for sid, rid in enumerate(room_ids):
            info = rooms[rid]
            cx, cy = float(info["center"][0]), float(info["center"][1])
            # Default target: room center
            x, y = cx, cy

            # Optional: place near doorway (inset into the room) instead of center.
            # This can avoid trivially coinciding with benchmarks where leak.pos is at room center.
            if self._placement == "door_inset":
                ds = sorted(doors_by_room.get(rid, []), key=lambda e: e.get("door_id", ""))
                if ds:
                    g = ds[0].get("geom", {}) if isinstance(ds[0], dict) else {}
                    p0 = g.get("p0", None) if isinstance(g, dict) else None
                    p1 = g.get("p1", None) if isinstance(g, dict) else None
                    if isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple)) and len(p0) >= 2 and len(p1) >= 2:
                        xm = 0.5 * (float(p0[0]) + float(p1[0]))
                        ym = 0.5 * (float(p0[1]) + float(p1[1]))
                        vx = cx - xm
                        vy = cy - ym
                        norm = math.hypot(vx, vy)
                        if norm > 1e-6:
                            vx /= norm
                            vy /= norm
                            x = xm + self._inset_m * vx
                            y = ym + self._inset_m * vy

            # Deterministic small jitter per room (break symmetry, avoid exact room center)
            # Use hash of room_id (stable) for reproducibility.
            h = abs(hash(str(rid))) % 1000003
            rng = random.Random(h)
            ang = rng.random() * 2.0 * math.pi
            rad = self._jitter_m * (0.3 + 0.7 * rng.random())
            x += rad * math.cos(ang)
            y += rad * math.sin(ang)

            # If invalid, fallback to center
            if not self._is_free(x, y):
                x, y = cx, cy

            sensors.append({
                "id": int(sid),
                "room_id": str(rid),
                "pos": [float(x), float(y), float(mount_z)],
                "alarm": 0,
                "t_first_alarm": None,
                "y_meas": 0.0,
                "y_raw": 0.0,
            })

        return sensors

    def step(self, t: float, dt: float, gas_query_fn) -> Dict[str, Any]:
        """Advance Tier-1 network by one control step.

        gas_query_fn signature: (x,y,z,t) -> y_raw (float)
        """
        t = float(t)
        dt = float(dt)

        alarm_ids: List[int] = []
        alarm_events: List[Dict[str, Any]] = []

        for k, s in enumerate(self.sensors):
            x, y, z = float(s["pos"][0]), float(s["pos"][1]), float(s["pos"][2])
            y_raw = float(gas_query_fn(x, y, z, t))
            meas = self._emu[k].step(y_raw=y_raw, dt=dt)
            y_meas = float(meas["y_meas"])

            s["y_raw"] = float(y_raw)
            s["y_meas"] = float(y_meas)

            # Debounce + optional latching
            if self._alarm_state[k] == 0:
                if y_meas >= self._alarm_thr:
                    self._t_above[k] += dt
                    if self._t_above[k] >= self._debounce_s:
                        self._alarm_state[k] = 1
                        s["alarm"] = 1
                        if self._t_first_alarm[k] is None:
                            self._t_first_alarm[k] = t
                            s["t_first_alarm"] = float(t)
                        alarm_events.append({"id": int(k), "t": float(t), "y_meas": float(y_meas)})
                else:
                    self._t_above[k] = 0.0
                    s["alarm"] = 0

            else:
                # already alarmed
                s["alarm"] = 1
                if not self._latch:
                    if y_meas <= self._reset_thr:
                        self._t_below[k] += dt
                        if self._t_below[k] >= self._release_s:
                            self._alarm_state[k] = 0
                            s["alarm"] = 0
                            self._t_below[k] = 0.0
                            self._t_above[k] = 0.0
                    else:
                        self._t_below[k] = 0.0

            if int(s["alarm"]) == 1:
                alarm_ids.append(int(k))

        return {
            "t": float(t),
            "alarm_ids": alarm_ids,
            "alarm_events": alarm_events,
            "sensors": self.sensors,
            # For downstream convenience
            "n_sensors": int(len(self.sensors)),
            "n_alarms": int(len(alarm_ids)),
            "alarm_threshold": float(self._alarm_thr),
        }
