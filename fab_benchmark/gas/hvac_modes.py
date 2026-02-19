"""
HVAC modes (v3): deterministic toy flow fields for fast gas surrogate.

Supported IDs (unchanged):
  drift_pos_x, drift_neg_x, drift_pos_y, vortex_ccw, vortex_cw, time_switch

What's improved vs v1:
- Vortex is now a Rankine vortex around a configurable center (default: (0,0)).
  This avoids the non-physical "constant speed at all radii" issue in v1.
- Adds a helper `backtrace_xy(...)` that analytically backtraces particles for drift/vortex,
  enabling more accurate advection without per-step integration.
"""
from __future__ import annotations

from typing import Dict, Tuple
import math


HVAC_MODE_IDS = ("drift_pos_x", "drift_neg_x", "vortex_ccw", "vortex_cw", "drift_pos_y", "time_switch")


def _vortex_params(mode_params: Dict) -> Tuple[float, float, float, float]:
    """
    Vortex parameters (all optional):
      vortex_center: [cx, cy]  (default [0,0])
      vortex_core_radius: r0   (default 3.0 m)
      vortex_max_speed: vmax   (default uses vortex_strength or 0.35 m/s)
    """
    c = mode_params.get("vortex_center", (0.0, 0.0))
    try:
        cx, cy = float(c[0]), float(c[1])
    except Exception:
        cx, cy = 0.0, 0.0
    r0 = float(mode_params.get("vortex_core_radius", mode_params.get("vortex_core", 3.0)))
    vmax = float(mode_params.get("vortex_max_speed", mode_params.get("vortex_strength", 0.35)))
    r0 = max(1e-3, r0)
    vmax = max(0.0, vmax)
    return cx, cy, r0, vmax


def flow_velocity(x: float, y: float, t: float, theta: str, mode_params: Dict) -> Tuple[float, float]:
    """
    Returns 2D flow velocity (vx, vy) in m/s at point (x,y) and time t for HVAC mode theta.
    """
    drift_speed = float(mode_params.get("drift_speed", 0.60))
    time_switch_t = float(mode_params.get("time_switch_t", 30.0))

    if theta == "drift_pos_x":
        return (drift_speed, 0.0)
    if theta == "drift_neg_x":
        return (-drift_speed, 0.0)
    if theta == "drift_pos_y":
        return (0.0, drift_speed)

    if theta == "vortex_ccw" or theta == "vortex_cw":
        cx, cy, r0, vmax = _vortex_params(mode_params)
        dx = float(x) - cx
        dy = float(y) - cy
        r = math.hypot(dx, dy) + 1e-9

        # Rankine vortex: inside core v_theta ~ r, outside v_theta ~ 1/r
        if r <= r0:
            v_theta = vmax * (r / r0)
        else:
            v_theta = vmax * (r0 / r)

        # Tangential unit vector
        if theta == "vortex_ccw":
            tx, ty = (-dy / r), (dx / r)
        else:
            tx, ty = (dy / r), (-dx / r)

        return (v_theta * tx, v_theta * ty)

    if theta == "time_switch":
        # drift_pos_x before time_switch_t, then vortex_ccw afterwards (same params)
        if t < time_switch_t:
            return (drift_speed, 0.0)
        return flow_velocity(x, y, t, "vortex_ccw", mode_params)

    raise ValueError(f"Unknown HVAC mode id: {theta}")


def backtrace_xy(x: float, y: float, t: float, theta: str, mode_params: Dict, dt: float) -> Tuple[float, float]:
    """
    Analytic backtrace for a particle located at (x,y) at time t, going back by dt seconds,
    under the toy flow field defined by (theta, mode_params).

    This is used by the gas model to compute where the air parcel came from.
    """
    dt = float(dt)
    if dt <= 0.0:
        return (float(x), float(y))

    drift_speed = float(mode_params.get("drift_speed", 0.60))
    time_switch_t = float(mode_params.get("time_switch_t", 30.0))

    if theta == "drift_pos_x":
        return (float(x) - drift_speed * dt, float(y))
    if theta == "drift_neg_x":
        return (float(x) + drift_speed * dt, float(y))
    if theta == "drift_pos_y":
        return (float(x), float(y) - drift_speed * dt)

    if theta == "vortex_ccw" or theta == "vortex_cw":
        cx, cy, r0, vmax = _vortex_params(mode_params)
        dx = float(x) - cx
        dy = float(y) - cy
        r = math.hypot(dx, dy)
        if r <= 1e-9 or vmax <= 0.0:
            return (float(x), float(y))

        ang = math.atan2(dy, dx)

        # For a pure tangential Rankine vortex, radius stays constant, so angular speed is constant.
        if r <= r0:
            omega = vmax / r0
        else:
            omega = vmax * r0 / (r * r)

        # Forward ccw is +omega; backtrace subtracts forward rotation.
        sgn = 1.0 if theta == "vortex_ccw" else -1.0
        ang_b = ang - sgn * omega * dt

        xb = cx + r * math.cos(ang_b)
        yb = cy + r * math.sin(ang_b)
        return (float(xb), float(yb))

    if theta == "time_switch":
        # Backtrace through a regime change:
        # t < time_switch_t => drift only
        # t >= time_switch_t => vortex only
        if t <= time_switch_t:
            return backtrace_xy(x, y, t, "drift_pos_x", mode_params, dt)

        dt_after = min(dt, float(t) - float(time_switch_t))  # spent in vortex regime
        dt_before = dt - dt_after                            # spent in drift regime

        x1, y1 = backtrace_xy(x, y, t, "vortex_ccw", mode_params, dt_after)
        if dt_before > 1e-12:
            x2, y2 = backtrace_xy(x1, y1, float(time_switch_t), "drift_pos_x", mode_params, dt_before)
            return (float(x2), float(y2))
        return (float(x1), float(y1))

    # Fallback: Euler backtrace using local velocity
    vx, vy = flow_velocity(x, y, t, theta, mode_params)
    return (float(x) - float(vx) * dt, float(y) - float(vy) * dt)
