"""
Discrete posterior update (v2) over HypothesisGrid.

Changes vs v1 (important for e1/F2):
- Uses the same *continuous leak* surrogate as GT-A v3 by numerically integrating
  contributions from recent emission times (windowed). This ensures concentration
  stays highest near the source and reduces "single puff" artifacts.
- Uses hvac_modes.backtrace_xy (analytic backtrace) instead of a single vx,vy drift.

Key feature for "Ours": optional explicit sensor lag model:
  s_{t+1} = s_t + (dt/tau) * (c_pred - s_t)
Likelihood compares measurement y_meas against s_t (predicted sensor output).

Ablation "Ours_NoDelayModel": set use_delay_model=False so predicted output is c_pred directly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import time
import math

import numpy as np

from .hypothesis_grid import HypothesisGrid
from fab_benchmark.gas.gt_a import GasModelA
from fab_benchmark.gas.hvac_modes import backtrace_xy


# Fixed default (Appendix A belief_defaults)
SIGMA_MODEL = 0.05
TOPK_DEFAULT = 10


class Posterior:
    def __init__(
        self,
        grid: HypothesisGrid,
        scenario_spec: Dict[str, Any],
        sim_params: Dict[str, Any],
        use_delay_model: bool = True,
        **_kwargs: Any,
    ):
        # Accept **_kwargs for forward/backward compatibility with older patches.
        self.grid = grid
        self.scenario_spec = scenario_spec
        self.sim_params = sim_params
        self.use_delay_model = bool(use_delay_model)

        self.P = np.ones((grid.N,), dtype=np.float64)
        self.P /= np.sum(self.P)

        # predicted sensor internal state per hypothesis
        self.S = np.zeros((grid.N,), dtype=np.float64)

        self._mode_params = dict((scenario_spec.get("hvac", {}) or {}).get("mode_params", {}) or {})
        self._theta = str((scenario_spec.get("hvac", {}) or {}).get("theta_true", "drift_pos_x"))
        self._gas_model = GasModelA(mode_params=self._mode_params)

        # We assume nominal q for inference (not estimated); keep fixed to scenario default.
        self._q_assumed = float((scenario_spec.get("leak", {}) or {}).get("q", 0.2))
        self._start_time = float((scenario_spec.get("leak", {}) or {}).get("start_time", 0.0))

        self._p_cut = float((sim_params.get("belief", {}) or {}).get("p_cut", 0.95))
        self._z_bin_height = float((sim_params.get("belief", {}) or {}).get("z_bin_height", 0.5))
        self._cell_stride = int((sim_params.get("belief", {}) or {}).get("cell_stride", 2))
        self._topk = TOPK_DEFAULT

        # sensor dynamics (assumed known)
        self._tau = float(((scenario_spec.get("sensor", {}) or {}).get("tau", 2.0)))
        self._t_prev: Optional[float] = None

    def reset(self, seed: int = 0) -> None:
        self.P[:] = 1.0
        self.P /= np.sum(self.P)
        self.S[:] = 0.0
        self._t_prev = None

    def _predict_concentration_vec(self, *, x: float, y: float, z: float, t: float) -> np.ndarray:
        """Vectorized concentration prediction at (x,y,z,t) for all hypotheses."""
        pos = self.grid.positions  # float32 [N,3]
        c_pred = np.zeros((pos.shape[0],), dtype=np.float64)

        valid = np.isfinite(pos[:, 0])
        if not np.any(valid):
            return c_pred

        sx = pos[valid, 0].astype(np.float64)
        sy = pos[valid, 1].astype(np.float64)
        sz = pos[valid, 2].astype(np.float64)

        if t < self._start_time:
            return c_pred

        dt_since = max(0.0, float(t - self._start_time))
        window = float(self._mode_params.get("release_window_s", getattr(self._gas_model, "release_window_s", 20.0)))
        window = max(1e-6, window)
        T = min(dt_since, window)

        # Integration settings
        n = int(self._mode_params.get("n_time_samples", getattr(self._gas_model, "n_time_samples", 12)))
        n = max(1, min(32, n))
        dtau = T / float(n) if T > 0 else 0.0

        D = float(getattr(self._gas_model, "D", 0.20))
        D_z = float(getattr(self._gas_model, "D_z", 0.03))
        lam = float(getattr(self._gas_model, "lambda_decay", 0.01))
        source_r = float(getattr(self._gas_model, "source_radius", 1.0))
        clamp_min = float(getattr(self._gas_model, "clamp_min", 0.0))
        clamp_max = float(getattr(self._gas_model, "clamp_max", 10.0))
        base_sigma_z2 = float(self._mode_params.get("sigma_z2_base", 0.6 ** 2))
        scale = float(self._mode_params.get("scale_factor", 50.0))

        # Vertical offset per hypothesis
        dz = float(z) - sz

        # Numerical midpoint integration over tau in [0, T]
        acc = np.zeros((sx.shape[0],), dtype=np.float64)
        if dtau > 0.0:
            for i in range(n):
                tau = (i + 0.5) * dtau
                xb, yb = backtrace_xy(float(x), float(y), float(t), self._theta, self._mode_params, float(tau))

                dx = xb - sx
                dy = yb - sy

                sigma2 = source_r ** 2 + 2.0 * D * float(tau)
                sigma2 = max(sigma2, 1e-6)
                norm_xy = 1.0 / (2.0 * math.pi * sigma2)
                g_xy = norm_xy * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))

                sigma_z2 = base_sigma_z2 + 2.0 * D_z * float(tau)
                sigma_z2 = max(sigma_z2, 1e-6)
                g_z = (1.0 / math.sqrt(2.0 * math.pi * sigma_z2)) * np.exp(-(dz * dz) / (2.0 * sigma_z2))

                acc += math.exp(-lam * float(tau)) * g_xy * g_z

            c = float(self._q_assumed) * acc * float(dtau) * float(scale)
            # Near-field peak bump (keeps peak at leak even with strong advection)
            peak_boost = float(getattr(self._gas_model, "source_peak_boost", 0.0))
            peak_sigma = float(getattr(self._gas_model, "source_peak_sigma", 0.0))
            peak_dt = float(getattr(self._gas_model, "source_peak_dt_s", 0.0))
            if peak_boost > 0.0 and peak_sigma > 1e-6 and peak_dt > 0.0:
                dx0 = float(x) - sx
                dy0 = float(y) - sy
                sigma2p = max(peak_sigma ** 2, 1e-6)
                norm_xy_p = 1.0 / (2.0 * math.pi * sigma2p)
                g_xy_p = norm_xy_p * np.exp(-(dx0 * dx0 + dy0 * dy0) / (2.0 * sigma2p))
                g_z0 = (1.0 / np.sqrt(2.0 * math.pi * base_sigma_z2)) * np.exp(-(dz * dz) / (2.0 * base_sigma_z2))
                c = c + peak_boost * float(self._q_assumed) * g_xy_p * g_z0 * peak_dt * float(scale)

        else:
            # Before leak start (or extremely early), predict ~0 everywhere.
            c = acc

        c = np.clip(c, clamp_min, clamp_max)
        c_pred[valid] = c
        return c_pred

    def update(self, t: float, pose: Dict[str, float], measurement: Dict[str, float]) -> Tuple[Dict[str, Any], float]:
        """Update posterior given measurement. Returns (belief_summary, inference_ms)."""
        t0 = time.perf_counter()

        x_r = float(pose["x"])
        y_r = float(pose["y"])
        z_r = float(pose.get("z", 0.35))
        # use measured output (includes drift/noise)
        y_obs = float(measurement["y_meas"])

        # dt for sensor dynamics propagation
        if self._t_prev is None:
            control_hz = float(self.sim_params.get("control_hz", 10.0))
            dt_step = 1.0 / max(1e-9, control_hz)
        else:
            dt_step = max(0.0, float(t - self._t_prev))
        self._t_prev = float(t)

        sigma_meas = float((self.scenario_spec.get("sensor", {}) or {}).get("sigma", 0.03))
        sigma = math.sqrt(max(1e-9, sigma_meas * sigma_meas + SIGMA_MODEL * SIGMA_MODEL))

        # Predicted raw concentration at current pose for each hypothesis
        c_pred = self._predict_concentration_vec(x=x_r, y=y_r, z=z_r, t=float(t))

        # propagate predicted sensor state S
        valid = np.isfinite(self.grid.positions[:, 0])
        if self.use_delay_model and self._tau > 1e-9:
            alpha = dt_step / self._tau
            alpha = max(0.0, min(1.0, float(alpha)))
            self.S[valid] = (1.0 - alpha) * self.S[valid] + alpha * c_pred[valid]
            self.S[~valid] = 0.0
            y_pred = self.S
        else:
            y_pred = c_pred  # no-delay ablation

        # Gaussian likelihood
        resid = (y_obs - y_pred) / sigma
        loglik = -0.5 * resid * resid - math.log(sigma + 1e-12)

        # Stable update
        m = float(np.max(loglik))
        w = np.exp(loglik - m)
        self.P *= w
        s = float(np.sum(self.P))
        if s <= 0 or (not np.isfinite(s)):
            self.P[:] = 1.0 / self.P.size
        else:
            self.P /= s

        entropy = float(-np.sum(self.P * np.log(self.P + 1e-12)))
        cs_size = self.grid.credible_set_size(self.P, self._p_cut)

        cell_edge = float(self.grid.resolution) * float(self._cell_stride)
        z_bin_height = float(self._z_bin_height)
        cell_volume = (cell_edge ** 2) * z_bin_height
        credible_volume = float(cs_size) * float(cell_volume)

        room_mass = self.grid.room_mass(self.P)

        true_room_mass = None
        try:
            true_rid = (self.scenario_spec.get("leak", {}) or {}).get("room_id", "")
            true_room_mass = float(room_mass.get(true_rid, 0.0))
        except Exception:
            true_room_mass = None

        topk = self.grid.topk(self.P, k=self._topk)
        summary: Dict[str, Any] = {
            "t": float(t),
            "entropy": float(entropy),
            "credible_set_size": int(cs_size),
            "credible_volume": float(credible_volume),
            "topk": topk,
            "room_mass": {k: float(v) for k, v in room_mass.items()},
        }
        if true_room_mass is not None:
            summary["true_room_mass"] = float(true_room_mass)

        t1 = time.perf_counter()
        return summary, float((t1 - t0) * 1000.0)

    def map_estimate(self) -> Tuple[List[float], str]:
        if self.P.size == 0:
            return [0.0, 0.0, 0.0], "unknown"
        idx = int(np.argmax(self.P))
        pos = self.grid.positions[idx].tolist()
        rid = self.grid.room_ids[idx]
        if not all(np.isfinite(pos)):
            return [float("nan"), float("nan"), float("nan")], "no_leak"
        return [float(pos[0]), float(pos[1]), float(pos[2])], str(rid)
