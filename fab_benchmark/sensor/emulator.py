"""
Sensor emulator (v2).

Why v2:
- v1 computed hazard from y_meas (which includes noise), so exposure/risk could be
  triggered by noise spikes.
- v1 used Euler alpha=dt/tau; v2 uses exact discretization alpha=1-exp(-dt/tau)
  for a first-order lag.
- v1 reset() did not restore bias to bias0.

Contract:
We keep original keys and ADD diagnostics. Safe superset.

Returned dict keys:
  y_raw: ground-truth concentration at sensor point (no noise/lag)
  y:     lagged concentration (sensor internal state)
  y_meas: noisy measurement = y + bias + noise
  hazard: stable hazard proxy (noise-free): max(0, y - y_safe)
  hazard_meas: noisy hazard: max(0, y_meas - y_safe)
  hazard_raw: physical hazard from truth: max(0, y_raw - y_safe)
  bias: drift bias
"""
from __future__ import annotations
from typing import Any, Dict
import math
import random


class SensorEmulator:
    def __init__(self, tau: float, sigma: float, drift: Dict[str, Any], y_safe: float = 0.10, seed: int = 0):
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.drift_enabled = int(drift.get("enabled", 0)) == 1
        self.drift_rate = float(drift.get("rate", 0.001))
        self._bias0 = float(drift.get("bias0", 0.0))
        self.bias = float(self._bias0)
        self.y_safe = float(y_safe)
        self._rng = random.Random(int(seed) + 999)
        self._y_lag = 0.0

    def reset(self, seed: int = 0) -> None:
        self._rng = random.Random(int(seed) + 999)
        self._y_lag = 0.0
        self.bias = float(self._bias0)

    def step(self, y_raw: float, dt: float) -> Dict[str, float]:
        y_raw = float(y_raw)
        dt = float(dt)

        # 1st-order lag (exact discretization)
        if self.tau <= 1e-9:
            self._y_lag = y_raw
        else:
            alpha = 1.0 - math.exp(-dt / max(1e-9, self.tau))
            alpha = max(0.0, min(1.0, float(alpha)))
            self._y_lag = (1.0 - alpha) * self._y_lag + alpha * y_raw

        # drift
        if self.drift_enabled:
            self.bias += self.drift_rate * dt

        # noise
        noise = self._rng.gauss(0.0, self.sigma) if self.sigma > 0 else 0.0
        y_meas = float(self._y_lag + self.bias + noise)

        # hazards
        hazard = max(0.0, float(self._y_lag - self.y_safe))          # stable, noise-free
        hazard_meas = max(0.0, float(y_meas - self.y_safe))          # noisy (for debug)
        hazard_raw = max(0.0, float(y_raw - self.y_safe))            # physical truth (for eval)

        return {
            "y_raw": float(y_raw),
            "y": float(self._y_lag),
            "y_meas": float(y_meas),
            "hazard": float(hazard),
            "hazard_meas": float(hazard_meas),
            "hazard_raw": float(hazard_raw),
            "bias": float(self.bias),
        }

    @property
    def y_lag(self) -> float:
        return float(self._y_lag)
