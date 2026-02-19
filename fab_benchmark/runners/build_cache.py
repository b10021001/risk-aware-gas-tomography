
"""
Build GT-B cache (v1).

CLI contract:
  python -m fab_benchmark.runners.build_cache --scenario f2 --K 20 --t_end 60 --out data/gtb_cache.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np

from fab_benchmark.runners.utils import ensure_dir, write_json


def build_cache(K: int, t_end: float, out_path: str) -> None:
    # Fixed appendix params (high_fidelity)
    dt_gas = 0.25
    D = 0.10
    lambda_decay = 0.01
    source_radius = 0.35
    vortex_strength = 0.55
    time_varying = 1

    z_layers = np.array([0.2, 1.0, 2.0], dtype=np.float32)
    times = np.arange(0.0, float(t_end) + 1e-9, dt_gas, dtype=np.float32)

    # Canonical XY grid around source at origin
    grid_resolution = 0.10
    H = 64
    W = 64
    origin = np.array([-3.2, -3.2], dtype=np.float32)  # covers [-3.2,3.2)
    xs = origin[0] + (np.arange(W, dtype=np.float32) + 0.5) * grid_resolution
    ys = origin[1] + (np.arange(H, dtype=np.float32) + 0.5) * grid_resolution
    X, Y = np.meshgrid(xs, ys)

    C = np.zeros((int(K), int(times.size), int(z_layers.size), int(H), int(W)), dtype=np.float32)

    # Precompute vertical profile factors for each z
    sigma_z2 = (0.55**2)
    g_z = np.exp(-((z_layers - 1.0)**2) / (2.0*sigma_z2)).astype(np.float32)  # center at z=1.0
    g_z = g_z / max(1e-9, float(np.max(g_z)))

    two_pi = 2.0 * math.pi

    for k in range(int(K)):
        phase = (k * 0.37) % two_pi
        for ti, t in enumerate(times.tolist()):
            # time-varying swirl angle
            if time_varying:
                amp = 1.0 + 0.2 * math.sin(two_pi * (t / 20.0) + phase)
            else:
                amp = 1.0
            angle = vortex_strength * amp * t  # radians-ish
            ca = math.cos(-angle)
            sa = math.sin(-angle)
            # backtrace rotation
            Xb = X * ca - Y * sa
            Yb = X * sa + Y * ca

            sigma2 = source_radius**2 + 2.0*D*t
            sigma2 = max(sigma2, 1e-6)
            norm_xy = 1.0 / (2.0*math.pi*sigma2)
            g_xy = norm_xy * np.exp(-(Xb*Xb + Yb*Yb)/(2.0*sigma2))
            decay = math.exp(-lambda_decay*t)
            base = (decay * g_xy * 80.0).astype(np.float32)  # scale

            for zi in range(z_layers.size):
                C[k, ti, zi, :, :] = base * g_z[zi]

    meta = {
        "version": "v1",
        "dt_gas": dt_gas,
        "D": D,
        "lambda_decay": lambda_decay,
        "source_radius": source_radius,
        "vortex_strength": vortex_strength,
        "time_varying": int(time_varying),
        "canonical_source": [0.0, 0.0, 1.0],
    }

    out_p = Path(out_path)
    ensure_dir(out_p.parent)
    np.savez_compressed(
        str(out_p),
        meta=np.array(meta, dtype=object),
        grid_resolution=np.array(grid_resolution, dtype=np.float32),
        origin=origin.astype(np.float32),
        z_layers=z_layers.astype(np.float32),
        times=times.astype(np.float32),
        C=C.astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, type=str, choices=["f1","f2","f3"])
    ap.add_argument("--K", required=True, type=int)
    ap.add_argument("--t_end", required=True, type=float)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    build_cache(K=args.K, t_end=args.t_end, out_path=args.out)


if __name__ == "__main__":
    main()
