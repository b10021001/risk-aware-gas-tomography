# -*- coding: utf-8 -*-
"""
Enhanced 2D plotting for FAB gas benchmark trials.

Features:
- Occupancy map: free=white, obstacle=black
- Optional gas concentration overlay (GT_A analytic model)
- Robot path colored by time
- Batch export: dump many run folders to a single output directory
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

# Headless-safe backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _repo_root_from_this_file() -> Path:
    # tools/plot_trial_2d_enhanced.py -> repo root is one level up from tools/
    return Path(__file__).resolve().parents[1]


# Ensure we can import fab_benchmark when the script is run from anywhere
_REPO_ROOT = _repo_root_from_this_file()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def safe_filename(s: str, max_len: int = 180) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)         # Windows-illegal
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_meta(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (meta, scenario_spec).

    Supported layouts:
    - run_dir/run_meta.json with meta["scenario_spec"]
    - run_dir/scenario_spec.json
    """
    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        meta = load_json(run_meta_path)
        scenario_spec = meta.get("scenario_spec", None)
        if scenario_spec is None:
            scenario_spec = meta.get("scenario", None)
        if scenario_spec is None:
            ss_path = run_dir / "scenario_spec.json"
            if ss_path.exists():
                scenario_spec = load_json(ss_path)
            else:
                raise FileNotFoundError(
                    f"Found {run_meta_path} but no scenario_spec inside, and no scenario_spec.json."
                )
        return meta, scenario_spec

    ss_path = run_dir / "scenario_spec.json"
    if ss_path.exists():
        meta = {}
        scenario_spec = load_json(ss_path)
        return meta, scenario_spec

    raise FileNotFoundError(f"Neither run_meta.json nor scenario_spec.json exists in: {run_dir}")


def load_trace_xy_time(trace_csv: Path):
    """
    Load time_s, x_m, y_m from trace.csv.
    Works without pandas.
    """
    data = np.genfromtxt(
        trace_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        invalid_raise=False,
    )
    if data.size == 0:
        raise ValueError(f"trace.csv seems empty: {trace_csv}")

    def col(name: str) -> np.ndarray:
        if data.ndim == 0:
            return np.asarray([data[name]], dtype=float)
        return np.asarray(data[name], dtype=float)

    t = col("time_s") if "time_s" in data.dtype.names else col("t")
    x = col("x_m") if "x_m" in data.dtype.names else col("x")
    y = col("y_m") if "y_m" in data.dtype.names else col("y")
    return t, x, y


def flow_velocity_vec(X: np.ndarray, Y: np.ndarray, t: float, theta: str, params: Dict[str, Any]):
    """
    Vectorized version of fab_benchmark.gas.hvac_modes.flow_velocity.
    """
    drift_speed = float(params.get("drift_speed", 0.25))
    vortex_strength = float(params.get("vortex_strength", 0.4))
    time_switch_t = float(params.get("time_switch_t", 30.0))

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if theta == "drift_pos_x":
        return drift_speed * np.ones_like(X), np.zeros_like(X)
    if theta == "drift_neg_x":
        return -drift_speed * np.ones_like(X), np.zeros_like(X)
    if theta == "drift_pos_y":
        return np.zeros_like(X), drift_speed * np.ones_like(X)
    if theta == "drift_neg_y":
        return np.zeros_like(X), -drift_speed * np.ones_like(X)

    if theta in ("vortex_ccw", "vortex_cw"):
        r = np.sqrt(X**2 + Y**2) + 1e-6
        k = vortex_strength * (1.0 if theta == "vortex_ccw" else -1.0)
        vx = -k * Y / r
        vy = k * X / r
        return vx, vy

    if theta == "time_switch":
        if float(t) < time_switch_t:
            return drift_speed * np.ones_like(X), np.zeros_like(X)
        r = np.sqrt(X**2 + Y**2) + 1e-6
        vx = -vortex_strength * Y / r
        vy = vortex_strength * X / r
        return vx, vy

    return np.zeros_like(X), np.zeros_like(X)


def compute_gas_field_on_grid(scenario_spec: Dict[str, Any], occ: np.ndarray, t: float, z_slice: float, log1p: bool):
    """
    Compute a 2D gas concentration field (H,W) using GT_A analytic model, mask obstacles as NaN.
    """
    leak = scenario_spec.get("leak", {})
    hvac = scenario_spec.get("hvac", {})
    if not leak.get("enabled", True):
        return np.where(occ == 1, np.nan, 0.0)

    t0 = float(leak.get("start_time", 0.0))
    if float(t) <= t0:
        return np.where(occ == 1, np.nan, 0.0)

    dt = float(t) - t0
    q = float(leak.get("q", 1.0))

    x0y0 = leak.get("xy", leak.get("pos_xy", (0.0, 0.0)))
    x0 = float(x0y0[0])
    y0 = float(x0y0[1])
    z0 = float(leak.get("z", 0.0))
    z_decay = float(leak.get("z_decay", 2.0))

    mode_params = hvac.get("mode_params", {})
    loss_rate = float(mode_params.get("loss_rate", 0.012))
    diffusion_rate = float(mode_params.get("diffusion_rate", 0.45))
    theta_true = hvac.get("theta_true", hvac.get("theta_obs", "drift_pos_x"))

    grid = scenario_spec.get("map", {})
    resolution = float(grid.get("resolution", 0.2))
    origin_xy = grid.get("origin_xy", [0.0, 0.0])
    ox = float(origin_xy[0])
    oy = float(origin_xy[1])

    H, W = occ.shape
    xs = ox + (np.arange(W) + 0.5) * resolution
    ys = oy + (np.arange(H) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)  # (H,W)

    vx, vy = flow_velocity_vec(X, Y, t, theta_true, mode_params)
    Xb = X - vx * dt
    Yb = Y - vy * dt

    dx = Xb - x0
    dy = Yb - y0

    sigma2 = 2.0 * diffusion_rate * dt + 1e-6
    r2 = dx * dx + dy * dy
    survival = math.exp(-loss_rate * dt)

    C = q * survival * np.exp(-0.5 * r2 / sigma2) / (2.0 * math.pi * sigma2)
    C *= np.exp(-abs(float(z_slice) - z0) / z_decay)

    C = np.where(occ == 1, np.nan, C)
    if log1p:
        C = np.log1p(C)
    return C


def reconstruct_occ_from_spec(scenario_spec: Dict[str, Any]):
    from fab_benchmark.scenarios.base_scenario import reconstruct_layout_from_spec

    occ, _rooms, _doors, topo = reconstruct_layout_from_spec(scenario_spec)

    grid = scenario_spec.get("map", {})
    res = float(grid.get("resolution", topo.get("resolution", 0.2)))
    origin_xy = grid.get("origin_xy", topo.get("origin_xy", [0.0, 0.0]))
    ox = float(origin_xy[0])
    oy = float(origin_xy[1])

    H, W = occ.shape
    xmin = ox
    xmax = ox + W * res
    ymin = oy
    ymax = oy + H * res
    return occ, (xmin, xmax, ymin, ymax)


def plot_one_run(
    run_dir: Path,
    out_path: Path,
    *,
    z_slice: float,
    t_field: str,
    show_gas: bool,
    gas_log1p: bool,
    path_max_points: int,
    dpi: int,
):
    meta, scenario_spec = load_run_meta(run_dir)

    trace_csv = run_dir / "trace.csv"
    if not trace_csv.exists():
        raise FileNotFoundError(f"Missing trace.csv in {run_dir}")

    t, x, y = load_trace_xy_time(trace_csv)

    if len(t) > path_max_points:
        idx = np.linspace(0, len(t) - 1, path_max_points).astype(int)
        t = t[idx]
        x = x[idx]
        y = y[idx]

    if t_field == "final":
        t_snap = float(t[-1])
    elif t_field == "mid":
        t_snap = float(t[len(t) // 2])
    else:
        t_snap = float(t_field)

    occ, extent = reconstruct_occ_from_spec(scenario_spec)
    xmin, xmax, ymin, ymax = extent

    free_img = (occ == 0).astype(float)  # free=1 white, obstacle=0 black, with cmap="gray"

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(
        free_img,
        cmap="gray",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    if show_gas:
        gas = compute_gas_field_on_grid(scenario_spec, occ, t_snap, z_slice=z_slice, log1p=gas_log1p)
        im = ax.imshow(
            gas,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            interpolation="nearest",
            alpha=0.55,
            cmap="inferno",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("gas (log1p)" if gas_log1p else "gas")

    pts = np.column_stack([x, y]).astype(float)
    if len(pts) >= 2:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", linewidths=1.6, alpha=0.95)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        cbar2 = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.08)
        cbar2.set_label("time (s)")

    ax.scatter([x[0]], [y[0]], s=40, marker="o", label="start")

    leak = scenario_spec.get("leak", {})
    if leak.get("enabled", True) and ("xy" in leak or "pos_xy" in leak):
        x0y0 = leak.get("xy", leak.get("pos_xy"))
        ax.scatter([float(x0y0[0])], [float(x0y0[1])], s=140, marker="*", label="leak")

    scenario_id = scenario_spec.get("scenario_id", "")
    doors_id = scenario_spec.get("doors", {}).get("pattern_id", "")
    hvac_theta = scenario_spec.get("hvac", {}).get("theta_true", "")
    policy = meta.get("policy_name", "")
    parts = [p for p in [policy, scenario_id, hvac_theta, doors_id] if p]
    ax.set_title(" | ".join(parts) if parts else run_dir.name)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def iter_run_dirs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted([p for p in root.glob(pattern) if p.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_dir", type=str, help="Single run directory containing trace.csv and run_meta.json")
    g.add_argument("--root", type=str, help="Root directory (e.g., results/exp_e1_lite) to batch-plot")

    ap.add_argument("--glob", type=str, default="Ours__*", help="Glob under --root (default: Ours__*)")
    ap.add_argument("--out", type=str, default="", help="Output PNG path for --run_dir mode")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder for --root mode (all PNGs go here)")

    ap.add_argument("--t_field", type=str, default="final", help="Gas snapshot time: final | mid | <float seconds>")
    ap.add_argument("--z_slice", type=float, default=0.3, help="Z slice for gas overlay (meters)")
    ap.add_argument("--no_gas", action="store_true", help="Disable gas overlay")
    ap.add_argument("--gas_linear", action="store_true", help="Use linear scale (default is log1p)")
    ap.add_argument("--path_max_points", type=int, default=5000, help="Downsample trace to at most N points")
    ap.add_argument("--dpi", type=int, default=180)

    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        out_path = Path(args.out) if args.out else (run_dir / "debug_map_enhanced.png")
        plot_one_run(
            run_dir,
            out_path,
            z_slice=args.z_slice,
            t_field=args.t_field,
            show_gas=(not args.no_gas),
            gas_log1p=(not args.gas_linear),
            path_max_points=args.path_max_points,
            dpi=args.dpi,
        )
        print(f"Saved: {out_path}")
        return

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "_maps_enhanced")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list(iter_run_dirs(root, args.glob))
    if not run_dirs:
        raise RuntimeError(f"No run dirs matched: {root} / {args.glob}")

    for k, run_dir in enumerate(run_dirs, start=1):
        try:
            meta, scenario_spec = load_run_meta(run_dir)
            scenario_id = scenario_spec.get("scenario_id", run_dir.name)
            doors_id = scenario_spec.get("doors", {}).get("pattern_id", "doors")
            hvac_theta = scenario_spec.get("hvac", {}).get("theta_true", "hvac")
            seed = scenario_spec.get("seed", meta.get("seed", ""))
            trial_id = meta.get("trial_id", "")
            policy = meta.get("policy_name", "")

            fname = safe_filename("__".join([p for p in [policy, scenario_id, hvac_theta, doors_id, str(trial_id), f"seed{seed}"] if p])) + ".png"
            out_path = out_dir / fname

            plot_one_run(
                run_dir,
                out_path,
                z_slice=args.z_slice,
                t_field=args.t_field,
                show_gas=(not args.no_gas),
                gas_log1p=(not args.gas_linear),
                path_max_points=args.path_max_points,
                dpi=args.dpi,
            )
            print(f"[{k}/{len(run_dirs)}] {run_dir.name} -> {out_path.name}")
        except Exception as e:
            print(f"[{k}/{len(run_dirs)}] FAILED {run_dir.name}: {e}")

    print(f"DONE. Output folder: {out_dir}")


if __name__ == "__main__":
    main()
