from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Ensure repo root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec


def read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_trace_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    cols: Dict[str, np.ndarray] = {}
    keys = list(rows[0].keys())
    for k in keys:
        out = []
        for r in rows:
            v = r.get(k, "")
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        cols[k] = np.array(out, dtype=np.float64)
    return cols


def accum_heat2d(x: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 180):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x_ok, y_ok, w_ok = x[ok], y[ok], w[ok]
    if len(x_ok) < 5:
        return np.zeros((bins, bins), dtype=np.float32), np.array([0, 1]), np.array([0, 1])
    H, xe, ye = np.histogram2d(x_ok, y_ok, bins=bins, weights=w_ok)
    return H.astype(np.float32), xe, ye


def gas_field_gtA(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    source_pos_xy: tuple[float, float],
    q: float,
    start_time: float,
    theta: str,
    mode_params: Optional[Dict[str, Any]] = None,
    diffusivity: float = 0.25,
    decay: float = 0.0,
) -> np.ndarray:
    """Vectorized version of fab_benchmark.gas.gt_a.GasModelA (approx).
    NOTE: This analytic plume does NOT account for walls/rooms. Use as an intuition overlay.
    """
    mode_params = mode_params or {}
    tau = max(float(t) - float(start_time), 1e-6)
    sigma = math.sqrt(2.0 * diffusivity * tau)

    x0, y0 = float(source_pos_xy[0]), float(source_pos_xy[1])
    dx0 = X - x0
    dy0 = Y - y0

    r = np.hypot(dx0, dy0)
    r_safe = np.maximum(r, 1e-6)

    v_drift = float(mode_params.get("v_drift", 0.30))
    k = float(mode_params.get("k", 0.45))

    vx = np.zeros_like(X, dtype=np.float64)
    vy = np.zeros_like(Y, dtype=np.float64)

    if theta == "drift_pos_x":
        vx[...] = v_drift
    elif theta == "drift_neg_x":
        vx[...] = -v_drift
    elif theta == "drift_pos_y":
        vy[...] = v_drift
    elif theta == "vortex_ccw":
        vx = k * (-dy0) / r_safe
        vy = k * (dx0) / r_safe
    elif theta == "vortex_cw":
        vx = k * (dy0) / r_safe
        vy = k * (-dx0) / r_safe
    elif theta == "time_switch":
        t_switch = float(mode_params.get("t_switch", 30.0))
        if float(t) < t_switch:
            vx[...] = v_drift
        else:
            vx = k * (-dy0) / r_safe
            vy = k * (dx0) / r_safe
    # else: unknown -> no advection

    adv_dx = dx0 - vx * tau
    adv_dy = dy0 - vy * tau

    r2 = adv_dx * adv_dx + adv_dy * adv_dy
    norm = float(q) / (2.0 * math.pi * (sigma * sigma))
    C = norm * np.exp(-r2 / (2.0 * sigma * sigma))
    if decay != 0.0:
        C *= math.exp(-decay * tau)
    return C


def plot_one(
    trial_dir: Path,
    out_png: Path,
    *,
    overlay_mode: str,
    heat_source: str,
    alpha_heat: float,
    gas_alpha: float,
    gas_log: bool,
    gas_t: float,
    time_lw: float,
    time_colorbar: bool,
) -> None:
    meta = read_json(trial_dir / "run_meta.json")
    scenario_spec = meta["scenario_spec"]
    scene = build_scene_dict_from_scenario_spec(scenario_spec)

    # map
    occ = np.array(scene["occupancy"], dtype=np.float32)  # 0 free, 1 wall
    res = float(scene["resolution"])
    ox, oy = float(scene["origin"][0]), float(scene["origin"][1])
    H, W = occ.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    # base map: free=white, wall=black
    free = (occ <= 0.5).astype(np.float32)

    # trace
    trace_path = Path(meta.get("paths", {}).get("trace_csv", str(trial_dir / "trace.csv")))
    trace = read_trace_csv(trace_path)
    if not trace:
        raise RuntimeError(f"Empty trace.csv: {trace_path}")

    # TIME: fix your 't' vs 'time' problem here
    if "t" in trace:
        t = trace["t"]
    elif "time" in trace:
        t = trace["time"]
    else:
        t = np.arange(len(next(iter(trace.values()))), dtype=np.float64)

    x = trace.get("x", None)
    y = trace.get("y", None)
    if x is None or y is None:
        raise RuntimeError(f"trace missing x/y: {trace_path}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(free, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # --- overlays ---
    if overlay_mode in ("evidence", "both"):
        if heat_source == "auto":
            w = trace.get("hazard", trace.get("y_meas", None))
            if w is None:
                w = trace.get("gas_gt", np.zeros_like(t))
        elif heat_source == "hazard":
            w = trace.get("hazard", None)
        elif heat_source == "y_meas":
            w = trace.get("y_meas", None)
        else:
            w = trace.get("gas_gt", None)

        if w is None:
            w = np.zeros_like(t)

        Hh, xe, ye = accum_heat2d(x, y, w, bins=180)
        if Hh.size > 1:
            # histogram2d uses x as first axis, so transpose for imshow
            cmap_h = plt.cm.inferno.copy()
            im_h = ax.imshow(
                Hh.T,
                origin="lower",
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                cmap=cmap_h,
                alpha=float(alpha_heat),
            )
            cbar = fig.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"evidence heat ({heat_source})")

    if overlay_mode in ("model", "both"):
        leak = scenario_spec.get("leak", {}) or {}
        hvac = scenario_spec.get("hvac", {}) or {}
        theta = hvac.get("theta_true", hvac.get("theta", None))
        mode_params = hvac.get("mode_params", {}) or {}

        if theta and isinstance(leak.get("pos", None), list) and len(leak["pos"]) >= 2:
            srcx, srcy = float(leak["pos"][0]), float(leak["pos"][1])
            q = float(leak.get("q", 1.0))
            start_time = float(leak.get("start_time", 0.0))

            if gas_t < 0:
                t_plot = float(np.nanmax(t)) if len(t) else 0.0
            else:
                t_plot = float(gas_t)

            xs = ox + (np.arange(W) + 0.5) * res
            ys = oy + (np.arange(H) + 0.5) * res
            X, Y = np.meshgrid(xs, ys)  # (H,W)

            C = gas_field_gtA(X, Y, t_plot, (srcx, srcy), q, start_time, str(theta), mode_params=mode_params)
            C = np.where(occ <= 0.5, C, np.nan)  # mask walls

            cmap_c = plt.cm.magma.copy()
            cmap_c.set_bad((0, 0, 0, 0))  # NaN transparent on walls

            if gas_log:
                C_plot = np.log10(C + 1e-12)
                label = f"log10 model C (t={t_plot:.1f}s)"
            else:
                C_plot = C
                label = f"model C (t={t_plot:.1f}s)"

            # robust clipping for readability
            vv = C_plot[np.isfinite(C_plot)]
            if vv.size > 20:
                vmin, vmax = np.percentile(vv, [5, 99])
            else:
                vmin, vmax = None, None

            im_c = ax.imshow(
                C_plot,
                origin="lower",
                extent=extent,
                cmap=cmap_c,
                alpha=float(gas_alpha),
                vmin=vmin,
                vmax=vmax,
            )
            cbar = fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.10)
            cbar.set_label(label)

            ax.scatter([srcx], [srcy], marker="*", s=220, edgecolors="k", linewidths=0.5, label="leak")

    # --- time-colored trajectory ---
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x2, y2, t2 = x[ok], y[ok], t[ok]
    if len(x2) >= 2:
        pts = np.column_stack([x2, y2]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, array=t2[:-1], cmap="viridis", linewidths=float(time_lw))
        lc.set_zorder(10)
        ax.add_collection(lc)
        ax.scatter([x2[0]], [y2[0]], s=70, marker="o", zorder=11, label="start")
        ax.scatter([x2[-1]], [y2[-1]], s=70, marker="s", zorder=11, label="end")
        if time_colorbar:
            cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.16)
            cb.set_label("time (s)")

    # title
    scenario_id = scenario_spec.get("scenario_id", trial_dir.name)
    doors_id = (scenario_spec.get("doors", {}) or {}).get("pattern_id", "")
    theta_title = (scenario_spec.get("hvac", {}) or {}).get("theta_true", "")
    ax.set_title(f"{scenario_id} | {theta_title} | {doors_id}")

    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="e.g. results\\exp_e1_lite")
    ap.add_argument("--glob", type=str, required=True, help="e.g. Ours__*")
    ap.add_argument("--out_dir", type=str, required=True, help="all PNGs go here")
    ap.add_argument("--overlay_mode", type=str, default="evidence", choices=["evidence", "model", "both"])
    ap.add_argument("--heat_source", type=str, default="hazard", choices=["hazard", "y_meas", "gas_gt", "auto"])
    ap.add_argument("--alpha_heat", type=float, default=0.35)
    ap.add_argument("--gas_alpha", type=float, default=0.45)
    ap.add_argument("--gas_log", type=int, default=1)
    ap.add_argument("--gas_t", type=float, default=-1.0, help="-1 uses end time")
    ap.add_argument("--time_lw", type=float, default=2.4)
    ap.add_argument("--time_colorbar", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    trial_dirs = sorted([p for p in root.glob(args.glob) if p.is_dir()])
    if not trial_dirs:
        raise SystemExit(f"No trials matched: {root / args.glob}")

    n = len(trial_dirs)
    fail = 0
    for i, td in enumerate(trial_dirs, 1):
        try:
            meta = read_json(td / "run_meta.json")
            spec = meta["scenario_spec"]
            scenario_id = spec.get("scenario_id", "")
            hvac = spec.get("hvac", {}) or {}
            theta = hvac.get("theta_true", hvac.get("theta", ""))
            doors = spec.get("doors", {}) or {}
            doors_id = doors.get("pattern_id", doors.get("id", ""))

            method = meta.get("policy_name", None) or td.name.split("__", 1)[0]
            seed = meta.get("seed", spec.get("seed", ""))

            fname = f"{method}__{scenario_id}__{theta}__{doors_id}__seed{seed}.png".replace(" ", "_")
            out_png = out_dir / fname

            if out_png.exists() and not args.overwrite:
                print(f"[{i}/{n}] SKIP (exists) {td.name}")
                continue

            plot_one(
                td,
                out_png,
                overlay_mode=args.overlay_mode,
                heat_source=args.heat_source,
                alpha_heat=args.alpha_heat,
                gas_alpha=args.gas_alpha,
                gas_log=bool(args.gas_log),
                gas_t=args.gas_t,
                time_lw=args.time_lw,
                time_colorbar=bool(args.time_colorbar),
            )
            print(f"[{i}/{n}] OK {td.name} -> {out_png}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{n}] FAILED {td.name}: {e}")

    print(f"DONE. ok={n-fail}, fail={fail}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
