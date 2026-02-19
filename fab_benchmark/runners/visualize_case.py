"""Visualize a trial (v1) for paper/demo: 2D topdown map + time-colored trajectory + gas evidence.

This version is designed to be interpretable:
- Occupancy (walls) + doors (open=green, closed=red)
- Trajectory colored by time, with start/end markers
- First detection marker (first hazard>0, else y_meas threshold)
- Leak truth marker (X) if available in scenario_spec
- Optional animation (GIF if imageio installed, otherwise PNG frames)

Usage examples:
  py -m fab_benchmark.runners.visualize_case --trial_dir <dir> --out results/_viz.png
  py -m fab_benchmark.runners.visualize_case --trial_dir <dir> --out results/_viz.png --anim_out results/_viz.gif

Notes:
- "Gas distribution" here is not CFD smoke. We visualize evidence over time by accumulating
  hazard (or y_meas) along visited positions into a 2D heatmap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from fab_benchmark.runners.utils import read_json
from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec


def _read_trace(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols: Dict[str, np.ndarray] = {}
    keys = list(rows[0].keys())
    for k in keys:
        try:
            cols[k] = np.array([float(rr.get(k, "nan")) for rr in rows], dtype=np.float64)
        except Exception:
            cols[k] = np.array([rr.get(k, "") for rr in rows], dtype=object)
    return cols


def _first_detect_idx(trace: Dict[str, np.ndarray], heat_source: str = "hazard") -> Tuple[Optional[int], str]:
    hz = trace.get("hazard", None)
    ym = trace.get("y_meas", None)

    if heat_source == "hazard" and isinstance(hz, np.ndarray):
        idx = np.where(np.isfinite(hz) & (hz > 0))[0]
        if len(idx) > 0:
            return int(idx[0]), "hazard>0"

    if heat_source == "y_meas" and isinstance(ym, np.ndarray):
        thr = np.nanpercentile(ym, 95)
        idx = np.where(np.isfinite(ym) & (ym >= thr))[0]
        if len(idx) > 0:
            return int(idx[0]), f"y_meas>=p95({thr:.3f})"

    if isinstance(hz, np.ndarray):
        idx = np.where(np.isfinite(hz) & (hz > 0))[0]
        if len(idx) > 0:
            return int(idx[0]), "hazard>0"
    if isinstance(ym, np.ndarray):
        thr = np.nanpercentile(ym, 95)
        idx = np.where(np.isfinite(ym) & (ym >= thr))[0]
        if len(idx) > 0:
            return int(idx[0]), f"y_meas>=p95({thr:.3f})"

    return None, "no_detect"


def _world_to_grid(x: np.ndarray, y: np.ndarray, origin_xy: Tuple[float, float], res: float) -> Tuple[np.ndarray, np.ndarray]:
    ox, oy = origin_xy
    j = (x - ox) / res
    i = (y - oy) / res
    return i, j


def _accum_heat2d(x: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 160):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x_ok, y_ok, w_ok = x[ok], y[ok], w[ok]
    if len(x_ok) < 5:
        return np.zeros((bins, bins), dtype=np.float32), np.array([0, 1]), np.array([0, 1])
    H, xe, ye = np.histogram2d(x_ok, y_ok, bins=bins, weights=w_ok)
    return H.astype(np.float32), xe, ye


def _draw_doors_on_map(ax, scenario_spec: Dict[str, Any], origin_xy: Tuple[float, float], res: float) -> None:
    doors = (scenario_spec.get("doors", {}) or {})
    geoms = doors.get("door_geoms", {}) or {}
    states = doors.get("states", {}) or {}
    ox, oy = origin_xy
    for did, g in geoms.items():
        p0 = g.get("p0", None); p1 = g.get("p1", None)
        if not (isinstance(p0, list) and isinstance(p1, list) and len(p0) == 2 and len(p1) == 2):
            continue
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        st = int(states.get(did, 1))
        i0, j0 = (y0 - oy) / res, (x0 - ox) / res
        i1, j1 = (y1 - oy) / res, (x1 - ox) / res
        ax.plot([j0, j1], [i0, i1], linewidth=4.0, color=("lime" if st == 1 else "red"), alpha=0.9)


def _plot_time_colored_traj(ax, j: np.ndarray, i: np.ndarray, t: np.ndarray, lw: float = 4.5) -> None:
    ok = np.isfinite(i) & np.isfinite(j) & np.isfinite(t)
    i2, j2, t2 = i[ok], j[ok], t[ok]
    if len(i2) < 2:
        return
    pts = np.column_stack([j2, i2]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t2[:-1], cmap="viridis", linewidths=lw)
    lc.set_zorder(10)
    ax.add_collection(lc)
    ax.scatter([j2[0]], [i2[0]], s=70, marker="o", zorder=11, label="start")
    ax.scatter([j2[-1]], [i2[-1]], s=70, marker="s", zorder=11, label="end")
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time (s)")


def make_static(trial_dir: Path, out_png: Path, heat_source: str = "hazard", alpha_heat: float = 0.35) -> None:
    meta = read_json(trial_dir / "run_meta.json")
    scenario_spec = meta["scenario_spec"]
    scene = build_scene_dict_from_scenario_spec(scenario_spec)

    trace = _read_trace(Path(meta["paths"]["trace_csv"]))
    if not trace:
        raise SystemExit("Empty trace.csv")

    occ = scene["occupancy"]
    res = float(scene["resolution"])
    origin = tuple(scene["origin"])

    t = trace["time"]; x = trace["x"]; y = trace["y"]

    if heat_source == "auto":
        w = trace.get("hazard", trace.get("y_meas", None))
    elif heat_source == "hazard":
        w = trace.get("hazard", None)
    else:
        w = trace.get("y_meas", None)
    if w is None:
        w = np.zeros_like(t)

    fig = plt.figure(figsize=(13, 8))
    ax_map = fig.add_subplot(2, 2, 1)
    ax_ent = fig.add_subplot(2, 2, 2)
    ax_expo = fig.add_subplot(2, 2, 3)
    ax_y = fig.add_subplot(2, 2, 4)

    ax_map.imshow(occ, origin="lower", cmap="gray_r")
    ax_map.set_title("Topdown: walls/doors + trajectory + gas evidence")
    ax_map.set_xlabel("grid j")
    ax_map.set_ylabel("grid i")

    _draw_doors_on_map(ax_map, scenario_spec, origin, res)

    Hh, xe, ye = _accum_heat2d(x, y, w, bins=180)
    if Hh.size > 1:
        extent = [
            (xe[0] - origin[0]) / res, (xe[-1] - origin[0]) / res,
            (ye[0] - origin[1]) / res, (ye[-1] - origin[1]) / res,
        ]
        ax_map.imshow(Hh.T, origin="lower", extent=extent, cmap="inferno", alpha=float(alpha_heat))

    i, j = _world_to_grid(x, y, origin, res)
    _plot_time_colored_traj(ax_map, j, i, t, lw=4.5)

    idx, rule = _first_detect_idx(trace, heat_source="hazard")
    if idx is not None:
        ax_map.scatter([j[idx]], [i[idx]], s=140, marker="*", zorder=12, label=f"first detect ({rule})")

    leak = (scenario_spec.get("leak", {}) or {})
    if isinstance(leak.get("pos", None), list) and len(leak["pos"]) >= 2:
        lx, ly = float(leak["pos"][0]), float(leak["pos"][1])
        li, lj = _world_to_grid(np.array([lx]), np.array([ly]), origin, res)
        ax_map.scatter([lj[0]], [li[0]], s=120, marker="X", zorder=12, label="leak (truth)")

    ax_map.legend(loc="upper right")

    ent = trace.get("entropy", None)
    expo = trace.get("exposure_integral", None)
    y_meas = trace.get("y_meas", None)
    hz = trace.get("hazard", None)

    if isinstance(ent, np.ndarray):
        ax_ent.plot(t, ent)
        ax_ent.set_title("Entropy")
        ax_ent.set_xlabel("time (s)")

    if isinstance(expo, np.ndarray):
        ax_expo.plot(t, expo)
        ax_expo.set_title("Exposure Integral")
        ax_expo.set_xlabel("time (s)")

    if isinstance(y_meas, np.ndarray):
        ax_y.plot(t, y_meas, label="y_meas")
        if isinstance(hz, np.ndarray):
            ax_y.plot(t, hz, label="hazard")
        ax_y.set_title("Gas signal")
        ax_y.set_xlabel("time (s)")
        ax_y.legend(loc="upper right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str, help="Output PNG path")
    ap.add_argument("--heat_source", default="hazard", choices=["hazard", "y_meas", "auto"])
    ap.add_argument("--alpha", default=0.35, type=float)
    args = ap.parse_args()
    make_static(Path(args.trial_dir), Path(args.out), heat_source=args.heat_source, alpha_heat=args.alpha)


if __name__ == "__main__":
    main()
