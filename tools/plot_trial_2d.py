# tools/plot_trial_2d.py
# Draw 2D map (free=white, obstacle=black), time-colored robot path, gas intensity overlay (if available),
# leak position, and Tier-1 points (if available).
#
# Usage:
#   py tools/plot_trial_2d.py --trace <trial_dir | trace.csv | run_meta.json> [--out out.png]
# Options:
#   --time_col auto|t|time|step|...
#   --gas_col  auto|gas|conc|ppm|...   (if missing -> no gas overlay)
#   --gas_mode none|scatter|grid       (grid = bin+blur from samples)
#
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec


# -------------------------
# IO helpers
# -------------------------
def _resolve_trace_and_meta(p: str) -> tuple[Path, Path]:
    p = Path(p)
    if p.exists() and p.is_dir():
        trace = p / "trace.csv"
        meta = p / "run_meta.json"
        if not meta.exists():
            meta = p / "scenario_spec.json"
        return trace, meta

    if p.exists() and p.is_file():
        name = p.name.lower()
        if name == "trace.csv":
            trace = p
            meta = p.parent / "run_meta.json"
            if not meta.exists():
                meta = p.parent / "scenario_spec.json"
            return trace, meta
        if name in ("run_meta.json", "scenario_spec.json"):
            return p.parent / "trace.csv", p

        trace = p
        meta = p.parent / "run_meta.json"
        if not meta.exists():
            meta = p.parent / "scenario_spec.json"
        return trace, meta

    raise FileNotFoundError(f"Path not found: {p}")


def _load_scenario_spec(meta_path: Path) -> dict:
    j = json.loads(meta_path.read_text(encoding="utf-8"))
    # run_meta.json format: {"scenario_spec": {...}, ...}
    if isinstance(j, dict) and isinstance(j.get("scenario_spec"), dict):
        return j["scenario_spec"]
    if isinstance(j, dict):
        return j
    raise ValueError(f"Unrecognized JSON format in {meta_path}")


def _read_trace(trace_csv: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read trace.csv into numpy arrays. Returns (data, columns)."""
    rows: list[dict[str, str]] = []
    with trace_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = list(rdr.fieldnames or [])
        for row in rdr:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"trace.csv empty: {trace_csv}")

    data: dict[str, list[float]] = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            v = row.get(c, "")
            try:
                data[c].append(float(v))
            except Exception:
                data[c].append(float("nan"))

    out = {c: np.asarray(v, dtype=float) for c, v in data.items()}
    return out, cols


# -------------------------
# Scene/map helpers
# -------------------------
def _infer_obstacle_mask(occ: np.ndarray) -> np.ndarray:
    """Robust to 0/1 convention: treat minority class as obstacles."""
    a = occ.astype(float)
    mean = float(np.mean(a))
    if mean <= 0.5:
        return a > 0.5
    return a <= 0.5


def _world_to_cell(x: float, y: float, ox: float, oy: float, res: float) -> tuple[int, int]:
    c = int(math.floor((x - ox) / res))
    r = int(math.floor((y - oy) / res))
    return r, c


def _pick_time_col(cols: list[str], preferred: str) -> str | None:
    if preferred and preferred.lower() != "auto":
        return preferred if preferred in cols else None
    for c in ["t", "time", "timestamp", "sim_time", "step", "frame", "tick"]:
        if c in cols:
            return c
    # fuzzy
    for c in cols:
        lc = c.lower()
        if lc in ("t", "time"):
            return c
        if "time" in lc or "stamp" in lc:
            return c
    return None


def _pick_gas_col(cols: list[str], preferred: str) -> str | None:
    if preferred and preferred.lower() != "auto":
        return preferred if preferred in cols else None
    # common exact names
    for c in ["gas", "conc", "concentration", "ppm", "smoke", "gas_ppm", "gas_conc", "c"]:
        if c in cols:
            return c
    # fuzzy names
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["gas", "conc", "ppm", "smoke"]):
            return c
    return None


def _extract_leak_xy(spec: dict) -> tuple[float, float] | None:
    # Prefer spec["leak"]["pos"]
    leak = spec.get("leak")
    if isinstance(leak, dict):
        pos = leak.get("pos") or leak.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                return float(pos[0]), float(pos[1])
            except Exception:
                pass
    # common fallbacks
    for k in ["leak_pos", "leak_position", "source_pos", "source_position"]:
        pos = spec.get(k)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                return float(pos[0]), float(pos[1])
            except Exception:
                pass
    return None


def _as_xy_list(obj: Any) -> list[tuple[float, float]]:
    """Try convert an object into list of (x,y). Accepts [x,y], [[x,y,z],...], dict with positions, etc."""
    out: list[tuple[float, float]] = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        # try common keys
        for k in ["positions", "pos", "points", "locations", "locs", "xy"]:
            if k in obj:
                return _as_xy_list(obj[k])
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return out
        # single point?
        if isinstance(obj[0], (int, float)) and len(obj) >= 2:
            try:
                out.append((float(obj[0]), float(obj[1])))
            except Exception:
                pass
            return out

        # list of points
        for it in obj:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                try:
                    out.append((float(it[0]), float(it[1])))
                except Exception:
                    continue
            elif isinstance(it, dict):
                out.extend(_as_xy_list(it))
        return out

    return out


def _extract_tier1_xy(spec: dict, scene: dict) -> list[tuple[float, float]]:
    # 1) from scene dict
    for k in ["tier1_positions", "tier_1_positions", "tier1_points", "tier1_xy", "tier1_locs", "tier1_locations"]:
        if k in scene:
            pts = _as_xy_list(scene.get(k))
            if pts:
                return pts

    # 2) from spec
    candidates: list[Any] = []
    for k in ["tier1", "tier_1", "tier1_positions", "tier_1_positions", "tier1_points", "tier1_locations", "sensors"]:
        if k in spec:
            candidates.append(spec.get(k))

    # nested patterns
    tiers = spec.get("tiers")
    if isinstance(tiers, dict):
        for kk in ["tier1", "tier_1", "1"]:
            if kk in tiers:
                candidates.append(tiers[kk])

    for obj in candidates:
        pts = _as_xy_list(obj)
        if pts:
            return pts

    return []


# -------------------------
# Gas overlay from samples (optional)
# -------------------------
def _box_filter_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Fast box filter sum using integral image. a: HxW"""
    if r <= 0:
        return a.copy()
    k = 2 * r + 1
    p = np.pad(a, ((r, r), (r, r)), mode="constant", constant_values=0.0)
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    # window sum for each pixel
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _blur_nan_mean(a: np.ndarray, r: int, passes: int = 2) -> np.ndarray:
    """NaN-aware repeated box blur to approximate smooth mean."""
    if passes <= 0 or r <= 0:
        return a
    val = np.nan_to_num(a, nan=0.0)
    w = (~np.isnan(a)).astype(float)
    sum0 = val * w
    w0 = w
    for _ in range(passes):
        sum1 = _box_filter_sum(sum0, r)
        w1 = _box_filter_sum(w0, r)
        mean = np.where(w1 > 0, sum1 / w1, np.nan)
        # keep weights for next pass
        sum0 = np.nan_to_num(mean, nan=0.0) * w1
        w0 = w1
    return np.where(w0 > 0, sum0 / w0, np.nan)


def render_one(trace_input: str, out_path: str = "", time_col: str = "auto", gas_col: str = "auto",
               gas_mode: str = "grid", dpi: int = 200) -> Path:
    trace_csv, meta_json = _resolve_trace_and_meta(trace_input)
    if not trace_csv.exists():
        raise FileNotFoundError(f"Missing trace.csv: {trace_csv}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing run_meta.json / scenario_spec.json: {meta_json}")

    spec = _load_scenario_spec(meta_json)
    scene = build_scene_dict_from_scenario_spec(spec)

    occ = np.array(scene["occupancy"])
    res = float(scene["resolution"])
    ox, oy = float(scene["origin"][0]), float(scene["origin"][1])
    H, W = occ.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    obs = _infer_obstacle_mask(occ)
    free = ~obs

    # base map: free=1(white), obstacle=0(black)
    base = free.astype(float)

    trace, cols = _read_trace(trace_csv)
    if "x" not in trace or "y" not in trace:
        raise RuntimeError(f"trace.csv missing x/y columns: {trace_csv}")

    xs = trace["x"]
    ys = trace["y"]
    n = min(len(xs), len(ys))
    xs = xs[:n]
    ys = ys[:n]

    # time for coloring
    tc = _pick_time_col(cols, time_col)
    if tc is not None and tc in trace:
        ts = trace[tc][:n]
        # if all nan -> fallback
        if np.all(np.isnan(ts)):
            ts = np.arange(n, dtype=float)
    else:
        ts = np.arange(n, dtype=float)

    # leak + tier1
    leak_xy = _extract_leak_xy(spec)
    tier1 = _extract_tier1_xy(spec, scene)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(base, origin="lower", extent=extent, interpolation="nearest", cmap="gray", vmin=0, vmax=1)

    # optional gas overlay
    gc = _pick_gas_col(cols, gas_col)
    if gas_mode.lower() != "none" and gc is not None and gc in trace:
        gv = trace[gc][:n]
        if not np.all(np.isnan(gv)):
            if gas_mode.lower() == "scatter":
                ax.scatter(xs, ys, c=gv, s=6, alpha=0.65, linewidths=0, cmap="inferno")
            else:
                # grid mode: bin samples into cells then blur
                gas_sum = np.zeros((H, W), dtype=float)
                gas_cnt = np.zeros((H, W), dtype=float)
                for x, y, g in zip(xs, ys, gv):
                    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(g)):
                        continue
                    r, c = _world_to_cell(float(x), float(y), ox, oy, res)
                    if 0 <= r < H and 0 <= c < W and bool(free[r, c]):
                        gas_sum[r, c] += float(g)
                        gas_cnt[r, c] += 1.0
                gas_mean = np.where(gas_cnt > 0, gas_sum / np.maximum(gas_cnt, 1e-9), np.nan)
                gas_mean[~free] = np.nan
                gas_smooth = _blur_nan_mean(gas_mean, r=2, passes=2)

                # robust contrast
                finite = gas_smooth[np.isfinite(gas_smooth)]
                if finite.size > 0:
                    vmin = np.percentile(finite, 5)
                    vmax = np.percentile(finite, 95)
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    ax.imshow(gas_smooth, origin="lower", extent=extent, cmap="inferno", alpha=0.55,
                              vmin=vmin, vmax=vmax, interpolation="nearest")

    # time-colored path
    pts = np.stack([xs, ys], axis=1)
    # remove NaN points
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    t2 = ts[ok]
    if len(pts) >= 2:
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        lc = LineCollection(segs, cmap="viridis")
        lc.set_array(t2[:-1])
        lc.set_linewidth(1.4)
        ax.add_collection(lc)

    # markers: start/end
    if len(pts) >= 1:
        ax.scatter([pts[0, 0]], [pts[0, 1]], s=45, marker="o", label="start")
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=55, marker="^", label="end")

    # leak marker
    if leak_xy is not None and all(math.isfinite(v) for v in leak_xy):
        ax.scatter([leak_xy[0]], [leak_xy[1]], s=160, marker="*", label="leak (source)")

    # tier1 marker
    if tier1:
        tx = [p[0] for p in tier1 if math.isfinite(p[0]) and math.isfinite(p[1])]
        ty = [p[1] for p in tier1 if math.isfinite(p[0]) and math.isfinite(p[1])]
        if tx:
            ax.scatter(tx, ty, s=35, marker="s", label="Tier1")

    title = f"{spec.get('scenario_id','?')} | {spec.get('hvac_theta_true', spec.get('hvac',{}).get('theta_true','?'))} | {spec.get('doors_pattern_id', spec.get('doors',{}).get('pattern_id','?'))}"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)

    out = Path(out_path) if out_path else (trace_csv.parent / "debug_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trial_dir OR trace.csv OR run_meta.json")
    ap.add_argument("--out", default="", help="output PNG path (optional)")
    ap.add_argument("--time_col", default="auto", help="auto or specify a column name")
    ap.add_argument("--gas_col", default="auto", help="auto or specify a column name")
    ap.add_argument("--gas_mode", default="grid", choices=["none", "scatter", "grid"])
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    out = render_one(args.trace, args.out, time_col=args.time_col, gas_col=args.gas_col, gas_mode=args.gas_mode, dpi=args.dpi)
    print("Saved:", out)


if __name__ == "__main__":
    main()
