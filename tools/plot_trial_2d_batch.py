"""Batch 2D plotting for FAB gas benchmark trials.

What it does
- Scans a results folder for trial subfolders (e.g. Ours__*, Coverage__*, ...)
- For each trial, draws:
  * Occupancy map: free=white, obstacle=black
  * (Optional) Gas concentration heatmap overlay + gas colorbar
  * (Optional) Robot path, optionally colored by time + time colorbar
  * Markers: start, leak, tier1
- Saves ALL images into ONE output folder.

This file is meant to be dropped into:
  C:\isaac_3\fab_gas_demo\tools\plot_trial_2d_batch.py

Example (Windows CMD):
  py tools\plot_trial_2d_batch.py ^
    --results_root results\exp_e1_lite ^
    --pattern "Ours__*" ^
    --out_dir results\exp_e1_lite\maps_2d_ours ^
    --show_gas 1 --gas_log 1 --gas_time_frac 0.5 ^
    --path_time_color 1 --colorbar 1 ^
    --dpi 200 --overwrite 1

Notes
- scenario_spec is read from run_meta.json (preferred) or scenario_spec.json if present.
- Gas model:
    * If data/gtb_cache.npz exists: uses GasModelBCache (fast, spatially-varying)
    * Else falls back to GasModelA (analytic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except Exception:
    make_axes_locatable = None


# --------------------------
# Repo imports (fab_benchmark)
# --------------------------
THIS_FILE = Path(__file__).resolve()


def _find_repo_root() -> Path:
    """Find the repo root that contains the 'fab_benchmark' package.

    This makes the script robust even if you run it from outside the repo
    directory, as long as the script itself lives under that repo.
    """

    # Typical layout: <repo>/tools/plot_trial_2d_batch.py
    # Start searching from the directory containing this file.
    start = THIS_FILE.parent
    for p in [start, *start.parents]:
        if (p / "fab_benchmark").is_dir():
            return p

    # Fallback: maybe user copied the script somewhere else but runs it while
    # being inside the repo.
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "fab_benchmark").is_dir():
            return p

    # Last resort: keep old behavior (may still fail later with a clear error).
    return THIS_FILE.parents[1] if len(THIS_FILE.parents) > 1 else THIS_FILE.parent


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec
except Exception as e:
    raise ImportError(
        "Cannot import fab_benchmark. Run this script from inside the fab_gas_demo repo, "
        "or make sure it is on PYTHONPATH.\n" + str(e)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_scenario_spec(trial_dir: Path) -> Dict[str, Any]:
    """Get scenario_spec from run_meta.json or scenario_spec.json."""
    run_meta = trial_dir / "run_meta.json"
    if run_meta.exists():
        d = _read_json(run_meta)
        if isinstance(d, dict) and "scenario_spec" in d and isinstance(d["scenario_spec"], dict):
            return d["scenario_spec"]

    scenario_spec = trial_dir / "scenario_spec.json"
    if scenario_spec.exists():
        return _read_json(scenario_spec)

    raise FileNotFoundError(f"No run_meta.json (with scenario_spec) or scenario_spec.json in {trial_dir}")


def _load_trace_csv(trial_dir: Path) -> pd.DataFrame:
    p = trial_dir / "trace.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing trace.csv in {trial_dir}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise ValueError(f"empty trace.csv in {trial_dir}")
    return df



def _load_summary_json(trial_dir: Path) -> Dict[str, Any]:
    p = trial_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}



def _trial_key_from_metrics_row(row: Dict[str, Any]) -> Optional[str]:
    """Derive trial folder name key from an aggregate metrics row.

    We prefer 'trace_path' because it is stable even if other IDs change.
    """
    tp = row.get("trace_path")
    if isinstance(tp, str) and tp.strip():
        # trace_path in metrics.csv uses backslashes; normalize.
        tp_norm = tp.replace('/', os.sep).replace('\\\\', os.sep).replace('\\', os.sep)
        try:
            return Path(tp_norm).parent.name
        except Exception:
            return None
    # Fallback: some metrics may include a 'trial_dir' field in other versions.
    td = row.get("trial_dir") or row.get("trial_name")
    if isinstance(td, str) and td.strip():
        return Path(td).name
    return None


def _load_metrics_index(metrics_csv: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load aggregate metrics.csv into a dict keyed by trial folder name.

    Returns empty dict if metrics_csv is None or not found.
    """
    if metrics_csv is None:
        return {}
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return {}

    try:
        mdf = pd.read_csv(metrics_csv)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in mdf.iterrows():
        row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.to_dict().items()}
        key = _trial_key_from_metrics_row(row)
        if key:
            out[key] = row
    return out


def _fmt_num(v: Any, fmt: str = "{:.2f}") -> str:
    try:
        if v is None:
            return "na"
        if isinstance(v, str) and not v.strip():
            return "na"
        return fmt.format(float(v))
    except Exception:
        return "na"


def _to_float(v: Any) -> Optional[float]:
    """Safe float conversion. Returns None if v is None/empty/NaN or not convertible."""
    try:
        if v is None:
            return None
        if isinstance(v, float):
            if np.isnan(v):
                return None
            return float(v)
        if isinstance(v, str):
            if not v.strip():
                return None
            return float(v)
        return float(v)
    except Exception:
        return None


def _first_nonnull(series: pd.Series) -> Any:
    for v in series.tolist():
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return v
    return None

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_xy_t(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xcol = _pick_col(df, ["x", "px", "robot_x", "pos_x"])
    ycol = _pick_col(df, ["y", "py", "robot_y", "pos_y"])
    if xcol is None or ycol is None:
        raise KeyError(f"trace.csv missing position columns. Have: {list(df.columns)}")

    tcol = _pick_col(df, ["t", "time", "sim_time", "timestamp"])  # some files use time
    if tcol is None:
        # fallback: use row index
        t = np.arange(len(df), dtype=np.float32)
    else:
        t = df[tcol].to_numpy(dtype=np.float32)

    x = df[xcol].to_numpy(dtype=np.float32)
    y = df[ycol].to_numpy(dtype=np.float32)

    # ensure finite
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if not np.any(m):
        raise ValueError("trace has no finite (x,y,t)")
    return x[m], y[m], t[m]


def _make_gas_model(prefer: str = "auto"):
    """Return a gas model instance.

    prefer:
      - "auto": try BCACHE then A
      - "bcache": force BCACHE (error if unavailable)
      - "a": force GasModelA
    """

    prefer = (prefer or "auto").lower()

    def try_bcache() -> Optional[object]:
        try:
            from fab_benchmark.gas.gt_b_cache import GasModelBCache
        except Exception:
            return None

        cache_path = REPO_ROOT / "data" / "gtb_cache.npz"
        if not cache_path.exists():
            return None

        try:
            return GasModelBCache(cache_path=str(cache_path), map_name="F2", resolution=0.5)
        except Exception:
            return None

    def make_a() -> object:
        from fab_benchmark.gas.gt_a import GasModelA
        # GasModelA expects a dict of HVAC mode parameters; it has sensible defaults
        # for missing keys, so an empty dict works fine for visualization.
        return GasModelA(mode_params={})

    if prefer in ("auto", "grid", "bcache", "b", "gtb"):
        gm = try_bcache()
        if gm is not None:
            return gm
        if prefer in ("bcache", "b", "gtb", "grid"):
            raise FileNotFoundError(
                "GasModelBCache requested but data/gtb_cache.npz not found or cannot be loaded. "
                "Either place gtb_cache.npz under data/ or use --gas_model a"
            )
        return make_a()

    if prefer in ("a", "analytic", "gta"):
        return make_a()

    # unknown: fallback to auto
    gm = try_bcache()
    return gm if gm is not None else make_a()


def _gas_query_generic(
    query_fn,
    *,
    hvac_theta: str,
    source_pos: Tuple[float, float, float],
    q: float,
    start_time: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Call a gas model query() across arrays, handling different signatures.

    Supports:
      - GasModelA.query(theta, source_pos, q, start_time, x, y, z, t)
      - GasModelBCache.query(x, y, z, t, source_pos, k=0)
    """

    import inspect

    sig = inspect.signature(query_fn)
    params = list(sig.parameters.values())
    names = [p.name for p in params]

    # Case 1: explicit theta/source_pos/q/start_time
    if "theta" in names or "hvac_theta" in names:
        # Prefer keyword call; if it fails, fallback positional.
        kwargs = {}
        if "theta" in names:
            kwargs["theta"] = hvac_theta
        if "hvac_theta" in names:
            kwargs["hvac_theta"] = hvac_theta
        if "source_pos" in names:
            kwargs["source_pos"] = source_pos
        if "q" in names:
            kwargs["q"] = float(q)
        if "start_time" in names:
            kwargs["start_time"] = float(start_time)

        # x/y/z/t
        if "x" in names:
            kwargs["x"] = x
        if "y" in names:
            kwargs["y"] = y
        if "z" in names:
            kwargs["z"] = z
        if "t" in names:
            kwargs["t"] = t

        return np.asarray(query_fn(**kwargs), dtype=np.float32)

    # Case 2: GasModelBCache-like signature: (x,y,z,t, source_pos, k=0)
    if ("x" in names) and ("y" in names) and ("z" in names) and ("t" in names) and ("source_pos" in names):
        kwargs = {"x": x, "y": y, "z": z, "t": t, "source_pos": source_pos}
        if "k" in names:
            kwargs["k"] = 0
        return np.asarray(query_fn(**kwargs), dtype=np.float32)

    # Fallback: try positional with common orders.
    # Try: (x,y,z,t)
    try:
        return np.asarray(query_fn(x, y, z, t), dtype=np.float32)
    except Exception:
        pass

    # Try: (x,y,z,t, source_pos)
    try:
        return np.asarray(query_fn(x, y, z, t, source_pos), dtype=np.float32)
    except Exception as e:
        raise TypeError(f"Unsupported gas model query signature: {sig}") from e


def _gas_model_a_grid(
    gas_model: object,
    *,
    theta: str,
    source_pos: Tuple[float, float, float],
    q: float,
    start_time: float,
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
) -> np.ndarray:
    """Vectorized re-implementation of fab_benchmark.gas.gt_a.GasModelA.query.

    The repo's GasModelA.query() is scalar-only (it uses Python 'if' and built-in
    max()), so calling it on arrays triggers "truth value of an array is
    ambiguous". For visualization we want a fast grid evaluation, so we mirror its
    math here.
    """

    sx, sy, sz = map(float, source_pos)
    t = float(t)

    if t < float(start_time):
        return np.zeros_like(X, dtype=np.float32)

    # Defaults match the class implementation.
    mode_params = getattr(gas_model, "mode_params", {}) or {}
    drift_speed = float(mode_params.get("drift_speed", 0.5))
    vortex_strength = float(mode_params.get("vortex_strength", 0.35))
    time_switch_t = float(mode_params.get("time_switch_t", 60.0))
    D = 0.1

    dt = max(t - float(start_time), 1e-9)

    r2 = (X - sx) ** 2 + (Y - sy) ** 2 + (0.0 - sz) ** 2
    base = (q / (4.0 * np.pi * D * dt)) * np.exp(-r2 / (4.0 * D * dt))
    C = base

    if theta == "drift_pos_x":
        C = C + drift_speed * (X - sx) / (dt * 10.0)
    elif theta == "drift_pos_y":
        C = C + drift_speed * (Y - sy) / (dt * 10.0)
    elif theta == "vortex_ccw":
        C = C + vortex_strength * ((X - sx) * (Y - sy)) / (dt * 50.0)
    elif theta == "vortex_cw":
        C = C - vortex_strength * ((X - sx) * (Y - sy)) / (dt * 50.0)
    elif theta == "time_switch":
        if t < time_switch_t:
            C = C + drift_speed * (X - sx) / (dt * 10.0)
        else:
            C = C + drift_speed * (Y - sy) / (dt * 10.0)

    return np.maximum(C, 0.0).astype(np.float32)



def _backtrace_xy_vec(x: np.ndarray, y: np.ndarray, dt: float, theta: str, t: float, mode_params: Dict[str, Any]):
    """
    Vectorized backtrace of (x,y) over dt seconds following the same logic as
    fab_benchmark.gas.hvac_modes.backtrace_xy.

    NOTE:
      - x, y are numpy arrays of the same shape.
      - dt is scalar (float).
      - theta is one of drift_* / vortex_* / time_switch.
    """
    dt = float(dt)
    if dt == 0.0:
        return x, y

    theta = str(theta)
    mode_params = mode_params or {}

    if theta in ("drift_pos_x", "drift_neg_x", "drift_pos_y", "drift_neg_y"):
        drift_speed = float(mode_params.get("drift_speed", 1.0))
        if theta == "drift_pos_x":
            vx, vy = +drift_speed, 0.0
        elif theta == "drift_neg_x":
            vx, vy = -drift_speed, 0.0
        elif theta == "drift_pos_y":
            vx, vy = 0.0, +drift_speed
        else:
            vx, vy = 0.0, -drift_speed
        # backtrace: x <- x - v*dt
        return x - vx * dt, y - vy * dt

    if theta in ("vortex_ccw", "vortex_cw"):
        omega = float(mode_params.get("vortex_omega", 0.2))
        cx, cy = mode_params.get("vortex_center", (0.0, 0.0))
        cx, cy = float(cx), float(cy)

        steps = int(max(2, min(20, abs(dt) * 4)))
        dt_step = dt / float(steps)

        xb = x.copy()
        yb = y.copy()
        for _ in range(steps):
            dx = xb - cx
            dy = yb - cy
            if theta == "vortex_ccw":
                vx = -omega * dy
                vy = +omega * dx
            else:
                vx = +omega * dy
                vy = -omega * dx
            xb = xb - vx * dt_step
            yb = yb - vy * dt_step
        return xb, yb

    if theta == "time_switch":
        # Same as hvac_modes.time_switch: drift_pos_x then vortex_ccw after t_switch
        t_switch = float(mode_params.get("time_switch_t", 20.0))
        if float(t) <= t_switch:
            return _backtrace_xy_vec(x, y, dt, "drift_pos_x", t, mode_params)

        dt_after = min(dt, max(0.0, float(t) - t_switch))
        dt_before = dt - dt_after
        xb, yb = _backtrace_xy_vec(x, y, dt_after, "vortex_ccw", t, mode_params)
        xb, yb = _backtrace_xy_vec(xb, yb, dt_before, "drift_pos_x", t_switch, mode_params)
        return xb, yb

    # Unknown / no-flow fallback
    return x, y


def _topology_factor_grid(gas_model: Any, X: np.ndarray, Y: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Vectorized room/door attenuation factor used by GasModelA (gt_a).

    Returns an array same shape as X/Y.
    """
    topo = getattr(gas_model, "topology", None)
    if not topo:
        return np.ones_like(X, dtype=float)

    room_idx = getattr(gas_model, "_room_idx", None)
    room_bboxes = getattr(gas_model, "_room_bboxes", None)
    room_centers = getattr(gas_model, "_room_centers", None)
    room_dist = getattr(gas_model, "_room_dist", None)
    room_hops = getattr(gas_model, "_room_hops", None)
    room_of = getattr(gas_model, "_room_of", None)
    nearest_room = getattr(gas_model, "_nearest_room", None)

    if not (room_idx and room_bboxes and room_centers and room_dist and room_hops and room_of and nearest_room):
        return np.ones_like(X, dtype=float)

    rid_s = room_of(float(sx), float(sy))
    if rid_s is None:
        rid_s = nearest_room(float(sx), float(sy))
    if rid_s is None or rid_s not in room_idx:
        return np.ones_like(X, dtype=float)

    # Use the same indexing as GasModelA._room_idx (built from sorted room ids)
    rids_sorted = sorted(room_idx.keys())
    centers = np.array([room_centers[rid] for rid in rids_sorted], dtype=float)  # (N,2)
    N = centers.shape[0]

    rid_grid = np.full(X.shape, -1, dtype=np.int32)

    # Assign by bbox when possible (fast + matches _room_of for most cells)
    for rid in rids_sorted:
        bbox = room_bboxes.get(rid, None)
        if bbox is None:
            continue
        x0, x1, y0, y1 = bbox
        mask = (rid_grid < 0) & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        rid_grid[mask] = int(room_idx[rid])

    # For corridor / unassigned cells: nearest room center (matches _nearest_room)
    if np.any(rid_grid < 0):
        cx = centers[:, 0]
        cy = centers[:, 1]
        # broadcast: (H,W,1) - (1,1,N) -> (H,W,N)
        dx = X[..., None] - cx[None, None, :]
        dy = Y[..., None] - cy[None, None, :]
        dist2 = dx * dx + dy * dy
        nn = np.argmin(dist2, axis=-1).astype(np.int32)
        # nn gives index within rids_sorted; convert to GasModelA room_idx value
        # room_idx values are also 0..N-1 aligned with rids_sorted, so we can use nn directly.
        rid_grid[rid_grid < 0] = nn[rid_grid < 0]

    # dist/hops matrices
    dist = np.asarray(room_dist, dtype=float)
    hops = np.asarray(room_hops, dtype=float)

    i_s = int(room_idx[rid_s])
    d = dist[i_s][rid_grid]
    h = hops[i_s][rid_grid]

    beta_len = float(topo.get("beta_len", 1.0))
    per_door = float(topo.get("per_door", 0.6))

    factor = np.exp(-beta_len * d) * (per_door ** h)
    factor = np.where(d > 1e8, 0.0, factor)
    factor = np.where(rid_grid == i_s, 1.0, factor)
    return factor


def _gas_model_a_v3_grid(
    gas_model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    z_query: float,
    t: float,
    theta: str,
    source_pos: Tuple[float, float, float],
    q: float,
    start_time: float,
) -> np.ndarray:
    """
    Compute a 2D gas grid using the same math as fab_benchmark.gas.gt_a.GasModelA.query,
    but vectorized over X/Y.
    """
    # --- Validate required attrs (otherwise fallback to slow generic query) ---
    required = ("mode_params", "D", "D_z", "source_radius_m", "base_sigma_z2", "lambda_decay", "scale_factor", "n_time_samples", "release_window_s", "clamp_min", "clamp_max")
    if not all(hasattr(gas_model, k) for k in required):
        # fallback: generic per-point query
        out = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    out[i, j] = float(gas_model.query(float(X[i, j]), float(Y[i, j]), float(z_query), float(t), str(theta), source_pos, float(q), float(start_time)))
                except Exception:
                    out[i, j] = 0.0
        return out

    mode_params = getattr(gas_model, "mode_params", {}) or {}
    t_eff = float(t) - float(mode_params.get("time_shift_s", 0.0))

    start_time = float(start_time)
    if t_eff < start_time:
        return np.zeros_like(X, dtype=float)

    dt_end = float(t_eff - start_time)
    T = min(dt_end, float(getattr(gas_model, "release_window_s")))
    if T <= 0.0:
        return np.zeros_like(X, dtype=float)

    n = int(getattr(gas_model, "n_time_samples"))
    n = max(1, min(64, n))
    dtau = T / float(n)
    taus = (np.arange(n, dtype=float) + 0.5) * dtau

    sx, sy, sz = float(source_pos[0]), float(source_pos[1]), float(source_pos[2])
    dz = float(z_query) - sz

    D = float(getattr(gas_model, "D"))
    D_z = float(getattr(gas_model, "D_z"))
    sigma0_2 = float(getattr(gas_model, "source_radius_m")) ** 2
    base_sigma_z2 = float(getattr(gas_model, "base_sigma_z2"))
    lam = float(getattr(gas_model, "lambda_decay"))
    scale = float(getattr(gas_model, "scale_factor"))

    # precompute topology attenuation (time-independent)
    topo_factor = _topology_factor_grid(gas_model, X, Y, sx, sy)

    # precompute vertical gaussian per tau (scalar per tau)
    sigma_z2 = base_sigma_z2 + 2.0 * D_z * taus
    g_z = (1.0 / np.sqrt(2.0 * np.pi * sigma_z2)) * np.exp(-(dz * dz) / (2.0 * sigma_z2))

    acc = np.zeros_like(X, dtype=float)
    for k, tau in enumerate(taus):
        tau = float(tau)
        xb, yb = _backtrace_xy_vec(X, Y, tau, theta, t_eff, mode_params)

        dx = xb - sx
        dy = yb - sy
        sigma2 = sigma0_2 + 2.0 * D * tau

        g_xy = (1.0 / (2.0 * np.pi * sigma2)) * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))
        acc += np.exp(-lam * tau) * g_xy * float(g_z[k])

    c = float(q) * acc * dtau * scale * topo_factor
    c = np.clip(c, float(getattr(gas_model, "clamp_min")), float(getattr(gas_model, "clamp_max")))
    return c


def _compute_gas_grid(
    scene: Dict[str, Any],
    scenario_spec: Dict[str, Any],
    trace_t: np.ndarray,
    *,
    gas_model: object,
    gas_time: Optional[float],
    gas_time_frac: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Return (gas_grid, t_query). gas_grid is HxW float (masked with NaNs in obstacles)."""

    occ = np.asarray(scene["occupancy"], dtype=np.uint8)
    H, W = occ.shape
    res = float(scene.get("resolution", 1.0))
    ox, oy = map(float, scene.get("origin", (0.0, 0.0)))

    # Choose query time
    t0 = float(np.nanmin(trace_t))
    t1 = float(np.nanmax(trace_t))

    if gas_time is not None:
        t_query = float(gas_time)
    elif gas_time_frac is not None:
        frac = float(gas_time_frac)
        frac = max(0.0, min(1.0, frac))
        t_query = t0 + frac * (t1 - t0)
    else:
        t_query = t1

    hvac_theta = (
        scenario_spec.get("hvac", {}).get("theta_true")
        or scenario_spec.get("hvac", {}).get("hvac_theta_true")
        or scenario_spec.get("hvac_theta_true")
        or "unknown"
    )

    leak = scenario_spec.get("leak", {})
    source_pos = leak.get("pos") or leak.get("source_pos")
    if source_pos is None:
        return None, t_query
    source_pos = (float(source_pos[0]), float(source_pos[1]), float(source_pos[2]) if len(source_pos) > 2 else 0.0)

    q = float(leak.get("q", leak.get("rate", 1.0)))
    start_time = float(leak.get("start_time", 0.0))

    # Cell centers
    xs = ox + (np.arange(W, dtype=np.float32) + 0.5) * res
    ys = oy + (np.arange(H, dtype=np.float32) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)

    # IMPORTANT: For correct visualization, GasModelA must be instantiated with the SAME
    # mode_params + topology (rooms/doors) as the simulator uses. Otherwise the plotted gas
    # field can look "wrong" (e.g., overly global, wrong maxima, no room attenuation).
    #
    # We therefore (re)build a per-trial GasModelA here using scenario_spec, then evaluate it
    # with a vectorized implementation of GasModelA.query.
    #
    # For other gas models (e.g. GasModelBCache), we fall back to calling query on a flattened grid.

    # Query height: use robot spawn z if available (matches measurement plane better than z=0).
    z_query = 0.0
    try:
        z_query = float(scenario_spec.get("robot", {}).get("spawn", {}).get("pos", [0.0, 0.0, 0.0])[2])
    except Exception:
        z_query = 0.0

    if gas_model.__class__.__name__ == "GasModelA":
        mode_params = dict(scenario_spec.get("hvac", {}).get("mode_params", {}) or {})
        topology = {
            "rooms": scenario_spec.get("rooms", {}) or {},
            "doors": scenario_spec.get("doors", {}) or {},
        }
        try:
            gm = gas_model.__class__(mode_params=mode_params, topology=topology)
        except Exception:
            gm = gas_model


        C = _gas_model_a_v3_grid(
            gm,
            X=X,
            Y=Y,
            z_query=z_query,
            t=float(t_query),
            theta=str(hvac_theta),
            source_pos=source_pos,
            q=q,
            start_time=start_time,
        ).astype(np.float32)
    else:
        x_flat = X.reshape(-1)
        y_flat = Y.reshape(-1)
        z_flat = np.full_like(x_flat, float(z_query), dtype=np.float32)
        t_flat = np.full_like(x_flat, float(t_query), dtype=np.float32)

        C_flat = _gas_query_generic(
            gas_model.query,
            hvac_theta=str(hvac_theta),
            source_pos=source_pos,
            q=q,
            start_time=start_time,
            x=x_flat,
            y=y_flat,
            z=z_flat,
            t=t_flat,
        )

        if C_flat is None:
            return None, t_query

        C = np.asarray(C_flat, dtype=np.float32).reshape(H, W)

    # Mask obstacles
    C = np.where(occ == 0, C, np.nan)
    return C, t_query


def _plot_trial(
    trial_dir: Path,
    out_path: Path,
    *,
    show_gas: bool,
    gas_model: object,
    gas_time: Optional[float],
    gas_time_frac: Optional[float],
    gas_log: bool,
    gas_alpha: float,
    gas_cmap: str,
    gas_log_range: float,
    path_time_color: bool,
    time_cmap: str,
    colorbar: bool,
    dpi: int,
    metrics_row: Optional[Dict[str, Any]] = None,
    annotate_metrics: bool = True,
    plot_pred: bool = True,
    pred_stride: int = 10,
    highlight_focus: bool = True,
    plot_goal: bool = True,
) -> None:

    scenario_spec = _load_scenario_spec(trial_dir)
    scene = build_scene_dict_from_scenario_spec(scenario_spec)

    df = _load_trace_csv(trial_dir)
    summary = _load_summary_json(trial_dir)

    # Prefer per-trial summary.json values; fall back to metrics row if needed.
    info: Dict[str, Any] = {}
    if metrics_row:
        info.update(metrics_row)
    if summary:
        info.update(summary)

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(float(v))
        except Exception:
            return default

    reach_flag = _to_int(info.get("reach_flag", 0), 0)
    localized_flag = _to_int(info.get("localized_flag", 0), 0)
    status = "SUCCESS" if reach_flag == 1 else "FAIL"
    title_text = f"{info.get('trial_id', trial_dir.name)} | {status} (reach={reach_flag}, loc={localized_flag})"

    x, y, t = _extract_xy_t(df)

    occ = np.asarray(scene["occupancy"], dtype=np.uint8)
    H, W = occ.shape
    res = float(scene.get("resolution", 1.0))
    ox, oy = map(float, scene.get("origin", (0.0, 0.0)))
    extent = [ox, ox + W * res, oy, oy + H * res]

    # base map: free white, obstacle black
    fig, ax = plt.subplots(figsize=(7.8, 6), dpi=dpi)
    ax.set_title(title_text, fontsize=10)
    fig.subplots_adjust(right=0.58)  # leave more room on the right for legend/metrics/colorbar
    ax.imshow(occ, cmap="gray_r", origin="lower", extent=extent, interpolation="nearest", vmin=0, vmax=1, zorder=0)

    # gas overlay
    gas_img = None
    gas_norm = None
    gas_t_query = None
    if show_gas:
        C, gas_t_query = _compute_gas_grid(
            scene,
            scenario_spec,
            t,
            gas_model=gas_model,
            gas_time=gas_time,
            gas_time_frac=gas_time_frac,
        )

        if C is not None:
            C = np.asarray(C, dtype=np.float32)
            C = np.where(np.isfinite(C) & (C > 0), C, np.nan)
            finite = C[np.isfinite(C)]
            if finite.size > 0:
                vmax = float(np.nanpercentile(finite, 99.5))
                vmax = max(vmax, 1e-12)
                if gas_log:
                    vmin = vmax / (10.0 ** float(gas_log_range))
                    vmin = max(vmin, float(np.nanmin(finite)))
                    # Hide very small concentrations to avoid tinting the whole map
                    C = np.where(C >= vmin, C, np.nan)
                    gas_norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    vmin = float(np.nanpercentile(finite, 5.0))
                    C = np.where(C >= vmin, C, np.nan)
                    gas_norm = Normalize(vmin=vmin, vmax=vmax)

                cmap_obj = plt.get_cmap(gas_cmap).copy()
                cmap_obj.set_bad((0, 0, 0, 0))  # NaNs transparent
                gas_img = ax.imshow(
                    C,
                    extent=extent,
                    origin="lower",
                    cmap=cmap_obj,
                    norm=gas_norm,
                    alpha=float(gas_alpha),
                    interpolation="bilinear",
                    zorder=1,
                )

    # path
    time_mappable = None
    if path_time_color and len(x) >= 2:
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        # color by segment time (use t[:-1])
        lc = LineCollection(segs, cmap=time_cmap, linewidths=2.2, zorder=2)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        time_mappable = lc
    else:
        ax.plot(x, y, color="tab:blue", linewidth=2.0, zorder=2)

    # markers
    ax.scatter([x[0]], [y[0]], s=90, c="tab:blue", marker="o", label="start", zorder=3)
    ax.scatter([x[-1]], [y[-1]], s=70, c="tab:orange", marker="s", label="end", zorder=3)

    leak = scenario_spec.get("leak", {})
    leak_pos = leak.get("pos")
    if leak_pos is not None:
        ax.scatter([leak_pos[0]], [leak_pos[1]], s=220, c="tab:orange", marker="*", label="leak", zorder=4)


    # tier1 sensors (support both legacy 'positions' and new 'sensors' list)
    tier1 = scenario_spec.get("tier1", {}) or {}
    sensors_xy: List[Tuple[float, float]] = []
    sensors_id: List[str] = []

    # Legacy: positions = [[x,y,z], ...]
    tpos = tier1.get("positions", [])
    if isinstance(tpos, list) and len(tpos) > 0 and isinstance(tpos[0], (list, tuple)):
        for i, p in enumerate(tpos):
            if len(p) >= 2:
                sensors_xy.append((float(p[0]), float(p[1])))
                sensors_id.append(f"t{i:02d}")

    # New: sensors = [{"id":..., "pos":[x,y,z]}, ...]
    tsens = tier1.get("sensors", [])
    if isinstance(tsens, list) and len(tsens) > 0 and isinstance(tsens[0], dict):
        sensors_xy = []
        sensors_id = []
        for s in tsens:
            pos = s.get("pos") or s.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                sensors_xy.append((float(pos[0]), float(pos[1])))
                sensors_id.append(str(s.get("id", f"t{len(sensors_id):02d}")))

    # plot all tier1 sensors (if any)
    if len(sensors_xy) > 0:
        xs = [p[0] for p in sensors_xy]
        ys = [p[1] for p in sensors_xy]
        ax.scatter(xs, ys, s=105, c="cyan", marker="^", label="tier1", zorder=4, alpha=0.85)

    # Highlight current focus sensor (from trace), if available
    if highlight_focus:
        fxcol = _pick_col(df, ["focus_x", "tier1_focus_x", "alarm_x"])
        fycol = _pick_col(df, ["focus_y", "tier1_focus_y", "alarm_y"])
        fidcol = _pick_col(df, ["focus_id", "tier1_focus_id", "alarm_id"])

        fx = _to_float(_first_nonnull(df[fxcol])) if fxcol is not None else None
        fy = _to_float(_first_nonnull(df[fycol])) if fycol is not None else None
        fid = str(_first_nonnull(df[fidcol])) if fidcol is not None else None

        # If only id is present, try to lookup from scenario sensors
        if (fx is None or fy is None) and fid is not None and len(sensors_id) == len(sensors_xy):
            try:
                j = sensors_id.index(fid)
                fx, fy = sensors_xy[j]
            except Exception:
                pass

        if fx is not None and fy is not None:
            ax.scatter([fx], [fy], s=170, c="red", marker="^", label="focus_tier1", zorder=5, edgecolors="k", linewidths=1.0)



    # Predicted source trail (from trace), if available
    if plot_pred:
        pxcol = _pick_col(df, ["pred_x", "map_x", "est_x", "belief_x"])
        pycol = _pick_col(df, ["pred_y", "map_y", "est_y", "belief_y"])
        if pxcol is not None and pycol is not None:
            px = df[pxcol].to_numpy(dtype=np.float32)
            py = df[pycol].to_numpy(dtype=np.float32)
            m2 = np.isfinite(px) & np.isfinite(py)
            if np.any(m2):
                px = px[m2]
                py = py[m2]
                stride = max(1, int(pred_stride))
                ax.scatter(px[::stride], py[::stride], s=18, c="lime", marker=".", alpha=0.7, label="pred (stride)", zorder=4)
                ax.scatter([px[-1]], [py[-1]], s=140, c="lime", marker="*", edgecolors="k", linewidths=0.8, label="pred_final", zorder=6)

    # Current / final goal (from trace), if available
    if plot_goal:
        gxcol = _pick_col(df, ["goal_x", "target_x", "nav_goal_x"])
        gycol = _pick_col(df, ["goal_y", "target_y", "nav_goal_y"])
        if gxcol is not None and gycol is not None:
            gx = _to_float(_first_nonnull(df[gxcol]))
            gy = _to_float(_first_nonnull(df[gycol]))
            # Plot last available goal
            try:
                gx = _to_float(df[gxcol].dropna().iloc[-1])
                gy = _to_float(df[gycol].dropna().iloc[-1])
            except Exception:
                pass
            if (gx is not None) and (gy is not None) and np.isfinite(gx) and np.isfinite(gy):
                ax.scatter([gx], [gy], s=120, c="yellow", marker="X", edgecolors="k", linewidths=0.8, label="goal", zorder=5)

    # Metrics annotation box (from aggregate metrics.csv), if available
    if annotate_metrics:
        rr = info if isinstance(info, dict) else {}
        reach = rr.get("reach_flag", "na")
        loc = rr.get("localized_flag", "na")
        tloc = rr.get("time_to_localize", None)
        terr = rr.get("final_error", None)
        mind = rr.get("min_dist_to_leak", None)
        coll = rr.get("collision_count", rr.get("collision_count", None))
        ttr = rr.get("time_to_reach", None)

        lines = [
            f"reach_flag: {reach}",
            f"localized:  {loc}",
            f"time_to_reach: {_fmt_num(ttr, '{:.1f}')} s",
            f"min_dist_to_leak: {_fmt_num(mind, '{:.2f}')} m",
            f"final_error: {_fmt_num(terr, '{:.2f}')} m",
            f"collisions: {coll if coll is not None else 'na'}",
            f"time_to_localize: {_fmt_num(tloc, '{:.1f}')} s",
        ]
        fig.text(
            0.80, 0.12,
            "\n".join(lines),
            transform=fig.transFigure,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.3', alpha=0.85),
        )

        # title
        sid = scenario_spec.get("scenario_id", "?")
        hvac_theta = (
            scenario_spec.get("hvac", {}).get("theta_true")
            or scenario_spec.get("hvac_theta_true")
            or "?"
        )
        doors_pat = scenario_spec.get("doors", {}).get("pattern_id") or scenario_spec.get("doors_pattern_id") or "?"
        ccount = rr.get("collision_count", "na")
        mind = rr.get("min_dist_to_leak", None)
        title_str = f"{sid} | {hvac_theta} | {doors_pat} | {status} | reach={reach} | loc={loc} | coll={ccount} | minD={_fmt_num(mind, '{:.1f}')}"
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", bbox_to_anchor=(1.55, 1.0), framealpha=0.85, borderaxespad=0.0)

        # colorbars
        if colorbar and make_axes_locatable is not None:
            divider = make_axes_locatable(ax)

            # Gas colorbar (right)
            if gas_img is not None:
                cax_g = divider.append_axes("right", size="4%", pad=0.06)
                cbg = fig.colorbar(gas_img, cax=cax_g)
                if gas_log:
                    cbg.set_label("gas concentration (log scale)")
                else:
                    cbg.set_label("gas concentration")
                if gas_t_query is not None:
                    cbg.ax.set_title(f"t={gas_t_query:.1f}s", fontsize=9)

            # Time colorbar (bottom)
            if time_mappable is not None:
                cax_t = divider.append_axes("bottom", size="4%", pad=0.55)
                cbt = fig.colorbar(time_mappable, cax=cax_t, orientation="horizontal")
                cbt.set_label("time (from trace)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _iter_trial_dirs(results_root: Path, pattern: str) -> List[Path]:
    pattern = pattern or "*"
    # Allow either "Ours__*" or "**/Ours__*" etc.
    # We expect trial folders directly under results_root.
    paths = sorted([p for p in results_root.glob(pattern) if p.is_dir()])
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--results_root", default=None, help="Root results folder, e.g. results\\exp_e1_lite")
    ap.add_argument("--results_dir", default=None, help="Alias of --results_root")
    ap.add_argument("--pattern", default="Ours__*", help="Glob pattern for trial dirs under results_root")
    ap.add_argument("--out_dir", required=True, help="Output directory (ALL images in one folder)")


    ap.add_argument("--metrics_csv", default="results\\exp_e1_main\\metrics.csv",
                help="Aggregate metrics.csv path for annotations (default: results\\exp_e1_main\\metrics.csv). "
                     "Set to empty to disable.")
    ap.add_argument("--annotate_metrics", type=int, default=1, help="1 to annotate reach/collisions/minDist etc from metrics_csv")
    ap.add_argument("--plot_pred", type=int, default=1, help="1 to plot predicted source trail (pred_x/pred_y if present in trace)")
    ap.add_argument("--pred_stride", type=int, default=10, help="Stride for plotting predicted points (every N steps)")
    ap.add_argument("--highlight_focus", type=int, default=1, help="1 to highlight focused Tier1 sensor (focus_x/y or focus_id if present)")
    ap.add_argument("--plot_goal", type=int, default=1, help="1 to plot final navigation goal (goal_x/goal_y if present)")
    

    ap.add_argument("--show_gas", type=int, default=1, help="1 to overlay gas heatmap")
    ap.add_argument("--gas_model", default="a", help="a|bcache|auto (auto uses bcache if available, else a)")
    ap.add_argument("--gas_mode", default=None, help="Alias of --gas_model (e.g., grid/analytic)")
    ap.add_argument("--gas_time", default=None, help="Gas query time in seconds (float). If omitted uses gas_time_frac or end")
    ap.add_argument("--gas_time_frac", type=float, default=None, help="Gas query time as fraction of trace [0..1]")
    ap.add_argument("--gas_log", type=int, default=1, help="1 to use log color scaling (LogNorm)")
    ap.add_argument("--gas_log_range", type=float, default=3.0, help="Dynamic range in decades for log scaling (default 3 => vmax/1e3)")
    ap.add_argument("--gas_alpha", type=float, default=0.55, help="Gas overlay alpha")
    ap.add_argument("--gas_cmap", default="magma_r", help="Matplotlib colormap for gas (high=dark, low=light)")
    ap.add_argument("--time_cmap", default="plasma_r", help="Colormap for path colored by time (short=light/yellow, long=dark/purple)")

    ap.add_argument("--path_time_color", type=int, default=1, help="1 to color path by time")
    ap.add_argument("--colorbar", type=int, default=1, help="1 to draw colorbars (time + gas)")

    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--overwrite", type=int, default=0)

    args = ap.parse_args()

    results_root = args.results_root or args.results_dir
    if results_root is None:
        ap.error("You must provide --results_root (or --results_dir)")
    results_root = Path(results_root)

    out_dir = Path(args.out_dir)
    pattern = args.pattern

    show_gas = bool(int(args.show_gas))
    path_time_color = bool(int(args.path_time_color))
    colorbar = bool(int(args.colorbar))

    gas_model_choice = args.gas_model
    if args.gas_mode is not None:
        gas_model_choice = args.gas_mode

    gas_time: Optional[float]
    if args.gas_time is None:
        gas_time = None
    else:
        try:
            gas_time = float(args.gas_time)
        except Exception:
            gas_time = None

    gas_time_frac = args.gas_time_frac

    gas_model = _make_gas_model(gas_model_choice) if show_gas else None


# Optional: load aggregate metrics for annotation (keyed by trial folder name)
    annotate_metrics = bool(int(args.annotate_metrics))
    metrics_csv_arg = str(args.metrics_csv) if args.metrics_csv is not None else ""
    metrics_csv_path = Path(metrics_csv_arg) if (metrics_csv_arg.strip() != "") else None
    metrics_idx = _load_metrics_index(metrics_csv_path) if annotate_metrics else {}

    trial_dirs = _iter_trial_dirs(results_root, pattern)
    if len(trial_dirs) == 0:
        print(f"No trial directories matched: {results_root} / {pattern}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for i, td in enumerate(trial_dirs, 1):
        name = td.name
        out_path = out_dir / f"{name}.png"

        if out_path.exists() and not bool(int(args.overwrite)):
            print(f"[{i}/{len(trial_dirs)}] SKIP {name} (exists)")
            continue

        try:
            _plot_trial(
                td,
                out_path,
                show_gas=show_gas,
                gas_model=gas_model,
                gas_time=gas_time,
                gas_time_frac=gas_time_frac,
                gas_log=bool(int(args.gas_log)),
                gas_alpha=float(args.gas_alpha),
                gas_cmap=str(args.gas_cmap),
                gas_log_range=float(args.gas_log_range),
                path_time_color=path_time_color,
                time_cmap=str(args.time_cmap),
                colorbar=colorbar,
                dpi=int(args.dpi),
                metrics_row=metrics_idx.get(name),
                annotate_metrics=annotate_metrics,
                plot_pred=bool(int(args.plot_pred)),
                pred_stride=int(args.pred_stride),
                highlight_focus=bool(int(args.highlight_focus)),
                plot_goal=bool(int(args.plot_goal)),
            )
            ok += 1
            print(f"[{i}/{len(trial_dirs)}] OK   {name} -> {out_path}")
        except Exception as e:
            fail += 1
            try:
                plt.close('all')
            except Exception:
                pass
            print(f"[{i}/{len(trial_dirs)}] FAIL {name}: {e}")

    print(f"DONE. ok={ok}, failed={fail}, out_dir={out_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())