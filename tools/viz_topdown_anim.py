#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_case_pretty_v3.py

修正：
- LiteBackend 的 __init__ 在你目前版本不接受 headless 參數，因此本版會自動相容：
  先嘗試 LiteBackend(headless=True)，失敗就改用 LiteBackend()。

其餘功能同 v2：
- 從 LiteBackend.load_scene(scenario_spec) 取得 occupancy / resolution / origin
- 牆黑地白 + 紅色粗線路徑 + 灰點 + hazard/y_meas 熱度
- 2x2 圖輸出

用法（cmd）：import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def _read_trace_csv(p: Path):
    rows = []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for rr in r:
            rows.append(rr)
    # required numeric columns
    def col(name, default=np.nan):
        out = []
        for rr in rows:
            try:
                out.append(float(rr.get(name, default)))
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=np.float64)

    t = col("time")
    x = col("x"); y = col("y"); z = col("z")
    hazard = col("hazard")
    y_meas = col("y_meas")
    entropy = col("entropy")
    exposure = col("exposure_integral")
    return t, x, y, z, hazard, y_meas, entropy, exposure

def _first_detect_idx(hazard: np.ndarray, y_meas: np.ndarray):
    idx = np.where(np.isfinite(hazard) & (hazard > 0))[0]
    if len(idx) > 0:
        return int(idx[0]), "hazard>0"
    # fallback: y_meas above 95th percentile
    thr = np.nanpercentile(y_meas, 95)
    idx = np.where(np.isfinite(y_meas) & (y_meas >= thr))[0]
    if len(idx) > 0:
        return int(idx[0]), f"y_meas>=p95({thr:.3f})"
    return None, "no_detect"

def _make_time_colored_line(ax, x, y, t, lw=2.0):
    # line segments colored by time
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap="viridis", linewidths=lw)
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time (s)")
    return lc

def _accumulated_hazard_heat(ax, x, y, v, bins=120):
    # build a simple accumulated heat map in XY using histogram weighted by v
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        return None
    H, xedges, yedges = np.histogram2d(
        x[np.isfinite(v)], y[np.isfinite(v)],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=v[np.isfinite(v)],
    )
    # show as image (transpose because histogram2d returns x as first axis)
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="inferno",
        alpha=0.35,
        aspect="equal",
    )
    return im

def make_static(meta, t, x, y, hazard, y_meas, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Topdown: trajectory (time-colored) + accumulated hazard")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # accumulated hazard heat (shows where gas was encountered over time)
    im = _accumulated_hazard_heat(ax, x, y, hazard)

    # trajectory colored by time
    _make_time_colored_line(ax, x, y, t, lw=2.5)

    # mark first detect
    idx, rule = _first_detect_idx(hazard, y_meas)
    if idx is not None:
        ax.scatter([x[idx]], [y[idx]], s=120, marker="*", zorder=5, label=f"first detect ({rule})")
    # mark leak (if present)
    leak = (meta.get("scenario_spec", {}) or {}).get("leak", {}) or {}
    if "pos" in leak and isinstance(leak["pos"], list) and len(leak["pos"]) >= 2:
        lx, ly = float(leak["pos"][0]), float(leak["pos"][1])
        ax.scatter([lx], [ly], s=100, marker="X", zorder=5, label="leak (truth)")

    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def make_gif(meta, t, x, y, hazard, y_meas, out_gif: Path, fps=10, stride=3):
    import imageio.v2 as imageio

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    idx_first, rule = _first_detect_idx(hazard, y_meas)
    frames = []
    N = len(t)
    frame_idxs = list(range(1, N, stride))
    if frame_idxs[-1] != N-1:
        frame_idxs.append(N-1)

    # precompute extent for consistent view
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    for k in frame_idxs:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.set_title(f"t={t[k]:.1f}s  (first detect: {('yes' if idx_first is not None and k>=idx_first else 'no')})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

        # heat: accumulated hazard up to time k (shows “how gas evidence appeared over time”)
        _accumulated_hazard_heat(ax, x[:k+1], y[:k+1], hazard[:k+1], bins=120)

        # path up to k
        ax.plot(x[:k+1], y[:k+1], linewidth=2.5)

        # current position
        ax.scatter([x[k]], [y[k]], s=35)

        # mark first detect + leak
        if idx_first is not None:
            ax.scatter([x[idx_first]], [y[idx_first]], s=110, marker="*", zorder=5)
        leak = (meta.get("scenario_spec", {}) or {}).get("leak", {}) or {}
        if "pos" in leak and isinstance(leak["pos"], list) and len(leak["pos"]) >= 2:
            lx, ly = float(leak["pos"][0]), float(leak["pos"][1])
            ax.scatter([lx], [ly], s=90, marker="X", zorder=5)

        fig.tight_layout()
        tmp = out_gif.parent / "_tmp_frame.png"
        fig.savefig(tmp, dpi=160)
        plt.close(fig)
        frames.append(imageio.imread(tmp))

    if (out_gif.parent / "_tmp_frame.png").exists():
        (out_gif.parent / "_tmp_frame.png").unlink()

    imageio.mimsave(out_gif, frames, fps=int(fps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    meta = _read_json(trial_dir / "run_meta.json")
    t, x, y, z, hazard, y_meas, entropy, exposure = _read_trace_csv(trial_dir / "trace.csv")

    out_dir = Path(args.out_dir)
    make_static(meta, t, x, y, hazard, y_meas, out_dir / "topdown_static.png")
    make_gif(meta, t, x, y, hazard, y_meas, out_dir / "topdown_anim.gif", fps=args.fps, stride=args.stride)
    print("[OK] wrote:", out_dir / "topdown_static.png")
    print("[OK] wrote:", out_dir / "topdown_anim.gif")

if __name__ == "__main__":
    main()

  cd /d C:\isaac_3\fab_gas_demo
  py tools\visualize_case_pretty_v3.py ^
    --trial_dir results\e8_best_isaac\Ours__F2__vortex_ccw__all_open__i013__seed13 ^
    --out results\e8_best_isaac\_viz_pretty.png ^
    --heat_source hazard
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# --- ensure project root on sys.path ---
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _read_trace(p: Path):
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise RuntimeError(f"Empty trace.csv: {p}")
    return rows


def _pick_key(rows, candidates):
    keys = set(rows[0].keys())
    for k in candidates:
        if k in keys:
            return k
    return None


def _col_float(rows, key):
    if key is None:
        return None
    out = []
    for rr in rows:
        try:
            out.append(float(rr[key]))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def _gaussian_kernel1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _blur2d(A, sigma=2.2, radius=7):
    k = _gaussian_kernel1d(sigma, radius)
    B = np.pad(A, ((0, 0), (radius, radius)), mode="edge")
    B = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, B)
    C = np.pad(B, ((radius, radius), (0, 0)), mode="edge")
    C = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, C)
    return C


def _init_lite_backend():
    from fab_benchmark.backends.lite_backend import LiteBackend  # type: ignore
    # Be compatible with different signatures
    try:
        return LiteBackend(headless=True)
    except TypeError:
        return LiteBackend()
    except Exception:
        # last resort: no args
        return LiteBackend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--heat_source", default="hazard", choices=["hazard", "y_meas", "auto"])
    ap.add_argument("--alpha", default=0.55, type=float)
    ap.add_argument("--lw", default=3.0, type=float)
    ap.add_argument("--dot_n", default=50, type=int)
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    out = Path(args.out)

    meta = _read_json(trial_dir / "run_meta.json")
    summ = _read_json(trial_dir / "summary.json")
    rows = _read_trace(trial_dir / "trace.csv")

    try:
        backend = _init_lite_backend()
    except Exception as e:
        raise SystemExit(
            "ERROR: 無法建立 LiteBackend。\n"
            f"PROJECT_ROOT={PROJECT_ROOT}\n"
            f"details: {e}"
        )

    scene = backend.load_scene(meta["scenario_spec"])
    if not isinstance(scene, dict):
        raise SystemExit(f"ERROR: load_scene returned {type(scene)} not dict")
    occ = scene.get("occupancy", None)
    if occ is None:
        raise SystemExit(f"ERROR: load_scene() keys={list(scene.keys())} (no occupancy)")

    occ = np.asarray(occ, dtype=np.float32)
    bg = 1.0 - np.clip(occ, 0.0, 1.0)  # floor white, walls black
    H, W = bg.shape

    # trace columns
    kgi = _pick_key(rows, ["grid_i", "i", "cell_i"])
    kgj = _pick_key(rows, ["grid_j", "j", "cell_j"])
    kx  = _pick_key(rows, ["x", "pos_x", "pose_x", "robot_x"])
    ky  = _pick_key(rows, ["y", "pos_y", "pose_y", "robot_y"])
    kt  = _pick_key(rows, ["t", "time", "time_s"])
    khaz = _pick_key(rows, ["hazard", "hazard_level"])
    kmeas = _pick_key(rows, ["y_meas", "meas", "gas", "gas_meas"])
    kent = _pick_key(rows, ["entropy"])
    kexp = _pick_key(rows, ["exposure_integral", "exposure"])

    gi = _col_float(rows, kgi)
    gj = _col_float(rows, kgj)
    xs = _col_float(rows, kx)
    ys = _col_float(rows, ky)
    ts = _col_float(rows, kt)
    haz = _col_float(rows, khaz) if khaz else None
    meas = _col_float(rows, kmeas) if kmeas else None
    ent = _col_float(rows, kent) if kent else None
    expi = _col_float(rows, kexp) if kexp else None

    # prefer grid coords for occupancy
    if gi is None or gj is None or not (np.isfinite(gi).any() and np.isfinite(gj).any()):
        origin = scene.get("origin", meta.get("scenario_spec", {}).get("map", {}).get("origin", [0.0, 0.0]))
        res = float(scene.get("resolution", meta.get("scenario_spec", {}).get("map", {}).get("resolution", 0.2)))
        x0, y0 = float(origin[0]), float(origin[1])
        if xs is None or ys is None:
            raise SystemExit("ERROR: trace.csv 沒有 grid_i/grid_j 也沒有 x/y")
        gj = (xs - x0) / res
        gi = (ys - y0) / res

    # heat source
    if args.heat_source == "hazard":
        val = haz
        vname = khaz or "hazard"
    elif args.heat_source == "y_meas":
        val = meas
        vname = kmeas or "y_meas"
    else:
        if haz is not None:
            val = haz
            vname = khaz or "hazard"
        elif meas is not None:
            val = meas
            vname = kmeas or "y_meas"
        else:
            val = None
            vname = "gas"
    if val is None:
        val = np.ones_like(gi, dtype=np.float64)

    # heatmap on grid
    heat = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for a, b, v in zip(gi, gj, val):
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(v)):
            continue
        ii = int(round(a)); jj = int(round(b))
        if 0 <= ii < H and 0 <= jj < W:
            heat[ii, jj] += max(0.0, float(v))
            cnt[ii, jj] += 1.0
    m = cnt > 0
    heat[m] = heat[m] / cnt[m]
    heat_s = _blur2d(heat, sigma=2.2, radius=7)
    mx = float(np.nanmax(heat_s)) if np.isfinite(heat_s).any() else 0.0
    if mx > 0:
        heat_s = heat_s / mx

    # leak & estimate (grid)
    leak_pos = meta.get("scenario_spec", {}).get("leak", {}).get("pos", None)
    est_pos = summ.get("final_estimate", {}).get("pos", None)
    origin = scene.get("origin", meta.get("scenario_spec", {}).get("map", {}).get("origin", [0.0, 0.0]))
    res = float(scene.get("resolution", meta.get("scenario_spec", {}).get("map", {}).get("resolution", 0.2)))
    x0, y0 = float(origin[0]), float(origin[1])
    leak_g = None; est_g = None
    if leak_pos is not None:
        leak_g = ((float(leak_pos[0]) - x0) / res, (float(leak_pos[1]) - y0) / res)  # (gj, gi)
    if est_pos is not None:
        est_g = ((float(est_pos[0]) - x0) / res, (float(est_pos[1]) - y0) / res)

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Map + Trajectory + Gas heat")
    ax1.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.imshow(heat_s, cmap="Reds", alpha=float(args.alpha), vmin=0, vmax=1)
    ax1.plot(gj, gi, color="red", linewidth=float(args.lw), label="trajectory")
    step = max(1, len(gj) // max(1, int(args.dot_n)))
    ax1.scatter(gj[::step], gi[::step], s=16, color="0.6", label="samples")
    ax1.scatter([gj[0]], [gi[0]], s=80, marker="o", color="black", label="start")
    ax1.scatter([gj[-1]], [gi[-1]], s=80, marker="s", color="black", label="end")
    if leak_g is not None:
        ax1.scatter([leak_g[0]], [leak_g[1]], s=160, marker="x", linewidths=3, color="lime", label="leak (true)")
    if est_g is not None:
        ax1.scatter([est_g[0]], [est_g[1]], s=200, marker="*", color="gold", edgecolors="k", linewidths=0.8, label="estimate (final)")
    ax1.set_xlabel("grid j"); ax1.set_ylabel("grid i")
    ax1.legend(loc="lower right", framealpha=0.85)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Entropy")
    if ent is not None and ts is not None:
        ax2.plot(ts, ent, linewidth=2.0); ax2.set_xlabel("time (s)")
    elif ent is not None:
        ax2.plot(np.arange(len(ent)), ent, linewidth=2.0); ax2.set_xlabel("step")
    else:
        ax2.text(0.5, 0.5, "entropy column not found", ha="center", va="center")
    ax2.set_ylabel("entropy")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Exposure Integral")
    if expi is not None and ts is not None:
        ax3.plot(ts, expi, linewidth=2.0); ax3.set_xlabel("time (s)")
    elif expi is not None:
        ax3.plot(np.arange(len(expi)), expi, linewidth=2.0); ax3.set_xlabel("step")
    else:
        ax3.text(0.5, 0.5, "exposure_integral column not found", ha="center", va="center")
    ax3.set_ylabel("exposure")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title(f"Measured gas ({vname})")
    series = haz if haz is not None else meas
    if args.heat_source == "y_meas" and meas is not None:
        series = meas
    if args.heat_source == "hazard" and haz is not None:
        series = haz
    if series is not None and ts is not None:
        ax4.plot(ts, series, linewidth=2.0); ax4.set_xlabel("time (s)")
    elif series is not None:
        ax4.plot(np.arange(len(series)), series, linewidth=2.0); ax4.set_xlabel("step")
    else:
        ax4.text(0.5, 0.5, "hazard/y_meas not found", ha="center", va="center")
    ax4.set_ylabel(vname)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=220)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
