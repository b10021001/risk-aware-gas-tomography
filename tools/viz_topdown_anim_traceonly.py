import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def read_trace(p: Path):
    rows = []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for rr in r:
            rows.append(rr)

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

def first_detect(hazard, y_meas):
    idx = np.where(np.isfinite(hazard) & (hazard > 0))[0]
    if len(idx) > 0:
        return int(idx[0]), "hazard>0"
    thr = np.nanpercentile(y_meas, 95)
    idx = np.where(np.isfinite(y_meas) & (y_meas >= thr))[0]
    if len(idx) > 0:
        return int(idx[0]), f"y_meas>=p95({thr:.3f})"
    return None, "no_detect"

def accumulated_heat(ax, x, y, w, bins=120):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if ok.sum() < 5:
        return None
    xmin, xmax = np.min(x[ok]), np.max(x[ok])
    ymin, ymax = np.min(y[ok]), np.max(y[ok])
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins,
                               range=[[xmin, xmax], [ymin, ymax]],
                               weights=w[ok])
    im = ax.imshow(H.T, origin="lower",
                   extent=[xmin, xmax, ymin, ymax],
                   cmap="inferno", alpha=0.35, aspect="equal")
    return im

def time_colored_path(ax, x, y, t):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[ok], y[ok], t[ok]
    if len(x) < 2:
        return
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap="viridis", linewidths=2.5)
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time (s)")

def make_static(trial_dir: Path, out_png: Path):
    meta = read_json(trial_dir / "run_meta.json")
    t,x,y,z,hazard,y_meas,entropy,exposure = read_trace(trial_dir / "trace.csv")

    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_title("Topdown: time-colored trajectory + accumulated hazard")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

    accumulated_heat(ax, x, y, hazard)
    time_colored_path(ax, x, y, t)

    idx, rule = first_detect(hazard, y_meas)
    if idx is not None:
        ax.scatter([x[idx]],[y[idx]], s=140, marker="*", zorder=5, label=f"first detect ({rule})")

    leak = (meta.get("scenario_spec", {}) or {}).get("leak", {}) or {}
    if isinstance(leak.get("pos", None), list) and len(leak["pos"]) >= 2:
        lx, ly = float(leak["pos"][0]), float(leak["pos"][1])
        ax.scatter([lx],[ly], s=120, marker="X", zorder=5, label="leak (truth)")

    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    trial_dir = Path(args.trial_dir)
    out_png = Path(args.out)
    if out_png.suffix.lower() != ".png":
        out_png = out_png.with_suffix(".png")
    make_static(trial_dir, out_png)
    print("[OK] wrote", out_png)

if __name__ == "__main__":
    main()
