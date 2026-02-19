import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def read_trace(p: Path):
    rows=[]
    with p.open("r", newline="") as f:
        r=csv.DictReader(f)
        for rr in r: rows.append(rr)
    def col(name):
        out=[]
        for rr in rows:
            try: out.append(float(rr.get(name,"nan")))
            except: out.append(np.nan)
        return np.array(out, dtype=np.float64)
    return col("time"), col("x"), col("y"), col("z"), col("hazard"), col("y_meas")

def first_detect(hazard, y_meas):
    idx=np.where(np.isfinite(hazard) & (hazard>0))[0]
    if len(idx)>0: return int(idx[0])
    thr=np.nanpercentile(y_meas,95)
    idx=np.where(np.isfinite(y_meas) & (y_meas>=thr))[0]
    return int(idx[0]) if len(idx)>0 else None

def draw_map(ax, meta):
    spec = meta.get("scenario_spec", {}) or {}
    mp = spec.get("map", {}) or {}
    occ = mp.get("occupancy", None)
    if occ is not None:
        occ = np.array(occ, dtype=np.uint8)
        res = float(mp.get("resolution", 0.1))
        origin = mp.get("origin", [0.0,0.0])
        ox, oy = float(origin[0]), float(origin[1])
        h, w = occ.shape
        extent = [ox, ox + w*res, oy, oy + h*res]
        ax.imshow(occ, origin="lower", extent=extent, cmap="gray_r", alpha=0.85, aspect="equal")
        return extent
    return None

def draw_doors(ax, meta):
    spec = meta.get("scenario_spec", {}) or {}
    doors = spec.get("doors", {}) or {}
    geoms = doors.get("door_geoms", {}) or {}
    states = doors.get("states", {}) or {}
    for did, g in geoms.items():
        p0 = g.get("p0", None); p1 = g.get("p1", None)
        if not (isinstance(p0,list) and isinstance(p1,list) and len(p0)==2 and len(p1)==2):
            continue
        x0,y0 = float(p0[0]), float(p0[1])
        x1,y1 = float(p1[0]), float(p1[1])
        st = int(states.get(did, 1))
        # open: green, closed: red
        ax.plot([x0,x1],[y0,y1], linewidth=4, alpha=0.9, color=("lime" if st==1 else "red"))

def draw_path(ax, t, x, y):
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t,x,y = t[ok], x[ok], y[ok]
    if len(x)<2: return
    pts = np.column_stack([x,y]).reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap="viridis", linewidths=4.5)
    lc.set_zorder(10)
    ax.add_collection(lc)
    ax.scatter([x[0]],[y[0]], s=80, marker="o", zorder=11, label="start")
    ax.scatter([x[-1]],[y[-1]], s=80, marker="s", zorder=11, label="end")

    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time (s)")

def draw_hazard_heat(ax, x, y, hazard, bins=140):
    ok = np.isfinite(x)&np.isfinite(y)&np.isfinite(hazard)
    if ok.sum()<5: return
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, weights=hazard[ok])
    ax.imshow(H.T, origin="lower", extent=[xe[0],xe[-1],ye[0],ye[-1]], cmap="inferno", alpha=0.35, aspect="equal")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    d=Path(args.trial_dir)
    meta=read_json(d/"run_meta.json")
    t,x,y,z,hazard,y_meas = read_trace(d/"trace.csv")

    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_title("Topdown: walls/doors + time-colored trajectory + hazard heat")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

    extent = draw_map(ax, meta)
    draw_doors(ax, meta)
    draw_hazard_heat(ax, x, y, hazard)
    draw_path(ax, t, x, y)

    idx = first_detect(hazard, y_meas)
    if idx is not None:
        ax.scatter([x[idx]],[y[idx]], s=140, marker="*", zorder=6, label="first detect (hazard>0)")
    leak = (meta.get("scenario_spec", {}) or {}).get("leak", {}) or {}
    if isinstance(leak.get("pos",None), list) and len(leak["pos"])>=2:
        lx,ly=float(leak["pos"][0]), float(leak["pos"][1])
        ax.scatter([lx],[ly], s=120, marker="X", zorder=6, label="leak (truth)")

    ax.legend(loc="upper right")
    fig.tight_layout()
    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("[OK] wrote", out)

if __name__=="__main__":
    main()
