import argparse
import matplotlib.pyplot as plt
from fab_benchmark.scenarios.base_scenario import generate_f2, build_scene_dict_from_scenario_spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", type=str, default="results/_preview_f2.png")
    args = ap.parse_args()

    spec = generate_f2(args.seed, rooms_n=14)
    scene = build_scene_dict_from_scenario_spec(spec)
    occ = scene["occupancy"]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(occ, origin="lower", cmap="gray_r")
    ax.set_title(f"F2 preview seed={args.seed} (walls/doors)")

    # doors
    doors = spec["doors"]["door_geoms"]
    states = spec["doors"]["states"]
    res = scene["resolution"]
    ox, oy = scene["origin"]
    for did, g in doors.items():
        p0, p1 = g["p0"], g["p1"]
        x0,y0 = p0; x1,y1 = p1
        j0 = (x0-ox)/res; i0 = (y0-oy)/res
        j1 = (x1-ox)/res; i1 = (y1-oy)/res
        ax.plot([j0,j1],[i0,i1], lw=3, color=("lime" if states[did]==1 else "red"))

    # spawn marker (optional)
    spawn_xy = None
    if "robot" in spec and "spawn" in spec["robot"] and "pos" in spec["robot"]["spawn"]:
        sx, sy, _ = spec["robot"]["spawn"]["pos"]
        spawn_xy = (sx, sy)
    # if no robot info, just skip drawing spawn (do NOT fallback to spec["rooms"])


    lx,ly,_ = spec["leak"]["pos"]
    if spawn_xy is not None:
        sx, sy = spawn_xy
        ax.scatter([(sx-ox)/res], [(sy-oy)/res], s=80, marker="o", label="spawn")

    ax.scatter([(lx-ox)/res], [(ly-oy)/res], s=90, marker="X", label="leak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
