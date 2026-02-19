
"""
Replay a completed Isaac trial in an interactive Isaac Sim session (v1).

CLI contract:
  python -m fab_benchmark.runners.replay_in_isaac --trial_dir results/isaac_cases/<id> --rate 1.0

Notes:
  - This is best-effort and depends on Isaac Sim APIs available in your version.
  - For video recording, prefer visualize_case.py which can optionally invoke Isaac capture.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from fab_benchmark.backends.isaac_backend import IsaacBackend
from fab_benchmark.runners.utils import read_json
from fab_benchmark.isaac.utils import set_prim_pose


def read_trace(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", required=True, type=str)
    ap.add_argument("--rate", required=True, type=float)
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    meta = read_json(trial_dir / "run_meta.json")
    sim_params = meta["sim_params"]
    scenario_spec = meta["scenario_spec"]

    # Force interactive rendering if possible
    sim_params = dict(sim_params)
    sim_params["isaac"] = dict(sim_params["isaac"])
    sim_params["isaac"]["headless"] = 0
    sim_params["isaac"]["render"] = 1

    backend = IsaacBackend(sim_params=sim_params)
    backend.load_scene(scenario_spec)
    backend.reset(seed=int(meta["experiment"]["seed"]), scene_spec=scenario_spec, init_pose={"x":0.0,"y":0.0,"z":0.35,"yaw":0.0})

    trace = read_trace(Path(meta["paths"]["trace_csv"]))
    rate = max(1e-3, float(args.rate))

    # Replay by setting robot pose at each trace row
    for row in trace:
        x = float(row["x"]); y = float(row["y"]); z = float(row["z"]); yaw = float(row["yaw"])
        try:
            set_prim_pose("/World/Robot/Body", position=(x,y,z), yaw=yaw)
        except Exception:
            pass
        # step world (dt not used; IsaacWorld step advances fixed)
        backend.step(dt=float(sim_params["dt_physics"]), substeps=int(sim_params["substeps"]))
        time.sleep((1.0/float(sim_params["control_hz"])) / rate)

    backend.close()


if __name__ == "__main__":
    main()
