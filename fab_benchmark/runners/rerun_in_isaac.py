
"""
Rerun selected cases in Isaac3D (v1).

CLI contract:
  python -m fab_benchmark.runners.rerun_in_isaac --case_list results/case_list.json --isaac_out results/isaac_cases/ --headless 1 --with_go2 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from fab_benchmark.runners.run_trial import run_trial
from fab_benchmark.runners.utils import load_yaml, ensure_dir


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_list", required=True, type=str)
    ap.add_argument("--isaac_out", required=True, type=str)
    ap.add_argument("--headless", required=True, type=int)
    ap.add_argument("--with_go2", required=True, type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    case_list = read_json(Path(args.case_list))
    cases = case_list.get("cases", [])
    out_base = Path(args.isaac_out)
    ensure_dir(out_base)

    for c in cases:
        trial_dir = Path(c["trial_dir"])
        meta_p = trial_dir / "run_meta.json"
        if not meta_p.exists():
            continue
        meta = read_json(meta_p)
        # Construct trial cfg from meta
        trial_cfg = {
            "version": "v1",
            "sim_params": meta["sim_params"],
            "experiment": {
                "exp_id": meta["experiment"]["exp_id"],
                "seed": int(meta["experiment"]["seed"]),
                "policy_name": meta["policy"]["policy_name"],
                "trial_id": meta["experiment"].get("trial_id", trial_dir.name),
                "scenario_spec": meta["scenario_spec"],
            },
            "policies": {meta["policy"]["policy_name"]: meta["policy"]["policy_cfg"]},
        }
        # preserve door mismatch flag if present
        if "door_mismatch" in meta.get("experiment", {}):
            trial_cfg["experiment"]["door_mismatch"] = int(meta["experiment"]["door_mismatch"])

        out_dir = out_base / trial_dir.name
        run_trial(trial_cfg, backend_name="isaac3d", headless=args.headless, with_go2=args.with_go2, out_dir=str(out_dir), force=args.force)


if __name__ == "__main__":
    main()
