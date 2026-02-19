
"""
Select interesting cases for Isaac rerun / visualization (v1).

CLI contract:
  python -m fab_benchmark.runners.select_cases --exp e1 --policy Ours --backend lite --top_k 10 --out results/case_list.json

Selection heuristic (deterministic):
  - prefer localized trials (localized_flag==1)
  - sort by final_error descending, break ties by exposure_total descending
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=str)
    ap.add_argument("--policy", required=True, type=str)
    ap.add_argument("--backend", required=True, type=str, choices=["lite","isaac3d"])
    ap.add_argument("--top_k", required=True, type=int)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    base = Path("results") / f"exp_{args.exp}_{args.backend}"
    if not base.exists():
        raise SystemExit(f"Results base not found: {base}")

    cases = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta_p = d / "run_meta.json"
        summ_p = d / "summary.json"
        if not meta_p.exists() or not summ_p.exists():
            continue
        meta = read_json(meta_p)
        if meta.get("policy", {}).get("policy_name") != args.policy:
            continue
        summ = read_json(summ_p)

        cases.append({
            "trial_dir": str(d),
            "trace_csv": meta.get("paths", {}).get("trace_csv"),
            "scene_usd": meta.get("paths", {}).get("scene_usd"),
            "scenario_hash": meta.get("scenario_hash"),
            "scenario_id": meta.get("scenario_spec", {}).get("scenario_id"),
            "scenario_family": meta.get("scenario_spec", {}).get("scenario_family"),
            "doors_pattern_id": meta.get("scenario_spec", {}).get("doors", {}).get("pattern_id"),
            "hvac_theta_true": meta.get("scenario_spec", {}).get("hvac", {}).get("theta_true"),
            "seed": meta.get("experiment", {}).get("seed"),
            "localized_flag": int(summ.get("localized_flag", 0)),
            "final_error": summ.get("final_error", None),
            "exposure_total": summ.get("exposure_total", None),
            "violation_flag": int(summ.get("violation_flag", 0)),
        })

    def key(c):
        # localized first (descending), then final_error desc, then exposure desc
        loc = int(c["localized_flag"])
        fe = c["final_error"]
        fev = float(fe) if fe is not None else -1.0
        ex = c["exposure_total"]
        exv = float(ex) if ex is not None else 0.0
        return (loc, fev, exv)

    cases.sort(key=key, reverse=True)
    selected = cases[:max(0, int(args.top_k))]

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"version":"v1","exp":args.exp,"policy":args.policy,"backend":args.backend,"cases":selected}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
