
"""
Run a full experiment sweep (v1).

CLI contract:
  python -m fab_benchmark.runners.run_experiment --exp e1 --config configs/exp_e1.yaml --backend lite|isaac3d --n_trials 50 --headless 1 --with_go2 0 [--resume] [--force]

Notes:
  - Deterministic seeds: same seed per (condition, trial_index) across methods.
  - Output dirs: results/exp_<exp>_<backend>/<trial_dir> (one level) for easy globbing.
"""
from __future__ import annotations

import argparse
import os
import sys
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy
import random

from fab_benchmark.runners.utils import load_yaml, validate_experiment_config, ValidationError, ensure_dir, exit_with
from fab_benchmark.runners.run_trial import run_trial
from fab_benchmark.scenarios.f1_small import make_scenario_spec as make_f1
from fab_benchmark.scenarios.f2_medium import make_scenario_spec as make_f2
from fab_benchmark.scenarios.f3_large import make_scenario_spec as make_f3
from fab_benchmark.scenarios.doors import apply_door_pattern


def _make_base_scenario(family: str, seed: int, params: Dict[str, Any]) -> Dict[str, Any]:
    if family == "F1":
        return make_f1(seed)
    if family == "F2":
        rooms_n = int(params.get("rooms_n", 14))
        return make_f2(seed, rooms_n=rooms_n)
    if family == "F3":
        rooms_n = int(params.get("rooms_n", 34))
        return make_f3(seed, rooms_n=rooms_n)
    raise ValueError(f"Unknown scenario_family: {family}")


def _apply_condition(spec: Dict[str, Any], hvac_theta_true: str, doors_pattern_id: str, seed: int) -> Dict[str, Any]:
    # Copy spec and override theta / doors
    out = copy.deepcopy(spec)
    out["hvac"]["theta_true"] = hvac_theta_true
    out["doors"]["pattern_id"] = doors_pattern_id
    door_ids = list(out["doors"]["door_geoms"].keys())
    out["doors"]["states"] = apply_door_pattern(doors_pattern_id, door_ids, seed)
    return out


def _add_door_mismatch(spec: Dict[str, Any], mismatch_seed: int, assumed_pattern: str = "all_open") -> Dict[str, Any]:
    out = copy.deepcopy(spec)
    door_ids = list(out["doors"]["door_geoms"].keys())
    assumed_states = apply_door_pattern(assumed_pattern, door_ids, mismatch_seed + 404)
    out["doors_assumed"] = {"pattern_id": assumed_pattern, "states": assumed_states}
    return out


def build_trial_cfg(exp_cfg: Dict[str, Any], sim_params: Dict[str, Any], policies: Dict[str, Any],
                    exp_id: str, policy_name: str, seed: int, scenario_spec: Dict[str, Any],
                    trial_id: str, extra_exp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {
        "version": "v1",
        "sim_params": sim_params,
        "experiment": {
            "exp_id": exp_id,
            "seed": int(seed),
            "policy_name": policy_name,
            "trial_id": trial_id,
            "scenario_spec": scenario_spec,
        },
        "policies": policies,
    }
    # append-only extra experiment fields (E3/E4/E5/E6/E7)
    cfg["experiment"].update(extra_exp)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=str)
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--backend", required=True, choices=["lite","isaac3d"])
    ap.add_argument("--n_trials", required=True, type=int)
    ap.add_argument("--headless", required=True, type=int)
    ap.add_argument("--with_go2", required=True, type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        validate_experiment_config(cfg, ctx="experiment_config", exp_cli=args.exp)
    except ValidationError as e:
        exit_with(2, f"[config validation error] {e}")

    sim_params = cfg["sim_params"]
    exp = cfg["experiment"]
    policies = cfg["policies"]

    # Ensure n_trials matches config
    if int(args.n_trials) != int(exp["trials_per_condition"]):
        exit_with(2, f"--n_trials must equal experiment.trials_per_condition ({exp['trials_per_condition']})")

    methods: List[str] = list(exp["methods"])
    sweep: Dict[str, Any] = exp["condition_sweep"]
    seed0 = int(exp.get("seed0", 0))

    scenario_families = sweep.get("scenario_families", ["F2"])
    scenario_params = sweep.get("scenario_params", {})
    hvac_modes = sweep.get("hvac_modes", ["drift_pos_x"])
    door_patterns = sweep.get("door_patterns", ["half_random"])

    # E3/E4 specific sweeps
    E_max_levels = sweep.get("E_max_levels", [1.50])
    delta_levels = sweep.get("delta_levels", [0.10])
    p_thr_levels = sweep.get("p_thr_levels", [0.50])
    cred_thr_levels = sweep.get("credible_volume_thr_levels", [0.80])

    # door mismatch (E2)
    door_mismatch = int(sweep.get("door_mismatch", 0)) == 1
    door_assumed_pattern = str(sweep.get("door_assumed_pattern", "all_open"))

    # gas truth mismatch (E5)
    gas_truth = str(sweep.get("gas_truth", "gt_a"))
    gtb_cache_path = sweep.get("gtb_cache_path", None)
    gtb_k = int(sweep.get("gtb_k", 0))

    # null leak (E6)
    null_leak = int(sweep.get("null_leak", 0)) == 1
    include_no_leak_hypothesis = int(sweep.get("include_no_leak_hypothesis", 0)) == 1

    # output base
    base_dir = Path("results") / f"exp_{args.exp}_{args.backend}"
    ensure_dir(base_dir)

    # Build condition product depending on exp
    # For most experiments: (family, hvac, doors, E_max, delta, p_thr, cred_thr)
    # We keep unused dims at singleton.
    conditions = list(itertools.product(
        scenario_families,
        hvac_modes,
        door_patterns,
        E_max_levels,
        delta_levels,
        p_thr_levels,
        cred_thr_levels,
    ))
    print(f"[DEBUG] exp={args.exp} | len(E_max_levels)={len(E_max_levels)} | len(delta_levels)={len(delta_levels)} | len(conditions)={len(conditions)}")

    for family, hvac, doors, E_max, delta, p_thr, cred_thr in conditions:
        # For each condition, build a base scenario (layout) per trial seed
        for trial_i in range(int(args.n_trials)):
            seed = seed0 + trial_i
            base_spec = _make_base_scenario(family, seed=seed, params=scenario_params.get(family, {}))
            spec = _apply_condition(base_spec, hvac_theta_true=hvac, doors_pattern_id=doors, seed=seed)

            # Apply experiment-specific modifications
            extra_exp: Dict[str, Any] = {}
            if args.exp == "e2" and door_mismatch:
                spec = _add_door_mismatch(spec, mismatch_seed=seed, assumed_pattern=door_assumed_pattern)
                extra_exp["door_mismatch"] = 1
            if args.exp == "e3":
                # risk parameter sweeps stored in policy cfg, but also record in exp meta
                extra_exp["risk_sweep"] = {"E_max": float(E_max), "delta": float(delta)}
            if args.exp == "e4":
                extra_exp["localization_criteria"] = {"p_thr_true_room": float(p_thr), "credible_volume_thr": float(cred_thr)}
            if args.exp == "e5":
                extra_exp["gas_truth"] = gas_truth
                extra_exp["gtb_cache_path"] = gtb_cache_path
                extra_exp["gtb_k"] = int(gtb_k)
            if args.exp == "e6" and null_leak:
                spec["leak"]["enabled"] = 0
                extra_exp["include_no_leak_hypothesis"] = 1 if include_no_leak_hypothesis else 0

            # Now run each method
            for method in methods:
                trial_id = f"{method}__{family}__{hvac}__{doors}__E{float(E_max):.2f}__d{float(delta):.2f}__p{float(p_thr):.2f}__c{float(cred_thr):.2f}__i{trial_i:03d}__seed{seed}"

                out_dir = base_dir / trial_id

                # Resume logic
                if args.resume and (out_dir / "summary.json").exists() and not args.force:
                    continue

                # Update policy cfg for E3 (E_max/delta)
                policies_local = copy.deepcopy(policies)
                if method in policies_local and args.exp == "e3":
                    policies_local[method]["E_max"] = float(E_max)
                    policies_local[method]["delta"] = float(delta)
                    # Keep risk_quantile consistent if not explicitly set
                    if "risk_quantile" not in policies_local[method]:
                        policies_local[method]["risk_quantile"] = max(0.0, min(1.0, 1.0 - float(delta)))
                
                if args.exp == "e4" and method == "Ours" and trial_i == 0: print("[DEBUG extra_exp]", extra_exp)
                trial_cfg = build_trial_cfg(
                    exp_cfg=cfg,
                    sim_params=sim_params,
                    policies=policies_local,
                    exp_id=args.exp,
                    policy_name=method,
                    seed=seed,
                    scenario_spec=spec,
                    trial_id=trial_id,
                    extra_exp=extra_exp,
                )

                run_trial(trial_cfg, backend_name=args.backend, headless=args.headless, with_go2=args.with_go2, out_dir=str(out_dir), force=args.force)


if __name__ == "__main__":
    main()
