"""
fab_benchmark.runners.utils

Full runner utility surface expected by run_trial.py and run_experiment.py.

Notes
- Configs are JSON-style YAML (valid JSON); parse with stdlib json (no PyYAML dependency).
- Provides CSV/JSON writers used by run_trial.
- Persists experiment.extra_exp for E2/E3/E4/E5/E6 auditing/post-processing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Trace CSV columns (v1 contract; exact order)
TRACE_COLUMNS_V1 = [
    "time",
    "x",
    "y",
    "z",
    "yaw",
    "v_cmd",
    "w_cmd",
    "y_raw",
    "y_lag",
    "y_meas",
    "hazard",
    "exposure_integral",
    "entropy",
    "credible_set_size",
    "credible_volume",
    "true_room_mass",
    "selected_action_id",
    "collision_flag",
    "collision_count",
    "inference_ms",
    "planning_ms",
    "candidate_count",
    "rollout_count",
    "policy_mode",
    "goal_x",
    "goal_y",
    "pred_x",
    "pred_y",
    "focus_id",
    "focus_x",
    "focus_y",
    "tier1_alarm_count",
    "tier1_alarm_ids",
    "tier1_y_max",
    "tier1_alarms",
    "dist_to_leak",
    "reach_now",
]



class ValidationError(Exception):
    pass


def exit_with(code: int, msg: str) -> None:
    sys.stderr.write(str(msg).rstrip() + "\n")
    raise SystemExit(int(code))


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a JSON-compatible YAML config. Benchmark configs are written in JSON syntax.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Config parse error (expected JSON-style YAML). File={path}. {e}") from e
    if isinstance(cfg, dict):
        cfg["__config_path__"] = str(p)
    return cfg


def validate_experiment_config(cfg: Dict[str, Any], ctx: str = "experiment_config", exp_cli: Optional[str] = None) -> None:
    if not isinstance(cfg, dict):
        raise ValidationError(f"{ctx}: config must be a dict")
    for k in ["version", "sim_params", "experiment", "policies"]:
        if k not in cfg:
            raise ValidationError(f"{ctx}: missing required key '{k}'")
    if str(cfg.get("version")) != "v1":
        raise ValidationError(f"{ctx}: version must be 'v1'")

    sim = cfg["sim_params"]
    if not isinstance(sim, dict):
        raise ValidationError(f"{ctx}:sim_params must be a dict")
    for k in ["t_end", "dt_physics", "substeps", "control_hz", "robot", "belief", "isaac"]:
        if k not in sim:
            raise ValidationError(f"{ctx}:sim_params: missing required key '{k}'")

    exp = cfg["experiment"]
    if not isinstance(exp, dict):
        raise ValidationError(f"{ctx}:experiment must be a dict")

    exp_id = exp.get("exp_id", None)
    if exp_id is None:
        raise ValidationError(f"{ctx}:experiment: missing required key 'exp_id'")
    if exp_cli is not None and str(exp_cli) != str(exp_id):
        raise ValidationError(f"{ctx}:experiment.exp_id='{exp_id}' must match CLI --exp '{exp_cli}'")

    for k in ["trials_per_condition", "methods", "condition_sweep", "seed0"]:
        if k not in exp:
            raise ValidationError(f"{ctx}:experiment: missing required key '{k}'")
    if not isinstance(exp["methods"], list) or len(exp["methods"]) == 0:
        raise ValidationError(f"{ctx}:experiment.methods must be a non-empty list")
    if not isinstance(exp["condition_sweep"], dict):
        raise ValidationError(f"{ctx}:experiment.condition_sweep must be a dict")

    if not isinstance(cfg["policies"], dict):
        raise ValidationError(f"{ctx}:policies must be a dict")


def validate_trial_config(cfg: Dict[str, Any], ctx: str = "trial_config") -> None:
    if not isinstance(cfg, dict):
        raise ValidationError(f"{ctx}: config must be a dict")
    for k in ["version", "sim_params", "experiment", "policies"]:
        if k not in cfg:
            raise ValidationError(f"{ctx}: missing required key '{k}'")
    if str(cfg.get("version")) != "v1":
        raise ValidationError(f"{ctx}: version must be 'v1'")

    sim = cfg["sim_params"]
    if not isinstance(sim, dict):
        raise ValidationError(f"{ctx}:sim_params must be a dict")
    for k in ["t_end", "dt_physics", "substeps", "control_hz", "robot", "belief", "isaac"]:
        if k not in sim:
            raise ValidationError(f"{ctx}:sim_params: missing required key '{k}'")

    exp = cfg["experiment"]
    if not isinstance(exp, dict):
        raise ValidationError(f"{ctx}:experiment must be a dict")
    for k in ["exp_id", "seed", "policy_name", "scenario_spec"]:
        if k not in exp:
            raise ValidationError(f"{ctx}:experiment: missing required key '{k}'")

    if not isinstance(cfg["policies"], dict):
        raise ValidationError(f"{ctx}:policies must be a dict")


def stable_hash_from_spec(spec: Dict[str, Any]) -> str:
    s = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def write_trace_csv(path: str, *args) -> None:
    """
    Backward-compatible trace.csv writer.

    Supported call signatures:
      1) write_trace_csv(path, rows)
      2) write_trace_csv(path, columns, rows)

    If columns are not provided, uses TRACE_COLUMNS_V1.
    Each row may be a dict (preferred) or a sequence aligned to columns.
    """
    # Parse args
    if len(args) == 1:
        columns = TRACE_COLUMNS_V1
        rows = args[0]
    elif len(args) == 2:
        columns = args[0]
        rows = args[1]
    else:
        raise TypeError("write_trace_csv(path, rows) or write_trace_csv(path, columns, rows)")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Normalize rows iterable
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get(c, "") for c in columns])
            else:
                # assume already ordered
                w.writerow(list(r))



def build_trial_cfg(
    exp_cfg: Dict[str, Any],
    sim_params: Dict[str, Any],
    policies: Dict[str, Any],
    exp_id: str,
    policy_name: str,
    seed: int,
    scenario_spec: Dict[str, Any],
    trial_id: str,
    extra_exp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if extra_exp is None:
        extra_exp = {}
    if policy_name not in policies:
        policies = dict(policies)
        policies[policy_name] = {}

    scenario_hash = stable_hash_from_spec(scenario_spec)

    cfg: Dict[str, Any] = {
        "version": "v1",
        "sim_params": sim_params,
        "experiment": {
            "exp_id": str(exp_id),
            "seed": int(seed),
            "trial_id": str(trial_id),
            "policy_name": str(policy_name),
            "scenario_spec": scenario_spec,
            "scenario_hash": scenario_hash,
            "extra_exp": extra_exp,
            "config_path": str(exp_cfg.get("__config_path__", "")) if isinstance(exp_cfg, dict) else "",
        },
        "policies": policies,
    }
    return cfg