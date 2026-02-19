
"""
Aggregation utilities (v1): compute metrics.csv and metrics_summary.csv.

Dependency policy: stdlib + numpy only.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import glob
import os
import math
import json
from pathlib import Path

import numpy as np


METRICS_COLUMNS_V1 = [
    "exp_id","backend","policy_name","scenario_family","scenario_id","seed",
    "doors_pattern_id","door_mismatch","hvac_theta_true",
    "localized_flag","time_to_localize","final_error",
    "exposure_total","violation_flag","collision_count",
    "wallclock_s","planning_ms_mean","inference_ms_mean",
    "scenario_hash","trace_path","risk_E_max","risk_delta",
    "reach_flag","time_to_reach","min_dist_to_leak",
]



def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_trial_dirs(glob_pattern: str) -> List[str]:
    # Expand glob, include dirs that contain summary.json
    candidates = glob.glob(glob_pattern, recursive=True)
    out = []
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "summary.json")):
            out.append(p)
    return sorted(out)


def bootstrap_ci(x: np.ndarray, seed: int = 0, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float,float,float]:
    """
    Returns (mean, lo, hi) bootstrap CI for mean.
    Deterministic by seed.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = x.size
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means.append(float(np.mean(x[idx])))
    means = np.array(means, dtype=np.float64)
    lo = float(np.quantile(means, alpha/2.0))
    hi = float(np.quantile(means, 1.0-alpha/2.0))
    return float(np.mean(x)), lo, hi


def aggregate_trials(trial_dirs: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for d in trial_dirs:
        summary_p = os.path.join(d, "summary.json")
        meta_p = os.path.join(d, "run_meta.json")
        if not os.path.exists(summary_p) or not os.path.exists(meta_p):
            continue
        summary = read_json(summary_p)
        meta = read_json(meta_p)
        if summary.get("version") != "v1" or meta.get("version") != "v1":
            continue
        row = {}
        for k in METRICS_COLUMNS_V1:
            row[k] = None
        # Fill from meta
        row["exp_id"] = meta["experiment"]["exp_id"]
        row["backend"] = meta["backend"]["name"]
        row["policy_name"] = meta["policy"]["policy_name"]
        row["scenario_family"] = meta["scenario_spec"]["scenario_family"]
        row["scenario_id"] = meta["scenario_spec"]["scenario_id"]
        row["seed"] = meta["experiment"]["seed"]
        row["doors_pattern_id"] = meta["scenario_spec"]["doors"]["pattern_id"]
        row["door_mismatch"] = int(meta["experiment"].get("door_mismatch", 0))
        row["hvac_theta_true"] = meta["scenario_spec"]["hvac"]["theta_true"]
        row["scenario_hash"] = meta["scenario_hash"]
        row["trace_path"] = meta["paths"]["trace_csv"]

        # E3: risk sweep params (E_max/delta) for tradeoff_grid/pareto_front
        try:
            pcfg = (meta.get("policy", {}).get("policy_cfg", {}) or {})
            if row.get("exp_id") == "e3":
                row["risk_E_max"] = pcfg.get("E_max", None)
                row["risk_delta"] = pcfg.get("delta", None)
        except Exception:
            pass

        # E3: risk sweep parameters (written by run_experiment.py as meta["experiment"]["extra_exp"]["risk_sweep"])
        try:
            rs = (meta.get("experiment", {}).get("extra_exp", {}) or {}).get("risk_sweep", {}) or {}
            if rs:
                row["risk_E_max"] = rs.get("E_max", None)
                row["risk_delta"] = rs.get("delta", None)
        except Exception:
            pass


        # Fill from summary
        row["localized_flag"] = int(summary.get("localized_flag", 0))
        row["time_to_localize"] = summary.get("time_to_localize", None)
        row["final_error"] = summary.get("final_error", None)
        row["exposure_total"] = summary.get("exposure_total", None)
        row["violation_flag"] = int(summary.get("violation_flag", 0))
        row["collision_count"] = summary.get("collision_count", None)
        row["wallclock_s"] = summary.get("wallclock_s", None)
        row["planning_ms_mean"] = summary.get("planning_ms_mean", None)
        row["inference_ms_mean"] = summary.get("inference_ms_mean", None)
        row["reach_flag"] = int(summary.get("reach_flag", 0) or 0)
        row["time_to_reach"] = summary.get("time_to_reach")
        row["min_dist_to_leak"] = summary.get("min_dist_to_leak")

        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (exp_id, backend, policy_name, scenario_family, hvac_theta_true, doors_pattern_id)
    and compute CI for key metrics.
    """
    # prepare groups
    def key(r):
        return (r["exp_id"], r["backend"], r["policy_name"], r["scenario_family"], r["hvac_theta_true"], r["doors_pattern_id"])
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: kv[0]):
        exp_id, backend, policy_name, scenario_family, hvac, doors = k
        # arrays
        loc = np.array([float(r["localized_flag"]) for r in rs], dtype=np.float64)
        tloc = np.array([float(r["time_to_localize"]) if r["time_to_localize"] is not None else float("nan") for r in rs], dtype=np.float64)
        ferr = np.array([float(r["final_error"]) if r["final_error"] is not None else float("nan") for r in rs], dtype=np.float64)
        expo = np.array([float(r["exposure_total"]) if r["exposure_total"] is not None else float("nan") for r in rs], dtype=np.float64)
        viol = np.array([float(r["violation_flag"]) for r in rs], dtype=np.float64)

        # CIs
        loc_mean, loc_lo, loc_hi = bootstrap_ci(loc, seed=0)
        # time_to_localize only for localized
        tloc_localized = tloc[np.isfinite(tloc)]
        t_mean, t_lo, t_hi = bootstrap_ci(tloc_localized, seed=1) if tloc_localized.size>0 else (float("nan"),float("nan"),float("nan"))
        ferr_valid = ferr[np.isfinite(ferr)]
        ferr_mean, ferr_lo, ferr_hi = bootstrap_ci(ferr_valid, seed=2) if ferr_valid.size>0 else (float("nan"),float("nan"),float("nan"))
        expo_valid = expo[np.isfinite(expo)]
        expo_mean, expo_lo, expo_hi = bootstrap_ci(expo_valid, seed=3) if expo_valid.size>0 else (float("nan"),float("nan"),float("nan"))
        viol_mean, viol_lo, viol_hi = bootstrap_ci(viol, seed=4)

        out.append({
            "exp_id": exp_id,
            "backend": backend,
            "policy_name": policy_name,
            "scenario_family": scenario_family,
            "hvac_theta_true": hvac,
            "doors_pattern_id": doors,
            "n": int(len(rs)),
            "localized_rate_mean": float(loc_mean),
            "localized_rate_ci95_lo": float(loc_lo),
            "localized_rate_ci95_hi": float(loc_hi),
            "time_to_localize_mean": float(t_mean),
            "time_to_localize_ci95_lo": float(t_lo),
            "time_to_localize_ci95_hi": float(t_hi),
            "final_error_mean": float(ferr_mean),
            "final_error_ci95_lo": float(ferr_lo),
            "final_error_ci95_hi": float(ferr_hi),
            "exposure_total_mean": float(expo_mean),
            "exposure_total_ci95_lo": float(expo_lo),
            "exposure_total_ci95_hi": float(expo_hi),
            "violation_rate_mean": float(viol_mean),
            "violation_rate_ci95_lo": float(viol_lo),
            "violation_rate_ci95_hi": float(viol_hi),
        })
    # E3 extra outputs: tradeoff_grid.csv and pareto_front.csv (prompt-required)
    try:
        if any(str(r.get("exp_id","")) == "e3" for r in rows):
            out_dir = os.path.join("results", "exp_e3_main")
            os.makedirs(out_dir, exist_ok=True)
            _e3_write_tradeoff_and_pareto(rows, out_dir)
    except Exception:
        pass

    return out

def _e3_write_tradeoff_and_pareto(rows, out_dir):
    import csv, math, random
    from statistics import median

    REQ_TRADEOFF_COLS = [
        "E_max","delta","method","median_time","median_error","viol_rate","mean_exposure",
        "ci95_time_low","ci95_time_high","ci95_error_low","ci95_error_high","ci95_exposure_low","ci95_exposure_high"
    ]
    REQ_PARETO_COLS = ["method","E_max","delta","median_time","median_error","viol_rate","mean_exposure"]

    def _bootstrap_ci(values, B=2000, seed=123):
        vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(vals) == 0:
            return (float("nan"), float("nan"))
        rnd = random.Random(seed)
        n = len(vals)
        stats = []
        for _ in range(B):
            samp = [vals[rnd.randrange(n)] for _ in range(n)]
            stats.append(sum(samp) / n)
        stats.sort()
        lo = stats[int(0.025 * (B - 1))]
        hi = stats[int(0.975 * (B - 1))]
        return (lo, hi)

    def _nondominated(points):
        nd = []
        for i, a in enumerate(points):
            dominated = False
            for j, b in enumerate(points):
                if i == j:
                    continue
                if (b["median_time"] <= a["median_time"] and
                    b["median_error"] <= a["median_error"] and
                    b["viol_rate"] <= a["viol_rate"] and
                    b["mean_exposure"] <= a["mean_exposure"] and
                    (b["median_time"] < a["median_time"] or
                     b["median_error"] < a["median_error"] or
                     b["viol_rate"] < a["viol_rate"] or
                     b["mean_exposure"] < a["mean_exposure"])):
                    dominated = True
                    break
            if not dominated:
                nd.append(a)
        nd.sort(key=lambda r: (r["method"], r["E_max"], r["delta"]))
        return nd

    # group by (method, E_max, delta)
    groups = {}
    for r in rows:
        if str(r.get("exp_id","")) != "e3":
            continue
        method = r.get("policy_name","UNKNOWN")
        try:
            # risk_sweep is stored in run_meta.extra_exp.risk_sweep by run_experiment.py
            E_max = float(r.get("risk_E_max"))
            delta = float(r.get("risk_delta"))
        except Exception:
            # if not present, skip (can't build grid)
            continue

        key = (method, E_max, delta)
        g = groups.setdefault(key, {"times": [], "errors": [], "viols": [], "expos": []})
        g["times"].append(float(r.get("time_to_localize", float("nan"))))
        g["errors"].append(float(r.get("final_error", float("nan"))))
        g["viols"].append(int(r.get("violation_flag", 0)))
        g["expos"].append(float(r.get("exposure_total", float("nan"))))

    trade_rows = []
    for (method, E_max, delta), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        times, errors, viols, expos = g["times"], g["errors"], g["viols"], g["expos"]
        med_t = float(median(times)) if len(times) else float("nan")
        med_e = float(median(errors)) if len(errors) else float("nan")
        viol_rate = float(sum(viols) / len(viols)) if len(viols) else float("nan")
        mean_expo = float(sum(expos) / len(expos)) if len(expos) else float("nan")
        ci_t = _bootstrap_ci(times, B=2000, seed=123)
        ci_e = _bootstrap_ci(errors, B=2000, seed=456)
        ci_x = _bootstrap_ci(expos, B=2000, seed=789)

        trade_rows.append({
            "E_max": E_max,
            "delta": delta,
            "method": method,
            "median_time": med_t,
            "median_error": med_e,
            "viol_rate": viol_rate,
            "mean_exposure": mean_expo,
            "ci95_time_low": ci_t[0],
            "ci95_time_high": ci_t[1],
            "ci95_error_low": ci_e[0],
            "ci95_error_high": ci_e[1],
            "ci95_exposure_low": ci_x[0],
            "ci95_exposure_high": ci_x[1],
        })

    if len(trade_rows) == 0:
        return  # nothing to write

    out_dir = str(out_dir)
    tradeoff_p = os.path.join(out_dir, "tradeoff_grid.csv")
    with open(tradeoff_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQ_TRADEOFF_COLS)
        w.writeheader()
        for tr in trade_rows:
            w.writerow({k: tr[k] for k in REQ_TRADEOFF_COLS})

    pareto = _nondominated([{
        "method": tr["method"],
        "E_max": tr["E_max"],
        "delta": tr["delta"],
        "median_time": tr["median_time"],
        "median_error": tr["median_error"],
        "viol_rate": tr["viol_rate"],
        "mean_exposure": tr["mean_exposure"],
    } for tr in trade_rows])

    pareto_p = os.path.join(out_dir, "pareto_front.csv")
    with open(pareto_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQ_PARETO_COLS)
        w.writeheader()
        for pr in pareto:
            w.writerow({k: pr[k] for k in REQ_PARETO_COLS})
