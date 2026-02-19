
"""
Run a single trial (v1).

CLI contract:
  python -m fab_benchmark.runners.run_trial --config <path> --backend lite|isaac3d --headless 1 --with_go2 0 --out <dir> [--force]

Exit codes:
  0 success
  2 config/schema validation error
  3 missing asset (e.g., Go2 USD)
  4 backend startup failure (Isaac not available)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

from fab_benchmark.runners.utils import (
    load_yaml, validate_trial_config, ValidationError, stable_hash_from_spec,
    ensure_dir, write_json, write_trace_csv, TRACE_COLUMNS_V1, exit_with
)
from fab_benchmark.backends.lite_backend import LiteBackend
from fab_benchmark.policies.registry import make_policy
from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec, reconstruct_layout_from_spec
from fab_benchmark.sensor.emulator import SensorEmulator
from fab_benchmark.belief.hypothesis_grid import HypothesisGrid
from fab_benchmark.belief.posterior import Posterior
from fab_benchmark.gas.gt_a import GasModelA
from fab_benchmark.gas.gt_b_cache import GTBCache


def _init_pose_from_scene(scene_dict: Dict[str, Any], scenario_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the robot initial pose.

    Priority:
      1) scenario_spec['robot']['spawn']['pos'] (or 'xyz') if provided
      2) scenario_spec['robot']['spawn']['room_id'] center if provided
      3) first room center (legacy fallback)
    """
    spawn = {}
    if isinstance(scenario_spec, dict):
        spawn = (scenario_spec.get("robot", {}) or {}).get("spawn", {}) or {}

    # 1) Direct position
    pos = spawn.get("pos") or spawn.get("xyz")
    yaw = float(spawn.get("yaw", 0.0) or 0.0)
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return {"x": float(pos[0]), "y": float(pos[1]), "z": 0.35, "yaw": yaw}

    # 2) Room center
    rooms = scene_dict.get("rooms", {}) or {}
    rid = spawn.get("room_id") or spawn.get("room")
    if rid and rid in rooms and "center" in rooms[rid]:
        cx, cy = rooms[rid]["center"]
        return {"x": float(cx), "y": float(cy), "z": 0.35, "yaw": yaw}

    # 3) Legacy fallback: first room center
    if rooms:
        rid0 = sorted(rooms.keys())[0]
        cx, cy = rooms[rid0]["center"]
        return {"x": float(cx), "y": float(cy), "z": 0.35, "yaw": yaw}

    return {"x": 0.5, "y": 0.5, "z": 0.35, "yaw": yaw}



def run_trial(cfg: Dict[str, Any], backend_name: str, headless: int, with_go2: int, out_dir: str, force: bool = False) -> None:
    print("[DEBUG run_trial keys]", list(cfg.get("experiment", {}).keys()))

    # Validate config
    try:
        validate_trial_config(cfg, ctx="trial_config")
    except ValidationError as e:
        exit_with(2, f"[config validation error] {e}")

    sim_params = cfg["sim_params"]
    exp = cfg["experiment"]
    policies_cfg = cfg["policies"]

    # Enforce headless flag consistency (fixed defaults expect 1)
    if int(headless) != int(sim_params["isaac"]["headless"]):
        exit_with(2, f"CLI --headless {headless} must match sim_params.isaac.headless {sim_params['isaac']['headless']}")

    scenario_spec = exp["scenario_spec"]
    scenario_hash = stable_hash_from_spec(scenario_spec)

    # Prepare output dir
    out = Path(out_dir)
    if out.exists() and not force:
        # allow if empty or already completed?
        if (out / "summary.json").exists():
            exit_with(2, f"Output dir already contains summary.json; use --force to overwrite: {out}")
    ensure_dir(out)

    trace_path = str(out / "trace.csv")
    summary_path = str(out / "summary.json")
    meta_path = str(out / "run_meta.json")
    scene_usd_path = str(out / "scene.usd") if backend_name == "isaac3d" else None

    # Doors mismatch handling (E2): if doors_assumed present, create assumed spec for policy planning only.
    scenario_spec_assumed = scenario_spec
    door_mismatch = 0
    if "doors_assumed" in scenario_spec:
        door_mismatch = 1
        scenario_spec_assumed = dict(scenario_spec)
        scenario_spec_assumed["doors"] = dict(scenario_spec["doors"])
        scenario_spec_assumed["doors"]["pattern_id"] = scenario_spec["doors_assumed"]["pattern_id"]
        scenario_spec_assumed["doors"]["states"] = scenario_spec["doors_assumed"]["states"]

    # Build scene dict for backend
    scene_dict = build_scene_dict_from_scenario_spec(scenario_spec)
    init_pose = _init_pose_from_scene(scene_dict, scenario_spec)

    # Backend instantiate
    backend_meta = {}
    backend = None
    try:
        if backend_name == "lite":
            backend = LiteBackend(sim_params=sim_params)
        elif backend_name == "isaac3d":
            # include go2 path if requested
            if int(with_go2) == 1:
                # allow user to specify path in cfg.sim_params.isaac.go2_usd_path
                go2_path = sim_params["isaac"].get("go2_usd_path", None)
                if not go2_path or not os.path.exists(go2_path):
                    exit_with(3, f"Go2 USD path not found. Set sim_params.isaac.go2_usd_path. Missing: {go2_path}")
            from fab_benchmark.backends.isaac_backend import IsaacBackend
            backend = IsaacBackend(sim_params=sim_params)
        else:
            exit_with(2, f"Unknown backend: {backend_name}")

        scene_info = backend.load_scene(scenario_spec)
        if isinstance(scene_info, dict) and "backend_meta" in scene_info:
            backend_meta = scene_info["backend_meta"]
    except ImportError as e:
        exit_with(4, f"Isaac backend not available: {e}")
    except SystemExit:
        raise
    except Exception as e:
        exit_with(4, f"Backend startup failure: {e}")

    # Reset backend
    backend.reset(seed=int(exp["seed"]), scene_spec=scenario_spec, init_pose=init_pose)

    # Save stage for Isaac
    if backend_name == "isaac3d" and scene_usd_path is not None:
        try:
            if hasattr(backend, "save_stage"):
                backend.save_stage(scene_usd_path)
        except Exception:
            pass

    # Gas truth model selection (E5)
    gas_truth_mode = exp.get("gas_truth", "gt_a")
    gas_a_truth = GasModelA(
        mode_params=scenario_spec["hvac"]["mode_params"],
        topology={"rooms": scene_dict.get("rooms", {}), "doors": scene_dict.get("doors", {})},
    )
    gtb_cache = None
    gtb_k = int(exp.get("gtb_k", 0))
    if gas_truth_mode == "gt_b_cache":
        cache_path = exp.get("gtb_cache_path", None)
        if not cache_path or not os.path.exists(cache_path):
            exit_with(2, f"E5 requires experiment.gtb_cache_path to existing cache npz; missing: {cache_path}")
        gtb_cache = GTBCache(cache_path)

    # Sensor emulator
    sensor = SensorEmulator(
        tau=float(scenario_spec["sensor"]["tau"]),
        sigma=float(scenario_spec["sensor"]["sigma"]),
        drift=scenario_spec["sensor"]["drift"],
        y_safe=0.03,
        seed=int(exp["seed"]),
    )
    
    no_leak_detected_flag = 0

    # Belief / posterior
    include_no_leak = int(exp.get("include_no_leak_hypothesis", 0)) == 1
    grid = HypothesisGrid.build(
        occupancy=scene_dict["occupancy"],
        resolution=float(scene_dict["resolution"]),
        origin=(float(scene_dict["origin"][0]), float(scene_dict["origin"][1])),
        rooms=scene_dict["rooms"],
        cell_stride=int(sim_params["belief"]["cell_stride"]),
        z_layers=[0.2, 1.0, 2.0],
        include_no_leak=include_no_leak,
    )
    use_delay_model = True
    if exp.get("policy_name") == "Ours_NoDelayModel":
        use_delay_model = False
    posterior = Posterior(grid=grid, scenario_spec=scenario_spec, sim_params=sim_params, use_delay_model=use_delay_model)
    posterior.reset(seed=int(exp["seed"]))

    # Policy
    policy_name = exp["policy_name"]
    if policy_name not in policies_cfg:
        exit_with(2, f"policies config missing entry for '{policy_name}'")
    policy_cfg = policies_cfg[policy_name]
    policy = make_policy(policy_name, policy_cfg=policy_cfg, sim_params=sim_params)
    policy.reset(seed=int(exp["seed"]), scenario_spec=scenario_spec_assumed)

    # Localization criteria (E4)
    loc_cfg = exp.get("localization_criteria", {})
    p_thr_true_room = float(loc_cfg.get("p_thr_true_room", 0.50))
    credible_volume_thr = float(loc_cfg.get("credible_volume_thr", 0.80))

    # Main loop
    t_end = float(sim_params["t_end"])
    control_hz = float(sim_params["control_hz"])
    dt_control = 1.0 / control_hz
    dt_physics = float(sim_params["dt_physics"])
    substeps = int(sim_params["substeps"])
    physics_steps_per_control = max(1, int(round(dt_control / dt_physics)))

    theta_true = scenario_spec["hvac"]["theta_true"]
    leak = scenario_spec["leak"]
    leak_enabled = int(leak["enabled"]) == 1
    source_pos = tuple(float(x) for x in leak["pos"])
    q_true = float(leak["q"])
    start_time = float(leak["start_time"])

    # ------------------------------------------------------------------
    # Tier-1 fixed sensor alarms (event-driven, outside the robot).
    # Policy MUST only react to notified Tier-1 sensors.
    # ------------------------------------------------------------------
    tier1_spec = scenario_spec.get("tier1", {}) if isinstance(scenario_spec, dict) else {}
    tier1_enabled = int(tier1_spec.get("enabled", 0)) == 1
    tier1_sensors = tier1_spec.get("sensors", []) if isinstance(tier1_spec, dict) else []
    tier1_alarm_cfg = tier1_spec.get("alarm", {}) if isinstance(tier1_spec, dict) else {}
    tier1_alarm_mode = str(tier1_alarm_cfg.get("mode", "nearest_to_leak"))
    tier1_k = int(tier1_alarm_cfg.get("k", 3))
    tier1_threshold = float(
        tier1_alarm_cfg.get(
            "threshold",
            tier1_alarm_cfg.get(
                "alarm_threshold",
                exp.get("tier1_alarm_threshold", 1e-3),
            ),
        )
    )
    # Determine which Tier-1 sensors are "notify-able" for this scenario.
    tier1_monitor_ids: List[str] = []
    if tier1_enabled and isinstance(tier1_sensors, list) and len(tier1_sensors) > 0:
        if tier1_alarm_mode == "nearest_to_leak" and leak_enabled:
            # Environment-side rule (NOT used by policy directly):
            # only the k sensors closest to the leak are assumed to trigger.
            lx, ly = float(source_pos[0]), float(source_pos[1])
            tmp = []
            for s in tier1_sensors:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id", ""))
                pos = s.get("pos", None)
                if not sid or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                sx, sy = float(pos[0]), float(pos[1])
                tmp.append((math.hypot(sx - lx, sy - ly), sid))
            tmp.sort(key=lambda x: x[0])
            tier1_monitor_ids = [sid for _, sid in tmp[: max(1, tier1_k)]]
        elif tier1_alarm_mode in ("notified", "preselected"):
            notified_ids = tier1_alarm_cfg.get("notified_ids", None)
            if isinstance(notified_ids, list) and len(notified_ids) > 0:
                tier1_monitor_ids = [str(sid) for sid in notified_ids]
            else:
                tier1_monitor_ids = [str(s.get("id", "")) for s in tier1_sensors[: max(1, tier1_k)] if isinstance(s, dict)]
        else:
            # Default: monitor all tier1 sensors (but still only latch up to K alarms).
            tier1_monitor_ids = [str(s.get("id", "")) for s in tier1_sensors if isinstance(s, dict)]

    tier1_alarm_state: Dict[str, float] = {}  # sid -> first_trigger_time

    exposure_integral = 0.0
    # Reach evaluation accumulators
    min_dist_to_leak = float('inf')
    reach_flag = 0
    time_to_reach = None
    localized_flag = 0
    time_to_localize = None

    rows = []
    wallclock_t0 = time.perf_counter()

    last_action = {"type":"velocity","v":0.0,"w":0.0,"action_id":"init"}
    selected_action_id = last_action["action_id"]

    stop_reason: Optional[str] = None
    for step in range(int(round(t_end * control_hz)) + 1):
        t = step * dt_control

        state = backend.get_state()
        pose = {"x": state["x"], "y": state["y"], "z": state["z"], "yaw": state["yaw"]}

        # truth gas
        if not leak_enabled:
            y_raw = 0.0
        else:
            if gas_truth_mode == "gt_a":
                y_raw = gas_a_truth.query(pose["x"], pose["y"], pose["z"], t, theta_true, source_pos, q_true, start_time)
            else:
                # GT-B cache ignores theta and uses canonical source
                assert gtb_cache is not None
                y_raw = gtb_cache.query(pose["x"], pose["y"], pose["z"], t, source_pos, k=gtb_k)

        meas = sensor.step(y_raw=y_raw, dt=dt_control)

        # Inference
        belief_summary, inference_ms = posterior.update(t=t, pose=pose, measurement=meas)

        # --- Tier-1 alarms (fixed sensors) ---
        # Only include sensors that are configured as "notified" (monitor_ids).
        alarms_out = []
        tier1_y_by_id = {}  # sid -> y_s at current t
        if tier1_enabled and leak_enabled and (t >= start_time) and tier1_monitor_ids:
            # For monitored sensors, trigger alarm if their *raw* concentration crosses the threshold.
            # We query the same gas truth model at the sensor's location.
            # Note: these are fixed environmental sensors (Tier-1), not the robot's onboard sensor.
            by_id = {str(s.get("id")): s for s in tier1_sensors if isinstance(s, dict)}
            for sid in tier1_monitor_ids:
                s = by_id.get(str(sid), None)
                if s is None:
                    continue
                spos = s.get("pos", None)
                if not isinstance(spos, (list, tuple)) or len(spos) < 3:
                    continue
                sx, sy, sz = float(spos[0]), float(spos[1]), float(spos[2])
                if gas_truth_mode == "gt_a":
                    y_s = gas_a_truth.query(sx, sy, sz, t, theta_true, source_pos, q_true, start_time)
                else:
                    assert gtb_cache is not None
                    y_s = gtb_cache.query(sx, sy, sz, t, source_pos, k=gtb_k)
                try:
                    tier1_y_by_id[str(sid)] = float(y_s)
                except Exception:
                    tier1_y_by_id[str(sid)] = float("nan")
                if (float(y_s) >= tier1_threshold) and (str(sid) not in tier1_alarm_state):
                    tier1_alarm_state[str(sid)] = float(t)

            # Build alarms list (active alarms)
            for sid, t0 in sorted(tier1_alarm_state.items(), key=lambda kv: kv[1]):
                s = next((ss for ss in tier1_sensors if isinstance(ss, dict) and str(ss.get("id")) == sid), None)
                if s is None:
                    continue
                pos3 = s.get("pos", [float("nan"), float("nan"), float("nan")])
                alarms_out.append({"id": sid, "pos": [float(pos3[0]), float(pos3[1]), float(pos3[2])], "t": float(t0)})

        belief_summary = dict(belief_summary)
        belief_summary["tier1"] = {"enabled": int(tier1_enabled), "threshold": float(tier1_threshold), "alarms": alarms_out, "y_by_id": tier1_y_by_id}

        # --- MAP estimate for per-step trace and for policy final-approach ---
        est_pos_step, est_room_step = posterior.map_estimate()
        belief_summary["map_estimate"] = {"room_id": str(est_room_step), "pos": [float(x) for x in est_pos_step]}
        # Convenience: top room mass
        rm = belief_summary.get("room_mass", {})
        if isinstance(rm, dict) and rm:
            max_room_id = max(rm.keys(), key=lambda k: float(rm.get(k, 0.0)))
            max_room_mass = float(rm.get(max_room_id, 0.0))
        else:
            max_room_id = ""
            max_room_mass = float("nan")
        belief_summary["max_room_id"] = str(max_room_id)
        belief_summary["max_room_mass"] = float(max_room_mass)
        policy.record_inference_ms(inference_ms)

        # Policy action (measure planning time per-step)
        _pt0 = time.perf_counter()
        policy_action = policy.step(t=t, pose=pose, measurement=meas, belief_summary=belief_summary)
        _pt1 = time.perf_counter()
        planning_ms_step = float((_pt1 - _pt0) * 1000.0)
        policy.record_planning_ms(planning_ms_step)
        # Clip
        v_max = float(sim_params["robot"]["v_max"])
        w_max = float(sim_params["robot"]["w_max"])
        v = max(-v_max, min(v_max, float(policy_action.get("v", 0.0))))
        w = max(-w_max, min(w_max, float(policy_action.get("w", 0.0))))
        selected_action_id = str(policy_action.get("action_id", "na"))

        policy_mode = str(policy_action.get("policy_mode", ""))

        # Goal (support both goal_xy and goal_x/goal_y)
        goal_x = float("nan"); goal_y = float("nan")
        goal_xy = policy_action.get("goal_xy", None)
        if isinstance(goal_xy, (list, tuple)) and len(goal_xy) >= 2:
            try:
                goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
            except Exception:
                goal_x, goal_y = float("nan"), float("nan")
        else:
            gx = policy_action.get("goal_x", None)
            gy = policy_action.get("goal_y", None)
            if gx is not None and gy is not None:
                try:
                    goal_x, goal_y = float(gx), float(gy)
                except Exception:
                    goal_x, goal_y = float("nan"), float("nan")

        # Focus target (support both focus dict and focus_id/x/y)
        focus_id = ""; focus_x = float("nan"); focus_y = float("nan")
        focus_info = policy_action.get("focus", None)
        if isinstance(focus_info, dict):
            try:
                focus_id = str(focus_info.get("id", ""))
                focus_x = float(focus_info.get("x", float("nan")))
                focus_y = float(focus_info.get("y", float("nan")))
            except Exception:
                focus_id = ""; focus_x = float("nan"); focus_y = float("nan")
        else:
            fid = policy_action.get("focus_id", "")
            fx = policy_action.get("focus_x", None)
            fy = policy_action.get("focus_y", None)
            if str(fid):
                focus_id = str(fid)
                try:
                    focus_x = float(fx) if fx is not None else float("nan")
                    focus_y = float(fy) if fy is not None else float("nan")
                except Exception:
                    focus_x = float("nan"); focus_y = float("nan")
        action = {"type":"velocity","v":v,"w":w,"action_id":selected_action_id}

        backend.apply_action(action)

        # Step physics
        for _ in range(physics_steps_per_control):
            backend.step(dt_physics, substeps)

        # Exposure integral (using hazard from meas which includes noise/drift)
        exposure_integral += float(meas["hazard"]) * dt_control

        # Localization check
        true_room_mass = float(belief_summary.get("true_room_mass", 0.0))
        credible_volume = float(belief_summary["credible_volume"])

        if leak_enabled:
            if localized_flag == 0 and (credible_volume <= credible_volume_thr) and (true_room_mass >= p_thr_true_room):
                localized_flag = 1
                time_to_localize = float(t)
        else:
            # Null-leak detection (E6): if no-leak hypothesis is used, detect when its mass is high
            if include_no_leak and no_leak_detected_flag == 0:
                nm = float(belief_summary.get("room_mass", {}).get("no_leak", 0.0))
                if nm >= 0.90:
                    no_leak_detected_flag = 1

        # Trace row
        # Reach metrics: distance to leak in XY (for evaluation / demos)
        dx_leak = float(pose["x"]) - float(source_pos[0])
        dy_leak = float(pose["y"]) - float(source_pos[1])
        dist_leak = math.sqrt(dx_leak*dx_leak + dy_leak*dy_leak)
        if dist_leak < min_dist_to_leak:
            min_dist_to_leak = float(dist_leak)
        reach_radius = float(sim_params.get("eval", {}).get("reach_radius", 1.0))
        if reach_flag == 0 and dist_leak <= reach_radius:
            reach_flag = 1
            time_to_reach = float(t)
        row = {
            "time": float(t),
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "yaw": float(pose["yaw"]),
            "v_cmd": float(v),
            "w_cmd": float(w),
            "y_raw": float(meas["y_raw"]),
            "y_lag": float(meas["y"]),
            "y_meas": float(meas["y_meas"]),
            "hazard": float(meas["hazard"]),
            "exposure_integral": float(exposure_integral),
            "entropy": float(belief_summary["entropy"]),
            "credible_set_size": int(belief_summary["credible_set_size"]),
            "credible_volume": float(belief_summary["credible_volume"]),
            "true_room_mass": float(belief_summary.get("true_room_mass", float("nan"))),
            "selected_action_id": selected_action_id,
            "collision_flag": int(state.get("collision", 0)),
            "collision_count": int(state.get("collision_count", 0)),
            "inference_ms": float(inference_ms),
            "planning_ms": float(planning_ms_step),
            "candidate_count": int(policy.get_budget_stats()["candidates"]),
            "rollout_count": int(policy.get_budget_stats()["rollouts"]),

            # Extended trace
            "policy_mode": policy_mode,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "focus_id": focus_id,
            "focus_x": focus_x,
            "focus_y": focus_y,
            "tier1_alarm_count": int(len(alarms_out)) if tier1_enabled else 0,
            "tier1_alarm_ids": "|".join([str(a.get('id','')) for a in alarms_out]) if tier1_enabled else "",
            "tier1_alarms": json.dumps(alarms_out, ensure_ascii=False) if tier1_enabled else "[]",
            "dist_to_leak": float(dist_leak),
            "reach_now": int(dist_leak <= reach_radius),
            "tier1_y_max": float(max([v for v in tier1_y_by_id.values() if v == v], default=float('nan'))) if tier1_enabled else float('nan'),
            "pred_x": float(est_pos_step[0]) if len(est_pos_step) >= 1 else float("nan"),
            "pred_y": float(est_pos_step[1]) if len(est_pos_step) >= 2 else float("nan"),
            "pred_z": float(est_pos_step[2]) if len(est_pos_step) >= 3 else float("nan"),
            "pred_room": str(est_room_step),
            "max_room_id": str(max_room_id),
            "max_room_mass": float(max_room_mass),
        }
        rows.append(row)

        # Early stop: if policy declares done, stop the episode now.
        if int(policy_action.get("done", 0)) == 1 or selected_action_id.startswith("stop_"):
            stop_reason = str(policy_action.get("stop_reason", "policy_done"))
            break

    wallclock_t1 = time.perf_counter()
    wallclock_s = float(wallclock_t1 - wallclock_t0)

    # Final estimate + metrics
    est_pos, est_room = posterior.map_estimate()
    true_pos = source_pos
    if leak_enabled and all(np.isfinite(est_pos)):
        final_error = float(math.sqrt((est_pos[0]-true_pos[0])**2 + (est_pos[1]-true_pos[1])**2 + (est_pos[2]-true_pos[2])**2))
    else:
        final_error = None

    E_max = float(policy_cfg.get("E_max", 1.50)) if isinstance(policy_cfg, dict) else 1.50
    violation_flag = int(exposure_integral > E_max + 1e-12)
    collision_count = int(rows[-1]["collision_count"]) if rows else 0

    # Summaries
    summary = {
        "version": "v1",
        "localized_flag": int(localized_flag),
        "time_to_localize": time_to_localize,
        "final_estimate": {"room_id": est_room, "pos": [float(x) for x in est_pos]},
        "final_error": final_error,
        "exposure_total": float(exposure_integral),
        "violation_flag": int(violation_flag),
        "collision_count": int(collision_count),
        "wallclock_s": float(wallclock_s),
        "planning_ms_mean": float(policy.get_budget_stats()["planning_ms_mean"]),
        "inference_ms_mean": float(policy.get_budget_stats()["inference_ms_mean"]),
    "reach_flag": int(reach_flag),
    "time_to_reach": (None if time_to_reach is None else float(time_to_reach)),
    "min_dist_to_leak": (float("nan") if (min_dist_to_leak == float("inf")) else float(min_dist_to_leak)),
    "steps_executed": int(len(rows)),
    "stop_reason": (None if stop_reason is None else str(stop_reason)),
    }
    # Append-only fields for E6
    if include_no_leak:
        summary["include_no_leak_hypothesis"] = 1

    # Run meta (must be written)
    run_meta = {
        "version": "v1",
        "experiment": {
            "exp_id": exp["exp_id"],
            "seed": int(exp["seed"]),
            "trial_id": str(exp.get("trial_id", out.name)),
            "policy_name": policy_name,
            "door_mismatch": int(door_mismatch),
            # ✅ NEW: persist experiment-specific metadata for E2/E3/E4/E5/E6
            "extra_exp": {
            # E4
            "localization_criteria": cfg["experiment"].get("localization_criteria", {}),
            # E3
            "risk_sweep": cfg["experiment"].get("risk_sweep", {}),
            # E5
            "gas_truth": cfg["experiment"].get("gas_truth", ""),
            "gtb_cache_path": cfg["experiment"].get("gtb_cache_path", ""),
            "gtb_k": cfg["experiment"].get("gtb_k", None),
            # E6 (if present)
            "include_no_leak_hypothesis": cfg["experiment"].get("include_no_leak_hypothesis", None),
            },

            # ✅ NEW: keep config path if present (optional)
            "config_path": str(exp.get("config_path", "")),
        },

        "backend": {"name": backend_name, "isaac_headless": int(headless), "with_go2": int(with_go2)},
        "sim_params": sim_params,
        "scenario_spec": scenario_spec,
        "scenario_hash": scenario_hash,
        "policy": {"policy_name": policy_name, "policy_cfg": policy_cfg},
        "paths": {"trace_csv": trace_path, "summary_json": summary_path, "scene_usd": scene_usd_path},
        "backend_meta": backend_meta,
    }

    # Write files
    write_trace_csv(trace_path, TRACE_COLUMNS_V1, rows)
    write_json(summary_path, summary)
    write_json(meta_path, run_meta)

    # Close backend
    try:
        backend.close()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--backend", required=True, choices=["lite","isaac3d"])
    ap.add_argument("--headless", required=True, type=int)
    ap.add_argument("--with_go2", required=True, type=int)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_trial(cfg, backend_name=args.backend, headless=args.headless, with_go2=args.with_go2, out_dir=args.out, force=args.force)


if __name__ == "__main__":
    main()