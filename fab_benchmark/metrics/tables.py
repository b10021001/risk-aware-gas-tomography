
"""
Table schemas (v1).

This file centralizes the CSV column orders required by the prompt contract.
"""
from __future__ import annotations

METRICS_COLUMNS_V1 = [
    "exp_id","backend","policy_name","scenario_family","scenario_id","seed",
    "doors_pattern_id","door_mismatch","hvac_theta_true",
    "localized_flag","time_to_localize","final_error",
    "exposure_total","violation_flag","collision_count",
    "wallclock_s","planning_ms_mean","inference_ms_mean",
    "scenario_hash","trace_path",
]

TRACE_COLUMNS_V1 = [
    "time","x","y","z","yaw",
    "v_cmd","w_cmd",
    "y_raw","y_lag","y_meas",
    "hazard","exposure_integral",
    "entropy","credible_set_size","credible_volume","true_room_mass",
    "selected_action_id",
    "collision_flag","collision_count",
    "inference_ms","planning_ms","candidate_count","rollout_count",
]
