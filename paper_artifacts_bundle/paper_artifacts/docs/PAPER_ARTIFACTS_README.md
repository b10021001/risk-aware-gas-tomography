# PAPER ARTIFACTS (GT generator parameters + per-run logs)

This folder (`paper_artifacts/`) contains **lightweight, per-run artifacts** for auditability and regeneration of reported metrics/figures **without rerunning Isaac Sim**.

## Contents
- `paper_artifacts/configs/paper_E1_ID_config.yaml` : E1 (ID) experiment knobs + GT-generator parameter ranges
- `paper_artifacts/configs/paper_E2_OOD_config.yaml` : E2 (OOD) experiment knobs + GT-generator parameter ranges
- `paper_artifacts/configs/paper_E5_mismatch_tiers.yaml` : E5 mismatch tier definitions (golden set)

- `paper_artifacts/data/logs/paper_manifest_template.csv` : manifest schema mapping (experiment, method, trial_id) to file paths + seeds
- `paper_artifacts/data/logs/paper_gt_episode_template.json` : **per-trial GT episode record** schema (seed, mode/strength, doors, leak params, GT-only effects)
- `paper_artifacts/data/logs/paper_trace_template.csv` : time-stamped pose + hazard measurements schema
- `paper_artifacts/data/logs/paper_plan_template.csv` : evaluated candidates + selected segment per replanning cycle schema
- `paper_artifacts/data/logs/paper_posterior_summary_template.csv` : posterior summary schema (final μ/Σ, credible volume, etc.)
- `paper_artifacts/data/logs/paper_timing_template.csv` : module-level latency breakdown schema

- `paper_artifacts/tools/paper_regenerate_tables_figs.py` : script to regenerate tables/figures from per-run logs + manifest (no simulator rerun)

## How to use
1) Populate `paper_artifacts/data/logs/manifest.csv` using the provided template.
2) For each trial, provide a directory with:
   - `gt_episode.json`, `trace.csv`, `plan.csv`, `posterior_summary.csv`, `timing.csv`
3) Run:
```bash
python3 paper_artifacts/tools/paper_regenerate_tables_figs.py --manifest paper_artifacts/data/logs/manifest.csv --make_fig5
```

Large binary simulation assets (facility-scale digital twin) are intentionally excluded due to size/licensing constraints.
