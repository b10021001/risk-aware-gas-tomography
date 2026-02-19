# risk-aware-gas-tomography

Companion repository for the paper:  
**“Autonomous Risk-Aware Active Gas Tomography via Informative Planning with a Quadruped Robot.”**  
(IEEE Transactions on Robotics submission)

> This repository is provided for public access as a companion artifact.  
> **Note:** The paper does **not** assume public release of the full implementation; this repo focuses on public-facing configs/artifacts and documentation aligned with the repeatability checklist.

---

## 1. Summary

Toxic gas releases in semiconductor fabrication facilities are low-frequency but high-consequence events. After a Tier-1 fixed sensing network raises an alarm, operators need rapid and auditable evidence of (i) the likely leak source region and (ii) the hazard extent, while strictly limiting exposure.  

This project studies an **event-triggered Tier-2 quadruped system (Unitree Go2-class)** for **risk-aware active gas tomography**, coupling:

- **Online Bayesian data assimilation**: maintains a continuous 3D posterior over leak parameters, including source location, and produces uncertainty-aware concentration/risk summaries under realistic sensor dynamics and model mismatch.
- **Receding-horizon informative planning with probabilistic exposure constraints**: selects short-horizon motion segments to accelerate posterior contraction while enforcing chance constraints via a scenario-based empirical-quantile feasibility test.

The evaluation is **simulation-first** using a high-fidelity digital twin workflow (Isaac Sim + sensor emulation), and the paper provides an explicit **repeatability checklist** (module rates, interface contracts, hyperparameters, and logging artifacts).

---

## 2. What this repository contains (public artifact scope)

This repository contains **public-facing artifacts** that support transparency and repeatability, including:

- Experiment / method configuration files (e.g., planning horizons, chance-constraint tolerance δ, risk scenario counts, etc.)
- Tooling utilities used for packaging, bookkeeping, or documentation
- Minimal documentation for how artifacts map to the paper

⚠️ **Not included** (intentionally):
- Large simulation assets, generated results, or bulky caches (to keep the repository lightweight)
- Full end-to-end implementation if it contains environment-/hardware-specific dependencies not suitable for public release

If you need a minimal “paper companion” record, the essential purpose is: **the repo is public, the paper is citable, and the artifact structure is traceable.**

---

## 3. Key contributions (as described in the paper)

1) **Event-triggered risk-aware active gas tomography** for continuous 3D source inference  
2) **Online Bayesian data assimilation** with explicit handling of sensor dynamics and model mismatch  
3) **Risk-constrained informative planning** with interpretable probabilistic exposure feasibility (scenario/quantile test)  
4) **Modular Go2-centric system design + simulation-first evaluation workflow**, with an explicit repeatability checklist

---

## 4. Reported headline result (paper, E1 summary)

In facility-scale simulation experiments (digital twin with sensor emulation), with **600 independent trials per method** and a strict **60 s time cap**, the risk-aware planner reports:

- **Safe-success:** 81.8%  
- **Mission-level exposure violations:** 7.3%  

This substantially improves over informative planning without risk constraints (safe-success 29.0%, violations 67.8%), along with other baselines.

(See paper Table VIII / Fig. 4 for the full comparison.)

---

## 5. Repository structure (current)

- `configs/`  
  Configuration files used to lock experimental settings (planner/inference/risk parameters, seeds, etc.)
- `tools/`  
  Helper scripts/utilities for artifact management and documentation
- `fab_benchmark/`  
  Placeholder directory for benchmark-related artifacts. (Large assets may be excluded; see Section 2.)

> If you add more components later, keep a clean top-level layout (e.g., `docs/`, `figures/`, `scripts/`) and avoid committing large binary files.

---

## 6. Notes on repeatability & transparency

The paper emphasizes **repeatability via explicit interface contracts and configuration-locked evaluation**, including:
- module frequencies (inference vs planning rates),
- planning horizons and candidate budgets,
- chance-constraint tolerance δ,
- risk scenario sample counts,
- determinism controls (seeds),
- and structured logging artifacts for auditing and figure regeneration.

This repository is organized to reflect that philosophy: **configs are first-class artifacts**.

---

## 7. Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@article{risk_aware_gas_tomography_2026,
  title   = {Autonomous Risk-Aware Active Gas Tomography via Informative Planning with a Quadruped Robot},
  author  = {Anonymous Author(s)},
  journal = {IEEE Transactions on Robotics (under review)},
  year    = {2026}
}
