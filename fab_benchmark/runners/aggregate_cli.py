
"""
Aggregate trials into metrics.csv and metrics_summary.csv (v1).

CLI contract:
  python -m fab_benchmark.runners.aggregate_cli --glob "results/exp_e1_*/*" --out_dir results/agg_e1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from fab_benchmark.metrics.aggregate import METRICS_COLUMNS_V1, list_trial_dirs, aggregate_trials, summarize
from fab_benchmark.runners.utils import ensure_dir


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    args = ap.parse_args()

    trial_dirs = list_trial_dirs(args.glob)
    rows = aggregate_trials(trial_dirs)
    summ = summarize(rows)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    write_csv(out_dir / "metrics.csv", rows, METRICS_COLUMNS_V1)
    # summary columns
    if summ:
        cols = list(summ[0].keys())
    else:
        cols = []
    write_csv(out_dir / "metrics_summary.csv", summ, cols)


if __name__ == "__main__":
    main()
