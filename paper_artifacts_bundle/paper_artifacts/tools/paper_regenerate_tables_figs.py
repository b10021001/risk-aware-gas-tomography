#!/usr/bin/env python3
"""tools/regenerate_tables_figs.py

Regenerate key tables/figures from released per-run logs and the manifest.
This script does NOT rerun the simulator; it recomputes reported metrics from logs.

Expected:
  data/logs/manifest.csv
  - gt_episode.json: leak_s_gt_m
  - trace.csv: t_s, y_hazard
  - posterior_summary.csv: final mu_x,mu_y,mu_z (optional)

Outputs:
  out/E1_table.csv, out/E2_table.csv
  out/Fig5_ID_OOD_success_error.(png/pdf) (optional)
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_json(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def exposure_end_from_trace(trace: pd.DataFrame) -> float:
    if len(trace) < 2:
        return float('nan')
    dt = float(np.median(np.diff(trace['t_s'].to_numpy())))
    return float((trace['y_hazard'].to_numpy() * dt).sum())

def final_error_from_posterior(gt_s: np.ndarray, post: pd.DataFrame) -> float:
    if post is None or len(post) == 0:
        return float('nan')
    r = post.iloc[-1]
    mu = np.array([r.mu_x, r.mu_y, r.mu_z], dtype=float)
    return float(np.linalg.norm(mu - gt_s))

def summarize_trials(manifest_csv: Path, emax: float, time_cap: float, err_thr: float):
    man = pd.read_csv(manifest_csv)
    rows = []
    for _, r in man.iterrows():
        gt_path = Path(r['gt_episode_path'])
        trace_path = Path(r['trace_path'])
        post_path = Path(r['posterior_summary_path']) if isinstance(r.get('posterior_summary_path', ''), str) and r['posterior_summary_path'] else None

        gt = load_json(gt_path)
        gt_s = np.array(gt['leak_s_gt_m'], dtype=float)

        trace = pd.read_csv(trace_path)
        E_end = exposure_end_from_trace(trace)
        violation = bool(E_end > emax)

        post = pd.read_csv(post_path) if post_path and post_path.exists() else None
        err = final_error_from_posterior(gt_s, post)
        t_end = float(trace['t_s'].max())
        success = bool((err <= err_thr) and (t_end <= time_cap + 1e-6))
        safe_success = bool(success and (not violation))

        rows.append({
            'trial_id': r['trial_id'],
            'experiment': r['experiment'],
            'setting': r['setting'],
            'method': r['method'],
            'E_end': E_end,
            'violation': violation,
            'error': err,
            't_end': t_end,
            'success': success,
            'safe_success': safe_success
        })
    return pd.DataFrame(rows)

def table_from_summary(df: pd.DataFrame, setting: str):
    sub = df[df['setting']==setting]
    g = sub.groupby('method').agg(
        n=('trial_id','count'),
        success=('success','mean'),
        violation=('violation','mean'),
        safe_success=('safe_success','mean'),
        error_med=('error','median'),
        E_med=('E_end','median')
    ).reset_index()
    for c in ['success','violation','safe_success']:
        g[c] = 100.0*g[c]
    return g.sort_values('method')

def make_fig5(e1, e2, outdir: Path):
    methods = e1['method'].tolist()
    x = np.arange(len(methods))
    width = 0.35

    fig = plt.figure(figsize=(7.2,3.2))
    ax = fig.add_subplot(121)
    ax.bar(x - width/2, e1['success'], width, label='ID')
    ax.bar(x + width/2, e2['success'], width, label='OOD')
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Success rate (%)')
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.bar(x - width/2, e1['error_med'], width, label='ID')
    ax2.bar(x + width/2, e2['error_med'], width, label='OOD')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Median error (m)')
    ax2.grid(True, linestyle=':', linewidth=0.7)
    ax2.legend()

    fig.tight_layout()
    out_png = outdir/'Fig5_ID_OOD_success_error.png'
    out_pdf = outdir/'Fig5_ID_OOD_success_error.pdf'
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=str, default='data/logs/manifest.csv')
    ap.add_argument('--emax', type=float, default=60.0)
    ap.add_argument('--time_cap', type=float, default=60.0)
    ap.add_argument('--err_thr', type=float, default=2.0)
    ap.add_argument('--outdir', type=str, default='out')
    ap.add_argument('--make_fig5', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = summarize_trials(Path(args.manifest), args.emax, args.time_cap, args.err_thr)

    e1 = table_from_summary(df, 'ID')
    e2 = table_from_summary(df, 'OOD')

    e1.to_csv(outdir/'E1_table.csv', index=False)
    e2.to_csv(outdir/'E2_table.csv', index=False)

    if args.make_fig5:
        make_fig5(e1, e2, outdir)

    print('Wrote:', outdir/'E1_table.csv', outdir/'E2_table.csv')

if __name__ == '__main__':
    main()
