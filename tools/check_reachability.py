# -*- coding: utf-8 -*-
"""
check_reachability.py

Why you saw:
  ModuleNotFoundError: No module named 'fab_benchmark'
When you run:
  py tools\check_reachability.py ...

Because Python puts "tools/" (the script directory) on sys.path, not the repo root.
This script fixes that by inserting the repo root into sys.path at runtime.

What it does:
- For each run_meta.json matched by --glob:
  - load scenario_spec
  - build occupancy grid via build_scene_dict_from_scenario_spec(...)
  - BFS from robot spawn cell to mark reachable region
  - report whether the leak cell is reachable
  - also report the closest reachable grid cell to the leak (distance in meters)

This helps interpret p90:
- If ~16% of cases are unreachable, your oracle navigation reach_rate will cap ~0.84
  and min_dist_p90 will stay large no matter how good the policy is.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import deque
from typing import Any, Dict, Tuple, Optional

# Ensure repo root is importable: <repo>/tools/check_reachability.py -> root is ..
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec  # noqa: E402


def world_to_grid(x: float, y: float, origin_xy: Tuple[float, float], res: float) -> Tuple[int, int]:
    gx = (float(x) - float(origin_xy[0])) / float(res) - 0.5
    gy = (float(y) - float(origin_xy[1])) / float(res) - 0.5
    j = int(round(gx))
    i = int(round(gy))
    return (i, j)


def grid_to_world(i: int, j: int, origin_xy: Tuple[float, float], res: float) -> Tuple[float, float]:
    x = (float(j) + 0.5) * float(res) + float(origin_xy[0])
    y = (float(i) + 0.5) * float(res) + float(origin_xy[1])
    return (x, y)


def bfs_reachable(occ, start_ij: Tuple[int, int]):
    H = len(occ)
    W = len(occ[0]) if H > 0 else 0
    si, sj = start_ij
    vis = [[False] * W for _ in range(H)]
    if not (0 <= si < H and 0 <= sj < W) or occ[si][sj]:
        return vis
    q = deque()
    q.append((si, sj))
    vis[si][sj] = True
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        i, j = q.popleft()
        for di, dj in nbrs:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W and (not vis[ni][nj]) and (not occ[ni][nj]):
                vis[ni][nj] = True
                q.append((ni, nj))
    return vis


def closest_reachable_to_goal(vis, goal_ij: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    H = len(vis)
    W = len(vis[0]) if H > 0 else 0
    gi, gj = goal_ij
    best = None
    best_d2 = None
    for i in range(H):
        row = vis[i]
        for j in range(W):
            if not row[j]:
                continue
            d2 = (i - gi) ** 2 + (j - gj) ** 2
            if best is None or d2 < best_d2:
                best = (i, j)
                best_d2 = d2
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, type=str, help='e.g. "results/exp_e1_lite/Ours__*/run_meta.json"')
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("No files matched:", args.glob)
        return

    n = 0
    reachable = 0
    dist_m_list = []

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print("[WARN] failed to load:", p, e)
            continue

        spec = meta.get("scenario_spec", {})
        # spawn / leak
        try:
            spawn = spec["robot"]["spawn"]["pos"]
            sx, sy = float(spawn[0]), float(spawn[1])
        except Exception:
            continue

        try:
            leak = spec.get("leak", {})
            if int(leak.get("enabled", 1)) != 1:
                continue
            lp = leak.get("pos", None)
            lx, ly = float(lp[0]), float(lp[1])
        except Exception:
            continue

        scene = build_scene_dict_from_scenario_spec(spec)
        occ = scene["occupancy"]
        res = float(scene["resolution"])
        origin = (float(scene["origin"][0]), float(scene["origin"][1]))

        s_ij = world_to_grid(sx, sy, origin, res)
        g_ij = world_to_grid(lx, ly, origin, res)

        vis = bfs_reachable(occ, s_ij)
        n += 1

        gi, gj = g_ij
        is_reach = (0 <= gi < len(vis) and 0 <= gj < len(vis[0]) and vis[gi][gj])

        if is_reach:
            reachable += 1
            dist_m_list.append(0.0)
        else:
            best = closest_reachable_to_goal(vis, g_ij)
            if best is None:
                dist_m_list.append(float("inf"))
            else:
                bx, by = grid_to_world(best[0], best[1], origin, res)
                dist_m_list.append(float(math.hypot(bx - lx, by - ly)))

    # summary
    dist_sorted = sorted([d for d in dist_m_list if math.isfinite(d)])
    p50 = dist_sorted[int(0.5 * len(dist_sorted))] if dist_sorted else float("nan")
    p90 = dist_sorted[int(0.9 * len(dist_sorted))] if dist_sorted else float("nan")

    print(f"files={len(paths)} parsed={n}")
    print(f"reachable_rate={reachable/max(1,n):.3f} ({reachable}/{n})")
    print(f"closest_reachable_dist_p50={p50:.3f}m  p90={p90:.3f}m")


if __name__ == "__main__":
    main()
