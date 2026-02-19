# fab_benchmark/policies/ours_policy.py
# -----------------------------------------------------------------------------
# OursPolicy v2 (Tier1-focus + online posterior localization + robust navigation)
#
# Key requirements (per your paper workflow):
# 1) Coverage: when no Tier1 alarm -> cover rooms (tour of room-centers).
# 2) Tier1 focus priority (NO cheating):
#    - Only go to *notified* Tier1 alarms (from belief_summary["tier1"]["alarms"]).
#    - Hard lock: once focus target selected, do NOT get pulled away by local high
#      concentration / other candidates until focus is reached or timeout.
# 3) While moving, update prediction: use belief_summary posterior / map estimate.
#    - In SEEK mode, goal is always the current prediction (pred_x, pred_y) when available.
# 4) Stop when localized: when posterior is confident and robot reaches prediction region.
# 5) Keep anti-stuck and wall-avoidance (A* + LOS lookahead + recovery).
#
# Notes:
# - This policy NEVER reads ground-truth leak position.
# - Tier1 sensor coordinates are read from scenario_spec["tier1"]["sensors"].
# - Alarms are read from belief_summary["tier1"]["alarms"] (dict or list format).
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import time
from collections import deque
import heapq

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from fab_benchmark.scenarios.base_scenario import build_scene_dict_from_scenario_spec


# ------------------------- small helpers -------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _world_to_grid(x: float, y: float, res: float, origin: Tuple[float, float]) -> Tuple[int, int]:
    j = int(math.floor((x - origin[0]) / res))
    i = int(math.floor((y - origin[1]) / res))
    return i, j


def _grid_to_world(i: int, j: int, res: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    x = origin[0] + (j + 0.5) * res
    y = origin[1] + (i + 0.5) * res
    return x, y


def _neighbors8(i: int, j: int) -> List[Tuple[int, int]]:
    return [
        (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
        (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1),
    ]


def _is_free(free_mask, i: int, j: int) -> bool:
    try:
        return bool(free_mask[i][j])
    except Exception:
        return False


def _nearest_free_cell(free_mask, start: Tuple[int, int], max_radius: int = 60) -> Optional[Tuple[int, int]]:
    si, sj = start
    H = len(free_mask)
    W = len(free_mask[0]) if H > 0 else 0
    if 0 <= si < H and 0 <= sj < W and _is_free(free_mask, si, sj):
        return (si, sj)
    q = [(si, sj)]
    seen = {(si, sj)}
    while q:
        i, j = q.pop(0)
        if max(abs(i - si), abs(j - sj)) > max_radius:
            continue
        if 0 <= i < H and 0 <= j < W and _is_free(free_mask, i, j):
            return (i, j)
        for ni, nj in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
            if (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append((ni, nj))
    return None


def _astar_grid(
    free_mask,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diag: bool = True,
    max_expansions: int = 250000,
) -> Optional[List[Tuple[int, int]]]:
    H = len(free_mask)
    W = len(free_mask[0]) if H > 0 else 0
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return None
    if not _is_free(free_mask, start[0], start[1]):
        s2 = _nearest_free_cell(free_mask, start)
        if s2 is None:
            return None
        start = s2
    if not _is_free(free_mask, goal[0], goal[1]):
        g2 = _nearest_free_cell(free_mask, goal)
        if g2 is None:
            return None
        goal = g2

    def h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        di = abs(a[0] - b[0])
        dj = abs(a[1] - b[1])
        return math.hypot(di, dj)

    openh: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(openh, (h(start, goal), start))
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start: 0.0}

    expansions = 0
    while openh:
        _f, cur = heapq.heappop(openh)
        expansions += 1
        if expansions > max_expansions:
            return None
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        nei = _neighbors8(cur[0], cur[1]) if allow_diag else [(cur[0]-1,cur[1]),(cur[0]+1,cur[1]),(cur[0],cur[1]-1),(cur[0],cur[1]+1)]
        for ni, nj in nei:
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if not _is_free(free_mask, ni, nj):
                continue
            # no corner cutting
            if allow_diag and (ni != cur[0] and nj != cur[1]):
                if not (_is_free(free_mask, cur[0], nj) and _is_free(free_mask, ni, cur[1])):
                    continue
            step = 1.4142 if (ni != cur[0] and nj != cur[1]) else 1.0
            ng = gscore[cur] + step
            nxt = (ni, nj)
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openh, (ng + h(nxt, goal), nxt))
    return None




def _inflate_obstacles_from_free(free: np.ndarray, radius_cells: int) -> np.ndarray:
    """Return a *safer* free mask by dilating obstacles by `radius_cells`.

    This is an approximation of footprint clearance: it encourages A* and line-of-sight smoothing
    to keep some distance from walls, reducing wall-collisions and corner-cutting issues.
    """
    free = np.asarray(free, dtype=bool)
    if radius_cells <= 0:
        return free.copy()

    occ = ~free  # True where obstacle
    inflated = occ.copy()
    h, w = occ.shape[:2]

    r = int(radius_cells)
    r2 = r * r
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r2:
                continue
            # OR shifted obstacle mask into inflated
            src_i0 = max(0, -di)
            src_i1 = min(h, h - di)
            dst_i0 = max(0, di)
            dst_i1 = min(h, h + di)

            src_j0 = max(0, -dj)
            src_j1 = min(w, w - dj)
            dst_j0 = max(0, dj)
            dst_j1 = min(w, w + dj)

            inflated[dst_i0:dst_i1, dst_j0:dst_j1] |= occ[src_i0:src_i1, src_j0:src_j1]

    return ~inflated


def _manhattan_dist_to_obstacle(obstacle: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance (in grid cells) to the nearest obstacle cell."""
    obstacle = np.asarray(obstacle, dtype=bool)
    h, w = obstacle.shape[:2]
    big = np.int32(10 ** 9)
    dist = np.full((h, w), big, dtype=np.int32)

    q = deque()
    obs_i, obs_j = np.where(obstacle)
    for i, j in zip(obs_i.tolist(), obs_j.tolist()):
        dist[i, j] = 0
        q.append((i, j))

    if not q:
        # No obstacles? (shouldn't happen) -> treat as very far.
        dist[:] = 10 ** 6
        return dist

    while q:
        i, j = q.popleft()
        nd = dist[i, j] + 1
        if i > 0 and dist[i - 1, j] > nd:
            dist[i - 1, j] = nd
            q.append((i - 1, j))
        if i + 1 < h and dist[i + 1, j] > nd:
            dist[i + 1, j] = nd
            q.append((i + 1, j))
        if j > 0 and dist[i, j - 1] > nd:
            dist[i, j - 1] = nd
            q.append((i, j - 1))
        if j + 1 < w and dist[i, j + 1] > nd:
            dist[i, j + 1] = nd
            q.append((i, j + 1))

    return dist

def _los_free(free_mask, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Bresenham line-of-sight on grid centers."""
    (i0, j0) = a
    (i1, j1) = b
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = dj - di
    i, j = i0, j0
    H = len(free_mask)
    W = len(free_mask[0]) if H > 0 else 0
    while True:
        if not (0 <= i < H and 0 <= j < W and _is_free(free_mask, i, j)):
            return False
        if (i, j) == (i1, j1):
            break
        e2 = 2 * err
        if e2 > -di:
            err -= di
            j += sj
        if e2 < dj:
            err += dj
            i += si
    return True


# ------------------------- policy -------------------------

@dataclass
class _FocusTarget:
    sensor_id: str
    xy: Tuple[float, float]
    t_evt: float


class OursPolicy:
    def __init__(self, policy_cfg: Optional[Dict[str, Any]] = None, sim_params: Optional[Dict[str, Any]] = None, **kwargs):
        # Compat: some runner/registry versions pass the policy config under `cfg`.
        if policy_cfg is None and isinstance(kwargs.get("cfg"), dict):
            policy_cfg = kwargs.get("cfg")
        policy_cfg = policy_cfg or {}
        # Keep a single source of truth for config. Some code paths use `self._cfg`,
        # while others use `self.cfg`. Point both to the same dict to avoid silent defaults.
        self._cfg: Dict[str, Any] = dict(policy_cfg)
        self.cfg = self._cfg
        # Motion (runner also clamps)
        self._v_max = float(policy_cfg.get("v_max", 3.7))
        self._w_max = float(policy_cfg.get("w_max", 2.0))
        self.sim_params = sim_params
        # Coverage
        self._visit_tol = float(policy_cfg.get("visit_tol", 0.7))
        self._tour: List[str] = []
        self._tour_idx = 0
        self._visited_rooms: set = set()

        # Focus priority
        self._focus_on_tier1 = bool(policy_cfg.get("focus_on_tier1", True))
        self._focus_reach_m = float(policy_cfg.get("focus_reach_m", 1.2))
        self._focus_hold_s = float(policy_cfg.get("focus_hold_s", 1.0))
        self._focus_lock_timeout_s = float(policy_cfg.get("focus_lock_timeout_s", 30.0))
        self._focus_queue_max = int(policy_cfg.get("focus_queue_max", 8))

        # Seek / stop
        self._pred_accept_m = float(policy_cfg.get("pred_accept_m", 1.0))
        self._stop_hold_s = float(policy_cfg.get("stop_hold_s", 1.0))
        self._stop_credible_volume_thr = float(policy_cfg.get("stop_credible_volume_thr", 0.80))
        self._stop_room_mass_thr = float(policy_cfg.get("stop_room_mass_thr", 0.65))

        # Navigation
        self._replan_s = float(policy_cfg.get("nav_replan_s", 0.8))
        self._lookahead_cells = int(policy_cfg.get("lookahead_cells", 12))
        self._goal_snap_max_cells = int(policy_cfg.get("goal_snap_max_cells", 60))

        # Anti-stuck
        self._stuck_window_s = float(policy_cfg.get("stuck_window_s", 1.2))
        self._stuck_dist_m = float(policy_cfg.get("stuck_dist_m", 0.15))
        self._recover_s = float(policy_cfg.get("recover_s", 0.8))
        self._recover_w = float(policy_cfg.get("recover_w", 1.2))
        self._recover_back_s = float(policy_cfg.get("recover_back_s", 0.3))
        self._recover_back_v = float(policy_cfg.get("recover_back_v", 0.3))

        self._stuck_yaw_rad = float(policy_cfg.get("stuck_yaw_rad", 0.35))
        self._stuck_cooldown_s = float(policy_cfg.get("stuck_cooldown_s", 1.0))
        self._last_recovery_t = -1e9

        # Seek gating: avoid chasing a meaningless early MAP estimate before any evidence
        self._seek_requires_alarm = bool(policy_cfg.get("seek_requires_alarm", True))
        self._seek_credible_volume_max = float(policy_cfg.get("seek_credible_volume_max", 120.0))
        self._seek_room_mass_min = float(policy_cfg.get("seek_room_mass_min", 0.35))
        self._seek_fallback_time_s = float(policy_cfg.get("seek_fallback_time_s", 20.0))

        self._ever_had_alarm = False
        self._recover_steps_left = 0
        self._recover_back_steps_left = 0

        # Runtime state
        self._scene: Dict[str, Any] = {}
        self._free = None
        self._origin = (0.0, 0.0)
        self._res = 1.0
        self._tier1_pos: Dict[str, Tuple[float, float]] = {}

        self._mode = "coverage"  # coverage | focus | seek | done
        self._focus: Optional[_FocusTarget] = None
        self._focus_start_t: float = -1.0
        self._focus_reach_t: float = -1.0
        self._focus_done: bool = False
        self._focus_checked_ids: set[str] = set()

        self._goal_xy: Optional[Tuple[float, float]] = None
        self._nav_path: Optional[List[Tuple[int, int]]] = None
        self._nav_path_t: float = -1.0

        self._pose_hist: List[Tuple[float, float, float, float]] = []  # (t,x,y,yaw)
        self._last_recovery_t = -1e9
        self._ever_had_alarm = False

        self._stop_reach_t: float = -1.0

        # Budget / debug stats (used by runner to write metrics/summary CSV).
        # Keep keys stable so downstream scripts don't break.
        self._budget_stats: Dict[str, int] = {
            "steps": 0,
            "candidates": 0,
            "focus_selected": 0,
            "replans": 0,
            "recoveries": 0,
        }

        # Timing stats (populated via runner hooks)
        self._planning_ms_list: List[float] = []
        self._inference_ms_list: List[float] = []

    # runner hooks
    def reset(self, scenario_spec: Dict[str, Any], seed: int = 0) -> None:
        # Build scene layout dict (keys differ across repo versions)
        self._scene = build_scene_dict_from_scenario_spec(scenario_spec)

        # Reset budget stats for this episode
        for k in list(self._budget_stats.keys()):
            self._budget_stats[k] = 0
        # Clear timing stats for this episode
        self._planning_ms_list.clear()
        self._inference_ms_list.clear()
        import numpy as np
        occ = self._scene.get('occupancy', None)
        if occ is None:
            occ = self._scene.get('occ', None)
        if occ is not None and not isinstance(occ, np.ndarray):
            occ = np.asarray(occ)
        free = self._scene.get('free_mask', None)
        if free is None:
            # occupancy convention in this codebase: 0=free, 1=wall
            if occ is None:
                raise KeyError('scene_dict missing occupancy/free_mask')
            free = (occ == 0)
        elif not isinstance(free, np.ndarray):
            free = np.asarray(free).astype(bool)
        self._free = free
        origin = self._scene.get('origin_xy', None)
        if origin is None:
            origin = self._scene.get('origin', [0.0, 0.0])
        self._origin = (float(origin[0]), float(origin[1]))
        self._res = float(self._scene.get('resolution', scenario_spec.get('map',{}).get('resolution', 0.1)))

        # Deterministic RNG (used for recovery wiggles).
        self._rng = np.random.default_rng(int(seed) + 12345)

        # Safer navigation: inflate obstacles for planning so A* / line-of-sight doesn't hug walls.
        robot_radius_m = float(self._cfg.get("robot_radius_m", 0.35))
        nav_margin_m = float(self._cfg.get("nav_safety_margin_m", 0.10))
        infl_cells = int(math.ceil((robot_radius_m + nav_margin_m) / max(self._res, 1e-6)))
        infl_cells = max(0, min(infl_cells, 10))
        self._inflation_cells = infl_cells
        self._free_plan = _inflate_obstacles_from_free(self._free, infl_cells)

        # Precompute (approx.) distance-to-obstacle in grid-cells (Manhattan) for speed limiting near walls.
        self._dist_to_obst = _manhattan_dist_to_obstacle(~self._free)
        self._speed_clear_min_m = float(self._cfg.get("speed_clear_min_m", 0.50))
        self._speed_clear_max_m = float(self._cfg.get("speed_clear_max_m", 1.00))
        self._speed_scale_min = float(self._cfg.get("speed_scale_min", 0.25))

        # Current active navigation mask (may fall back to raw free mask if inflation disconnects the map).
        self._free_active = self._free_plan

        rooms = self._scene.get("rooms", {}) or {}
        self._tour = list(rooms.keys())
        # deterministic order: nearest-neighbor from first room center
        if len(self._tour) > 1:
            centers = {rid: tuple(rooms[rid]["center"]) for rid in self._tour if "center" in rooms[rid]}
            if centers:
                start = self._tour[0]
                cur = start
                un = set(self._tour)
                order = []
                while un:
                    order.append(cur)
                    un.remove(cur)
                    if not un:
                        break
                    cx, cy = centers.get(cur, centers.get(start))
                    best = None
                    bestd = 1e18
                    for r in un:
                        rx, ry = centers.get(r, (cx, cy))
                        d = (rx - cx) ** 2 + (ry - cy) ** 2
                        if d < bestd:
                            bestd = d
                            best = r
                    cur = best if best is not None else list(un)[0]
                self._tour = order

        self._tour_idx = 0
        self._visited_rooms = set()
        self._tier1_pos = {}
        tier1 = scenario_spec.get("tier1", {}) or {}
        for s in (tier1.get("sensors", []) or []):
            try:
                sid = str(s.get("id"))
                px, py = float(s["pos"][0]), float(s["pos"][1])
                self._tier1_pos[sid] = (px, py)
            except Exception:
                pass

        self._mode = "coverage"
        self._focus = None
        self._focus_start_t = -1.0
        self._focus_reach_t = -1.0
        self._focus_done = False
        self._focus_checked_ids = set()
        self._goal_xy = None
        self._nav_path = None
        self._nav_path_t = -1.0
        self._pose_hist = []
        self._recover_steps_left = 0
        self._recover_back_steps_left = 0
        self._stop_reach_t = -1.0

    # ------------------------- parsing -------------------------

    def _parse_tier1_alarms(self, belief_summary: Optional[Dict[str, Any]]) -> List[_FocusTarget]:
        """Return list of focus targets (newest first). Supports dict or list formats."""
        if not belief_summary or not isinstance(belief_summary, dict):
            return []
        tier1 = belief_summary.get("tier1", {}) or {}
        alarms = tier1.get("alarms", None)

        items: List[_FocusTarget] = []

        # dict format: {sensor_id_or_room_id: {t:..., ...}} or {id: t}
        if isinstance(alarms, dict):
            for key, info in alarms.items():
                sid = str(key)
                t_evt = -1.0
                if isinstance(info, dict):
                    try:
                        t_evt = float(info.get("t", -1.0))
                    except Exception:
                        t_evt = -1.0
                else:
                    try:
                        t_evt = float(info)
                    except Exception:
                        t_evt = -1.0
                if sid in self._tier1_pos:
                    items.append(_FocusTarget(sensor_id=sid, xy=self._tier1_pos[sid], t_evt=t_evt))
        # list format: [{"id":..., "pos":[x,y,z], "t":...}, ...]
        elif isinstance(alarms, list):
            for a in alarms:
                if not isinstance(a, dict):
                    continue
                sid = str(a.get("id", ""))
                t_evt = float(a.get("t", -1.0))
                if "pos" in a and a["pos"] is not None:
                    try:
                        px, py = float(a["pos"][0]), float(a["pos"][1])
                        items.append(_FocusTarget(sensor_id=sid or sid, xy=(px, py), t_evt=t_evt))
                        continue
                    except Exception:
                        pass
                if sid in self._tier1_pos:
                    items.append(_FocusTarget(sensor_id=sid, xy=self._tier1_pos[sid], t_evt=t_evt))

        # newest first
        items.sort(key=lambda z: z.t_evt, reverse=True)
        return items[: max(0, self._focus_queue_max)]

    def _extract_prediction(self, belief_summary: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        """Try to get current MAP prediction from belief_summary."""
        if not belief_summary or not isinstance(belief_summary, dict):
            return None

        # common: {"map_estimate": {"pos":[x,y]}} or {"map_estimate":[x,y]}
        me = belief_summary.get("map_estimate", None)
        if isinstance(me, dict) and "pos" in me:
            try:
                return (float(me["pos"][0]), float(me["pos"][1]))
            except Exception:
                pass
        if isinstance(me, (list, tuple)) and len(me) >= 2:
            try:
                return (float(me[0]), float(me[1]))
            except Exception:
                pass

        # candidates: [{"pos":[x,y], "score":...}, ...]
        cand = belief_summary.get("candidates", None) or belief_summary.get("topk", None)
        if isinstance(cand, list) and len(cand) > 0:
            try:
                self._budget_stats["candidates"] += int(len(cand))
            except Exception:
                pass
            best = None
            bests = -1e18
            for c in cand:
                if not isinstance(c, dict):
                    continue
                sc = c.get("score", 0.0)
                try:
                    sc = float(sc)
                except Exception:
                    sc = 0.0
                if sc > bests and "pos" in c and c["pos"] is not None:
                    try:
                        px, py = float(c["pos"][0]), float(c["pos"][1])
                        best = (px, py)
                        bests = sc
                    except Exception:
                        pass
            if best is not None:
                return best

        # fallback: posterior_room_mass -> room center
        rm = belief_summary.get("room_mass", None)
        if rm is None:
            rm = belief_summary.get("posterior_room_mass", None)
        if isinstance(rm, dict) and rm:
            # pick max mass room
            rid = None
            best = -1e18
            for k, v in rm.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv > best:
                    best = fv
                    rid = str(k)
            if rid is not None:
                rooms = self._scene.get("rooms", {}) or {}
                if rid in rooms and "center" in rooms[rid]:
                    cx, cy = rooms[rid]["center"]
                    return (float(cx), float(cy))

        return None

    def _belief_confidence(self, belief_summary: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        """Return (credible_volume, max_room_mass)."""
        if not belief_summary or not isinstance(belief_summary, dict):
            return (1e9, 0.0)
        cv = belief_summary.get("credible_volume", 1e9)
        try:
            cv = float(cv)
        except Exception:
            cv = 1e9
        maxm = 0.0
        rm = belief_summary.get("posterior_room_mass", None)
        if isinstance(rm, dict) and rm:
            for v in rm.values():
                try:
                    maxm = max(maxm, float(v))
                except Exception:
                    pass
        return (cv, maxm)

    # ------------------------- navigation / control -------------------------

    def _plan_path(self, here_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> Optional[List[Tuple[int, int]]]:
        si, sj = _world_to_grid(here_xy[0], here_xy[1], self._res, self._origin)
        gi, gj = _world_to_grid(goal_xy[0], goal_xy[1], self._res, self._origin)

        # Try inflated free-mask first (keeps distance from walls). If that disconnects the map for this
        # start/goal pair, fall back to the raw free mask so we still have a path.
        candidates: List[np.ndarray] = []
        if getattr(self, "_free_plan", None) is not None:
            candidates.append(self._free_plan)
        candidates.append(self._free)

        for free in candidates:
            start = _nearest_free_cell(free, (si, sj), max_radius=self._goal_snap_max_cells)
            goal = _nearest_free_cell(free, (gi, gj), max_radius=self._goal_snap_max_cells)
            if start is None or goal is None:
                continue
            path = _astar_grid(free, start, goal, allow_diag=True)
            if path is not None:
                self._free_active = free
                return path

        return None

    def _pick_waypoint(self, path: List[Tuple[int, int]], here_cell: Tuple[int, int]) -> Tuple[int, int]:
        """LOS lookahead: choose farthest visible waypoint up to lookahead_cells."""
        if not path:
            return here_cell
        # find nearest index along path
        # (simple: choose first cell in path that equals here_cell; else use 0)
        idx = 0
        for k in range(len(path)):
            if path[k] == here_cell:
                idx = k
                break
        best = path[min(len(path) - 1, idx + 1)]
        max_k = min(len(path) - 1, idx + self._lookahead_cells)
        for k in range(idx + 1, max_k + 1):
            if _los_free(self._free_active, here_cell, path[k]):
                best = path[k]
        return best

    def _is_stuck(self, t: float, x: float, y: float, yaw: float) -> bool:
        # Cooldown after recovery to avoid immediate re-trigger.
        if (t - self._last_recovery_t) < self._stuck_cooldown_s:
            self._pose_hist.append((t, x, y, yaw))
            while len(self._pose_hist) > 0 and (t - self._pose_hist[0][0]) > self._stuck_window_s:
                self._pose_hist.pop(0)
            return False

        self._pose_hist.append((t, x, y, yaw))
        # Prune old history to a sliding window
        while len(self._pose_hist) > 0 and (t - self._pose_hist[0][0]) > self._stuck_window_s:
            self._pose_hist.pop(0)

        if len(self._pose_hist) < 2:
            return False

        t0, x0, y0, yaw0 = self._pose_hist[0]
        span = t - t0
        # Require ~one full window of history; avoids false stuck at episode start and while turning-in-place.
        if span < 0.9 * self._stuck_window_s:
            return False

        d = math.hypot(x - x0, y - y0)
        dyaw = abs(_wrap_pi(yaw - yaw0))
        return (d < self._stuck_dist_m) and (dyaw < self._stuck_yaw_rad)

    def _compute_vw(self, here_xy: Tuple[float, float], yaw: float, waypoint_xy: Tuple[float, float]) -> Tuple[float, float]:
        dx = waypoint_xy[0] - here_xy[0]
        dy = waypoint_xy[1] - here_xy[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        target = math.atan2(dy, dx)
        err = _wrap_pi(target - yaw)

        # turn-in-place if large heading error
        if abs(err) > 0.8:
            v = 0.0
            w = _clamp(1.5 * err, -self._w_max, self._w_max)
            return v, w

        v = _clamp(1.2 * dist, 0.0, self._v_max)
        w = _clamp(1.2 * err, -self._w_max, self._w_max)

        # Slow down near walls to reduce collision likelihood (based on distance-to-obstacle map).
        if getattr(self, "_dist_to_obst", None) is not None:
            i, j = _world_to_grid(here_xy[0], here_xy[1], self._res, self._origin)
            if 0 <= i < self._dist_to_obst.shape[0] and 0 <= j < self._dist_to_obst.shape[1]:
                clear_m = float(self._dist_to_obst[i, j]) * float(self._res)
                lo = float(getattr(self, "_speed_clear_min_m", 0.5))
                hi = float(getattr(self, "_speed_clear_max_m", 1.0))
                min_scale = float(getattr(self, "_speed_scale_min", 0.25))
                if clear_m <= lo:
                    scale = min_scale
                elif clear_m >= hi:
                    scale = 1.0
                else:
                    scale = min_scale + (1.0 - min_scale) * ((clear_m - lo) / max(hi - lo, 1e-6))
                v *= float(scale)

        return v, w

    # ------------------------- main step -------------------------

    def step(self, obs: Optional[Dict[str, Any]] = None, *, t: Optional[float] = None,
             dt: Optional[float] = None, pose: Optional[Dict[str, Any]] = None,
             measurement: Optional[Dict[str, Any]] = None,
             belief_summary: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Runner-compatible step.
    
        This repo's runner calls: step(t=..., pose=..., measurement=..., belief_summary=...).
        Some older tooling calls: step(obs_dict).
        We support both without changing the core logic.
        """
        if obs is None:
            obs = {}
            if t is not None:
                obs["t"] = t
            if dt is not None:
                obs["dt"] = dt
            if pose is not None:
                obs["pose"] = pose
            if measurement is not None:
                obs["measurement"] = measurement
            if belief_summary is not None:
                obs["belief_summary"] = belief_summary
            # allow dt via kwargs (some runners pass dt)
            if "dt" in kwargs and "dt" not in obs:
                obs["dt"] = kwargs["dt"]
        return self._step_obs(obs)
    
    def _step_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return action dict: {"v":..., "w":..., "action_id":..., "goal_xy":..., ...}"""
        t = float(obs.get("t", 0.0))
        dt = float(obs.get("dt", 0.1))
        pose = obs.get("pose", {}) or {}
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        yaw = float(pose.get("yaw", 0.0))
        belief = obs.get("belief_summary", None)

        here_xy = (x, y)

        # book-keeping
        try:
            self._budget_stats["steps"] += 1
        except Exception:
            pass

        # ---------------- focus: select / lock ----------------
        if self._focus_on_tier1:
            alarms_all = self._parse_tier1_alarms(belief)
            if alarms_all:
                self._ever_had_alarm = True

            # Keep the full list for debugging/trace, but only consider *unchecked* alarms for focusing.
            alarms_pending = [a for a in alarms_all if a.sensor_id not in self._focus_checked_ids]

            # Acquire focus only once per episode (paper intent):
            # after we investigate the alarm vicinity, we should trust the posterior and go to MAP.
            if (not self._focus_done) and (self._focus is None) and alarms_pending:
                self._focus = alarms_pending[0]
                try:
                    self._budget_stats["focus_selected"] += 1
                except Exception:
                    pass
                self._focus_start_t = t
                self._focus_reach_t = -1.0
                self._mode = "focus"
                self._nav_path = None
                self._nav_path_t = -1.0

            # If active focus: keep lock until reach/hold or timeout.
            if self._focus is not None:
                # timeout -> mark this alarm as checked, then fall back to seeking posterior MAP
                if self._focus_start_t > 0 and (t - self._focus_start_t) > self._focus_lock_timeout_s:
                    try:
                        self._focus_checked_ids.add(self._focus.sensor_id)
                    except Exception:
                        pass
                    self._focus = None
                    self._focus_done = True
                    self._mode = "seek"
                    self._nav_path = None
                    self._nav_path_t = -1.0
                    self._focus_reach_t = -1.0

# ---------------- decide goal ----------------
        pred_xy = self._extract_prediction(belief)
        credible_v, max_room_mass = self._belief_confidence(belief)

        # Mode priority:
        # 1) focus (must go to focus_tier1)
        # 2) seek (goal = prediction)
        # 3) coverage (tour rooms)
        if self._focus is not None:
            self._mode = "focus"
            self._goal_xy = self._focus.xy
        else:
            # If we have a prediction, we may seek it — but avoid chasing a meaningless early MAP
            # before any Tier-1 alarm / evidence has arrived.
            seek_allowed = pred_xy is not None
            if seek_allowed and self._seek_requires_alarm and (not self._ever_had_alarm):
                # No Tier-1 alarm yet: do coverage first. Allow seek only after a fallback time AND
                # the belief has concentrated into a small number of rooms.
                if (t < self._seek_fallback_time_s) or (max_room_mass < self._seek_room_mass_min):
                    seek_allowed = False
            if seek_allowed:
                self._mode = "seek"
                self._goal_xy = pred_xy
            else:
                self._mode = "coverage"
                self._goal_xy = self._next_coverage_goal(here_xy)

        # ---------------- stop logic (after focus) ----------------
        # stop only when not in focus lock, and prediction exists
        if self._mode != "focus" and pred_xy is not None:
            if (credible_v <= self._stop_credible_volume_thr) or (max_room_mass >= self._stop_room_mass_thr):
                if math.hypot(here_xy[0] - pred_xy[0], here_xy[1] - pred_xy[1]) <= self._pred_accept_m:
                    if self._stop_reach_t < 0:
                        self._stop_reach_t = t
                    if (t - self._stop_reach_t) >= self._stop_hold_s:
                        self._mode = "done"
                        self._mode = "done"
                        v0, w0 = 0.0, 0.0
                        # use _pack_action so trace fields stay consistent
                        a = self._pack_action(t, here_xy, pred_xy, v0, w0, credible_v, max_room_mass, action_id="stop_found")
                        a["done"] = 1
                        a["stop_reason"] = "confidence_stop"
                        return a
                else:
                    self._stop_reach_t = -1.0
            else:
                self._stop_reach_t = -1.0

        # ---------------- focus reach check (hold) ----------------
        if self._mode == "focus" and self._focus is not None:
            d_focus = math.hypot(here_xy[0] - self._focus.xy[0], here_xy[1] - self._focus.xy[1])
            if d_focus <= self._focus_reach_m:
                if self._focus_reach_t < 0:
                    self._focus_reach_t = t
                if (t - self._focus_reach_t) >= self._focus_hold_s:
                    # focus satisfied -> mark checked and continue seek to posterior MAP
                    try:
                        self._focus_checked_ids.add(self._focus.sensor_id)
                    except Exception:
                        pass
                    self._focus_done = True
                    self._focus = None
                    self._mode = "seek"
                    self._nav_path = None
                    self._nav_path_t = -1.0
            else:
                self._focus_reach_t = -1.0

        # ---------------- anti-stuck recovery ----------------
        if self._recover_steps_left > 0:
            self._recover_steps_left -= 1
            if self._recover_back_steps_left > 0:
                self._recover_back_steps_left -= 1
                v, w = -self._recover_back_v, 0.0
            else:
                v, w = 0.0, float(getattr(self, '_recover_w_sign', 1.0)) * self._recover_w
            return self._pack_action(t, here_xy, pred_xy, v, w, credible_v, max_room_mass)

        if self._is_stuck(t, x, y, yaw):
            try:
                self._budget_stats["recoveries"] += 1
            except Exception:
                pass
            self._recover_steps_left = int(max(1, round(self._recover_s / max(1e-6, dt))))
            self._recover_back_steps_left = int(max(0, round(self._recover_back_s / max(1e-6, dt))))
            self._last_recovery_t = t
            self._pose_hist.clear()
            self._nav_path = None
            self._nav_path_t = -1.0
            # Choose a random rotation direction (deterministic via seed) and start by backing up if requested.
            try:
                self._recover_w_sign = -1.0 if float(self._rng.random()) < 0.5 else 1.0
            except Exception:
                self._recover_w_sign = 1.0

            v0, w0 = 0.0, float(self._recover_w_sign) * self._recover_w
            if self._recover_back_steps_left > 0:
                v0, w0 = -self._recover_back_v, 0.0
                self._recover_back_steps_left -= 1
            self._recover_steps_left = max(0, self._recover_steps_left - 1)

            return self._pack_action(t, here_xy, pred_xy, v0, w0, credible_v, max_room_mass, action_id="recover")

        # ---------------- plan / follow ----------------
        if self._goal_xy is None:
            return self._pack_action(t, here_xy, pred_xy, 0.0, 0.0, credible_v, max_room_mass, action_id="idle")

        # replan periodically or if no path
        if (self._nav_path is None) or (self._nav_path_t < 0) or ((t - self._nav_path_t) >= self._replan_s):
            try:
                self._budget_stats["replans"] += 1
            except Exception:
                pass
            pth = self._plan_path(here_xy, self._goal_xy)
            self._nav_path = pth
            self._nav_path_t = t

        # if still no path, turn in place to explore
        if not self._nav_path:
            v, w = 0.0, 0.6
            return self._pack_action(t, here_xy, pred_xy, v, w, credible_v, max_room_mass, action_id="no_path")

        here_cell = _world_to_grid(here_xy[0], here_xy[1], self._res, self._origin)
        wp_cell = self._pick_waypoint(self._nav_path, here_cell)
        wp_xy = _grid_to_world(wp_cell[0], wp_cell[1], self._res, self._origin)

        v, w = self._compute_vw(here_xy, yaw, wp_xy)

        # slow near goal
        dg = math.hypot(here_xy[0] - self._goal_xy[0], here_xy[1] - self._goal_xy[1])
        if dg < 1.2:
            v *= 0.5
        if dg < 0.6:
            v *= 0.3

        return self._pack_action(t, here_xy, pred_xy, v, w, credible_v, max_room_mass)

    # ------------------------- coverage goal -------------------------

    def _next_coverage_goal(self, here_xy: Tuple[float, float]) -> Tuple[float, float]:
        rooms = self._scene.get("rooms", {}) or {}
        if not self._tour:
            return here_xy
        # pick next unvisited room in tour; reset if all visited
        for k in range(len(self._tour)):
            rid = self._tour[(self._tour_idx + k) % len(self._tour)]
            if rid in self._visited_rooms:
                continue
            self._tour_idx = (self._tour_idx + k) % len(self._tour)
            cx, cy = rooms[rid]["center"]
            # mark visited if close enough
            if math.hypot(here_xy[0] - cx, here_xy[1] - cy) <= self._visit_tol:
                self._visited_rooms.add(rid)
                self._tour_idx = (self._tour_idx + 1) % len(self._tour)
            return (float(cx), float(cy))

        # all visited -> restart
        self._visited_rooms = set()
        cx, cy = rooms[self._tour[self._tour_idx]]["center"]
        return (float(cx), float(cy))

    # ------------------------- pack action -------------------------

    def _pack_action(
        self,
        t: float,
        here_xy: Tuple[float, float],
        pred_xy: Optional[Tuple[float, float]],
        v: float,
        w: float,
        credible_v: float,
        max_room_mass: float,
        action_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pack an action dict in a runner-friendly format.

        The runner *always* reads:
          - v, w
          - action_id
          - policy_mode

        For debugging/plotting, it may also read:
          - goal_xy (preferred) OR goal_x/goal_y
          - focus (preferred) OR focus_id/focus_x/focus_y
        """
        aid = action_id
        if aid is None:
            if self._mode == "focus":
                aid = "dispatch_tier1"
            elif self._mode == "seek":
                aid = "seek_pred"
            elif self._mode == "coverage":
                aid = "coverage"
            else:
                aid = self._mode

        goal_xy = None
        if self._goal_xy is not None:
            goal_xy = [float(self._goal_xy[0]), float(self._goal_xy[1])]

        pred_xy_out = None
        if pred_xy is not None:
            pred_xy_out = [float(pred_xy[0]), float(pred_xy[1])]

        focus_out = None
        if self._focus is not None:
            focus_out = {"id": self._focus.sensor_id, "x": float(self._focus.xy[0]), "y": float(self._focus.xy[1])}

        out = {
            "v": float(_clamp(v, -self._v_max, self._v_max)),
            "w": float(_clamp(w, -self._w_max, self._w_max)),
            "action_id": str(aid),
            "policy_mode": str(self._mode),

            # Debug
            "goal_xy": goal_xy,
            "goal_x": float(goal_xy[0]) if goal_xy is not None else float("nan"),
            "goal_y": float(goal_xy[1]) if goal_xy is not None else float("nan"),
            "pred_xy": pred_xy_out,
            "pred_x": float(pred_xy_out[0]) if pred_xy_out is not None else float("nan"),
            "pred_y": float(pred_xy_out[1]) if pred_xy_out is not None else float("nan"),
            "focus": focus_out,
            "focus_id": str(focus_out.get("id")) if isinstance(focus_out, dict) else "",
            "focus_done": int(self._focus_done),
            "focus_checked_n": int(len(self._focus_checked_ids)),
            "focus_x": float(focus_out.get("x")) if isinstance(focus_out, dict) else float("nan"),
            "focus_y": float(focus_out.get("y")) if isinstance(focus_out, dict) else float("nan"),
            "credible_volume": float(credible_v),
            "max_room_mass": float(max_room_mass),
        }
        return out

    # Optional hooks used by runner (safe no-ops)
    
    # --- Runner hooks --------------------------------------------------------

    def record_planning_ms(self, ms: float) -> None:
        """
        Called by the runner each control step to record wall-clock planning time (ms).
        """
        try:
            v = float(ms)
        except Exception:
            return
        self._planning_ms_list.append(v)

    def record_inference_ms(self, ms: float) -> None:
        """
        Called by the runner each control step to record wall-clock inference time (ms).
        """
        try:
            v = float(ms)
        except Exception:
            return
        self._inference_ms_list.append(v)

    def get_budget_stats(self) -> Dict[str, float]:
        """
        Runner-facing stats used for experiment logging.
        Must always return numeric values (no None) to avoid downstream float(None) failures.
        """
        candidates = int(getattr(self, "_budget_candidates", 0) or 0)
        rollouts = int(getattr(self, "_budget_rollouts", 0) or 0)

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        return {
            "candidates": candidates,
            "rollouts": rollouts,
            "planning_ms_mean": _mean(self._planning_ms_list),
            "inference_ms_mean": _mean(self._inference_ms_list),
        }