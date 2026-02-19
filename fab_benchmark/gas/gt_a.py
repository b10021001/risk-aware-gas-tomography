"""
Gas generator GT-A (v3): fast 2.5D surrogate with more realistic "continuous leak".

Compared to v1:
- Treats the leak as CONTINUOUS emission starting at start_time (not a single puff),
  by numerically integrating contributions from recent emission times.
- Uses analytic advection backtrace (drift/vortex) via hvac_modes.backtrace_xy.
- Optional topology attenuation: reduces concentration across multiple rooms/doors
  using an open-door room graph (cheap obstacle awareness).

This is still a surrogate (not CFD), but behaves much closer to "a leak keeps emitting"
and makes door patterns matter in the gas field.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any
import math
import heapq

from .hvac_modes import backtrace_xy


Z_LAYERS = [0.2, 1.0, 2.0]


def _room_bboxes(rooms: Dict[str, Any]) -> Dict[str, Tuple[float, float, float, float]]:
    b = {}
    for rid, r in rooms.items():
        poly = r.get("poly", None)
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        b[str(rid)] = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
    return b


def _room_of(x: float, y: float, bboxes: Dict[str, Tuple[float, float, float, float]]) -> Optional[str]:
    # First try strict bbox membership.
    for rid, (xmin, xmax, ymin, ymax) in bboxes.items():
        if (x >= xmin) and (x <= xmax) and (y >= ymin) and (y <= ymax):
            return rid

    # Fallback: assign corridor / in-between points to the nearest room center.
    # This avoids returning 0 concentration purely because a point lies just outside a bbox.
    if not bboxes:
        return None
    best_rid: Optional[str] = None
    best_d2 = float("inf")
    for rid, (xmin, xmax, ymin, ymax) in bboxes.items():
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_rid = rid
    return best_rid



def _nearest_room(x: float, y: float, rooms: Dict[str, Any]) -> Optional[str]:
    best = None
    best_d2 = 1e18
    for rid, r in rooms.items():
        try:
            cx, cy = float(r["center"][0]), float(r["center"][1])
        except Exception:
            continue
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = str(rid)
    return best


def _build_room_graph(rooms: Dict[str, Any], doors: Dict[str, Any]):
    centers = {str(rid): (float(r["center"][0]), float(r["center"][1])) for rid, r in rooms.items() if "center" in r}
    adj = {rid: [] for rid in centers.keys()}
    for did, d in doors.items():
        try:
            ra = str(d["room_a"]); rb = str(d["room_b"])
            st = int(d.get("state", 1))
        except Exception:
            continue
        if st != 1:
            continue
        if ra not in centers or rb not in centers:
            continue
        ax, ay = centers[ra]
        bx, by = centers[rb]
        w = float(math.hypot(ax - bx, ay - by))
        adj[ra].append((rb, w))
        adj[rb].append((ra, w))
    return centers, adj


def _all_pairs_shortest_paths(centers: Dict[str, Tuple[float, float]], adj: Dict[str, Any]):
    rids = sorted(list(centers.keys()))
    idx = {rid: i for i, rid in enumerate(rids)}
    n = len(rids)
    INF = 1e18
    dist = [[INF] * n for _ in range(n)]
    hops = [[10**9] * n for _ in range(n)]
    for s in rids:
        si = idx[s]
        dist[si][si] = 0.0
        hops[si][si] = 0
        pq = [(0.0, 0, s)]
        while pq:
            dcur, hcur, u = heapq.heappop(pq)
            ui = idx[u]
            if dcur > dist[si][ui] + 1e-9:
                continue
            for v, w in adj.get(u, []):
                vi = idx[v]
                nd = dcur + float(w)
                nh = hcur + 1
                if (nd < dist[si][vi] - 1e-9) or (abs(nd - dist[si][vi]) <= 1e-9 and nh < hops[si][vi]):
                    dist[si][vi] = nd
                    hops[si][vi] = nh
                    heapq.heappush(pq, (nd, nh, v))
    return rids, idx, dist, hops


class GasModelA:
    def __init__(
        self,
        mode_params: Dict,
        D: float = 0.20,
        lambda_decay: float = 0.01,
        clamp_min: float = 0.0,
        clamp_max: float = 10.0,
        source_radius: float = 1.0,
        # v3 additions (all optional)
        release_window_s: float = 20.0,
        n_time_samples: int = 12,
        D_z: float = 0.03,
        topology: Optional[Dict[str, Any]] = None,
    ):
        self.mode_params = dict(mode_params)
        self.D = float(D)
        self.lambda_decay = float(lambda_decay)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.source_radius = float(source_radius)

        self.release_window_s = float(release_window_s)
        self.n_time_samples = int(max(1, n_time_samples))
        self.D_z = float(D_z)

        # Enforce a near-source peak so concentration is highest at the leak (not only downwind).
        # This is a small, isotropic near-field bump added on top of the advected plume.
        self.source_peak_boost = float(self.mode_params.get("source_peak_boost", 1.0))
        self.source_peak_sigma = float(self.mode_params.get("source_peak_sigma", 0.6))  # meters
        self.source_peak_dt_s = float(self.mode_params.get("source_peak_dt_s", 0.5))  # seconds

        # Optional cheap topology awareness
        self._rooms = None
        self._doors = None
        self._room_bboxes = None
        self._room_centers = None
        self._room_ids = None
        self._room_idx = None
        self._room_dist = None
        self._room_hops = None

        if isinstance(topology, dict):
            rooms = topology.get("rooms", None)
            doors = topology.get("doors", None)
            if isinstance(rooms, dict) and isinstance(doors, dict) and len(rooms) > 0:
                self._rooms = rooms
                self._doors = doors
                self._room_bboxes = _room_bboxes(rooms)
                centers, adj = _build_room_graph(rooms, doors)
                rids, idx, dist, hops = _all_pairs_shortest_paths(centers, adj)
                self._room_centers = centers
                self._room_ids = rids
                self._room_idx = idx
                self._room_dist = dist
                self._room_hops = hops

    def _topology_factor(self, x: float, y: float, sx: float, sy: float) -> float:
        """
        Cheap "walls/doors matter" attenuation.
        - Find room of query point and room of source point (via room bbox).
        - Compute shortest open-door room-graph distance.
        - Apply attenuation exp(-beta_len * dist) * (per_door ** hops)
        """
        if self._rooms is None or self._room_bboxes is None or self._room_idx is None:
            return 1.0

        rid_q = _room_of(x, y, self._room_bboxes)
        rid_s = _room_of(sx, sy, self._room_bboxes)

        # Corridor/unassigned: map to nearest room center
        if rid_q is None:
            rid_q = _nearest_room(x, y, self._rooms)
        if rid_s is None:
            rid_s = _nearest_room(sx, sy, self._rooms)
        if rid_q is None or rid_s is None:
            return 1.0

        if rid_q == rid_s:
            return 1.0

        if rid_s not in self._room_idx or rid_q not in self._room_idx:
            return 1.0

        si = self._room_idx[rid_s]
        qi = self._room_idx[rid_q]
        d = float(self._room_dist[si][qi])
        h = int(self._room_hops[si][qi])

        if not math.isfinite(d) or d > 1e17:
            return 0.0

        beta_len = float(self.mode_params.get("topology_beta_len", 0.03))
        per_door = float(self.mode_params.get("topology_per_door", 0.80))
        per_door = max(0.0, min(1.0, per_door))

        return float(math.exp(-beta_len * d) * (per_door ** max(0, h)))

    def query(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        theta: str,
        source_pos: Tuple[float, float, float],
        q: float,
        start_time: float,
    ) -> float:
        """
        Concentration at (x,y,z) time t for source at source_pos with continuous release rate q.
        """
        if t < start_time:
            return 0.0

        x = float(x); y = float(y); z = float(z); t = float(t)
        sx, sy, sz = float(source_pos[0]), float(source_pos[1]), float(source_pos[2])
        q = float(q)

        dt_since = max(0.0, float(t - start_time))
        window = float(self.mode_params.get("release_window_s", self.release_window_s))
        window = max(1e-6, window)
        T = min(dt_since, window)

        # Choose number of samples proportional to window, but capped.
        n = int(self.mode_params.get("n_time_samples", self.n_time_samples))
        n = max(1, min(32, n))
        # Midpoint samples in [0, T]
        dtau = T / float(n)
        taus = [(i + 0.5) * dtau for i in range(n)]

        # Vertical kernel (allow mild vertical diffusion over time via D_z)
        dz = z - sz
        base_sigma_z2 = float(self.mode_params.get("sigma_z2_base", 0.6 ** 2))

        acc = 0.0
        for tau in taus:
            # backtrace parcel at age tau
            xb, yb = backtrace_xy(x, y, t, theta, self.mode_params, tau)

            dx = xb - sx
            dy = yb - sy

            sigma2 = self.source_radius ** 2 + 2.0 * self.D * float(tau)
            sigma2 = max(sigma2, 1e-6)
            norm_xy = 1.0 / (2.0 * math.pi * sigma2)
            g_xy = norm_xy * math.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))

            sigma_z2 = base_sigma_z2 + 2.0 * self.D_z * float(tau)
            sigma_z2 = max(sigma_z2, 1e-6)
            g_z = (1.0 / math.sqrt(2.0 * math.pi * sigma_z2)) * math.exp(-(dz * dz) / (2.0 * sigma_z2))

            acc += math.exp(-self.lambda_decay * float(tau)) * g_xy * g_z

        # Numerical integral over tau in [0, T]
        scale = float(self.mode_params.get("scale_factor", 50.0))
        c = q * acc * dtau * scale

        # Near-field peak bump (helps ensure max at leak, even with strong advection).
        c_peak = 0.0
        if self.source_peak_boost > 0.0 and self.source_peak_sigma > 1e-6:
            dx0 = x - sx
            dy0 = y - sy
            sigma2p = max(self.source_peak_sigma ** 2, 1e-6)
            norm_xy_p = 1.0 / (2.0 * math.pi * sigma2p)
            g_xy_p = norm_xy_p * math.exp(-(dx0 * dx0 + dy0 * dy0) / (2.0 * sigma2p))
            # Use the base vertical kernel (tau≈0)
            g_z0 = (1.0 / math.sqrt(2.0 * math.pi * base_sigma_z2)) * math.exp(-(dz * dz) / (2.0 * base_sigma_z2))
            peak_dt = max(0.0, float(self.source_peak_dt_s))
            c_peak = float(self.source_peak_boost) * q * g_xy_p * g_z0 * peak_dt * scale

        # Topology attenuation (optional)
        topo = self._topology_factor(x, y, sx, sy)
        c = (c + c_peak) * topo

        # Clamp
        c = max(self.clamp_min, min(self.clamp_max, c))
        return float(c)