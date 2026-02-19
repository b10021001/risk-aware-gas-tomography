
"""
Scenario generation and scene building (v1).

Contract note:
- scenario_spec is JSON-serializable and follows the schema in prompt.
- backend.load_scene(scene_spec) expects an occupancy grid and topology outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math
import random

import numpy as np

from .doors import apply_door_pattern


# Fixed geometry defaults (Appendix B)
GEOM_DEFAULTS = {
    "resolution": 0.20,
    "wall_thickness": 0.12,
    "wall_height": 3.20,
    "corridor_width_f1_f2": 2.00,
    "corridor_width_f3": 2.20,
    "door_width": 1.20,
    "door_height": 2.20,
    "door_thickness": 0.05,
}

HVAC_MODE_IDS = ["drift_pos_x","drift_neg_x","vortex_ccw","vortex_cw","drift_pos_y","time_switch"]


def grid_to_world(i: int, j: int, resolution: float, origin: Tuple[float,float]) -> Tuple[float,float]:
    # cell center world coordinates
    return (origin[0] + (j + 0.5) * resolution, origin[1] + (i + 0.5) * resolution)


def world_to_grid(x: float, y: float, resolution: float, origin: Tuple[float,float]) -> Tuple[int,int]:
    j = int(math.floor((x - origin[0]) / resolution))
    i = int(math.floor((y - origin[1]) / resolution))
    return i, j


def carve_rect(occ: np.ndarray, i0: int, j0: int, i1: int, j1: int, value: int = 0) -> None:
    # inclusive-exclusive i1, j1
    H, W = occ.shape
    i0 = max(0, min(H, i0)); i1 = max(0, min(H, i1))
    j0 = max(0, min(W, j0)); j1 = max(0, min(W, j1))
    occ[i0:i1, j0:j1] = value


def _free_mask(occ: np.ndarray) -> np.ndarray:
    # In this benchmark: 0 = free, 1 = wall/blocked
    return (occ == 0)

def _connected_components_free(occ: np.ndarray) -> Tuple[int, np.ndarray]:
    free = _free_mask(occ)
    H, W = occ.shape
    comp = -np.ones((H, W), dtype=np.int32)
    cid = 0
    from collections import deque
    for i in range(H):
        for j in range(W):
            if free[i, j] and comp[i, j] == -1:
                q = deque([(i, j)])
                comp[i, j] = cid
                while q:
                    a, b = q.popleft()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and free[na, nb] and comp[na, nb] == -1:
                            comp[na, nb] = cid
                            q.append((na, nb))
                cid += 1
    return cid, comp

def _carve_l_corridor(occ: np.ndarray, a: Tuple[int,int], b: Tuple[int,int], corridor_w_cells: int) -> None:
    ai, aj = a
    bi, bj = b
    bend = (ai, bj)
    j0, j1 = sorted([aj, bend[1]])
    carve_rect(occ, ai - corridor_w_cells//2, j0, ai + corridor_w_cells//2 + 1, j1 + 1, value=0)
    i0, i1 = sorted([bend[0], bi])
    carve_rect(occ, i0, bj - corridor_w_cells//2, i1 + 1, bj + corridor_w_cells//2 + 1, value=0)

def _ensure_connected_free_space(occ: np.ndarray, corridor_w_cells: int, anchor: Optional[Tuple[int,int]] = None) -> None:
    """
    Ensure all FREE cells (occ==0) are in one connected component by carving L-corridors.
    If anchor is provided and lies on a FREE cell, we connect everything to anchor's component.
    """
    free = (occ == 0)
    H, W = occ.shape
    comp = -np.ones((H, W), dtype=np.int32)

    from collections import deque

    # label components
    cid = 0
    for i in range(H):
        for j in range(W):
            if free[i, j] and comp[i, j] == -1:
                q = deque([(i, j)])
                comp[i, j] = cid
                while q:
                    a, b = q.popleft()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and free[na, nb] and comp[na, nb] == -1:
                            comp[na, nb] = cid
                            q.append((na, nb))
                cid += 1

    if cid <= 1:
        return

    # representative (mean cell) per component
    reps = []
    for k in range(cid):
        idx = np.argwhere(comp == k)
        if idx.size == 0:
            continue
        mi, mj = idx.mean(axis=0)
        reps.append((k, int(mi), int(mj)))

    if not reps:
        return

    # choose base component
    base_cid = reps[0][0]
    if anchor is not None:
        ai, aj = int(anchor[0]), int(anchor[1])
        if 0 <= ai < H and 0 <= aj < W and free[ai, aj]:
            base_cid = int(comp[ai, aj])

    # pick base cell as the representative of base_cid
    base_cell = None
    for k, ri, rj in reps:
        if k == base_cid:
            base_cell = (ri, rj)
            break
    if base_cell is None:
        base_cell = (reps[0][1], reps[0][2])

    # connect every other component to base
    for k, ri, rj in reps:
        if k == base_cid:
            continue
        _carve_l_corridor(occ, base_cell, (ri, rj), corridor_w_cells)

def build_nav_graph_from_rooms(rooms: Dict[str, Any], doors: Dict[str, Any]) -> Dict[str, Any]:
    nodes = []
    edges = []
    # room center nodes
    for rid, r in rooms.items():
        nodes.append({"id": f"room_{rid}", "x": float(r["center"][0]), "y": float(r["center"][1]), "type": "room_center", "room_id": rid})
    # door nodes
    for did, d in doors.items():
        p0 = d["geom"]["p0"]; p1 = d["geom"]["p1"]
        cx = 0.5*(p0[0] + p1[0]); cy = 0.5*(p0[1] + p1[1])
        nodes.append({"id": f"door_{did}", "x": float(cx), "y": float(cy), "type": "door", "room_id": ""})
    # edges: connect door to its two rooms
    # (simple; planners may ignore)
    node_index = {n["id"]: n for n in nodes}
    def dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
    for did, d in doors.items():
        u = f"door_{did}"
        ra = f"room_{d['room_a']}"
        rb = f"room_{d['room_b']}"
        if ra in node_index:
            edges.append({"u": u, "v": ra, "length": dist(node_index[u], node_index[ra])})
        if rb in node_index:
            edges.append({"u": u, "v": rb, "length": dist(node_index[u], node_index[rb])})
    return {"nodes": nodes, "edges": edges}


# ---------------------------
# F1/F2/F3 generators
# ---------------------------

def _gen_grid_layout(seed: int, rooms_n: int, grid_shape: Tuple[int,int], room_size_range_cells: Tuple[Tuple[int,int],Tuple[int,int]],
                     map_size: Tuple[int,int], resolution: float, corridor_w_m: float,
                     doors_fraction: float, max_doors: int,
                     require_loop: bool) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Create a deterministic grid-of-rooms layout and a door library (geoms/connectivity only; door states applied later).
    Returns:
      occ_base: occupancy with rooms/corridors carved, but WITHOUT door closures (doors are open holes)
      rooms: dict room_id -> poly/center/area + a cell bbox for internal use
      doors: dict door_id -> room_a/room_b/state (placeholder 1) + geom
    """
    rng = random.Random(seed)
    H, W = map_size
    origin = (0.0, 0.0)
    occ = np.ones((H, W), dtype=np.uint8)

    gw, gh = grid_shape
    # grid cell size in world meters (allocate evenly)
    cell_w = W / gw
    cell_h = H / gh

    door_width = GEOM_DEFAULTS["door_width"]
    door_thickness = GEOM_DEFAULTS["door_thickness"]
    door_height = GEOM_DEFAULTS["door_height"]

    corridor_w_cells = max(2, int(round(corridor_w_m / resolution)))

    rooms: Dict[str, Any] = {}
    room_cells: Dict[str, Tuple[int,int,int,int]] = {}  # (i0,j0,i1,j1)

    room_ids = []
    # allocate rooms into grid slots
    slots = [(gx, gy) for gy in range(gh) for gx in range(gw)]
    rng.shuffle(slots)
    for k in range(rooms_n):
        gx, gy = slots[k % len(slots)]
        # room size in cells
        (wmin,wmax),(hmin,hmax) = room_size_range_cells
        rw = rng.randint(wmin, wmax)
        rh = rng.randint(hmin, hmax)
        # slot boundaries in cells
        j_slot0 = int(gx * cell_w)
        j_slot1 = int((gx+1) * cell_w)
        i_slot0 = int(gy * cell_h)
        i_slot1 = int((gy+1) * cell_h)

        # leave margins for corridors
        margin = corridor_w_cells + 2
        j0 = min(max(j_slot0 + margin, 0), W-1)
        i0 = min(max(i_slot0 + margin, 0), H-1)
        j1 = min(j0 + rw, j_slot1 - margin)
        i1 = min(i0 + rh, i_slot1 - margin)
        if j1 <= j0 + 2:  # fallback
            j1 = min(j_slot1 - margin, W-1)
        if i1 <= i0 + 2:
            i1 = min(i_slot1 - margin, H-1)

        rid = f"R{k:02d}"
        room_ids.append(rid)
        carve_rect(occ, i0, j0, i1, j1, value=0)
        # compute poly corners in world meters
        x0,y0 = grid_to_world(i0, j0, resolution, origin)
        x1,y1 = grid_to_world(i1-1, j1-1, resolution, origin)
        # poly in CCW
        poly = [[x0-resolution*0.5, y0-resolution*0.5],
                [x1+resolution*0.5, y0-resolution*0.5],
                [x1+resolution*0.5, y1+resolution*0.5],
                [x0-resolution*0.5, y1+resolution*0.5]]
        cx = 0.5*(poly[0][0]+poly[1][0])
        cy = 0.5*(poly[0][1]+poly[2][1])
        area = abs((poly[1][0]-poly[0][0])*(poly[2][1]-poly[1][1]))
        rooms[rid] = {"poly": poly, "center": [cx, cy], "area": float(area)}
        room_cells[rid] = (i0, j0, i1, j1)

    # corridors: carve horizontal/vertical corridors connecting room centers to form a grid connectivity
    # build adjacency candidate edges between nearby rooms by grid proximity
    # We'll approximate room "slot" by its center cell.
    centers_cell: Dict[str, Tuple[int,int]] = {}
    for rid, (i0,j0,i1,j1) in room_cells.items():
        ci = (i0+i1)//2
        cj = (j0+j1)//2
        centers_cell[rid] = (ci,cj)

    # candidate edges based on nearest neighbors in manhattan distance
    # Build a simple kNN graph: for each room, connect to up to 3 nearest rooms.
    candidates: List[Tuple[str,str]] = []
    for a in room_ids:
        ai,aj = centers_cell[a]
        dists = []
        for b in room_ids:
            if a==b: 
                continue
            bi,bj = centers_cell[b]
            d = abs(ai-bi)+abs(aj-bj)
            dists.append((d,b))
        dists.sort(key=lambda x:x[0])
        for _, b in dists[:3]:
            pair = tuple(sorted((a,b)))
            if pair not in candidates:
                candidates.append(pair)

    # ensure connectivity with spanning tree
    rng.shuffle(candidates)
    parent = {rid: rid for rid in room_ids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb:
            parent[rb]=ra
            return True
        return False

    edges: List[Tuple[str,str]] = []
    for a,b in candidates:
        if union(a,b):
            edges.append((a,b))
    # Add extra edges for loops
    if require_loop:
        extra_needed = max(1, min(3, len(candidates)-len(edges)))
    else:
        extra_needed = 0
    remaining = [e for e in candidates if e not in edges]
    rng.shuffle(remaining)
    edges.extend(remaining[:extra_needed])

    # Cap by doors_fraction and max_doors while keeping at least spanning tree size
    min_edges = max(1, len(room_ids)-1)
    target = int(round(doors_fraction * len(edges)))
    target = max(min_edges, target)
    target = min(max_doors, target)
    if len(edges) > target:
        rng.shuffle(edges)
        # keep first target but ensure connectivity remains (approx; since tree edges included, ok)
        edges = edges[:target]

    # carve corridors between connected rooms and define doors
    doors: Dict[str, Any] = {}
    door_id_list = []
    for k,(a,b) in enumerate(edges):
        ai,aj = centers_cell[a]
        bi,bj = centers_cell[b]
        # carve L-shaped corridor: horizontal then vertical
        # choose bend at (ai,bj)
        bend_i, bend_j = ai, bj
        # carve along ai to bend_j
        j_start, j_end = sorted([aj, bend_j])
        i_center = ai
        carve_rect(occ, i_center - corridor_w_cells//2, j_start, i_center + corridor_w_cells//2 + 1, j_end+1, value=0)
        # carve along bend_i to bi at bend_j
        i_start, i_end = sorted([bend_i, bi])
        j_center = bend_j
        carve_rect(occ, i_start, j_center - corridor_w_cells//2, i_end+1, j_center + corridor_w_cells//2 + 1, value=0)

        # define a door at the approximate midpoint of corridor near room a boundary.
        # We'll place door at room a boundary along direction toward b.
        # Determine which axis dominates
        dx = bj - aj
        dy = bi - ai
        if abs(dx) >= abs(dy):
            # horizontal connection: door at room a right/left wall
            i0,j0,i1,j1 = room_cells[a]
            if dx > 0:
                j_door = j1  # right wall outside room
                i_door = (i0+i1)//2
                # p0-p1 along y (vertical door)
                x_d = grid_to_world(i_door, j_door, resolution, origin)[0]
                y_d = grid_to_world(i_door, j_door, resolution, origin)[1]
                p0 = [x_d, y_d - door_width/2]
                p1 = [x_d, y_d + door_width/2]
            else:
                j_door = j0-1
                i_door = (i0+i1)//2
                x_d,y_d = grid_to_world(i_door, max(0,j_door), resolution, origin)
                p0 = [x_d, y_d - door_width/2]
                p1 = [x_d, y_d + door_width/2]
        else:
            # vertical connection: door at room a top/bottom wall
            i0,j0,i1,j1 = room_cells[a]
            if dy > 0:
                i_door = i1
                j_door = (j0+j1)//2
                x_d,y_d = grid_to_world(i_door, j_door, resolution, origin)
                p0 = [x_d - door_width/2, y_d]
                p1 = [x_d + door_width/2, y_d]
            else:
                i_door = i0-1
                j_door = (j0+j1)//2
                x_d,y_d = grid_to_world(max(0,i_door), j_door, resolution, origin)
                p0 = [x_d - door_width/2, y_d]
                p1 = [x_d + door_width/2, y_d]

        did = f"D{k:02d}"
        door_id_list.append(did)
        doors[did] = {
            "room_a": a,
            "room_b": b,
            "state": 1,
            "geom": {"p0": [float(p0[0]),float(p0[1])],
                     "p1": [float(p1[0]),float(p1[1])],
                     "z_bottom": 0.0,
                     "height": float(door_height),
                     "thickness": float(door_thickness)}
        }

    # ensure free-space connectivity (paper-grade)
    _ensure_connected_free_space(occ, corridor_w_cells)

    return occ, rooms, doors


def generate_f1(seed: int) -> Dict[str, Any]:
    """
    Deterministic F1 small scenario_spec.
    """
    resolution = GEOM_DEFAULTS["resolution"]
    map_size = (80, 80)  # H,W
    occ, rooms, doors = _gen_grid_layout(
        seed=seed,
        rooms_n=6,
        grid_shape=(3,2),
        room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
        map_size=map_size,
        resolution=resolution,
        corridor_w_m=GEOM_DEFAULTS["corridor_width_f1_f2"],
        doors_fraction=0.35,
        max_doors=8,
        require_loop=True,
    )
    # choose door pattern default
    pattern_id = "all_open"
    door_states = apply_door_pattern(pattern_id, list(doors.keys()), seed)
    for did, s in door_states.items():
        doors[did]["state"] = int(s)

    # pick a leak room and position at room center (z within z_layers)
    leak_room = sorted(list(rooms.keys()))[0]
    leak_pos = [float(rooms[leak_room]["center"][0]), float(rooms[leak_room]["center"][1]), 1.0]
    scenario_spec = {
        "scenario_family": "F1",
        "scenario_id": f"f1_{seed:04d}",
        "generator": {"seed": int(seed), "params": {}},
        "map": {"resolution": float(resolution), "size": [int(map_size[0]), int(map_size[1])]},
        "doors": {
            "pattern_id": pattern_id,
            "states": {k:int(v) for k,v in door_states.items()},
            "door_geoms": {k: doors[k]["geom"] for k in doors.keys()},
        },
        "hvac": {
            "theta_true": "drift_pos_x",
            "theta_library": ["drift_pos_x","vortex_ccw","time_switch"],
            "mode_params": {
                "drift_speed": 0.60,
                "vortex_strength": 0.35,
                "time_switch_t": 30.0,
            },
        },
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.2, "start_time": 0.0},
        "sensor": {"tau": 2.0, "sigma": 0.03, "drift": {"enabled": 0, "rate": 0.001, "bias0": 0.0}},
    }
    return scenario_spec


def generate_f2(seed: int, rooms_n: int = 14) -> Dict[str, Any]:
    resolution = GEOM_DEFAULTS["resolution"]
    map_size = (160, 160)
    occ, rooms, doors = _gen_grid_layout(
        seed=seed,
        rooms_n=rooms_n,
        grid_shape=(4,4),
        room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
        map_size=map_size,
        resolution=resolution,
        corridor_w_m=GEOM_DEFAULTS["corridor_width_f1_f2"],
        doors_fraction=0.35,
        max_doors=10,
        require_loop=True,
    )
    # door pattern default: half_random
    pattern_id = "half_random"
    door_states = apply_door_pattern(pattern_id, list(doors.keys()), seed)
    for did, s in door_states.items():
        doors[did]["state"] = int(s)
    # robot spawn: deterministic "lobby" room (first room by id)
    spawn_room = sorted(list(rooms.keys()))[0]
    spawn_xy = rooms[spawn_room]["center"]
    robot_spawn = [float(spawn_xy[0]), float(spawn_xy[1]), 0.2]

    # Build room adjacency graph from doors (regardless of door state; geometry connectivity)
    adj = {rid: [] for rid in rooms.keys()}
    for did, d in doors.items():
        a = d.get("room_a"); b = d.get("room_b")
        if a in adj and b in adj:
            adj[a].append(b); adj[b].append(a)

    # BFS hop-distance from spawn_room to define reachability
    from collections import deque
    dist_hop = {spawn_room: 0}
    q = deque([spawn_room])
    while q:
        r0 = q.popleft()
        for r1 in adj.get(r0, []):
            if r1 not in dist_hop:
                dist_hop[r1] = dist_hop[r0] + 1
                q.append(r1)

    reachable = [r for r in rooms.keys() if r in dist_hop]

    # Leak room: reachable AND far enough (avoid trivial leak=start)
    rng = random.Random(seed + 202)
    min_leak_dist_m = 8.0

    def room_dist(rid):
        cx, cy = rooms[rid]["center"]
        return math.hypot(cx - spawn_xy[0], cy - spawn_xy[1])

    cands = [rid for rid in reachable if room_dist(rid) >= min_leak_dist_m]

    if not cands:
        reachable_sorted = sorted(reachable, key=lambda rid: (dist_hop.get(rid, -1), room_dist(rid)), reverse=True)
        cands = reachable_sorted[:max(1, min(5, len(reachable_sorted)))]

    leak_room = rng.choice(cands) if cands else rng.choice(sorted(list(rooms.keys())))
    leak_pos = [float(rooms[leak_room]["center"][0]), float(rooms[leak_room]["center"][1]), 1.0]
    scenario_spec = {
        "scenario_family": "F2",
        "scenario_id": f"f2_{seed:04d}",
        "generator": {"seed": int(seed), "params": {"rooms_n": int(rooms_n)}},
        "map": {"resolution": float(resolution), "size": [int(map_size[0]), int(map_size[1])]},
        "doors": {
            "pattern_id": pattern_id,
            "states": {k:int(v) for k,v in door_states.items()},
            "door_geoms": {k: doors[k]["geom"] for k in doors.keys()},
        },
        "hvac": {
            "theta_true": "drift_pos_x",
            "theta_library": ["drift_pos_x","vortex_ccw","time_switch"],
            "mode_params": {
                "drift_speed": 0.60,
                "vortex_strength": 0.35,
                "time_switch_t": 30.0,
            },
        },
        "robot": {"spawn": {"room_id": spawn_room, "pos": robot_spawn}},
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.2, "start_time": 5.0},
        "sensor": {"tau": 2.0, "sigma": 0.03, "drift": {"enabled": 0, "rate": 0.001, "bias0": 0.0}},
    }
    return scenario_spec


def generate_f3(seed: int, rooms_n: int = 34) -> Dict[str, Any]:
    resolution = GEOM_DEFAULTS["resolution"]
    map_size = (240, 240)
    occ, rooms, doors = _gen_grid_layout(
        seed=seed,
        rooms_n=rooms_n,
        grid_shape=(6,6),
        room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
        map_size=map_size,
        resolution=resolution,
        corridor_w_m=GEOM_DEFAULTS["corridor_width_f3"],
        doors_fraction=0.40,
        max_doors=24,
        require_loop=True,
    )
    pattern_id = "half_random"
    door_states = apply_door_pattern(pattern_id, list(doors.keys()), seed)
    for did, s in door_states.items():
        doors[did]["state"] = int(s)

    rng = random.Random(seed + 303)
    leak_room = rng.choice(sorted(list(rooms.keys())))
    leak_pos = [float(rooms[leak_room]["center"][0]), float(rooms[leak_room]["center"][1]), 1.0]
    scenario_spec = {
        "scenario_family": "F3",
        "scenario_id": f"f3_{seed:04d}",
        "generator": {"seed": int(seed), "params": {"rooms_n": int(rooms_n)}},
        "map": {"resolution": float(resolution), "size": [int(map_size[0]), int(map_size[1])]},
        "doors": {
            "pattern_id": pattern_id,
            "states": {k:int(v) for k,v in door_states.items()},
            "door_geoms": {k: doors[k]["geom"] for k in doors.keys()},
        },
        "hvac": {
            "theta_true": "drift_pos_x",
            "theta_library": ["drift_pos_x","vortex_ccw","time_switch"],
            "mode_params": {
                "drift_speed": 0.60,
                "vortex_strength": 0.35,
                "time_switch_t": 30.0,
            },
        },
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.2, "start_time": 0.0},
        "sensor": {"tau": 2.0, "sigma": 0.03, "drift": {"enabled": 0, "rate": 0.001, "bias0": 0.0}},
    }
    return scenario_spec


def reconstruct_layout_from_spec(scenario_spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Deterministically reconstruct layout (occupancy + rooms + doors + nav_graph) from scenario_spec.

    Note: door STATES come from scenario_spec["doors"]["states"] (true doors).
    """
    fam = scenario_spec["scenario_family"]
    seed = int(scenario_spec["generator"]["seed"])
    resolution = float(scenario_spec["map"]["resolution"])
    H, W = int(scenario_spec["map"]["size"][0]), int(scenario_spec["map"]["size"][1])

    if fam == "F1":
        occ, rooms, doors = _gen_grid_layout(
            seed=seed,
            rooms_n=6,
            grid_shape=(3,2),
            room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
            map_size=(H,W),
            resolution=resolution,
            corridor_w_m=GEOM_DEFAULTS["corridor_width_f1_f2"],
            doors_fraction=0.35,
            max_doors=8,
            require_loop=True,
        )
    elif fam == "F2":
        rooms_n = int(scenario_spec["generator"]["params"].get("rooms_n", 14))
        occ, rooms, doors = _gen_grid_layout(
            seed=seed,
            rooms_n=rooms_n,
            grid_shape=(4,4),
            room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
            map_size=(H,W),
            resolution=resolution,
            corridor_w_m=GEOM_DEFAULTS["corridor_width_f1_f2"],
            doors_fraction=0.35,
            max_doors=10,
            require_loop=True,
        )
    elif fam == "F3":
        rooms_n = int(scenario_spec["generator"]["params"].get("rooms_n", 34))
        occ, rooms, doors = _gen_grid_layout(
            seed=seed,
            rooms_n=rooms_n,
            grid_shape=(6,6),
            room_size_range_cells=((int(4/resolution), int(8/resolution)), (int(4/resolution), int(8/resolution))),
            map_size=(H,W),
            resolution=resolution,
            corridor_w_m=GEOM_DEFAULTS["corridor_width_f3"],
            doors_fraction=0.40,
            max_doors=24,
            require_loop=True,
        )
    else:
        raise ValueError(f"Unknown scenario_family: {fam}")

    # Apply true door states and geoms from spec
    door_states = scenario_spec["doors"]["states"]
    door_geoms = scenario_spec["doors"]["door_geoms"]
    for did in list(doors.keys()):
        if did in door_states:
            doors[did]["state"] = int(door_states[did])
        if did in door_geoms:
            doors[did]["geom"] = door_geoms[did]

    # Apply door closures to occupancy: block a small rectangle around door center when closed
    origin = (0.0, 0.0)
    for did, d in doors.items():
        if int(d["state"]) == 1:
            continue
        p0 = d["geom"]["p0"]; p1 = d["geom"]["p1"]
        cx = 0.5*(p0[0]+p1[0]); cy = 0.5*(p0[1]+p1[1])
        # block 2x2 cells around center
        ci, cj = world_to_grid(cx, cy, resolution, origin)
        carve_rect(occ, ci-1, cj-1, ci+2, cj+2, value=1)

    nav_graph = build_nav_graph_from_rooms(rooms, doors)
    return occ, rooms, doors, nav_graph


def build_scene_dict_from_scenario_spec(scenario_spec: Dict[str, Any]) -> Dict[str, Any]:
    occ, rooms, doors, nav_graph = reconstruct_layout_from_spec(scenario_spec)
    resolution = float(scenario_spec["map"]["resolution"])
    origin = [0.0, 0.0]
    # doors contract wants dict door_id -> {room_a, room_b, state, geom}
    doors_out = {}
    for did, d in doors.items():
        doors_out[did] = {
            "room_a": d["room_a"],
            "room_b": d["room_b"],
            "state": int(d["state"]),
            "geom": d["geom"],
        }
    return {
        "occupancy": occ,
        "resolution": resolution,
        "origin": origin,
        "nav_graph": nav_graph,
        "rooms": rooms,
        "doors": doors_out,
        "isaac_stage_path": None,
    }


# =====================================================================
# F2 SINGLE CONNECTED OVERRIDE (paper-grade, long connected corridor)
# - Keeps the rest of the file intact.
# - Overrides only generate_f2 and reconstruct_layout_from_spec.
# =====================================================================

def _build_f2_single_connected(seed: int, resolution: float, map_size: Tuple[int,int]) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Deterministic single connected long-corridor layout.
    occ: 0=free, 1=wall.
    """
    H, W = map_size
    occ = np.ones((H, W), dtype=np.uint8)

    cw = max(3, int(round(GEOM_DEFAULTS["corridor_width_f1_f2"] / resolution)))
    pad = max(2, cw)

    # Outer loop
    carve_rect(occ, pad, pad, pad+cw, W-pad, value=0)
    carve_rect(occ, H-pad-cw, pad, H-pad, W-pad, value=0)
    carve_rect(occ, pad, pad, H-pad, pad+cw, value=0)
    carve_rect(occ, pad, W-pad-cw, H-pad, W-pad, value=0)

    # Central spine
    mid_j = W//2
    carve_rect(occ, pad, mid_j - cw//2, H-pad, mid_j + cw//2 + 1, value=0)

    # Horizontal branches
    for frac in (0.30, 0.55, 0.75):
        i = int(H*frac)
        carve_rect(occ, i - cw//2, pad, i + cw//2 + 1, W-pad, value=0)

    # Alcoves (dead-ends but connected)
    for (ci,cj) in [(int(H*0.20), int(W*0.20)), (int(H*0.80), int(W*0.80)), (int(H*0.50), int(W*0.80))]:
        carve_rect(occ, ci - cw, cj - cw, ci + cw, cj + cw, value=0)

    def cell_center(i,j):
        return grid_to_world(i, j, resolution, (0.0, 0.0))

    # Minimal room anchors
    room_cells = {
        "R00": (pad + cw//2, pad + cw//2),
        "R01": (pad + cw//2, W-pad - cw//2 - 1),
        "R02": (H//2, mid_j),
        "R03": (H-pad - cw//2 - 1, pad + cw//2),
        "R04": (H-pad - cw//2 - 1, W-pad - cw//2 - 1),
        "R05": (int(H*0.55), int(W*0.80)),
    }
    rooms: Dict[str, Any] = {}
    for rid, (i,j) in room_cells.items():
        x,y = cell_center(i,j)
        rooms[rid] = {"center": [float(x), float(y)]}

    def door_geom_at(i: int, j: int, vertical: bool) -> Dict[str, Any]:
        x,y = cell_center(i,j)
        half = (cw*resolution)/2.0
        if vertical:
            p0 = [x, y-half]; p1 = [x, y+half]
        else:
            p0 = [x-half, y]; p1 = [x+half, y]
        return {"p0": [float(p0[0]), float(p0[1])], "p1": [float(p1[0]), float(p1[1])],
                "z_bottom": 0.0, "height": float(GEOM_DEFAULTS["door_height"]), "thickness": float(GEOM_DEFAULTS["door_thickness"])}

    doors: Dict[str, Any] = {}
    door_list = [
        ("D00","R00","R02", int(H*0.30), mid_j, False),
        ("D01","R02","R01", int(H*0.55), mid_j, False),
        ("D02","R02","R03", int(H*0.75), mid_j, False),
        ("D03","R02","R04", H-pad-cw, mid_j, False),
        ("D04","R02","R05", int(H*0.55), int(W*0.72), True),
    ]
    for did, ra, rb, i, j, vertical in door_list:
        doors[did] = {"room_a": ra, "room_b": rb, "state": 1, "geom": door_geom_at(i,j,vertical)}
    return occ, rooms, doors


# Keep a reference to the original reconstruct function
_old_reconstruct_layout_from_spec = reconstruct_layout_from_spec

def reconstruct_layout_from_spec(scenario_spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Override: if layout_id==F2_SINGLE, build deterministic connected layout.
    Otherwise delegate to the original function.
    """
    layout_id = (scenario_spec.get("generator", {}) or {}).get("params", {}).get("layout_id", "")
    if layout_id != "F2_SINGLE":
        return _old_reconstruct_layout_from_spec(scenario_spec)

    resolution = float(scenario_spec["map"]["resolution"])
    map_size = (int(scenario_spec["map"]["size"][0]), int(scenario_spec["map"]["size"][1]))
    occ, rooms, doors = _build_f2_single_connected(int((scenario_spec.get("generator", {}) or {}).get("seed", 0)), resolution, map_size)

    # Apply door states with thin blocking that preserves a 1-cell gap
    states = (scenario_spec.get("doors", {}) or {}).get("states", {}) or {}
    door_geoms = (scenario_spec.get("doors", {}) or {}).get("door_geoms", {}) or {}
    for did, geom in door_geoms.items():
        if int(states.get(did, 1)) == 1:
            continue
        p0 = geom["p0"]; p1 = geom["p1"]
        cx = 0.5*(p0[0]+p1[0]); cy = 0.5*(p0[1]+p1[1])
        ci, cj = world_to_grid(cx, cy, resolution, (0.0,0.0))
        vertical = abs(p0[0]-p1[0]) < abs(p0[1]-p1[1])
        if vertical:
            carve_rect(occ, ci-1, cj, ci+2, cj+1, value=1)
            carve_rect(occ, ci+1, cj, ci+2, cj+1, value=0)  # keep gap
        else:
            carve_rect(occ, ci, cj-1, ci+1, cj+2, value=1)
            carve_rect(occ, ci, cj+1, ci+1, cj+2, value=0)

    # Ensure final connectivity
    cw = max(3, int(round(GEOM_DEFAULTS["corridor_width_f1_f2"] / resolution)))
    _ensure_connected_free_space(occ, cw)

    topo = {"resolution": float(resolution), "origin": (0.0, 0.0)}
    return occ, rooms, doors, topo


def generate_f2(seed: int, rooms_n: int = 14) -> Dict[str, Any]:
    """Override: deterministic single connected corridor benchmark for F2."""
    resolution = GEOM_DEFAULTS["resolution"]
    map_size = (160, 160)

    occ, rooms, doors = _build_f2_single_connected(seed, resolution, map_size)

    pattern_id = "half_random"
    door_states = apply_door_pattern(pattern_id, list(doors.keys()), seed)
    for did, s in door_states.items():
        doors[did]["state"] = int(s)

    # Spawn at center anchor
    spawn_room = "R02"
    sx, sy = rooms[spawn_room]["center"]
    robot_spawn = [float(sx), float(sy), 0.2]

    # Leak far from spawn
    rng = random.Random(seed + 202)
    min_leak_dist_m = 14.0
    room_list = [r for r in rooms.keys() if r != spawn_room]
    dists = sorted([(r, math.hypot(rooms[r]["center"][0]-sx, rooms[r]["center"][1]-sy)) for r in room_list],
                   key=lambda x: x[1], reverse=True)
    top = [r for r,d in dists if d >= min_leak_dist_m]
    if not top:
        top = [dists[0][0]]
    top = top[:3]
    leak_room = rng.choice(top)
    lx, ly = rooms[leak_room]["center"]
    leak_pos = [float(lx), float(ly), 1.0]

    scenario_spec = {
        "scenario_family": "F2",
        "scenario_id": f"f2_{seed:04d}",
        "generator": {"seed": int(seed), "params": {"rooms_n": int(rooms_n), "layout_id": "F2_SINGLE"}},
        "map": {"resolution": float(resolution), "size": [int(map_size[0]), int(map_size[1])]},
        "rooms": rooms,
        "doors": {
            "pattern_id": pattern_id,
            "states": {k:int(v) for k,v in door_states.items()},
            "door_geoms": {k: doors[k]["geom"] for k in doors.keys()},
        },
        "hvac": {
            "theta_true": "drift_pos_x",
            "theta_library": ["drift_pos_x","vortex_ccw","time_switch"],
            "mode_params": {"drift_speed": 0.60, "vortex_strength": 0.35, "time_switch_t": 30.0},
        },
        "robot": {"spawn": {"room_id": spawn_room, "pos": robot_spawn}},
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.2, "start_time": 5.0},
        "sensor": {"tau": 2.0, "sigma": 0.03, "drift": {"enabled": 0, "rate": 0.001, "bias0": 0.0}},
    }
    return scenario_spec
