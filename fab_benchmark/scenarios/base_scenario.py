
"""
Scenario generation and scene building (v1).

Contract note:
- scenario_spec is JSON-serializable and follows the schema in prompt.
- backend.load_scene(scene_spec) expects an occupancy grid and topology outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
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
    # Default discrete map size (rows, cols) used by some grid-based
    # generators / helpers. Keeping this here avoids KeyError when code
    # paths expect GEOM_DEFAULTS['map_size'].
    "map_size": (160, 160),
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

# ---------------------------------------------------------------------
# Connectivity repair: ensure FREE space (occ==0) is a single component.
# Used by the F2_SINGLE family to guarantee global traversability.
# ---------------------------------------------------------------------
def _cc_label_free(occ: np.ndarray):
    free = (occ == 0)
    H, W = occ.shape
    comp = -np.ones((H, W), dtype=np.int32)
    from collections import deque
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
    return cid, comp

def _carve_l_corridor_free(occ: np.ndarray, a: Tuple[int,int], b: Tuple[int,int], corridor_w_cells: int) -> None:
    ai, aj = a
    bi, bj = b
    bend = (ai, bj)
    j0, j1 = sorted([aj, bend[1]])
    carve_rect(occ, ai - corridor_w_cells//2, j0, ai + corridor_w_cells//2 + 1, j1 + 1, value=0)
    i0, i1 = sorted([bend[0], bi])
    carve_rect(occ, i0, bj - corridor_w_cells//2, i1 + 1, bj + corridor_w_cells//2 + 1, value=0)

def _ensure_connected_free_space(occ: np.ndarray, corridor_w_cells: int) -> None:
    for _ in range(10):
        n, comp = _cc_label_free(occ)
        if n <= 1:
            return
        sizes = [(k, int((comp == k).sum())) for k in range(n)]
        base_cid = max(sizes, key=lambda x: x[1])[0]
        reps = []
        for k in range(n):
            idx = np.argwhere(comp == k)
            if idx.size == 0:
                continue
            mi, mj = idx.mean(axis=0)
            reps.append((k, int(mi), int(mj)))
        base_cell = next(((ri, rj) for k, ri, rj in reps if k == base_cid), (reps[0][1], reps[0][2]))
        for k, ri, rj in reps:
            if k == base_cid:
                continue
            _carve_l_corridor_free(occ, base_cell, (ri, rj), corridor_w_cells)


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


def generate_f2(seed: int, rooms_n: int = 14, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    # leak room: random room
    rng = random.Random(seed + 202)
    leak_room = rng.choice(sorted(list(rooms.keys())))
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
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.4, "start_time": 0.0},
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
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.4, "start_time": 0.0},
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
        params = (scenario_spec.get("generator", {}) or {}).get("params", {}) or {}
        layout_id = str(params.get("layout_id", "")).upper()
        rooms_n = int(params.get("rooms_n", 14))

        # IMPORTANT: keep reconstruction consistent with generate_f2().
        # If we generated F2_SINGLE (paper map), we must reconstruct the same layout here;
        # otherwise leak/tier1 coordinates won't match the occupancy used by the backend/policy.
        if layout_id == "F2_SINGLE":
            occ, rooms, doors = _build_f2_single_connected(
                seed=seed,
                resolution=resolution,
                map_size=(H, W),
                rooms_n=rooms_n,
            )
        else:
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
# F2 SINGLE CONNECTED (FINAL): seed-dependent connected map + stable doors
# =====================================================================

def _build_f2_single_connected(
    seed: int,
    resolution: float,
    map_size: Tuple[int, int],
    rooms_n: int = None,
    **_kwargs,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Seed-dependent connected long-corridor layout (paper-grade, map OOD).

    occ: 0=free, 1=wall. Backbone: outer loop + a jittered central spine.
    Variations: branch positions/spans, notches, alcoves, door locations.
    """
    H, W = map_size
    occ = np.ones((H, W), dtype=np.uint8)

    rng = random.Random(int(seed) * 1009 + 7)

    cw = max(3, int(round(GEOM_DEFAULTS["corridor_width_f1_f2"] / resolution)))
    pad = max(2, cw + 2)

    # Outer loop backbone (always connected)
    carve_rect(occ, pad, pad, pad+cw, W-pad, value=0)
    carve_rect(occ, H-pad-cw, pad, H-pad, W-pad, value=0)
    carve_rect(occ, pad, pad, H-pad, pad+cw, value=0)
    carve_rect(occ, pad, W-pad-cw, H-pad, W-pad, value=0)

    # Central spine with small horizontal jitter
    mid_j = W//2 + rng.randint(-cw, cw)
    mid_j = max(pad + cw, min(W - pad - cw - 1, mid_j))
    carve_rect(occ, pad, mid_j - cw//2, H-pad, mid_j + cw//2 + 1, value=0)

    # Choose 3 distinct branch y-levels
    pool = [int(H*p) for p in (0.22, 0.30, 0.38, 0.48, 0.55, 0.63, 0.72, 0.80)]
    rng.shuffle(pool)
    branches = []
    for y in pool:
        if all(abs(y - yy) >= int(0.12*H) for yy in branches):
            branches.append(y)
        if len(branches) == 3:
            break
    if len(branches) < 3:
        branches = (branches + pool)[:3]

    # Branches: variable spans and small notches (still connected)
    for i in branches:
        left = pad + rng.randint(0, int(0.15*W))
        right = W - pad - rng.randint(0, int(0.15*W))
        if rng.random() < 0.35:
            if rng.random() < 0.5:
                right = mid_j + rng.randint(cw, int(0.25*W))
            else:
                left = mid_j - rng.randint(cw, int(0.25*W))
        left = max(pad, min(W-pad-1, left))
        right = max(left+cw+2, min(W-pad, right))
        carve_rect(occ, i - cw//2, left, i + cw//2 + 1, right, value=0)

        if rng.random() < 0.5:
            notch_w = rng.randint(cw, 2*cw)
            notch_j = rng.randint(left+cw, max(left+cw+1, right-notch_w-cw))
            carve_rect(occ, i - cw//2, notch_j, i + cw//2 + 1, notch_j+notch_w, value=1)
            carve_rect(occ, i, notch_j, i+1, notch_j+notch_w, value=0)

    # Alcoves: attach small pockets to nearby free cells
    alcove_candidates = []
    for _ in range(100):
        ci = rng.randint(pad+cw, H-pad-cw-1)
        cj = rng.randint(pad+cw, W-pad-cw-1)
        if occ[ci, cj] == 0:
            continue
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = ci+di, cj+dj
            if pad <= ni < H-pad and pad <= nj < W-pad and occ[ni, nj] == 0:
                alcove_candidates.append((ci, cj, ni, nj))
                break
    rng.shuffle(alcove_candidates)
    k_alc = rng.randint(2, 6)
    for (ci, cj, ni, nj) in alcove_candidates[:k_alc]:
        neck_len = rng.randint(cw, 3*cw)
        if ci == ni:
            step = 1 if cj > nj else -1
            j0 = min(cj, cj - step*neck_len)
            j1 = max(cj, cj - step*neck_len)
            carve_rect(occ, ci - cw//2, j0, ci + cw//2 + 1, j1 + 1, value=0)
            box_w = rng.randint(2*cw, 4*cw)
            bj0 = max(pad, min(W-pad-box_w, cj - box_w//2))
            carve_rect(occ, ci - 2*cw, bj0, ci + 2*cw, bj0 + box_w, value=0)
        else:
            step = 1 if ci > ni else -1
            i0 = min(ci, ci - step*neck_len)
            i1 = max(ci, ci - step*neck_len)
            carve_rect(occ, i0, cj - cw//2, i1 + 1, cj + cw//2 + 1, value=0)
            box_h = rng.randint(2*cw, 4*cw)
            bi0 = max(pad, min(H-pad-box_h, ci - box_h//2))
            carve_rect(occ, bi0, cj - 2*cw, bi0 + box_h, cj + 2*cw, value=0)

    # Ensure final connectivity
    _ensure_connected_free_space(occ, max(3, int(2*cw)))

    # Anchor rooms (for spawn/leak)
    def cell_center(i,j):
        return grid_to_world(i, j, resolution, (0.0, 0.0))

    anchors = [
        ("R00", pad + cw//2, pad + cw//2),
        ("R01", pad + cw//2, W-pad - cw//2 - 1),
        ("R02", H//2, mid_j),
        ("R03", H-pad - cw//2 - 1, pad + cw//2),
        ("R04", H-pad - cw//2 - 1, W-pad - cw//2 - 1),
        ("R05", branches[1], max(pad+cw, min(W-pad-cw-1, mid_j + rng.randint(int(0.12*W), int(0.22*W))))),
    ]
    rooms: Dict[str, Any] = {}
    for rid, ii, jj in anchors:
        x,y = cell_center(int(ii), int(jj))
        # Define a simple rectangular room polygon around the anchor center.
        # This is required by HypothesisGrid (expects r["poly"]).
        room_half = max(3.0, 3.0 * cw * resolution)  # meters, make it reasonably sized
        x0, y0 = float(x - room_half), float(y - room_half)
        x1, y1 = float(x + room_half), float(y + room_half)
        rooms[rid] = {
            "center": [float(x), float(y)],
            "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],  # CCW rectangle
            }


    # Door geometry: short segment of physical door width, centered at door cell.
    door_w_cells = max(2, int(round(GEOM_DEFAULTS["door_width"] / resolution)))
    half_cells = max(1, door_w_cells // 2)

    def door_geom_at(i: int, j: int) -> Dict[str, Any]:
        def run(ci, cj, di, dj, max_steps=200):
            steps=0
            while steps < max_steps:
                ni, nj = ci+di, cj+dj
                if not (0<=ni<H and 0<=nj<W): break
                if occ[ni,nj] == 1: break
                ci, cj = ni, nj
                steps += 1
            return steps
        wi = run(i,j,-1,0)+run(i,j,1,0)+1
        wj = run(i,j,0,-1)+run(i,j,0,1)+1
        vertical = (wi <= wj)

        if vertical:
            a_i, a_j = max(0, i-half_cells), j
            b_i, b_j = min(H-1, i+half_cells), j
        else:
            a_i, a_j = i, max(0, j-half_cells)
            b_i, b_j = i, min(W-1, j+half_cells)

        x0,y0 = grid_to_world(a_i, a_j, resolution, (0.0,0.0))
        x1,y1 = grid_to_world(b_i, b_j, resolution, (0.0,0.0))
        return {"p0":[float(x0),float(y0)], "p1":[float(x1),float(y1)],
                "z_bottom":0.0, "height":float(GEOM_DEFAULTS["door_height"]),
                "thickness":float(GEOM_DEFAULTS["door_thickness"])}

    door_candidates = []
    for frac in (0.30, 0.45, 0.55, 0.70):
        door_candidates.append((int(H*frac), mid_j))
    for bi in branches:
        door_candidates.append((bi, mid_j))
        door_candidates.append((bi, max(pad+cw, min(W-pad-cw-1, mid_j + rng.randint(int(0.10*W), int(0.20*W))))))

    rng.shuffle(door_candidates)
    min_sep = max(10, int(3.0*cw))
    chosen = []
    used = []
    k_doors = rng.randint(4, 8)
    for ii, jj in door_candidates:
        if any(abs(ii-ui)+abs(jj-uj) < min_sep for ui, uj in used):
            continue
        used.append((ii, jj))
        chosen.append((ii, jj))
        if len(chosen) >= k_doors:
            break

    doors: Dict[str, Any] = {}
    for idx, (ii, jj) in enumerate(chosen):
        did = f"D{idx:02d}"
        ra, rb = "R02", rng.choice([r for r in rooms.keys() if r != "R02"])
        doors[did] = {"room_a": ra, "room_b": rb, "state": 1, "geom": door_geom_at(ii, jj)}

    return occ, rooms, doors


_old_reconstruct_layout_from_spec = reconstruct_layout_from_spec

def reconstruct_layout_from_spec(scenario_spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    layout_id = (scenario_spec.get("generator", {}) or {}).get("params", {}).get("layout_id", "")
    if layout_id != "F2_SINGLE":
        return _old_reconstruct_layout_from_spec(scenario_spec)

    resolution = float(scenario_spec["map"]["resolution"])
    map_size = (int(scenario_spec["map"]["size"][0]), int(scenario_spec["map"]["size"][1]))
    occ, rooms, doors = _build_f2_single_connected(int((scenario_spec.get("generator", {}) or {}).get("seed", 0)), resolution, map_size)

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
            carve_rect(occ, ci+1, cj, ci+2, cj+1, value=0)
        else:
            carve_rect(occ, ci, cj-1, ci+1, cj+2, value=1)
            carve_rect(occ, ci, cj+1, ci+1, cj+2, value=0)

    _ensure_connected_free_space(occ, max(3, int(round(GEOM_DEFAULTS["corridor_width_f1_f2"]/resolution))*2))

    topo = {"resolution": float(resolution), "origin": (0.0, 0.0)}
    return occ, rooms, doors, topo



# ---------------------------------------------------------------------
# Tier-1 sensor placement (anti-cheat): random offsets + distance difficulty
# + triggerability check (covered by plume) for the notified sensors only.
# ---------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))

def _room_bbox(poly: List[List[float]]) -> Tuple[float,float,float,float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def _is_free_xy(occ: np.ndarray, x: float, y: float, resolution: float) -> bool:
    # World origin is (0,0) for all generators in this file.
    i, j = world_to_grid(float(x), float(y), float(resolution), (0.0, 0.0))
    if i < 0 or j < 0 or i >= occ.shape[0] or j >= occ.shape[1]:
        return False
    return int(occ[i, j]) == 0


def _is_free_disk(
    occ: np.ndarray,
    x: float,
    y: float,
    resolution: float,
    radius_m: float,
) -> bool:
    """Returns True if a disk footprint at (x,y) with radius `radius_m` lies entirely in free space.

    A point can be in a free cell but still be in collision once you consider the robot's footprint.
    """
    if radius_m <= 0.0:
        return _is_free_xy(occ, x, y, resolution)

    i0, j0 = world_to_grid(float(x), float(y), float(resolution), (0.0, 0.0))
    r = int(math.ceil(float(radius_m) / float(resolution)))
    h, w = occ.shape[:2]
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            ii = i0 + di
            jj = j0 + dj
            if ii < 0 or jj < 0 or ii >= h or jj >= w:
                return False
            if int(occ[ii, jj]) != 0:
                return False
    return True

def _sample_free_point_in_room(
    rng: random.Random,
    occ: np.ndarray,
    room_poly: List[List[float]],
    resolution: float,
    margin_m: float = 0.6,
    max_tries: int = 120,
) -> Tuple[float,float]:
    # Rooms in F2_SINGLE are rectangles; sample uniformly in bbox and require occ-free.
    x0, y0, x1, y1 = _room_bbox(room_poly)
    x0 += float(margin_m); y0 += float(margin_m)
    x1 -= float(margin_m); y1 -= float(margin_m)
    if x1 <= x0 or y1 <= y0:
        # fallback: no margin
        x0, y0, x1, y1 = _room_bbox(room_poly)
    for _ in range(int(max_tries)):
        x = x0 + (x1 - x0) * rng.random()
        y = y0 + (y1 - y0) * rng.random()
        if _is_free_xy(occ, x, y, resolution):
            return (float(x), float(y))
    # Fallback: pick nearest free cell around bbox center
    cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
    if _is_free_xy(occ, cx, cy, resolution):
        return (float(cx), float(cy))
    # brute: spiral search nearby
    for rad in (0.5, 1.0, 1.5, 2.0, 3.0):
        for k in range(24):
            ang = 2.0 * math.pi * (k / 24.0)
            x = cx + rad * math.cos(ang)
            y = cy + rad * math.sin(ang)
            if _is_free_xy(occ, x, y, resolution):
                return (float(x), float(y))
    return (float(cx), float(cy))

def _max_conc_over_window_gt_a(
    rooms: Dict[str, Any],
    doors: Dict[str, Any],
    hvac_theta: str,
    hvac_mode_params: Dict[str, Any],
    leak_pos: List[float],
    q: float,
    start_time: float,
    x: float, y: float, z: float,
    t0: float,
    t1: float,
    n: int = 14,
) -> float:
    # Import locally to avoid circular import in some environments.
    try:
        from ..gas.gt_a import GasModelA
    except Exception:
        return 0.0
    gm = GasModelA(mode_params=dict(hvac_mode_params), topology={"rooms": rooms, "doors": doors})
    mx = 0.0
    if n <= 1:
        ts = [float(t1)]
    else:
        ts = [float(t0 + (t1 - t0) * (k / (n - 1))) for k in range(n)]
    for t in ts:
        c = float(gm.query(x=x, y=y, z=z, t=t, theta=hvac_theta, source_pos=tuple(leak_pos), q=float(q), start_time=float(start_time)))
        if c > mx:
            mx = c
    return float(mx)

def _build_tier1_no_cheat(
    *,
    seed: int,
    occ: np.ndarray,
    rooms: Dict[str, Any],
    doors: Dict[str, Any],
    resolution: float,
    leak_pos: List[float],
    leak_q: float,
    leak_start_time: float,
    hvac_theta: str,
    hvac_mode_params: Dict[str, Any],
    k: int = 3,
    # anti-cheat / difficulty controls
    d_min: float = 4.0,
    d_max: float = 24.0,
    exclusion_r: float = 2.0,
    alarm_threshold: float = 1e-3,
    margin_m: float = 0.6,
    time_window: Tuple[float,float] = (10.0, 70.0),
) -> Dict[str, Any]:
    """Return tier1 spec:
    - one sensor per room, placed at a random occ-free point inside the room bbox
    - NOTIFIED sensors (k of them) are selected to satisfy:
        distance in [d_min, d_max], max_conc>=alarm_threshold over a time window,
        and never within exclusion_r of the leak (anti-cheat)
    - sensors list is reordered so first k entries are the notified sensors
      (runner treats first k as monitor set when alarm.mode != 'nearest_to_leak').
    """
    rng = random.Random(int(seed) * 917 + 41)
    lx, ly = float(leak_pos[0]), float(leak_pos[1])
    alarm_threshold_eff = float(alarm_threshold)

    # 1) Place one sensor per room (random, occ-free)
    sensors = []
    for rid in sorted(rooms.keys()):
        r = rooms[rid]
        poly = r.get("poly", None)
        if not isinstance(poly, list) or len(poly) < 4:
            # fallback to room center
            cx, cy = float(r["center"][0]), float(r["center"][1])
            sensors.append({"id": str(rid), "pos": [cx, cy, 1.0]})
            continue
        sx, sy = _sample_free_point_in_room(rng, occ, poly, resolution, margin_m=margin_m)
        # Anti-cheat: keep ANY sensor from being exactly on top of leak.
        # (This is mild; the strong constraints apply to the notified subset.)
        for _ in range(40):
            if math.hypot(sx - lx, sy - ly) >= float(exclusion_r):
                break
            sx, sy = _sample_free_point_in_room(rng, occ, poly, resolution, margin_m=margin_m)
        sensors.append({"id": str(rid), "pos": [float(sx), float(sy), 1.0]})

    # 2) Build candidates for NOTIFIED sensors (distance + triggerability)
    t0 = float(leak_start_time) + float(time_window[0])
    t1 = float(leak_start_time) + float(time_window[1])

    cand = []
    for s in sensors:
        sid = str(s.get("id", ""))
        pos = s.get("pos", None)
        if not sid or not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue
        sx, sy = float(pos[0]), float(pos[1])
        d = float(math.hypot(sx - lx, sy - ly))
        if d < float(d_min) or d > float(d_max):
            continue
        if d < float(exclusion_r):
            continue
        mx = _max_conc_over_window_gt_a(
            rooms=rooms, doors=doors,
            hvac_theta=hvac_theta, hvac_mode_params=hvac_mode_params,
            leak_pos=leak_pos, q=float(leak_q), start_time=float(leak_start_time),
            x=sx, y=sy, z=1.0,
            t0=t0, t1=t1, n=14,
        )
        if mx >= float(alarm_threshold_eff):
            cand.append((d, sid, mx))

    # If not enough candidates, relax d_max (and slightly d_min) conservatively.
    relax_steps = 0
    while len(cand) < max(1, int(k)) and relax_steps < 4:
        relax_steps += 1
        d_min2 = max(1.0, float(d_min) * (0.85 ** relax_steps))
        d_max2 = float(d_max) * (1.15 ** relax_steps)
        cand = []
        for s in sensors:
            sid = str(s.get("id", ""))
            pos = s.get("pos", None)
            if not sid or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            sx, sy = float(pos[0]), float(pos[1])
            d = float(math.hypot(sx - lx, sy - ly))
            if d < d_min2 or d > d_max2 or d < float(exclusion_r):
                continue
            mx = _max_conc_over_window_gt_a(
                rooms=rooms, doors=doors,
                hvac_theta=hvac_theta, hvac_mode_params=hvac_mode_params,
                leak_pos=leak_pos, q=float(leak_q), start_time=float(leak_start_time),
                x=sx, y=sy, z=1.0,
                t0=t0, t1=t1, n=14,
            )
            if mx >= float(alarm_threshold_eff):
                cand.append((d, sid, mx))

    # 3) Select notified sensors with distance difficulty (near/mid/far stratification)
    cand.sort(key=lambda x: x[0])
    notified: List[str] = []
    if len(cand) > 0:
        kk = max(1, int(k))
        if kk == 1:
            notified = [cand[len(cand)//2][1]]
        else:
            # split by distance quantiles into kk bins
            ds = [c[0] for c in cand]
            lo, hi = ds[0], ds[-1]
            # Avoid degenerate bins
            if hi - lo < 1e-6:
                notified = [c[1] for c in cand[:kk]]
            else:
                for b in range(kk):
                    a = lo + (hi - lo) * (b / kk)
                    bnd = lo + (hi - lo) * ((b + 1) / kk)
                    bucket = [c for c in cand if (c[0] >= a and (c[0] < bnd or (b == kk-1 and c[0] <= bnd)))]
                    if not bucket:
                        bucket = cand
                    pick = bucket[rng.randrange(len(bucket))]
                    if pick[1] not in notified:
                        notified.append(pick[1])
                # ensure size == kk
                if len(notified) < kk:
                    for _, sid, _ in cand:
                        if sid not in notified:
                            notified.append(sid)
                        if len(notified) >= kk:
                            break


    # If selection failed (rare), fall back to the strongest sensor and
    # slightly lower the threshold so that at least one alarm can fire within the episode.
    if len(notified) == 0 and len(sensors) > 0:
        best_sid = None
        best_mx = -1.0
        # consider any sensor not too close to the leak
        for s in sensors:
            sid = str(s.get("id", ""))
            pos = s.get("pos", None)
            if (not sid) or (not isinstance(pos, (list, tuple))) or len(pos) < 2:
                continue
            sx, sy = float(pos[0]), float(pos[1])
            d = float(math.hypot(sx - lx, sy - ly))
            if d < float(exclusion_r):
                continue
            mx = _max_conc_over_window_gt_a(
                rooms=rooms, doors=doors,
                hvac_theta=hvac_theta, hvac_mode_params=hvac_mode_params,
                leak_pos=leak_pos, q=float(leak_q), start_time=float(leak_start_time),
                x=sx, y=sy, z=1.0,
                t0=t0, t1=t1, n=14,
            )
            if mx > best_mx:
                best_mx = float(mx)
                best_sid = sid
        if best_sid is not None:
            notified = [best_sid]
            if best_mx > 0.0:
                alarm_threshold_eff = min(alarm_threshold_eff, 0.8 * best_mx)

    # 4) Reorder sensors so notified sensors are first (runner monitors first k when mode != nearest_to_leak)
    notified_set = set(notified)
    sensors_sorted = [s for s in sensors if str(s.get("id","")) in notified_set] +                      [s for s in sensors if str(s.get("id","")) not in notified_set]

    return {
        "enabled": 1,
        "sensors": sensors_sorted,
        "alarm": {
            "mode": "notified",  # important: runner will monitor first k sensors
            "k": int(k),
            "threshold": float(alarm_threshold_eff),
            "d_min": float(d_min),
            "d_max": float(d_max),
            "exclusion_r": float(exclusion_r),
            "time_window_s": [float(time_window[0]), float(time_window[1])],
            "selection": "stratified_distance",
            "notified_ids": list(notified),
        },
        "placement": {
            "per_room": 1,
            "margin_m": float(margin_m),
            "random_seed": int(seed) * 917 + 41,
        },
    }


def generate_f2(seed: int, rooms_n: int = 14, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolution = float(GEOM_DEFAULTS.get("resolution", 0.20))
    map_size = tuple(GEOM_DEFAULTS.get("map_size", (160, 160)))
    occ, rooms, doors = _build_f2_single_connected(seed=seed, resolution=resolution, map_size=map_size, rooms_n=rooms_n)

    pattern_id = "half_random"
    door_states = apply_door_pattern(pattern_id, list(doors.keys()), seed)
    for did, s in door_states.items():
        doors[did]["state"] = int(s)

    # Robot spawn: randomize to avoid always starting in the same corner.
    # NOTE: run_trial.py must respect scenario_spec["robot"]["spawn"] for this to take effect.
    # ---------------------------------------------------------------------
    # Robot spawn
    #
    # NOTE:
    # Room centers can be too close to walls for the robot footprint, which can
    # lead to "collision forever" and zero motion. We therefore sample a free
    # point inside a (reasonably large) room with a clearance margin.
    # ---------------------------------------------------------------------
    rng_spawn = random.Random(seed + 101)
    params = params or {}


    robot_radius_m = float(params.get("robot_radius_m", 0.35))
    # Keep some distance to walls/room boundaries when spawning.
    spawn_margin_hi_m = float(params.get("spawn_margin_hi_m", robot_radius_m + 0.20))
    spawn_margin_lo_m = float(params.get("spawn_margin_lo_m", max(0.10, robot_radius_m + 0.05)))
    spawn_disk_r_m = float(params.get("spawn_disk_r_m", robot_radius_m + 0.05))

    def _room_min_dim_m(room: Dict) -> float:
        poly = room.get("poly", None)
        if not poly:
            return 0.0
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return float(min(max(xs) - min(xs), max(ys) - min(ys)))

    room_ids = sorted(rooms.keys())
    roomy = [rid for rid in room_ids if _room_min_dim_m(rooms[rid]) >= 2.2 * spawn_margin_hi_m]
    candidates = roomy if roomy else room_ids
    rng_spawn.shuffle(candidates)

    spawn_room = None
    spawn_xy = None

    for rid in candidates:
        room = rooms[rid]
        for margin_m in (spawn_margin_hi_m, spawn_margin_lo_m, 0.0):
            pt = _sample_free_point_in_room(rng_spawn, occ, room["poly"], resolution, margin_m=margin_m, max_tries=200)
            if pt is None:
                continue
            sx, sy = float(pt[0]), float(pt[1])
            if _is_free_disk(occ, sx, sy, resolution, radius_m=spawn_disk_r_m):
                spawn_room = rid
                spawn_xy = (sx, sy)
                break
        if spawn_xy is not None:
            break

    # Absolute fallback: pick any safe free cell in the whole map.
    if spawn_xy is None:
        free_ij = np.argwhere(occ == 0)
        if len(free_ij) > 0:
            for _ in range(2000):
                ii, jj = free_ij[rng_spawn.randrange(len(free_ij))]
                sx = (float(jj) + 0.5) * float(resolution)
                sy = (float(ii) + 0.5) * float(resolution)
                if _is_free_disk(occ, sx, sy, resolution, radius_m=spawn_disk_r_m):
                    spawn_xy = (sx, sy)
                    spawn_room = spawn_room or room_ids[0]
                    break

    if spawn_room is None:
        spawn_room = rng_spawn.choice(room_ids)
    if spawn_xy is None:
        sx, sy = rooms[spawn_room]["center"]
        spawn_xy = (float(sx), float(sy))

    sx, sy = spawn_xy
    robot_spawn = [float(sx), float(sy), 0.2]

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

    # Tier-1 sensors: random placement (anti-cheat) + stratified distance difficulty.
    # Only the NOTIFIED sensors (first k in the list) are expected to trigger alarms.
    tier1_spec = _build_tier1_no_cheat(
        seed=seed,
        occ=occ,
        rooms=rooms,
        doors=doors,
        resolution=resolution,
        leak_pos=leak_pos,
        leak_q=0.4,
        leak_start_time=5.0,
        hvac_theta="drift_pos_x",
        hvac_mode_params={"drift_speed": 0.60, "vortex_strength": 0.35, "time_switch_t": 30.0},
        k=3,
        d_min=4.0,
        d_max=26.0,
        exclusion_r=2.5,
        alarm_threshold=1e-3,
        margin_m=0.6,
        time_window=(6.0, 55.0),  # ensure alarm can trigger within t_end=60s
    )

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
        "leak": {"enabled": 1, "room_id": leak_room, "pos": leak_pos, "q": 0.4, "start_time": 5.0},
        "sensor": {"tau": 2.0, "sigma": 0.03, "drift": {"enabled": 0, "rate": 0.001, "bias0": 0.0}},
        "tier1": tier1_spec,

    }
    return scenario_spec