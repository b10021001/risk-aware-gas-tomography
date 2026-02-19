
"""
Isaac scene builder (v1).

Builds:
- Walls (from occupancy grid, coarse aggregated blocks)
- Doors (colliders toggled by door state)
- Ground plane

Note: This is designed for feasibility + collision correctness (R0). It is not optimized for
rendering fidelity.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import math

import numpy as np

from .utils import add_fixed_cuboid, add_xform


def build_scene(scene_dict: Dict[str, Any], parent_path: str = "/World/Env", block_stride: int = 4) -> Dict[str, Any]:
    """
    Build scene into current stage.

    Returns backend_meta dict that can be stored into run_meta.json["backend_meta"].
    """
    occ: np.ndarray = scene_dict["occupancy"]
    res: float = float(scene_dict["resolution"])
    origin = (float(scene_dict["origin"][0]), float(scene_dict["origin"][1]))
    doors: Dict[str, Any] = scene_dict["doors"]

    # Create parent prims
    add_xform(parent_path)
    walls_path = parent_path + "/Walls"
    doors_path = parent_path + "/Doors"
    add_xform(walls_path)
    add_xform(doors_path)

    # Ground plane (Isaac has a helper)
    try:
        from omni.isaac.core.objects import GroundPlane
        GroundPlane(prim_path="/World/GroundPlane", name="GroundPlane", z_position=0.0)
    except Exception:
        # if helper missing, skip (physics still OK if stage has default ground)
        pass

    H, W = occ.shape
    wall_h = 3.20  # Appendix B
    # Build coarse blocks for walls to reduce prim count:
    # For each stride-block fully occupied, create one cuboid.
    stride = max(1, int(block_stride))
    wall_count = 0
    for bi in range(0, H, stride):
        for bj in range(0, W, stride):
            block = occ[bi:min(H,bi+stride), bj:min(W,bj+stride)]
            if block.size == 0:
                continue
            if int(block.min()) == 1:  # fully blocked
                # block center in world
                ci = bi + block.shape[0]/2.0
                cj = bj + block.shape[1]/2.0
                x = origin[0] + (cj) * res
                y = origin[1] + (ci) * res
                sx = block.shape[1] * res
                sy = block.shape[0] * res
                path = f"{walls_path}/W_{bi}_{bj}"
                add_fixed_cuboid(path=path, position=(float(x), float(y), wall_h/2.0), size=(float(sx), float(sy), float(wall_h)))
                wall_count += 1

    # Doors: add colliders only for closed doors
    door_count = 0
    for did, d in doors.items():
        if int(d.get("state", 1)) == 1:
            continue
        geom = d["geom"]
        p0 = geom["p0"]; p1 = geom["p1"]
        z0 = float(geom["z_bottom"])
        height = float(geom["height"])
        thickness = float(geom["thickness"])
        cx = 0.5*(p0[0]+p1[0])
        cy = 0.5*(p0[1]+p1[1])
        # door width is distance between p0 and p1 in XY
        width = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        # Determine orientation: if segment is mostly vertical, door plane normal is x; else y.
        dx = abs(p1[0]-p0[0])
        dy = abs(p1[1]-p0[1])
        # We approximate door as a thin cuboid aligned with world axes.
        if dx >= dy:
            size = (float(width), float(thickness), float(height))
        else:
            size = (float(thickness), float(width), float(height))
        path = f"{doors_path}/Door_{did}"
        add_fixed_cuboid(path=path, position=(float(cx), float(cy), z0 + height/2.0), size=size)
        door_count += 1

    return {
        "isaac_scene_parent": parent_path,
        "isaac_wall_block_stride": int(stride),
        "isaac_wall_prim_count": int(wall_count),
        "isaac_closed_door_prim_count": int(door_count),
    }
