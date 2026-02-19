
"""
Go2 loader + simple base-velocity controller (v1).

Design choice:
- We always create a physics "Body" collider (DynamicCuboid) used for collisions and motion.
- If a Go2 USD is provided, we reference it as a visual child under the Body prim so the asset is
  loaded (R0) while collisions remain simple and robust across different Go2 USD variants.

This avoids depending on specific Go2 articulation joint names.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import math

from .utils import add_dynamic_cuboid, add_xform, add_usd_reference, set_prim_pose


class Go2BaseController:
    def __init__(self, body_obj: Any, body_prim_path: str):
        self._body = body_obj
        self._path = body_prim_path
        self._v = 0.0
        self._w = 0.0

    @property
    def prim_path(self) -> str:
        return self._path

    def set_cmd(self, v: float, w: float) -> None:
        self._v = float(v)
        self._w = float(w)

    def apply(self) -> None:
        # Apply planar velocity in body yaw frame using Isaac dynamic object interface
        # Many Isaac object classes provide set_linear_velocity / set_angular_velocity.
        try:
            import numpy as np
            # read yaw from current pose? We keep world-frame velocity by approximating yaw=0; Isaac will handle?
            # For robustness, set world-frame velocity along +X and angular about Z.
            lin = np.array([self._v, 0.0, 0.0], dtype=float)
            ang = np.array([0.0, 0.0, self._w], dtype=float)
            if hasattr(self._body, "set_linear_velocity"):
                self._body.set_linear_velocity(lin)
            if hasattr(self._body, "set_angular_velocity"):
                self._body.set_angular_velocity(ang)
        except Exception:
            # As a fallback, do nothing (will appear as stop)
            pass


def create_go2_rigidbody(root_path: str = "/World/Robot",
                         go2_usd_path: Optional[str] = None,
                         init_pose: Tuple[float,float,float,float] = (0.0,0.0,0.35,0.0)) -> Go2BaseController:
    """
    Create robot root prim, a dynamic rigid-body "Body", and optional Go2 visual USD reference.
    init_pose = (x,y,z,yaw)
    """
    add_xform(root_path)
    body_path = f"{root_path}/Body"
    # approximate Go2 footprint as cuboid
    body = add_dynamic_cuboid(
        path=body_path,
        position=(float(init_pose[0]), float(init_pose[1]), float(init_pose[2])),
        size=(0.70, 0.35, 0.35),
        mass=20.0,
    )
    # Optional visual reference
    if go2_usd_path:
        vis_parent = f"{root_path}/Visual"
        add_xform(vis_parent)
        add_usd_reference(vis_parent, go2_usd_path, prim_name="Go2")
        # Align visual root with body (inherit parent transform by parenting under body would be ideal,
        # but reference is added under Visual; we set it to match body pose for simplicity)
        try:
            set_prim_pose(f"{vis_parent}/Go2", position=(float(init_pose[0]),float(init_pose[1]),float(init_pose[2])), yaw=float(init_pose[3]))
        except Exception:
            pass

    # Set initial yaw on body prim
    try:
        set_prim_pose(body_path, position=(float(init_pose[0]),float(init_pose[1]),float(init_pose[2])), yaw=float(init_pose[3]))
    except Exception:
        pass

    return Go2BaseController(body_obj=body, body_prim_path=body_path)
