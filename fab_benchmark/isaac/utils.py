
"""
Isaac helper utilities (v1). Keep dependencies inside Isaac environment.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import math

def ensure_omni_imports():
    try:
        import omni  # noqa: F401
        return True
    except Exception:
        return False


def add_fixed_cuboid(
    path: str,
    position: Tuple[float, float, float],
    size: Tuple[float, float, float],
    color: Optional[Tuple[float, float, float]] = None,
) -> Any:
    """
    Add a fixed (static) cuboid with collision at path.
    IMPORTANT: Use UsdGeom.Cube.size (double) + xform scale (vec3) to represent (sx,sy,sz).
    """
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()

    # Create a Cube prim directly at `path` (matches your error path /World/Env/Walls/W_0_0)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetPrim().GetAttribute("size").Set(1.0)
  # MUST be double

    xformable = UsdGeom.Xformable(cube.GetPrim())
    # Clear existing ops to avoid stacking if re-created
    try:
        xformable.ClearXformOpOrder()
    except Exception:
        pass
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xformable.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))

    prim = cube.GetPrim()

    # Collision (static collider: no rigid body needed)
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    # Optional display color (non-critical)
    if color is not None:
        try:
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        except Exception:
            pass

    return prim


def add_dynamic_cuboid(
    path: str,
    position: Tuple[float, float, float],
    size: Tuple[float, float, float],
    mass: float = 1.0,
    color: Optional[Tuple[float, float, float]] = None,
) -> Any:
    """
    Add a dynamic cuboid (rigid body) at path.
    """
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetPrim().GetAttribute("size").Set(1.0)


    xformable = UsdGeom.Xformable(cube.GetPrim())
    try:
        xformable.ClearXformOpOrder()
    except Exception:
        pass
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xformable.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))

    prim = cube.GetPrim()

    # Collision
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    # Rigid body + mass
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    if color is not None:
        try:
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        except Exception:
            pass

    return prim



def add_xform(path: str) -> Any:
    from omni.isaac.core.prims import XFormPrim
    return XFormPrim(prim_path=path, name=path.split("/")[-1])


def add_usd_reference(parent_path: str, usd_path: str, prim_name: str = "Go2") -> str:
    """
    Reference an external USD into stage under parent_path/prim_name.
    Returns the prim path of referenced asset.
    """
    from omni.isaac.core.utils.stage import add_reference_to_stage
    prim_path = f"{parent_path}/{prim_name}"
    add_reference_to_stage(usd_path, prim_path)
    return prim_path


def set_prim_pose(prim_path: str, position: Tuple[float,float,float], yaw: float) -> None:
    """
    Set prim pose (x,y,z) and yaw about Z.
    """
    from omni.isaac.core.utils.prims import set_prim_world_pose
    # Quaternion from yaw
    import numpy as np
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    quat = np.array([0.0, 0.0, sy, cy])  # (x,y,z,w)
    set_prim_world_pose(prim_path, position=position, orientation=quat)


def get_prim_pose(prim_path: str):
    """
    Return ((x,y,z), yaw) in world frame, without relying on deprecated omni.isaac.core APIs.
    """
    import math
    import omni.usd
    from pxr import Usd, UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    # Compute local-to-world transform at default time
    xf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # translation
    t = xf.ExtractTranslation()
    x, y, z = float(t[0]), float(t[1]), float(t[2])

    # yaw from rotation matrix (Z-up)
    r = xf.ExtractRotationMatrix()
    # yaw = atan2(r10, r00) (Z-axis yaw)
    yaw = math.atan2(float(r[1][0]), float(r[0][0]))

    return (x, y, z), yaw

def ensure_go2_collision_proxy(
    go2_root_path: str,
    proxy_name: str = "CollisionProxy",
    size_xyz=(0.60, 0.30, 0.35),     # (x,y,z) meters for box proxy
    offset_xyz=(0.0, 0.0, 0.18),     # lift a bit above ground
) -> str:
    """
    Create a simple box collider under Go2 root to replace missing collision meshes.
    Returns proxy prim path: f"{go2_root_path}/{proxy_name}"
    """
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    proxy_path = f"{go2_root_path}/{proxy_name}"

    # Create cube as proxy collider
    cube = UsdGeom.Cube.Define(stage, proxy_path)
    # Cube.size is double; keep it 1.0 then use scale for dimensions
    cube.GetPrim().GetAttribute("size").Set(1.0)

    xformable = UsdGeom.Xformable(cube.GetPrim())
    try:
        xformable.ClearXformOpOrder()
    except Exception:
        pass

    # Offset in Go2 local frame (child prim inherits Go2 transform)
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2])))
    xformable.AddScaleOp().Set(Gf.Vec3d(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))

    prim = cube.GetPrim()

    # Collision API
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    # Mark as kinematic rigid body so PhysX will generate contacts but it won't fall
    UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True)
    rb.CreateDisableGravityAttr(True)

    return proxy_path
