
"""
Gas generator GT-B cache (v1) for E5.

Fixed keys required by prompt:
  meta, grid_resolution, origin, z_layers, times, C

Interpretation in this implementation:
- Cache is canonical around source at (0,0) in XY. For an arbitrary source position (sx,sy),
  we query at relative coordinates (x-sx, y-sy) and bilinear sample C.

- C shape: (K, T, Z, H, W) float32
  K: number of cached flow realizations
  T: time steps
  Z: z layers
  H,W: XY grid
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import json
import math

import numpy as np


class GTBCache:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        # Required keys
        self.meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]
        self.grid_resolution = float(data["grid_resolution"])
        self.origin = tuple(float(x) for x in data["origin"])
        self.z_layers = [float(z) for z in data["z_layers"]]
        self.times = data["times"].astype(np.float32)
        self.C = data["C"].astype(np.float32)  # (K,T,Z,H,W)
        self.K = int(self.C.shape[0])
        self.T = int(self.C.shape[1])
        self.Z = int(self.C.shape[2])
        self.H = int(self.C.shape[3])
        self.W = int(self.C.shape[4])

    def _sample_xy(self, k: int, ti: int, zi: int, x: float, y: float) -> float:
        """
        Bilinear sample at world (x,y) in canonical grid.
        Grid index (i,j) from origin + resolution.
        """
        # convert to grid coords
        fx = (x - self.origin[0]) / self.grid_resolution
        fy = (y - self.origin[1]) / self.grid_resolution
        j0 = int(math.floor(fx))
        i0 = int(math.floor(fy))
        if i0 < 0 or i0 >= self.H-1 or j0 < 0 or j0 >= self.W-1:
            return 0.0
        tx = fx - j0
        ty = fy - i0
        c00 = self.C[k, ti, zi, i0, j0]
        c10 = self.C[k, ti, zi, i0, j0+1]
        c01 = self.C[k, ti, zi, i0+1, j0]
        c11 = self.C[k, ti, zi, i0+1, j0+1]
        c0 = c00*(1-tx) + c10*tx
        c1 = c01*(1-tx) + c11*tx
        return float(c0*(1-ty) + c1*ty)

    def query(self, x: float, y: float, z: float, t: float,
              source_pos: Tuple[float,float,float], k: int = 0) -> float:
        """
        Query concentration using cache for a source at source_pos.
        """
        k = int(k) % max(1, self.K)
        # map time to nearest index
        ti = int(np.clip(np.searchsorted(self.times, np.float32(t), side="right") - 1, 0, self.T-1))
        # nearest z layer
        zi = int(np.argmin(np.abs(np.array(self.z_layers, dtype=np.float32) - np.float32(z))))
        sx, sy, sz = float(source_pos[0]), float(source_pos[1]), float(source_pos[2])
        xr = float(x - sx)
        yr = float(y - sy)
        # canonical grid assumes source at (0,0); ignore sz mismatch and use zi for vertical
        return self._sample_xy(k, ti, zi, xr, yr)
