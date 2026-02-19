
"""
Hypothesis grid for discrete posterior (v1).

Fixed concept:
  hypotheses = free cells (with stride) × z_layers
  optional no-leak hypothesis for E6.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math

import numpy as np


@dataclass
class HypothesisGrid:
    positions: np.ndarray  # (N,3) float32
    room_ids: List[str]    # len N
    resolution: float
    origin: Tuple[float,float]
    cell_stride: int
    z_layers: List[float]
    include_no_leak: bool = False

    @classmethod
    def build(cls, occupancy: np.ndarray, resolution: float, origin: Tuple[float,float],
              rooms: Dict[str, Any], cell_stride: int = 2, z_layers: Optional[List[float]] = None,
              include_no_leak: bool = False) -> "HypothesisGrid":
        if z_layers is None:
            z_layers = [0.2, 1.0, 2.0]
        H, W = occupancy.shape
        stride = max(1, int(cell_stride))

        # Precompute room bounding boxes in grid for quick assignment:
        # We'll assign by point-in-rect approximation using poly bbox.
        room_polys = {rid: r["poly"] for rid, r in rooms.items()}
        room_bboxes = {}
        for rid, poly in room_polys.items():
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            room_bboxes[rid] = (min(xs), min(ys), max(xs), max(ys))

        def room_of(x: float, y: float) -> str:
            for rid, (x0,y0,x1,y1) in room_bboxes.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return rid
            return "corridor"

        pos_list = []
        room_list = []
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                if int(occupancy[i, j]) != 0:
                    continue
                x = origin[0] + (j + 0.5) * resolution
                y = origin[1] + (i + 0.5) * resolution
                rid = room_of(x, y)
                for z in z_layers:
                    pos_list.append((x, y, float(z)))
                    room_list.append(rid)

        if include_no_leak:
            pos_list.append((float("nan"), float("nan"), float("nan")))
            room_list.append("no_leak")

        positions = np.array(pos_list, dtype=np.float32)
        return cls(
            positions=positions,
            room_ids=room_list,
            resolution=float(resolution),
            origin=(float(origin[0]), float(origin[1])),
            cell_stride=int(stride),
            z_layers=[float(z) for z in z_layers],
            include_no_leak=bool(include_no_leak),
        )

    @property
    def N(self) -> int:
        return int(self.positions.shape[0])

    def credible_set_size(self, posterior: np.ndarray, p_cut: float) -> int:
        """Return the size of the "credible" region.

        Semantics: count hypotheses whose posterior probability >= p_cut.
        If none meet the threshold (e.g., early uniform belief), fall back to
        including the MAP cell so this is never zero.
        """
        if posterior.size == 0:
            return 0
        p = np.asarray(posterior, dtype=float)
        thr = float(p_cut)
        mx = float(np.max(p))
        thr_eff = thr if mx >= thr else mx
        return int(np.sum(p >= thr_eff))

    def topk(self, posterior: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        k = int(k)
        if k <= 0 or posterior.size == 0:
            return []
        idx = np.argsort(-posterior)[:k]
        out = []
        for ii in idx:
            out.append({"pos": [float(x) for x in self.positions[ii].tolist()], "p": float(posterior[ii])})
        return out

    def room_mass(self, posterior: np.ndarray) -> Dict[str, float]:
        masses: Dict[str, float] = {}
        for rid, p in zip(self.room_ids, posterior.tolist()):
            masses[rid] = masses.get(rid, 0.0) + float(p)
        return masses