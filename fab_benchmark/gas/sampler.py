
"""
Sampling utilities (v1) for leak positions/rooms.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, List
import random


def choose_leak_room(seed: int, room_ids: List[str]) -> str:
    rng = random.Random(seed + 17)
    return rng.choice(sorted(room_ids)) if room_ids else "corridor"


def sample_q(seed: int, q_log10_range: Tuple[float,float] = (-2.0, 0.0)) -> float:
    rng = random.Random(seed + 23)
    lo, hi = float(q_log10_range[0]), float(q_log10_range[1])
    import math
    v = lo + (hi-lo)*rng.random()
    return float(10.0**v)
