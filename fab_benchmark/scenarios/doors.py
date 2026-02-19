
"""
Door patterns (v1): all_open, half_random, bottleneck_closed
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import random


DOOR_PATTERN_IDS = ("all_open", "half_random", "bottleneck_closed")


def apply_door_pattern(pattern_id: str, door_ids: List[str], seed: int) -> Dict[str, int]:
    rng = random.Random(seed + 1337)
    if pattern_id == "all_open":
        return {d: 1 for d in door_ids}
    if pattern_id == "half_random":
        states = {d: 1 for d in door_ids}
        if len(door_ids) == 0:
            return states
        k = max(1, len(door_ids) // 2)
        closed = rng.sample(door_ids, k=k)
        for d in closed:
            states[d] = 0
        return states
    if pattern_id == "bottleneck_closed":
        states = {d: 1 for d in door_ids}
        if len(door_ids) == 0:
            return states
        # Choose a deterministic "bottleneck" door: smallest id after shuffle for diversity
        shuffled = door_ids[:]
        rng.shuffle(shuffled)
        bottleneck = sorted(shuffled)[0]
        states[bottleneck] = 0
        return states
    raise ValueError(f"Unknown door pattern_id: {pattern_id}")
