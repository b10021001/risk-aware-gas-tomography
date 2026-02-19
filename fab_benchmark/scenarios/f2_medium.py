
"""
F2 medium scenario (v1).
"""
from __future__ import annotations
from typing import Any, Dict
from .base_scenario import generate_f2

def make_scenario_spec(seed: int, rooms_n: int = 14) -> Dict[str, Any]:
    return generate_f2(seed=seed, rooms_n=rooms_n)
