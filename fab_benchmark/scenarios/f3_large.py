
"""
F3 large scenario (v1).
"""
from __future__ import annotations
from typing import Any, Dict
from .base_scenario import generate_f3

def make_scenario_spec(seed: int, rooms_n: int = 34) -> Dict[str, Any]:
    return generate_f3(seed=seed, rooms_n=rooms_n)
