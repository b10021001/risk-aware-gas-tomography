
"""
F1 small scenario (v1).
"""
from __future__ import annotations
from typing import Any, Dict
from .base_scenario import generate_f1

def make_scenario_spec(seed: int) -> Dict[str, Any]:
    return generate_f1(seed)
