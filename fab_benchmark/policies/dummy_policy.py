
"""
Dummy policy (v1): stay still.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .base_policy import Policy


class DummyPolicy(Policy):
    name = "Dummy"

    def step(self, t: float, pose: Dict[str, float], measurement: Dict[str, float], belief_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"type": "velocity", "v": 0.0, "w": 0.0, "action_id": "dummy_stop"}
