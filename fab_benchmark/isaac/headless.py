
"""
Isaac Sim bootstrap utilities (v1).

This module is only used when backend == "isaac3d".
It assumes execution in an Isaac Sim Python environment where omni.isaac.* is available.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

def make_simulation_app(headless: bool = True, render: bool = False, enable_rtx: bool = False):
    """
    Create a SimulationApp. Raises ImportError if Isaac is not available.
    """
    try:
        from omni.isaac.kit import SimulationApp
    except Exception as e:
        raise ImportError("Isaac Sim Python packages not found. Run within Isaac Sim python environment.") from e

    # NOTE: Isaac uses a 'headless' flag; render can be enabled if headless is False.
    # Keep defaults per prompt: headless=1, render=0, enable_rtx=0
    cfg = {
        "headless": bool(headless),
        "renderer": "RayTracedLighting" if enable_rtx else "Hydra",
    }
    # Some versions accept additional flags; we pass minimal.
    app = SimulationApp(cfg)
    return app
