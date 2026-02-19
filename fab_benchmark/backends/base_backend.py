
"""
Backend interface contract (v1).

Do NOT rename public class/methods. Only append.
"""
from __future__ import annotations
from typing import Any, Dict


class Backend:
    """
    Abstract backend API used by runners.

    Methods:
      - init(sim_params)
      - load_scene(scene_spec) -> scene_dict
      - reset(seed, scene_spec, init_pose)
      - apply_action(action)
      - step(dt, substeps)
      - get_state() -> dict
      - close()
    """
    def __init__(self, sim_params: Dict[str, Any]):
        self.init(sim_params)

    def init(self, sim_params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def load_scene(self, scene_spec: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self, seed: int, scene_spec: Dict[str, Any], init_pose: Dict[str, Any]) -> None:
        raise NotImplementedError

    def apply_action(self, action: Dict[str, Any]) -> None:
        raise NotImplementedError

    def step(self, dt: float, substeps: int) -> None:
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
