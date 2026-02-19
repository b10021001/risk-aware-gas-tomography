
"""
Policy registry (v1).
"""
from __future__ import annotations
from typing import Any, Dict, Type

from .base_policy import Policy
from .dummy_policy import DummyPolicy
from .coverage_policy import CoveragePolicy
from .greedy_policy import GreedyPolicy
from .infotaxis_policy import InfotaxisPolicy
from .informative_no_risk import InformativeNoRiskPolicy
from .ours_policy import OursPolicy
from .ours_no_delay_model import OursNoDelayModelPolicy


_REGISTRY: Dict[str, Type[Policy]] = {
    "Dummy": DummyPolicy,
    "Coverage": CoveragePolicy,
    "Greedy": GreedyPolicy,
    "Infotaxis": InfotaxisPolicy,
    "Informative_NoRisk": InformativeNoRiskPolicy,
    "Ours": OursPolicy,
    "Ours_NoDelayModel": OursNoDelayModelPolicy,
}


def make_policy(policy_name: str, policy_cfg: Dict[str, Any], sim_params: Dict[str, Any]) -> Policy:
    if policy_name not in _REGISTRY:
        raise ValueError(f"Unknown policy_name '{policy_name}'. Known: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[policy_name](policy_cfg=policy_cfg, sim_params=sim_params)
