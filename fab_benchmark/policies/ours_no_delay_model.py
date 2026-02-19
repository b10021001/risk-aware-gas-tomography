
"""
Ours_NoDelayModel policy (v1).

Policy behavior is identical to OursPolicy; the ablation is implemented in the inference module
(Posterior(use_delay_model=False)) selected by runners when policy_name=="Ours_NoDelayModel".
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .ours_policy import OursPolicy


class OursNoDelayModelPolicy(OursPolicy):
    name = "Ours_NoDelayModel"
