"""Unified DELTA-ZSC implementation.

The active method is derived from one latent-variable principle:
observable partner responses and privileged counterfactual action values are
conditionally independent emissions of the same exchangeable coordination
mode.  Deployment uses response evidence only; training additionally uses
counterfactual decision emissions.  Policy adaptation is the analytic solution
of a KL-constrained mirror-improvement problem.
"""

from .config import METHOD_VERSION, UnifiedConfig, load_config
from .model import BasePolicyModel, LatentCoordinationModel, UnifiedAgent
from .types import AgentState, UnifiedTrainState

__all__ = [
    "AgentState",
    "BasePolicyModel",
    "LatentCoordinationModel",
    "METHOD_VERSION",
    "UnifiedAgent",
    "UnifiedConfig",
    "UnifiedTrainState",
    "load_config",
]
