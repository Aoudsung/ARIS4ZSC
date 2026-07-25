"""Device rollout, prefit, and adaptation training utilities."""

from .adaptation import (
    adaptation_loss,
    clipped_actor_loss,
    environment_minibatches,
    generalized_advantage_estimate,
    make_adaptation_optimizer,
)
from .prefit import auxiliary_losses, make_prefit_optimizer, prefit_loss
from .rollout import RolloutBatch, RolloutState, collect_rollout, initialize_rollout

__all__ = [
    "RolloutBatch",
    "RolloutState",
    "adaptation_loss",
    "auxiliary_losses",
    "clipped_actor_loss",
    "collect_rollout",
    "generalized_advantage_estimate",
    "environment_minibatches",
    "initialize_rollout",
    "make_adaptation_optimizer",
    "make_prefit_optimizer",
    "prefit_loss",
]
