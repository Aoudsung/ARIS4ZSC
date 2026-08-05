"""Registered decision-continuation estimand shared by every DEPI path.

This module is deliberately simulator agnostic.  Simulator-specific collectors
must use :func:`discounted_reward_increment` and persist the accompanying
contract so comparator, training, M1, and mechanism artifacts cannot silently
change the value being estimated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


RAW_REWARD_DEFINITION = "raw_simulator_team_reward"
TERMINAL_HANDLING = "mask_rewards_at_and_after_first_terminal"
CONTINUATION_FIT_KEY_DOMAIN = "depi_continuation_fit_v1"
CONTINUATION_EVALUATION_KEY_DOMAIN = "depi_continuation_evaluation_v1"


@dataclass(frozen=True, slots=True)
class ContinuationContract:
    gamma: float
    horizon: int
    raw_reward_definition: str = RAW_REWARD_DEFINITION
    terminal_handling: str = TERMINAL_HANDLING
    continuation_policy_fingerprint: str = "runtime-bound"
    fit_key_domain: str = CONTINUATION_FIT_KEY_DOMAIN
    evaluation_key_domain: str = CONTINUATION_EVALUATION_KEY_DOMAIN

    def __post_init__(self) -> None:
        if not (0.0 < float(self.gamma) <= 1.0):
            raise ValueError("Continuation gamma must be in (0, 1].")
        if int(self.horizon) <= 0:
            raise ValueError("Continuation horizon must be positive.")
        if self.raw_reward_definition != RAW_REWARD_DEFINITION:
            raise ValueError("DEPI uses the registered raw simulator team reward only.")
        if self.terminal_handling != TERMINAL_HANDLING:
            raise ValueError("DEPI terminal handling differs from the registration.")
        for value in (
            self.continuation_policy_fingerprint,
            self.fit_key_domain,
            self.evaluation_key_domain,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("Continuation contract string fields must be non-empty.")

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def discounted_reward_increment(
    reward: Any, *, active: Any, step: int | Any, gamma: float
) -> Any:
    """Return the registered ``active * gamma**step * raw_reward`` increment."""

    import jax.numpy as jnp

    discount = jnp.power(
        jnp.asarray(gamma, dtype=jnp.float32),
        jnp.asarray(step, dtype=jnp.float32),
    )
    return jnp.where(
        jnp.asarray(active, dtype=jnp.bool_),
        discount * jnp.asarray(reward, dtype=jnp.float32),
        0.0,
    )


def validate_continuation_contract(
    payload: Mapping[str, Any], *, expected: ContinuationContract
) -> None:
    """Fail closed on any estimand/provenance difference."""

    if not isinstance(payload, Mapping) or dict(payload) != dict(expected.to_mapping()):
        raise ValueError("Continuation contract is incompatible with this run.")


__all__ = [
    "CONTINUATION_EVALUATION_KEY_DOMAIN",
    "CONTINUATION_FIT_KEY_DOMAIN",
    "ContinuationContract",
    "RAW_REWARD_DEFINITION",
    "TERMINAL_HANDLING",
    "discounted_reward_increment",
    "validate_continuation_contract",
]
