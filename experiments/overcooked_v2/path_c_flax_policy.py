"""Flax policy wrapper for official recurrent OvercookedV2 actors.

The remote experiments package owns the network definition.  This module does
not copy or reinterpret that definition: a small adapter supplies the official
hidden-state initializer and apply call, while this wrapper binds parameters,
checks their canonical hash, carries recurrent state, and applies the registered
temperature-one categorical action rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.overcooked_v2.path_c_official_artifact import (
    FLAX_WEIGHTS_HASH_DOMAIN,
    OFFICIAL_ACTION_RULE_ID,
    flax_weights_sha256,
    load_flax_parameter_tree,
)


@runtime_checkable
class OfficialFlaxNetworkAdapter(Protocol):
    """Exact official-network calls supplied after remote wiring verification."""

    def initial_state(self, batch_size: int) -> Any:
        ...

    def apply_actor(
        self,
        params: Mapping[str, Any],
        recurrent_state: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any]:
        """Return ``(next_recurrent_state, categorical_logits)``."""


@dataclass(frozen=True)
class CallableOfficialFlaxNetworkAdapter:
    """Explicit callback adapter for an official experiments-package network."""

    initial_state_fn: Callable[[int], Any]
    apply_actor_fn: Callable[[Mapping[str, Any], Any, Any, Any], tuple[Any, Any]]

    def __post_init__(self) -> None:
        if not callable(self.initial_state_fn) or not callable(self.apply_actor_fn):
            raise TypeError("Official Flax network adapter callbacks must be callable.")

    def initial_state(self, batch_size: int) -> Any:
        return self.initial_state_fn(batch_size)

    def apply_actor(
        self,
        params: Mapping[str, Any],
        recurrent_state: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any]:
        return self.apply_actor_fn(
            params,
            recurrent_state,
            observation,
            episode_start,
        )


@dataclass(frozen=True)
class OfficialFlaxPolicyStepV1:
    action: Any
    next_recurrent_state: Any
    logits: Any


@dataclass(frozen=True)
class OfficialFlaxPolicy:
    """One immutable official policy plus its exact network-call adapter."""

    params: Mapping[str, Any]
    network: OfficialFlaxNetworkAdapter
    expected_model_weights_sha256: str
    action_rule: str = OFFICIAL_ACTION_RULE_ID
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping) or not self.params:
            raise ValueError("Official Flax policy requires a non-empty parameter tree.")
        if not isinstance(self.network, OfficialFlaxNetworkAdapter):
            raise TypeError("Official Flax policy requires the registered network adapter.")
        if self.action_rule != OFFICIAL_ACTION_RULE_ID:
            raise ValueError("Official Flax policy action rule changed.")
        if not math.isclose(float(self.temperature), 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Official Flax categorical evaluation uses temperature one.")
        actual = flax_weights_sha256(self.params)
        if actual != self.expected_model_weights_sha256:
            raise ValueError("Loaded Flax parameters do not match their registered hash.")

    @property
    def model_weights_hash_domain(self) -> str:
        return FLAX_WEIGHTS_HASH_DOMAIN

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        checkpoint_format: str,
        parameter_tree_path: Sequence[str],
        network: OfficialFlaxNetworkAdapter,
        expected_model_weights_sha256: str,
    ) -> "OfficialFlaxPolicy":
        params = load_flax_parameter_tree(
            checkpoint_path,
            checkpoint_format=checkpoint_format,
            parameter_tree_path=parameter_tree_path,
        )
        return cls(
            params=params,
            network=network,
            expected_model_weights_sha256=expected_model_weights_sha256,
        )

    def initial_state(self, batch_size: int) -> Any:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Official Flax policy batch size must be positive.")
        return self.network.initial_state(batch_size)

    def act(
        self,
        observation: Any,
        episode_start: Any,
        recurrent_state: Any,
        random_key: Any,
    ) -> OfficialFlaxPolicyStepV1:
        """Carry hidden state one step and sample from official actor logits."""

        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:  # pragma: no cover - remote runtime dependency
            raise RuntimeError("Official Flax policy evaluation requires JAX.") from error
        next_state, logits = self.network.apply_actor(
            self.params,
            recurrent_state,
            observation,
            episode_start,
        )
        logits_array = jnp.asarray(logits)
        if logits_array.ndim < 1 or int(logits_array.shape[-1]) != 6:
            raise ValueError("Official actor must return six categorical logits.")
        if not bool(np.asarray(jnp.all(jnp.isfinite(logits_array)))):
            raise ValueError("Official actor logits must be finite.")
        action = jax.random.categorical(random_key, logits_array, axis=-1)
        return OfficialFlaxPolicyStepV1(
            action=action,
            next_recurrent_state=next_state,
            logits=logits_array,
        )
