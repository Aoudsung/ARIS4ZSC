"""Thin runtime protocols separating Path C from a concrete environment."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorizedCookingEnvironment(Protocol):
    """A vector environment exposing two local-observation seats."""

    num_envs: int
    observation_shape: tuple[int, ...]
    num_actions: int
    episode_steps: int

    def reset(self, key: Any) -> tuple[Any, Any]:
        """Return ``(state, observations[environment, seat, ...])``."""

    def step(
        self, state: Any, joint_actions: Any, key: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        """Advance all environments and return state, observations, reward, done, info."""


@runtime_checkable
class BackboneDock(Protocol):
    """Supply exact official-network semantics without importing them in core code."""

    def official_parameter_tree(self, checkpoint_ref: Any) -> Mapping[str, Any]:
        ...

    def network_dimensions(self) -> Mapping[str, Any]:
        ...

    def action_order(self) -> Sequence[str]:
        ...

    def initial_recurrent_state(self, batch_size: int) -> Any:
        ...

    def explicit_parameter_mapping(
        self, target_params: Mapping[str, Any]
    ) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Map every copied target leaf to one exact official leaf."""


@runtime_checkable
class FrozenPartner(Protocol):
    """One immutable partner policy with explicit recurrent state and randomness."""

    def act(self, observation: Any, carry: Any, key: Any) -> tuple[Any, Any]:
        ...
