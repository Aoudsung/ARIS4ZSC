"""Minimal resettable benchmark interfaces for SRVF-MAPPO interventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class InterventionSnapshot:
    """Opaque reset point for repeating interventions at the same public state."""

    state_g: torch.Tensor
    opaque_state: Any
    observation: Mapping[str, Any] = field(default_factory=dict)
    phase_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkStep:
    """One environment step emitted by a resettable benchmark adapter."""

    observation: Mapping[str, Any]
    reward: float
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    state_g: torch.Tensor | None = None
    affordance: Any = None


class PartnerPolicy(Protocol):
    """Policy interface for source partners used in IRF table construction."""

    def reset(self, seed: Optional[int] = None) -> Any:
        """Reset optional recurrent partner state."""

    def act(
        self,
        observation: Mapping[str, Any],
        state: Any = None,
        rng: Any = None,
    ) -> Tuple[int, Any]:
        """Return ``(partner_action, next_state)``."""


class ResettableBenchmarkAdapter(Protocol):
    """Protocol required for controlled ``do(a)`` interventions."""

    num_actions: int

    def reset(self, seed: int) -> Mapping[str, Any]:
        """Reset to a fresh episode state."""

    def snapshot(self, *, phase_id: str = "unknown") -> InterventionSnapshot:
        """Return an opaque state snapshot that can be restored exactly."""

    def restore(self, snapshot: InterventionSnapshot) -> Mapping[str, Any]:
        """Restore an earlier snapshot and return its public observation."""

    def step(self, ego_action: int, partner_action: int) -> BenchmarkStep:
        """Execute one simultaneous ego/partner primitive action step."""


class BenchmarkAdapter(ABC):
    """Backward-compatible benchmark adapter base with optional snapshot support."""

    benchmark_name: str
    num_actions: int

    @abstractmethod
    def reset(self, seed: int) -> Mapping[str, Any]:
        """Reset the benchmark and return the current public observation."""

    @abstractmethod
    def step(self, ego_action: int, partner_action: int) -> BenchmarkStep:
        """Execute one simultaneous ego/partner primitive action step."""

    @abstractmethod
    def extract_affordance(self) -> Any:
        """Return the current public affordance/chart state."""

    def snapshot(self, *, phase_id: str = "unknown") -> InterventionSnapshot:
        """Return an exact reset point when the concrete adapter supports it."""

        raise NotImplementedError(
            "This adapter must implement exact snapshot()/restore() before it can collect "
            "SRVF-MAPPO source tables"
        )

    def restore(self, snapshot: InterventionSnapshot) -> Mapping[str, Any]:
        """Restore an exact reset point when the concrete adapter supports it."""

        raise NotImplementedError(
            "This adapter must implement exact snapshot()/restore() before it can collect "
            "SRVF-MAPPO source tables"
        )

    def legal_action_mask(self) -> torch.Tensor:
        return torch.ones(self.num_actions, dtype=torch.bool)

    def sample_random_action(self, generator: Optional[torch.Generator] = None) -> int:
        action = torch.randint(0, self.num_actions, size=(), generator=generator)
        return int(action.item())

    def close(self) -> None:
        """Release adapter resources, if any."""


def discounted_returns(
    rewards: torch.Tensor,
    gamma: float,
    dones: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-step discounted return targets."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [B, T]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if dones is not None and dones.shape != rewards.shape:
        raise ValueError("dones must have the same shape as rewards")
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    for time_idx in reversed(range(rewards.shape[1])):
        if dones is not None:
            running = running * (1.0 - dones[:, time_idx].to(dtype=rewards.dtype))
        running = rewards[:, time_idx] + gamma * running
        returns[:, time_idx] = running
    return returns
