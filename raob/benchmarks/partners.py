"""Partner-policy specifications for classic SRVF-MAPPO experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PartnerSpec:
    """Serializable description of one benchmark partner artifact."""

    benchmark: str
    partner_id: str
    split: str
    artifact: str
    layout: str | None = None
    group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "manifest"

    def to_json(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "partner_id": self.partner_id,
            "split": self.split,
            "artifact": self.artifact,
            "layout": self.layout,
            "group": self.group,
            "metadata": self.metadata,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PartnerSpec":
        return cls(
            benchmark=str(data["benchmark"]),
            partner_id=str(data["partner_id"]),
            split=str(data["split"]),
            artifact=str(data["artifact"]),
            layout=None if data.get("layout") is None else str(data.get("layout")),
            group=None if data.get("group") is None else str(data.get("group")),
            metadata=dict(data.get("metadata", {})),
            source=str(data.get("source", "manifest")),
        )


class RandomPartnerPolicy:
    """Uniform random partner used only for smoke tests and diagnostics."""

    def __init__(self, num_actions: int = 6) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.num_actions = int(num_actions)
        self._generator = torch.Generator()

    def reset(self, seed: int | None = None) -> torch.Generator:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        return self._generator

    def act(
        self,
        observation: Mapping[str, Any],
        state: Any = None,
        rng: Any = None,
    ) -> tuple[int, Any]:
        del observation, rng
        generator = state if isinstance(state, torch.Generator) else self._generator
        action = torch.randint(0, self.num_actions, size=(), generator=generator)
        return int(action.item()), generator


def make_partner_policy(
    spec: PartnerSpec,
    *,
    num_actions: int = 6,
) -> RandomPartnerPolicy:
    """Instantiate lightweight non-GOAT smoke partners.

    Real classic Overcooked partners are loaded through `goat_classic.py`.
    """

    loader = str(spec.metadata.get("loader", ""))
    if spec.artifact == "random" or loader == "random":
        return RandomPartnerPolicy(num_actions=num_actions)
    raise ValueError(
        "unsupported generic partner spec; use GOATClassicPartnerPolicy for classic artifacts"
    )
