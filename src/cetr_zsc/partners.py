"""Static parent and checkpoint sampling for training rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TRAINING_SUPPORT_MECHANISMS
from .manifest import normalized_mechanism
from .runner import PartnerFunctions


@dataclass(frozen=True, slots=True)
class PartnerPoolMember:
    run_id: str
    parent_training_run_id: str
    mechanism: str
    checkpoint_stage: float
    checkpoint: Any
    parent_index: int
    stage_slot: int
    probability: float


@dataclass(frozen=True, slots=True)
class PartnerPool:
    members: tuple[PartnerPoolMember, ...]
    parent_ids: tuple[str, ...]
    parent_mechanisms: tuple[str, ...]
    parent_nominal_weights: tuple[float, ...]
    parent_members: tuple[tuple[int, int, int], ...]


def build_training_partner_pool(config: Any, manifest: Any) -> PartnerPool:
    rows = tuple(manifest.by_role("development_support"))
    if not rows:
        raise ValueError("CETR training support is empty.")

    stages = tuple(float(value) for value in config.partner_pool.checkpoint_stages)
    if stages != (0.0, 0.5, 1.0):
        raise ValueError("Training support stages must be 0/0.5/1.")
    stage_slots = {stage: index for index, stage in enumerate(stages)}

    grouped: dict[str, dict[str, dict[float, Any]]] = {}
    parent_mechanisms: dict[str, str] = {}
    for row in rows:
        mechanism = normalized_mechanism(row.generation_mechanism)
        if mechanism not in TRAINING_SUPPORT_MECHANISMS:
            raise ValueError(f"Unregistered training support mechanism: {mechanism}.")
        stage = float(row.checkpoint_stage)
        if stage not in stage_slots:
            raise ValueError(f"Unregistered support checkpoint stage: {row.run_id}")
        if row.owner_seed_index is not None:
            raise ValueError("Training support partners must be owner-free.")
        parent_id = str(row.parent_training_run_id)
        previous_mechanism = parent_mechanisms.setdefault(parent_id, mechanism)
        if previous_mechanism != mechanism:
            raise ValueError(
                "Training support parent_training_run_id must be globally unique "
                f"across mechanisms; parent {parent_id} appears under "
                f"{previous_mechanism} and {mechanism}."
            )
        by_parent = grouped.setdefault(mechanism, {})
        by_stage = by_parent.setdefault(parent_id, {})
        if by_stage and parent_id in by_parent and stage in by_stage:
            raise ValueError(
                f"Training support parent {parent_id} repeats stage {stage}."
            )
        by_stage[stage] = row

    mechanisms = tuple(sorted(grouped))
    required = set(TRAINING_SUPPORT_MECHANISMS)
    if set(mechanisms) != required:
        raise ValueError("Training support must contain all four mechanisms.")
    required_parents = 4 if config.run_kind == "formal" else 1
    for mechanism in mechanisms:
        observed = len(grouped[mechanism])
        if observed != required_parents:
            if config.run_kind == "formal":
                raise ValueError(
                    "Formal support requires exactly 4 parents per mechanism."
                )
            raise ValueError(
                "Development and mechanical support require exactly 1 parent per mechanism."
            )

    parent_ids: list[str] = []
    parent_mechanisms: list[str] = []
    parent_nominal_weights: list[float] = []
    parent_members: list[tuple[int, int, int]] = []
    members: list[PartnerPoolMember] = []

    mechanism_count = len(mechanisms)
    for mechanism in mechanisms:
        parent_order = tuple(sorted(grouped[mechanism]))
        parent_probability = 1.0 / float(mechanism_count) / float(len(parent_order))
        for parent_id in parent_order:
            stage_rows = grouped[mechanism][parent_id]
            if set(stage_rows) != set(stages):
                raise ValueError(
                    f"Training support parent {parent_id} lacks stages 0/0.5/1."
                )
            parent_index = len(parent_ids)
            parent_ids.append(parent_id)
            parent_mechanisms.append(mechanism)
            parent_nominal_weights.append(float(parent_probability))
            member_indexes: list[int] = []
            for stage in stages:
                row = stage_rows[stage]
                member_index = len(members)
                member_indexes.append(member_index)
                members.append(
                    PartnerPoolMember(
                        run_id=str(row.run_id),
                        parent_training_run_id=parent_id,
                        mechanism=mechanism,
                        checkpoint_stage=stage,
                        checkpoint=row.checkpoint,
                        parent_index=parent_index,
                        stage_slot=stage_slots[stage],
                        probability=float(parent_probability / 3.0),
                    )
                )
            parent_members.append(tuple(member_indexes))

    if abs(sum(member.probability for member in members) - 1.0) > 1.0e-8:
        raise AssertionError("Partner-pool probabilities must sum to one.")
    return PartnerPool(
        members=tuple(members),
        parent_ids=tuple(parent_ids),
        parent_mechanisms=tuple(parent_mechanisms),
        parent_nominal_weights=tuple(parent_nominal_weights),
        parent_members=tuple(parent_members),
    )


__all__ = [
    "PartnerPool",
    "PartnerPoolMember",
    "build_training_partner_pool",
]
