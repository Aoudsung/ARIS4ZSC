"""Partner lineage and split validation for unified DELTA-ZSC."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


ACTIVE_ROLES = (
    "development_support",
    "calibration",
    "confirmatory",
)


def normalized_mechanism(value: str) -> str:
    aliases = {
        "rnn-sp": "sp",
        "sp": "sp",
        "rnn-op": "op",
        "op": "op",
        "state-augmented": "sa",
        "sa": "sa",
        "rnn-sa": "sa",
        "fcp": "fcp",
        "rnn-fcp": "fcp",
    }
    lowered = str(value).strip().lower()
    return aliases.get(lowered, lowered)


def _role_rows(manifest: Any, role: str) -> tuple[Any, ...]:
    return tuple(manifest.by_role(role))


def _active_lineage_disjoint(manifest: Any) -> None:
    by_role = {role: _role_rows(manifest, role) for role in ACTIVE_ROLES}
    for left_index, left_role in enumerate(ACTIVE_ROLES):
        for right_role in ACTIVE_ROLES[left_index + 1 :]:
            left = by_role[left_role]
            right = by_role[right_role]
            left_checkpoints = {str(row.checkpoint_sha256) for row in left}
            right_checkpoints = {str(row.checkpoint_sha256) for row in right}
            if left_checkpoints & right_checkpoints:
                raise ValueError(
                    f"Unified partner checkpoints overlap across {left_role}/{right_role}."
                )
            left_parents = {str(row.parent_training_run_id) for row in left}
            right_parents = {str(row.parent_training_run_id) for row in right}
            if left_parents & right_parents:
                raise ValueError(
                    f"Unified partner parents overlap across {left_role}/{right_role}."
                )
            left_groups = {
                str(row.co_training_group_id)
                for row in left
                if row.co_training_group_id is not None
            }
            right_groups = {
                str(row.co_training_group_id)
                for row in right
                if row.co_training_group_id is not None
            }
            if left_groups & right_groups:
                raise ValueError(
                    f"Unified co-training groups overlap across {left_role}/{right_role}."
                )


def _validate_support(rows: tuple[Any, ...], *, formal: bool) -> None:
    if not rows:
        raise ValueError("Unified training support is empty.")
    if any(row.owner_seed_index is not None for row in rows):
        raise ValueError("Unified support partners must be shared and owner-free.")
    mechanisms = {normalized_mechanism(row.generation_mechanism) for row in rows}
    if not {"sp", "op"} <= mechanisms:
        raise ValueError("Unified support must include independent SP and OP parents.")
    if not mechanisms <= {"sp", "op"}:
        raise ValueError(
            "The registered unified training distribution contains only SP/OP support."
        )
    by_parent: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_parent[str(row.parent_training_run_id)].append(row)
    for parent, values in by_parent.items():
        stages = {float(row.checkpoint_stage) for row in values}
        if stages != {0.0, 0.5, 1.0}:
            raise ValueError(
                f"Support parent {parent} must contribute stages 0/0.5/1."
            )
        mechanisms_for_parent = {
            normalized_mechanism(row.generation_mechanism) for row in values
        }
        if len(mechanisms_for_parent) != 1:
            raise ValueError(f"Support parent {parent} changes mechanism across stages.")
    if formal:
        parents_by_mechanism = Counter(
            normalized_mechanism(values[0].generation_mechanism)
            for values in by_parent.values()
        )
        if parents_by_mechanism["sp"] < 10 or parents_by_mechanism["op"] < 10:
            raise ValueError(
                "Formal unified support requires at least ten independent SP and OP parents."
            )


def _validate_final_panel(
    rows: tuple[Any, ...],
    *,
    role: str,
    formal: bool,
    required_per_mechanism: int,
) -> None:
    if not rows:
        raise ValueError(f"Unified {role} partner panel is empty.")
    parents = [str(row.parent_training_run_id) for row in rows]
    if len(parents) != len(set(parents)):
        raise ValueError(f"Unified {role} panel requires independent parent blocks.")
    if any(float(row.checkpoint_stage) != 1.0 for row in rows):
        raise ValueError(f"Unified {role} panel uses final checkpoints only.")
    if any(row.owner_seed_index is not None for row in rows):
        raise ValueError(f"Unified {role} partners must be owner-free.")
    if formal:
        counts = Counter(
            normalized_mechanism(row.generation_mechanism) for row in rows
        )
        required = {"sp", "op", "sa", "fcp"}
        if not required <= set(counts):
            raise ValueError(
                f"Formal {role} panel must cover SP, OP, SA, and FCP."
            )
        if min(counts[name] for name in required) < int(required_per_mechanism):
            raise ValueError(
                f"Formal {role} panel needs at least {required_per_mechanism} "
                "independent parents per mechanism."
            )


def validate_unified_manifest(
    manifest: Any,
    *,
    formal: bool,
    require_support: bool = False,
    require_calibration: bool = False,
    require_confirmatory: bool = False,
) -> None:
    """Validate active roles without inheriting retired DEPI requirements."""

    from src.path_c.experiment import validate_partner_manifest

    validate_partner_manifest(manifest)
    _active_lineage_disjoint(manifest)
    if require_support:
        _validate_support(_role_rows(manifest, "development_support"), formal=formal)
    if require_calibration:
        _validate_final_panel(
            _role_rows(manifest, "calibration"),
            role="calibration",
            formal=formal,
            required_per_mechanism=5,
        )
    if require_confirmatory:
        _validate_final_panel(
            _role_rows(manifest, "confirmatory"),
            role="confirmatory",
            formal=formal,
            required_per_mechanism=4,
        )


__all__ = [
    "ACTIVE_ROLES",
    "normalized_mechanism",
    "validate_unified_manifest",
]
