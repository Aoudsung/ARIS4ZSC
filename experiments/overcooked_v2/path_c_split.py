from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, ClassVar, Iterable, Mapping

from experiments.overcooked_v2.path_c_seed import (
    OCV2_EXECUTION_SEED_VERSION,
    validate_unique_execution_seed_mapping,
)


SPLIT_MANIFEST_SCHEMA_VERSION = "path_c_split_manifest_v2"
SPLIT_GROUP_SCHEMA_VERSION = "path_c_split_group_v1"
CROSS_FIT_SCHEMA_VERSION = "path_c_role_cross_fit_v1"
NUMERIC_SEED_SCHEDULE_SCHEMA_VERSION = "path_c_numeric_seed_schedule_v2"
SPLIT_ROLES = ("train", "design", "calibration", "locked_audit")
MIN_GROUPS_PER_MECHANISM = 4
PREFERRED_GROUPS_PER_MECHANISM = 5

_ISOLATION_FIELDS = (
    "identity_group",
    "style_group",
    "seed_group",
    "layout_group",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    observed = set(payload)
    unknown = sorted(observed.difference(expected))
    missing = sorted(expected.difference(observed))
    if unknown:
        raise ValueError(f"{name} contains unknown key(s): " + ", ".join(unknown))
    if missing:
        raise ValueError(
            f"{name} is missing required key(s): " + ", ".join(missing)
        )


@dataclass(frozen=True, slots=True)
class SplitGroupV1:
    """One indivisible scientific group before role assignment.

    layout_group identifies a concrete layout template and is role-isolated.
    layout_stratum is a preregistered control stratum that may be represented by
    different layout templates in different roles. This distinction lets the
    primary identity shift condition on layout while a separate secondary view
    measures layout shift.
    """

    group_id: str
    mechanism: str
    identity_group: str
    style_group: str
    seed_group: str
    layout_group: str
    layout_stratum: str
    schema_version: str = SPLIT_GROUP_SCHEMA_VERSION

    _MAPPING_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "group_id",
            "mechanism",
            "identity_group",
            "style_group",
            "seed_group",
            "layout_group",
            "layout_stratum",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_GROUP_SCHEMA_VERSION:
            raise ValueError(
                f"SplitGroupV1 schema_version must be {SPLIT_GROUP_SCHEMA_VERSION!r}."
            )
        for field_name in (
            "group_id",
            "mechanism",
            *_ISOLATION_FIELDS,
            "layout_stratum",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"Split group field {field_name} must be a non-empty, "
                    "whitespace-trimmed string."
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SplitGroupV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Split group must be a mapping.")
        _require_exact_keys(payload, cls._MAPPING_KEYS, "Split group")
        return cls(
            group_id=payload["group_id"],
            mechanism=payload["mechanism"],
            identity_group=payload["identity_group"],
            style_group=payload["style_group"],
            seed_group=payload["seed_group"],
            layout_group=payload["layout_group"],
            layout_stratum=payload["layout_stratum"],
            schema_version=payload["schema_version"],
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "mechanism": self.mechanism,
            "identity_group": self.identity_group,
            "style_group": self.style_group,
            "seed_group": self.seed_group,
            "layout_group": self.layout_group,
            "layout_stratum": self.layout_stratum,
        }


@dataclass(frozen=True, slots=True)
class _IsolationComponent:
    group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CoverageTarget:
    kind: str
    mechanism: str
    layout_stratum: str
    required_role_mask: int


@dataclass(frozen=True, slots=True)
class SplitManifestV1:
    """Deterministic four-role, group-disjoint Path C split manifest."""

    groups: tuple[SplitGroupV1, ...]
    assignments: tuple[tuple[str, str], ...]
    primary_layout_strata: tuple[tuple[str, str], ...]
    secondary_layout_strata: tuple[tuple[str, str | None], ...]
    manifest_seed: int = 0
    cross_fit_folds: int = 5
    evaluation_seeds_per_group: int = 3
    schema_version: str = SPLIT_MANIFEST_SCHEMA_VERSION

    _SERIALIZED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "roles",
            "manifest_seed",
            "groups",
            "assignments",
            "feasibility",
            "primary_identity_shift",
            "secondary_layout_shift",
            "cross_fitting",
            "cross_fitting_sha256",
            "numeric_seed_schedule",
            "numeric_seed_schedule_sha256",
            "sha256",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(
            self,
            "assignments",
            tuple((str(group_id), str(role)) for group_id, role in self.assignments),
        )
        object.__setattr__(
            self,
            "primary_layout_strata",
            tuple(
                (str(mechanism), str(layout_stratum))
                for mechanism, layout_stratum in self.primary_layout_strata
            ),
        )
        object.__setattr__(
            self,
            "secondary_layout_strata",
            tuple(
                (
                    str(mechanism),
                    None if layout_stratum is None else str(layout_stratum),
                )
                for mechanism, layout_stratum in self.secondary_layout_strata
            ),
        )
        if self.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "SplitManifestV1 schema_version must be "
                f"{SPLIT_MANIFEST_SCHEMA_VERSION!r}."
            )
        if isinstance(self.manifest_seed, bool) or not isinstance(
            self.manifest_seed, Integral
        ):
            raise TypeError("manifest_seed must be an integer.")
        if int(self.manifest_seed) < 0:
            raise ValueError("manifest_seed must be non-negative.")
        if isinstance(self.cross_fit_folds, bool) or not isinstance(
            self.cross_fit_folds, Integral
        ):
            raise TypeError("cross_fit_folds must be an integer.")
        if int(self.cross_fit_folds) < 2:
            raise ValueError("cross_fit_folds must be at least two.")
        if (
            isinstance(self.evaluation_seeds_per_group, bool)
            or not isinstance(self.evaluation_seeds_per_group, Integral)
            or int(self.evaluation_seeds_per_group) <= 0
        ):
            raise ValueError("evaluation_seeds_per_group must be a positive integer.")
        _validate_manifest(self)

    @classmethod
    def build(
        cls,
        groups: Iterable[SplitGroupV1 | Mapping[str, Any]],
        *,
        manifest_seed: int = 0,
        cross_fit_folds: int = 5,
        evaluation_seeds_per_group: int = 3,
    ) -> "SplitManifestV1":
        if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, Integral):
            raise TypeError("manifest_seed must be an integer.")
        if int(manifest_seed) < 0:
            raise ValueError("manifest_seed must be non-negative.")
        if isinstance(cross_fit_folds, bool) or not isinstance(
            cross_fit_folds, Integral
        ):
            raise TypeError("cross_fit_folds must be an integer.")
        if int(cross_fit_folds) < 2:
            raise ValueError("cross_fit_folds must be at least two.")
        if (
            isinstance(evaluation_seeds_per_group, bool)
            or not isinstance(evaluation_seeds_per_group, Integral)
            or int(evaluation_seeds_per_group) <= 0
        ):
            raise ValueError("evaluation_seeds_per_group must be a positive integer.")
        normalized = []
        for item in groups:
            if isinstance(item, SplitGroupV1):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(SplitGroupV1.from_mapping(item))
            else:
                raise TypeError(
                    "groups must contain SplitGroupV1 objects or strict mappings."
                )
        ordered_groups = tuple(sorted(normalized, key=lambda item: item.group_id))
        _validate_unique_groups(ordered_groups)
        if not ordered_groups:
            raise ValueError("A split manifest requires at least one group.")

        components = _isolation_components(ordered_groups)
        independent_counts = _independent_group_counts(ordered_groups, components)
        insufficient = {
            mechanism: count
            for mechanism, count in independent_counts.items()
            if count < MIN_GROUPS_PER_MECHANISM
        }
        if insufficient:
            details = ", ".join(
                f"{mechanism}={count}"
                for mechanism, count in sorted(insufficient.items())
            )
            raise ValueError(
                "Each mechanism needs at least four independent isolation groups; "
                + details
            )

        primary_layout_strata, secondary_layout_strata = _select_layout_strata(
            ordered_groups,
            components,
        )
        assignments = _deterministic_assignments(
            ordered_groups,
            components,
            primary_layout_strata,
            secondary_layout_strata,
            manifest_seed=int(manifest_seed),
        )
        return cls(
            groups=ordered_groups,
            assignments=assignments,
            primary_layout_strata=primary_layout_strata,
            secondary_layout_strata=secondary_layout_strata,
            manifest_seed=int(manifest_seed),
            cross_fit_folds=int(cross_fit_folds),
            evaluation_seeds_per_group=int(evaluation_seeds_per_group),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SplitManifestV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Split manifest must be a mapping.")
        _require_exact_keys(payload, cls._SERIALIZED_KEYS, "Split manifest")
        group_payloads = payload["groups"]
        if isinstance(group_payloads, (str, bytes)):
            raise TypeError("Split manifest groups must be a sequence.")
        try:
            groups = tuple(
                SplitGroupV1.from_mapping(item) for item in group_payloads
            )
        except TypeError as exc:
            raise TypeError("Split manifest groups must be a sequence.") from exc
        cross_fitting = payload["cross_fitting"]
        if not isinstance(cross_fitting, Mapping):
            raise TypeError("cross_fitting must be a mapping.")
        if "n_folds" not in cross_fitting:
            raise ValueError("cross_fitting is missing n_folds.")
        rebuilt = cls.build(
            groups,
            manifest_seed=payload["manifest_seed"],
            cross_fit_folds=cross_fitting["n_folds"],
            evaluation_seeds_per_group=payload["numeric_seed_schedule"][
                "seeds_per_group"
            ],
        )
        if _canonical_json(rebuilt.to_mapping()) != _canonical_json(dict(payload)):
            raise ValueError(
                "Split manifest does not match its deterministic assignments, "
                "derived descriptions, or hashes."
            )
        return rebuilt

    @property
    def assignment_by_group(self) -> dict[str, str]:
        return dict(self.assignments)

    def role_for(self, group_id: str) -> str:
        try:
            return self.assignment_by_group[str(group_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown split group: {group_id!r}") from exc

    def groups_for_role(self, role: str) -> tuple[SplitGroupV1, ...]:
        _validate_role(role)
        assignments = self.assignment_by_group
        return tuple(
            group for group in self.groups if assignments[group.group_id] == role
        )

    @property
    def independent_group_count_by_mechanism(self) -> dict[str, int]:
        return _independent_group_counts(
            self.groups,
            _isolation_components(self.groups),
        )

    @property
    def preferred_spare_satisfied_by_mechanism(self) -> dict[str, bool]:
        return {
            mechanism: count >= PREFERRED_GROUPS_PER_MECHANISM
            for mechanism, count in sorted(
                self.independent_group_count_by_mechanism.items()
            )
        }

    def _split_core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roles": list(SPLIT_ROLES),
            "manifest_seed": int(self.manifest_seed),
            "evaluation_seeds_per_group": int(self.evaluation_seeds_per_group),
            "groups": [group.to_mapping() for group in self.groups],
            "assignments": [
                {"group_id": group_id, "role": role}
                for group_id, role in self.assignments
            ],
            "primary_layout_strata": {
                mechanism: layout_stratum
                for mechanism, layout_stratum in self.primary_layout_strata
            },
            "secondary_layout_strata": {
                mechanism: layout_stratum
                for mechanism, layout_stratum in self.secondary_layout_strata
            },
        }

    @property
    def split_core_sha256(self) -> str:
        return _canonical_sha256(self._split_core_payload())

    def cross_fitting_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_FIT_SCHEMA_VERSION,
            "scope": "within_role_only",
            "roles": list(SPLIT_ROLES),
            "unit": "episode_uid",
            "n_folds": int(self.cross_fit_folds),
            "fold_derivation": (
                "sha256(split_core_sha256, manifest_seed, role, episode_uid) "
                "modulo n_folds"
            ),
            "cross_role_pooling_allowed": False,
            "split_core_sha256": self.split_core_sha256,
        }

    @property
    def cross_fitting_sha256(self) -> str:
        return _canonical_sha256(self.cross_fitting_payload())

    def cross_fit_fold(self, role: str, episode_uid: str) -> int:
        _validate_role(role)
        if not isinstance(episode_uid, str) or not episode_uid:
            raise ValueError("episode_uid must be a non-empty string.")
        digest = _canonical_sha256(
            {
                "cross_fitting_sha256": self.cross_fitting_sha256,
                "manifest_seed": int(self.manifest_seed),
                "role": role,
                "episode_uid": episode_uid,
            }
        )
        return int(digest[:16], 16) % int(self.cross_fit_folds)

    def numeric_seed_schedule_payload(self) -> dict[str, Any]:
        used: set[int] = set()
        entries: list[dict[str, Any]] = []
        for group in self.groups:
            seeds: list[int] = []
            for seed_index in range(int(self.evaluation_seeds_per_group)):
                collision_index = 0
                while True:
                    digest = _canonical_sha256({
                        "split_core_sha256": self.split_core_sha256,
                        "manifest_seed": int(self.manifest_seed),
                        "group_id": group.group_id,
                        "seed_index": seed_index,
                        "collision_index": collision_index,
                    })
                    numeric_seed = int(digest[:16], 16)
                    if numeric_seed not in used:
                        break
                    collision_index += 1
                used.add(numeric_seed)
                seeds.append(numeric_seed)
            execution_seed_by_canonical = validate_unique_execution_seed_mapping(
                seeds,
                name=f"numeric seed schedule for group {group.group_id!r}",
            )
            entries.append({
                "group_id": group.group_id,
                "role": self.role_for(group.group_id),
                "seed_group": group.seed_group,
                "numeric_seeds": seeds,
                "ocv2_execution_seeds": [
                    execution_seed_by_canonical[seed] for seed in seeds
                ],
            })
        validate_unique_execution_seed_mapping(
            (
                seed
                for entry in entries
                for seed in entry["numeric_seeds"]
            ),
            name="complete numeric seed schedule",
        )
        return {
            "schema_version": NUMERIC_SEED_SCHEDULE_SCHEMA_VERSION,
            "derivation": (
                "sha256(split_core_sha256,manifest_seed,group_id,seed_index,"
                "collision_index) first_unsigned_64_bits"
            ),
            "seeds_per_group": int(self.evaluation_seeds_per_group),
            "ocv2_execution_seed_version": OCV2_EXECUTION_SEED_VERSION,
            "entries": entries,
        }

    @property
    def numeric_seed_schedule_sha256(self) -> str:
        return _canonical_sha256(self.numeric_seed_schedule_payload())

    def numeric_seeds_for_group(self, group_id: str) -> tuple[int, ...]:
        for entry in self.numeric_seed_schedule_payload()["entries"]:
            if entry["group_id"] == str(group_id):
                return tuple(map(int, entry["numeric_seeds"]))
        raise KeyError(f"Unknown split group: {group_id!r}")

    def validate_numeric_seed(self, group_id: str, numeric_seed: int) -> None:
        if isinstance(numeric_seed, bool) or not isinstance(numeric_seed, Integral):
            raise TypeError("numeric_seed must be an integer.")
        if int(numeric_seed) not in self.numeric_seeds_for_group(group_id):
            raise ValueError(
                "Numeric seed is not registered for its frozen split group."
            )

    def canonical_payload(self) -> dict[str, Any]:
        primary = dict(self.primary_layout_strata)
        secondary = dict(self.secondary_layout_strata)
        return {
            "schema_version": self.schema_version,
            "roles": list(SPLIT_ROLES),
            "manifest_seed": int(self.manifest_seed),
            "groups": [group.to_mapping() for group in self.groups],
            "assignments": [
                {"group_id": group_id, "role": role}
                for group_id, role in self.assignments
            ],
            "feasibility": {
                "minimum_independent_groups_per_mechanism": (
                    MIN_GROUPS_PER_MECHANISM
                ),
                "preferred_independent_groups_per_mechanism": (
                    PREFERRED_GROUPS_PER_MECHANISM
                ),
                "independent_group_count_by_mechanism": (
                    self.independent_group_count_by_mechanism
                ),
                "preferred_spare_satisfied_by_mechanism": (
                    self.preferred_spare_satisfied_by_mechanism
                ),
                "fail_closed": True,
            },
            "primary_identity_shift": {
                "priority": "primary",
                "evaluation_role": "locked_audit",
                "shift_axes": [
                    "identity_group",
                    "style_group",
                    "seed_group",
                ],
                "condition_on": ["mechanism", "layout_stratum"],
                "layout_template_is_role_isolated": True,
                "layout_shift_must_not_be_pooled": True,
                "layout_stratum_by_mechanism": primary,
            },
            "secondary_layout_shift": {
                "priority": "secondary",
                "evaluation_role": "locked_audit",
                "shift_axes": ["layout_group", "layout_stratum"],
                "condition_on": ["mechanism"],
                "may_replace_primary_identity_shift": False,
                "layout_stratum_by_mechanism": secondary,
                "available_by_mechanism": {
                    mechanism: layout_stratum is not None
                    for mechanism, layout_stratum in self.secondary_layout_strata
                },
            },
            "cross_fitting": self.cross_fitting_payload(),
            "cross_fitting_sha256": self.cross_fitting_sha256,
            "numeric_seed_schedule": self.numeric_seed_schedule_payload(),
            "numeric_seed_schedule_sha256": self.numeric_seed_schedule_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "sha256": self.sha256}


def _validate_role(role: str) -> None:
    if role not in SPLIT_ROLES:
        raise ValueError(f"role must be one of {SPLIT_ROLES!r}; got {role!r}.")


def _validate_unique_groups(groups: tuple[SplitGroupV1, ...]) -> None:
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Split group_id values must be unique.")


def _isolation_components(
    groups: tuple[SplitGroupV1, ...],
) -> tuple[_IsolationComponent, ...]:
    if not groups:
        return ()
    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field_name in _ISOLATION_FIELDS:
        first: dict[str, int] = {}
        for index, group in enumerate(groups):
            value = getattr(group, field_name)
            if value in first:
                union(first[value], index)
            else:
                first[value] = index

    components: dict[int, list[str]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group.group_id)
    return tuple(
        _IsolationComponent(tuple(sorted(group_ids)))
        for group_ids in sorted(
            components.values(),
            key=lambda values: tuple(sorted(values)),
        )
    )


def _independent_group_counts(
    groups: tuple[SplitGroupV1, ...],
    components: tuple[_IsolationComponent, ...],
) -> dict[str, int]:
    by_id = {group.group_id: group for group in groups}
    mechanisms = sorted({group.mechanism for group in groups})
    return {
        mechanism: sum(
            any(by_id[group_id].mechanism == mechanism for group_id in component.group_ids)
            for component in components
        )
        for mechanism in mechanisms
    }


def _layout_stratum_component_counts(
    groups: tuple[SplitGroupV1, ...],
    components: tuple[_IsolationComponent, ...],
) -> dict[tuple[str, str], int]:
    by_id = {group.group_id: group for group in groups}
    keys = sorted(
        {(group.mechanism, group.layout_stratum) for group in groups}
    )
    return {
        key: sum(
            any(
                (
                    by_id[group_id].mechanism,
                    by_id[group_id].layout_stratum,
                )
                == key
                for group_id in component.group_ids
            )
            for component in components
        )
        for key in keys
    }


def _select_layout_strata(
    groups: tuple[SplitGroupV1, ...],
    components: tuple[_IsolationComponent, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str | None], ...]]:
    counts = _layout_stratum_component_counts(groups, components)
    mechanisms = sorted({group.mechanism for group in groups})
    primary = []
    secondary = []
    for mechanism in mechanisms:
        candidates = [
            (layout_stratum, count)
            for (candidate_mechanism, layout_stratum), count in counts.items()
            if candidate_mechanism == mechanism
        ]
        primary_candidates = [
            item for item in candidates if item[1] >= MIN_GROUPS_PER_MECHANISM
        ]
        if not primary_candidates:
            raise ValueError(
                "Primary identity shift requires at least four independent groups "
                "for one shared layout_stratum per mechanism; "
                f"mechanism {mechanism!r} has none."
            )
        primary_stratum = sorted(
            primary_candidates,
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        primary.append((mechanism, primary_stratum))
        secondary_candidates = [
            item for item in candidates if item[0] != primary_stratum
        ]
        secondary_stratum = (
            sorted(
                secondary_candidates,
                key=lambda item: (-item[1], item[0]),
            )[0][0]
            if secondary_candidates
            else None
        )
        secondary.append((mechanism, secondary_stratum))
    return tuple(primary), tuple(secondary)


def _component_matches(
    component: _IsolationComponent,
    by_id: Mapping[str, SplitGroupV1],
    mechanism: str,
    layout_stratum: str,
) -> bool:
    return any(
        by_id[group_id].mechanism == mechanism
        and by_id[group_id].layout_stratum == layout_stratum
        for group_id in component.group_ids
    )


def _deterministic_assignments(
    groups: tuple[SplitGroupV1, ...],
    components: tuple[_IsolationComponent, ...],
    primary_layout_strata: tuple[tuple[str, str], ...],
    secondary_layout_strata: tuple[tuple[str, str | None], ...],
    *,
    manifest_seed: int,
) -> tuple[tuple[str, str], ...]:
    by_id = {group.group_id: group for group in groups}
    full_role_mask = (1 << len(SPLIT_ROLES)) - 1
    locked_audit_mask = 1 << SPLIT_ROLES.index("locked_audit")
    targets = [
        _CoverageTarget("primary", mechanism, layout_stratum, full_role_mask)
        for mechanism, layout_stratum in primary_layout_strata
    ]
    targets.extend(
        _CoverageTarget(
            "secondary",
            mechanism,
            layout_stratum,
            locked_audit_mask,
        )
        for mechanism, layout_stratum in secondary_layout_strata
        if layout_stratum is not None
    )
    targets_tuple = tuple(
        sorted(
            targets,
            key=lambda item: (item.kind, item.mechanism, item.layout_stratum),
        )
    )

    def target_indexes(component: _IsolationComponent) -> tuple[int, ...]:
        return tuple(
            index
            for index, target in enumerate(targets_tuple)
            if _component_matches(
                component,
                by_id,
                target.mechanism,
                target.layout_stratum,
            )
        )

    def component_digest(component: _IsolationComponent) -> str:
        return _canonical_sha256(
            {
                "manifest_seed": int(manifest_seed),
                "group_ids": list(component.group_ids),
            }
        )

    ordered_components = tuple(
        sorted(
            components,
            key=lambda component: (
                -len(target_indexes(component)),
                component_digest(component),
                component.group_ids,
            ),
        )
    )
    hits = tuple(target_indexes(component) for component in ordered_components)
    n_components = len(ordered_components)
    remaining = [
        [0 for _ in range(n_components + 1)] for _ in targets_tuple
    ]
    for index in range(n_components - 1, -1, -1):
        for target_index in range(len(targets_tuple)):
            remaining[target_index][index] = (
                remaining[target_index][index + 1]
                + int(target_index in hits[index])
            )

    failed_states: set[tuple[int, tuple[int, ...]]] = set()

    def solve(
        index: int,
        coverage: tuple[int, ...],
        role_counts: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        complete = all(
            coverage[target_index] & target.required_role_mask
            == target.required_role_mask
            for target_index, target in enumerate(targets_tuple)
        )
        if complete:
            tail = []
            mutable_counts = list(role_counts)
            for component in ordered_components[index:]:
                role_index = min(
                    range(len(SPLIT_ROLES)),
                    key=lambda candidate: (
                        mutable_counts[candidate],
                        _canonical_sha256(
                            {
                                "manifest_seed": int(manifest_seed),
                                "group_ids": list(component.group_ids),
                                "role": SPLIT_ROLES[candidate],
                            }
                        ),
                    ),
                )
                mutable_counts[role_index] += 1
                tail.append(role_index)
            return tuple(tail)
        if index >= n_components:
            return None
        state_key = (index, coverage)
        if state_key in failed_states:
            return None
        for target_index, target in enumerate(targets_tuple):
            missing_mask = target.required_role_mask & ~coverage[target_index]
            if remaining[target_index][index] < missing_mask.bit_count():
                failed_states.add(state_key)
                return None

        component = ordered_components[index]
        needed_by_role = []
        for role_index in range(len(SPLIT_ROLES)):
            role_mask = 1 << role_index
            needed = sum(
                bool(
                    targets_tuple[target_index].required_role_mask & role_mask
                    and not coverage[target_index] & role_mask
                )
                for target_index in hits[index]
            )
            needed_by_role.append(int(needed))
        role_order = sorted(
            range(len(SPLIT_ROLES)),
            key=lambda role_index: (
                -needed_by_role[role_index],
                role_counts[role_index],
                _canonical_sha256(
                    {
                        "manifest_seed": int(manifest_seed),
                        "group_ids": list(component.group_ids),
                        "role": SPLIT_ROLES[role_index],
                    }
                ),
            ),
        )
        if max(needed_by_role, default=0) == 0:
            role_order = role_order[:1]

        for role_index in role_order:
            role_mask = 1 << role_index
            updated_coverage = list(coverage)
            for target_index in hits[index]:
                updated_coverage[target_index] |= role_mask
            updated_counts = list(role_counts)
            updated_counts[role_index] += 1
            suffix = solve(
                index + 1,
                tuple(updated_coverage),
                tuple(updated_counts),
            )
            if suffix is not None:
                return (role_index, *suffix)
        failed_states.add(state_key)
        return None

    role_indexes = solve(
        0,
        tuple(0 for _ in targets_tuple),
        tuple(0 for _ in SPLIT_ROLES),
    )
    if role_indexes is None:
        raise ValueError(
            "No deterministic four-role assignment satisfies mechanism coverage, "
            "layout-controlled primary identity shift, and group isolation."
        )
    assignment_by_group = {}
    for component, role_index in zip(
        ordered_components,
        role_indexes,
        strict=True,
    ):
        role = SPLIT_ROLES[role_index]
        for group_id in component.group_ids:
            assignment_by_group[group_id] = role
    return tuple(sorted(assignment_by_group.items()))


def _validate_manifest(manifest: SplitManifestV1) -> None:
    _validate_unique_groups(manifest.groups)
    if not manifest.groups:
        raise ValueError("A split manifest requires at least one group.")
    group_ids = {group.group_id for group in manifest.groups}
    assignment_ids = [group_id for group_id, _role in manifest.assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("Split manifest assignments contain duplicate group IDs.")
    if set(assignment_ids) != group_ids:
        raise ValueError("Split manifest assignments must cover every group exactly once.")
    assignments = manifest.assignment_by_group
    for role in assignments.values():
        _validate_role(role)

    components = _isolation_components(manifest.groups)
    for component in components:
        roles = {assignments[group_id] for group_id in component.group_ids}
        if len(roles) != 1:
            raise ValueError(
                "Identity/style/seed/layout-linked groups cannot cross split roles."
            )
    for field_name in _ISOLATION_FIELDS:
        roles_by_value: dict[str, set[str]] = {}
        for group in manifest.groups:
            roles_by_value.setdefault(getattr(group, field_name), set()).add(
                assignments[group.group_id]
            )
        crossing = sorted(
            value for value, roles in roles_by_value.items() if len(roles) > 1
        )
        if crossing:
            raise ValueError(
                f"{field_name} values cross roles: " + ", ".join(crossing)
            )

    counts = _independent_group_counts(manifest.groups, components)
    insufficient = {
        mechanism: count
        for mechanism, count in counts.items()
        if count < MIN_GROUPS_PER_MECHANISM
    }
    if insufficient:
        raise ValueError(
            "Each mechanism needs at least four independent isolation groups."
        )
    mechanisms = sorted(counts)
    primary = dict(manifest.primary_layout_strata)
    secondary = dict(manifest.secondary_layout_strata)
    if len(primary) != len(manifest.primary_layout_strata) or len(secondary) != len(
        manifest.secondary_layout_strata
    ):
        raise ValueError(
            "Primary and secondary layout-stratum descriptions cannot repeat a mechanism."
        )
    if sorted(primary) != mechanisms or sorted(secondary) != mechanisms:
        raise ValueError(
            "Primary and secondary layout-stratum descriptions must cover every mechanism."
        )
    for mechanism in mechanisms:
        primary_stratum = primary[mechanism]
        primary_roles = {
            assignments[group.group_id]
            for group in manifest.groups
            if group.mechanism == mechanism
            and group.layout_stratum == primary_stratum
        }
        if primary_roles != set(SPLIT_ROLES):
            raise ValueError(
                "Primary identity shift must cover all four roles inside one "
                f"layout_stratum for mechanism {mechanism!r}."
            )
        mechanism_roles = {
            assignments[group.group_id]
            for group in manifest.groups
            if group.mechanism == mechanism
        }
        if mechanism_roles != set(SPLIT_ROLES):
            raise ValueError(
                f"Mechanism {mechanism!r} does not cover all four split roles."
            )
        secondary_stratum = secondary[mechanism]
        if secondary_stratum is not None:
            if secondary_stratum == primary_stratum:
                raise ValueError(
                    "Secondary layout shift must use a different layout_stratum "
                    "from the primary identity shift."
                )
            has_locked_secondary = any(
                group.mechanism == mechanism
                and group.layout_stratum == secondary_stratum
                and assignments[group.group_id] == "locked_audit"
                for group in manifest.groups
            )
            if not has_locked_secondary:
                raise ValueError(
                    "A declared secondary layout shift needs a locked-audit group."
                )


__all__ = [
    "CROSS_FIT_SCHEMA_VERSION",
    "MIN_GROUPS_PER_MECHANISM",
    "PREFERRED_GROUPS_PER_MECHANISM",
    "SPLIT_GROUP_SCHEMA_VERSION",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SPLIT_ROLES",
    "SplitGroupV1",
    "SplitManifestV1",
]
