from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

from experiments.overcooked_v2.path_c_protocol import (
    ActingEpisodeLog,
    ActingEpisodeSpec,
    FORMAL_ACTING_POLICY_NAMES,
    FORMAL_LOCKED_ACTING_POLICY_NAMES,
    FormalActingPolicyFactoryEntry,
    FormalActingPolicyFactoryRegistry,
    PathCActingEnvironment,
    PathCActingPolicy,
    PathCActingRunner,
    ReturnProbeBudgetCurve,
    ReturnProbeBudgetPoint,
    build_return_probe_budget_curve,
)
from experiments.overcooked_v2.path_c_split import SplitManifestV1


RETURN_POINT_LEDGER_SCHEMA = "path_c_return_point_ledger_v2"
MEASUREMENT_SCHEMA = "path_c_measurement_v3"
_SPLIT_ROLES = frozenset({"design", "locked_audit"})
_POINT_KEYS = frozenset(
    {
        "split_role",
        "episode_uid",
        "source_episode_log_sha256",
        "source_episode_log",
        "policy_name",
        "split_group_id",
        "mechanism",
        "identity_group",
        "style_group",
        "layout_group",
        "seed_group",
        "seed",
        "budget",
        "raw_return",
        "probes_used",
        "realized_probe_cost",
        "probe_cost_per_use",
        "episode_environment_steps",
        "training_environment_steps",
        "gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
    }
)
_LEDGER_CORE_KEYS = frozenset(
    {
        "schema_version",
        "split_role",
        "probe_budget_grid",
        "normalization_rule",
        "normalization_lower",
        "normalization_upper",
        "probe_cost_per_use",
        "split_manifest_sha256",
        "numeric_seed_schedule_sha256",
        "factory_registry_sha256",
        "environment_manifest_sha256",
        "policy_artifact_sha256_by_name",
        "points",
        "ledger_sha256",
    }
)
_BOUND_ENVELOPE_KEYS = frozenset(
    {
        "measurement_schema_version",
        "preregistration_sha256",
        "resolved_path_c_sha256",
        "semantic_bindings",
        "artifact_sha256",
    }
)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    observed = set(payload)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch; missing={missing}, unknown={unknown}.")


def _strict_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class BoundReturnProbeBudgetPointV1:
    """One hash-bound episode point from the common acting-policy runner."""

    split_role: str
    episode_uid: str
    source_episode_log_sha256: str
    source_episode_log: ActingEpisodeLog
    point: ReturnProbeBudgetPoint

    def __post_init__(self) -> None:
        if self.split_role not in _SPLIT_ROLES:
            raise ValueError("Return points are admitted only from design or locked_audit.")
        if not str(self.episode_uid).strip():
            raise ValueError("episode_uid must be non-empty.")
        if not _is_sha256(self.source_episode_log_sha256):
            raise ValueError("source_episode_log_sha256 must be a SHA-256 digest.")
        if not isinstance(self.source_episode_log, ActingEpisodeLog):
            raise TypeError("source_episode_log must be an ActingEpisodeLog.")
        if self.source_episode_log.sha256 != self.source_episode_log_sha256:
            raise ValueError("Source episode-log SHA-256 does not match its content.")
        if (
            self.source_episode_log.spec.split_role != self.split_role
            or self.source_episode_log.spec.episode_uid != self.episode_uid
        ):
            raise ValueError("Source episode log changed the point's split role or episode uid.")
        if self.point != self.source_episode_log.return_probe_budget_point():
            raise ValueError("Return point was not derived from its source episode log.")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "split_role": str(self.split_role),
            "episode_uid": str(self.episode_uid),
            "source_episode_log_sha256": str(self.source_episode_log_sha256),
            "source_episode_log": self.source_episode_log.canonical_payload(),
            "policy_name": str(self.point.policy_name),
            "split_group_id": str(self.point.split_group_id),
            "mechanism": str(self.point.mechanism),
            "identity_group": str(self.point.identity_group),
            "style_group": str(self.point.style_group),
            "layout_group": str(self.point.layout_group),
            "seed_group": str(self.point.seed_group),
            "seed": int(self.point.seed),
            "budget": int(self.point.budget),
            "raw_return": float(self.point.raw_return),
            "probes_used": int(self.point.probes_used),
            "realized_probe_cost": float(self.point.realized_probe_cost),
            "probe_cost_per_use": float(self.point.probe_cost_per_use),
            "episode_environment_steps": int(self.point.episode_environment_steps),
            "training_environment_steps": int(self.point.training_environment_steps),
            "gradient_updates": int(self.point.gradient_updates),
            "evaluation_environment_step_limit": int(
                self.point.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": str(self.point.evaluation_schedule_id),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundReturnProbeBudgetPointV1":
        if not isinstance(payload, Mapping):
            raise TypeError("A bound return point must be a mapping.")
        _require_exact_keys(payload, _POINT_KEYS, "return point")
        source_episode_log = ActingEpisodeLog.from_mapping(payload["source_episode_log"])
        point = ReturnProbeBudgetPoint(
            policy_name=str(payload["policy_name"]),
            split_group_id=str(payload["split_group_id"]),
            mechanism=str(payload["mechanism"]),
            identity_group=str(payload["identity_group"]),
            style_group=str(payload["style_group"]),
            layout_group=str(payload["layout_group"]),
            seed_group=str(payload["seed_group"]),
            seed=_strict_integer(payload["seed"], "return point seed"),
            budget=_strict_integer(payload["budget"], "return point budget"),
            raw_return=_finite_number(payload["raw_return"], "return point raw_return"),
            probes_used=_strict_integer(
                payload["probes_used"], "return point probes_used"
            ),
            realized_probe_cost=_finite_number(
                payload["realized_probe_cost"],
                "return point realized_probe_cost",
            ),
            probe_cost_per_use=_finite_number(
                payload["probe_cost_per_use"],
                "return point probe_cost_per_use",
            ),
            episode_environment_steps=_strict_integer(
                payload["episode_environment_steps"],
                "return point episode_environment_steps",
                minimum=1,
            ),
            training_environment_steps=_strict_integer(
                payload["training_environment_steps"],
                "return point training_environment_steps",
                minimum=1,
            ),
            gradient_updates=_strict_integer(
                payload["gradient_updates"],
                "return point gradient_updates",
            ),
            evaluation_environment_step_limit=_strict_integer(
                payload["evaluation_environment_step_limit"],
                "return point evaluation_environment_step_limit",
                minimum=1,
            ),
            evaluation_schedule_id=str(payload["evaluation_schedule_id"]),
        )
        return cls(
            split_role=str(payload["split_role"]),
            episode_uid=str(payload["episode_uid"]),
            source_episode_log_sha256=str(payload["source_episode_log_sha256"]),
            source_episode_log=source_episode_log,
            point=point,
        )


@dataclass(frozen=True)
class ReturnPointLedgerV1:
    """Strict raw episode ledger from which design and locked curves are rebuilt."""

    points: tuple[BoundReturnProbeBudgetPointV1, ...]
    split_role: str
    probe_budget_grid: tuple[int, ...]
    normalization_lower: float
    normalization_upper: float
    probe_cost_per_use: float
    split_manifest_sha256: str
    numeric_seed_schedule_sha256: str
    factory_registry_sha256: str
    environment_manifest_sha256: str
    policy_artifact_sha256_by_name: Mapping[str, str]
    normalization_rule: str = "affine_without_clipping"
    schema_version: str = RETURN_POINT_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RETURN_POINT_LEDGER_SCHEMA:
            raise ValueError("Return-point ledger has an unsupported schema version.")
        if self.split_role not in _SPLIT_ROLES:
            raise ValueError("Return-point ledger split_role must be design or locked_audit.")
        for name in (
            "split_manifest_sha256",
            "numeric_seed_schedule_sha256",
            "factory_registry_sha256",
            "environment_manifest_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"Return-point ledger {name} must be a SHA-256.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in self.probe_budget_grid
        ):
            raise TypeError("Return-point ledger budget grid must contain integers.")
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(
            self,
            "probe_budget_grid",
            tuple(int(value) for value in self.probe_budget_grid),
        )
        if not self.points:
            raise ValueError("Return-point ledger must contain episode points.")
        if self.normalization_rule != "affine_without_clipping":
            raise ValueError("Return-point ledger has an unknown normalization rule.")
        lower = float(self.normalization_lower)
        upper = float(self.normalization_upper)
        cost = float(self.probe_cost_per_use)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("Return-point ledger normalization requires finite upper > lower.")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("Return-point ledger probe cost must be finite and non-negative.")
        grid = self.probe_budget_grid
        if not grid or grid[0] != 0 or any(
            right <= left for left, right in zip(grid, grid[1:], strict=False)
        ):
            raise ValueError("Return-point ledger budget grid must start at zero and increase.")
        seen: set[tuple[str, str, str, int]] = set()
        source_hashes: set[str] = set()
        roles: set[str] = set()
        cells_by_policy: dict[str, set[tuple[str, str, int, int]]] = {}
        for record in self.points:
            point = record.point
            roles.add(record.split_role)
            if int(point.budget) not in grid:
                raise ValueError("A return point uses a budget outside the frozen grid.")
            if float(point.probe_cost_per_use) != cost:
                raise ValueError("A return point changed the frozen probe cost.")
            key = (
                record.split_role,
                str(point.policy_name),
                str(record.episode_uid),
                int(point.budget),
            )
            if key in seen:
                raise ValueError("Return-point ledger repeats a policy/episode/budget row.")
            seen.add(key)
            if record.source_episode_log_sha256 in source_hashes:
                raise ValueError("Return-point ledger reuses a source episode-log hash.")
            source_hashes.add(record.source_episode_log_sha256)
            policy_cells = cells_by_policy.setdefault(str(point.policy_name), set())
            policy_cell = (
                str(point.split_group_id),
                str(point.seed_group),
                int(point.seed),
                int(point.budget),
            )
            if policy_cell in policy_cells:
                raise ValueError(
                    "Return-point ledger repeats a policy/group/numeric-seed/budget cell."
                )
            policy_cells.add(policy_cell)
        if roles != {self.split_role}:
            raise ValueError("Every return point must match the ledger split_role.")
        policy_names = set(cells_by_policy)
        if set(self.policy_artifact_sha256_by_name) != policy_names:
            raise ValueError("Return-point ledger policy artifact hashes changed policy coverage.")
        if not all(
            _is_sha256(value)
            for value in self.policy_artifact_sha256_by_name.values()
        ):
            raise ValueError("Return-point ledger policy artifact hashes must be SHA-256.")
        reference_cells: set[tuple[str, str, int, int]] | None = None
        for policy_name, policy_cells in sorted(cells_by_policy.items()):
            if reference_cells is None:
                reference_cells = policy_cells
            elif policy_cells != reference_cells:
                raise ValueError(
                    "Every policy must cover the identical split-group, numeric-seed, "
                    "and probe-budget cells; mismatch at " + policy_name + "."
                )

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_role": self.split_role,
            "probe_budget_grid": list(self.probe_budget_grid),
            "normalization_rule": self.normalization_rule,
            "normalization_lower": float(self.normalization_lower),
            "normalization_upper": float(self.normalization_upper),
            "probe_cost_per_use": float(self.probe_cost_per_use),
            "split_manifest_sha256": str(self.split_manifest_sha256),
            "numeric_seed_schedule_sha256": str(self.numeric_seed_schedule_sha256),
            "factory_registry_sha256": str(self.factory_registry_sha256),
            "environment_manifest_sha256": str(self.environment_manifest_sha256),
            "policy_artifact_sha256_by_name": {
                str(name): str(value)
                for name, value in sorted(self.policy_artifact_sha256_by_name.items())
            },
            "points": [
                point.canonical_payload()
                for point in sorted(
                    self.points,
                    key=lambda item: (
                        item.split_role,
                        item.point.policy_name,
                        item.point.mechanism,
                        item.point.identity_group,
                        item.point.style_group,
                        item.point.layout_group,
                        item.point.seed_group,
                        item.point.seed,
                        item.point.budget,
                        item.episode_uid,
                    ),
                )
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.core_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self.core_payload(), "ledger_sha256": self.sha256}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReturnPointLedgerV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Return-point ledger must be a mapping.")
        observed = set(payload)
        if observed == set(_LEDGER_CORE_KEYS):
            core = dict(payload)
        elif observed == set(_LEDGER_CORE_KEYS | _BOUND_ENVELOPE_KEYS):
            if payload.get("measurement_schema_version") != MEASUREMENT_SCHEMA:
                raise ValueError("Return-point ledger has the wrong parent measurement schema.")
            for key in (
                "preregistration_sha256",
                "resolved_path_c_sha256",
                "artifact_sha256",
            ):
                if not _is_sha256(payload.get(key)):
                    raise ValueError(f"Return-point ledger {key} must be a SHA-256 digest.")
            if not isinstance(payload.get("semantic_bindings"), Mapping):
                raise TypeError("Return-point ledger semantic_bindings must be a mapping.")
            core = {key: payload[key] for key in _LEDGER_CORE_KEYS}
        else:
            missing = sorted(set(_LEDGER_CORE_KEYS).difference(observed))
            unknown = sorted(
                observed.difference(set(_LEDGER_CORE_KEYS | _BOUND_ENVELOPE_KEYS))
            )
            raise ValueError(
                "Return-point ledger key mismatch; "
                f"missing={missing}, unknown={unknown}."
            )
        if core["schema_version"] != RETURN_POINT_LEDGER_SCHEMA:
            raise ValueError("Return-point ledger has the wrong schema version.")
        if core["split_role"] not in _SPLIT_ROLES:
            raise ValueError("Return-point ledger split role is not registered.")
        raw_points = core["points"]
        if isinstance(raw_points, (str, bytes)) or not isinstance(raw_points, Sequence):
            raise TypeError("Return-point ledger points must be a sequence.")
        ledger = cls(
            points=tuple(
                BoundReturnProbeBudgetPointV1.from_mapping(item)
                for item in raw_points
            ),
            split_role=str(core["split_role"]),
            probe_budget_grid=tuple(
                _strict_integer(value, "return-point ledger budget")
                for value in core["probe_budget_grid"]
            ),
            normalization_rule=str(core["normalization_rule"]),
            normalization_lower=_finite_number(
                core["normalization_lower"], "return-point ledger normalization_lower"
            ),
            normalization_upper=_finite_number(
                core["normalization_upper"], "return-point ledger normalization_upper"
            ),
            probe_cost_per_use=_finite_number(
                core["probe_cost_per_use"], "return-point ledger probe_cost_per_use"
            ),
            split_manifest_sha256=str(core["split_manifest_sha256"]),
            numeric_seed_schedule_sha256=str(core["numeric_seed_schedule_sha256"]),
            factory_registry_sha256=str(core["factory_registry_sha256"]),
            environment_manifest_sha256=str(core["environment_manifest_sha256"]),
            policy_artifact_sha256_by_name=dict(
                core["policy_artifact_sha256_by_name"]
            ),
            schema_version=str(core["schema_version"]),
        )
        if core["ledger_sha256"] != ledger.sha256:
            raise ValueError("Return-point ledger SHA-256 does not match its content.")
        return ledger

    def validate_split_manifest(self, manifest: SplitManifestV1) -> None:
        """Verify that every episode is assigned to this role by the frozen split."""

        if not isinstance(manifest, SplitManifestV1):
            raise TypeError("manifest must be a SplitManifestV1.")
        if manifest.sha256 != self.split_manifest_sha256:
            raise ValueError("Return-point ledger changed the frozen split manifest.")
        if manifest.numeric_seed_schedule_sha256 != self.numeric_seed_schedule_sha256:
            raise ValueError("Return-point ledger changed the numeric seed schedule.")
        groups = {group.group_id: group for group in manifest.groups}
        assignments = manifest.assignment_by_group
        observed_group_ids: set[str] = set()
        observed_cells: set[tuple[str, str, int, int]] = set()
        for record in self.points:
            spec = record.source_episode_log.spec
            group = groups.get(spec.split_group_id)
            if group is None:
                raise ValueError(
                    f"Return episode references unknown split group {spec.split_group_id!r}."
                )
            if assignments.get(group.group_id) != self.split_role:
                raise ValueError("Return episode role disagrees with the split manifest.")
            observed_group_ids.add(str(group.group_id))
            observed = (
                spec.mechanism,
                spec.identity_group,
                spec.style_group,
                spec.seed_group,
                spec.layout_group,
            )
            expected = (
                group.mechanism,
                group.identity_group,
                group.style_group,
                group.seed_group,
                group.layout_group,
            )
            if observed != expected:
                raise ValueError(
                    "Return episode grouping fields disagree with its split-manifest group."
                )
            manifest.validate_numeric_seed(group.group_id, spec.seed)
            cell = (
                str(record.point.policy_name),
                str(group.group_id),
                int(spec.seed),
                int(spec.probe_budget),
            )
            if cell in observed_cells:
                raise ValueError(
                    "Return-point ledger repeats a policy/group/seed/budget cell."
                )
            observed_cells.add(cell)
        expected_group_ids = {
            str(group_id)
            for group_id, role in assignments.items()
            if str(role) == self.split_role
        }
        if observed_group_ids != expected_group_ids:
            missing = sorted(expected_group_ids.difference(observed_group_ids))
            unknown = sorted(observed_group_ids.difference(expected_group_ids))
            raise ValueError(
                "Return-point ledger must cover every frozen split group for its role; "
                f"missing={missing}, unknown={unknown}."
            )
        expected_cells = {
            (policy_name, group_id, int(seed), int(probe_budget))
            for policy_name in self.policy_artifact_sha256_by_name
            for group_id in expected_group_ids
            for seed in manifest.numeric_seeds_for_group(group_id)
            for probe_budget in self.probe_budget_grid
        }
        if observed_cells != expected_cells:
            missing_cells = sorted(expected_cells.difference(observed_cells))
            unknown_cells = sorted(observed_cells.difference(expected_cells))
            raise ValueError(
                "Return-point ledger group and seed schedule is incomplete; "
                f"missing={missing_cells}, unknown={unknown_cells}."
            )

    def curves(
        self,
        *,
        split_role: str,
        policy_names: Sequence[str] | None = None,
    ) -> tuple[ReturnProbeBudgetCurve, ...]:
        if split_role not in _SPLIT_ROLES:
            raise ValueError("Curve split_role must be design or locked_audit.")
        if split_role != self.split_role:
            raise ValueError("Curve split_role does not match the return-point ledger.")
        allowed = None if policy_names is None else set(map(str, policy_names))
        grouped: dict[
            tuple[str, str, str, str, str, str, str, int],
            list[ReturnProbeBudgetPoint],
        ] = {}
        for record in self.points:
            point = record.point
            if record.split_role != split_role:
                continue
            if allowed is not None and point.policy_name not in allowed:
                continue
            key = (
                str(point.policy_name),
                str(point.split_group_id),
                str(point.mechanism),
                str(point.identity_group),
                str(point.style_group),
                str(point.layout_group),
                str(point.seed_group),
                int(point.seed),
            )
            grouped.setdefault(key, []).append(point)
        if allowed is not None:
            missing = sorted(allowed.difference({key[0] for key in grouped}))
            if missing:
                raise ValueError(
                    f"Return-point ledger lacks {split_role} policy curve(s): {missing}."
                )
        curves = [
            build_return_probe_budget_curve(
                points,
                budget_grid=self.probe_budget_grid,
                normalization_lower=self.normalization_lower,
                normalization_upper=self.normalization_upper,
            )
            for _key, points in sorted(grouped.items())
        ]
        if not curves:
            raise ValueError(f"Return-point ledger has no {split_role} curves.")
        return tuple(curves)


def run_acting_policy_grid(
    *,
    split_role: str,
    episode_specs: Sequence[ActingEpisodeSpec],
    policy_factories: Mapping[
        str,
        Callable[[ActingEpisodeSpec], PathCActingPolicy]
        | FormalActingPolicyFactoryEntry,
    ],
    environment_factory: Callable[[ActingEpisodeSpec], PathCActingEnvironment],
    probe_cost_per_use: float,
    probe_budget_grid: Sequence[int],
    normalization_lower: float,
    normalization_upper: float,
    split_manifest: SplitManifestV1,
    factory_registry_sha256: str,
    environment_manifest_sha256: str,
    policy_artifact_sha256_by_name: Mapping[str, str],
) -> ReturnPointLedgerV1:
    """Run every policy through one runner over the same frozen episode cells."""

    if split_role not in _SPLIT_ROLES:
        raise ValueError("Acting grid split_role must be design or locked_audit.")
    if not episode_specs:
        raise ValueError("Acting grid requires episode specifications.")
    if not policy_factories:
        raise ValueError("Acting grid requires policy factories.")
    for name, value in (
        ("factory_registry_sha256", factory_registry_sha256),
        ("environment_manifest_sha256", environment_manifest_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"Acting grid {name} must be SHA-256.")
    if any(
        not isinstance(name, str) or not name.strip()
        for name in policy_factories
    ):
        raise ValueError("Acting grid policy names must be non-empty.")
    grid = tuple(_strict_integer(value, "acting grid budget") for value in probe_budget_grid)
    if not grid or grid[0] != 0 or any(
        right <= left for left, right in zip(grid, grid[1:], strict=False)
    ):
        raise ValueError("Acting grid budgets must start at zero and increase.")
    specs = tuple(episode_specs)
    if any(spec.split_role != split_role for spec in specs):
        raise ValueError("Every acting episode spec must match the requested split role.")
    if any(spec.probe_budget not in grid for spec in specs):
        raise ValueError("An acting episode spec uses a budget outside the frozen grid.")
    cells = [
        (
            spec.split_group_id,
            spec.mechanism,
            spec.identity_group,
            spec.style_group,
            spec.layout_group,
            spec.seed_group,
            spec.seed,
            spec.probe_budget,
        )
        for spec in specs
    ]
    if len(cells) != len(set(cells)):
        raise ValueError("Acting grid repeats a split/group/seed/budget cell.")
    expected_cells = {
        (*cell[:-1], budget)
        for cell in cells
        for budget in grid
    }
    if set(cells) != expected_cells:
        raise ValueError("Every acting group/seed cell must cover the complete budget grid.")
    training_steps = {spec.training_environment_steps for spec in specs}
    updates = {spec.gradient_updates for spec in specs}
    limits = {spec.evaluation_environment_step_limit for spec in specs}
    schedules = {spec.evaluation_schedule_id for spec in specs}
    if any(len(values) != 1 for values in (training_steps, updates, limits, schedules)):
        raise ValueError(
            "Acting grid specs must share training budget and evaluation schedule."
        )
    maximum_environment_steps = next(iter(limits))
    runner = PathCActingRunner(
        cost_per_probe=float(probe_cost_per_use),
        maximum_environment_steps=int(maximum_environment_steps),
    )
    records: list[BoundReturnProbeBudgetPointV1] = []
    for policy_name in sorted(map(str, policy_factories)):
        factory = policy_factories[policy_name]
        for spec in sorted(
            specs,
            key=lambda item: (
                item.split_group_id,
                item.mechanism,
                item.identity_group,
                item.style_group,
                item.layout_group,
                item.seed_group,
                item.seed,
                item.probe_budget,
                item.episode_uid,
            ),
        ):
            environment = environment_factory(spec)
            if not isinstance(environment, PathCActingEnvironment):
                raise TypeError("environment_factory did not return PathCActingEnvironment.")
            policy = (
                factory.build(spec, environment)
                if isinstance(factory, FormalActingPolicyFactoryEntry)
                else factory(spec)
            )
            if not isinstance(policy, PathCActingPolicy):
                raise TypeError(f"Policy factory {policy_name!r} did not return PathCActingPolicy.")
            if str(policy.policy_name) != policy_name:
                raise ValueError("Policy factory key differs from policy.policy_name.")
            if split_role == "locked_audit" and bool(policy.oracle_baseline):
                raise ValueError(
                    "Oracle acting policies are design-only and cannot enter the locked ledger."
                )
            episode = runner.run_episode(
                environment=environment,
                policy=policy,
                spec=spec,
            )
            records.append(
                BoundReturnProbeBudgetPointV1(
                    split_role=split_role,
                    episode_uid=spec.episode_uid,
                    source_episode_log_sha256=episode.sha256,
                    source_episode_log=episode,
                    point=episode.return_probe_budget_point(),
                )
            )
    ledger = ReturnPointLedgerV1(
        points=tuple(records),
        split_role=split_role,
        probe_budget_grid=grid,
        normalization_lower=float(normalization_lower),
        normalization_upper=float(normalization_upper),
        probe_cost_per_use=float(probe_cost_per_use),
        split_manifest_sha256=split_manifest.sha256,
        numeric_seed_schedule_sha256=split_manifest.numeric_seed_schedule_sha256,
        factory_registry_sha256=str(factory_registry_sha256),
        environment_manifest_sha256=str(environment_manifest_sha256),
        policy_artifact_sha256_by_name=dict(policy_artifact_sha256_by_name),
    )
    ledger.validate_split_manifest(split_manifest)
    return ledger


@dataclass(frozen=True)
class FormalActingBenchmarkLedgersV1:
    """Independent raw design and locked-audit ledgers from one factory registry."""

    design: ReturnPointLedgerV1
    locked_audit: ReturnPointLedgerV1
    factory_registry_manifest: Mapping[str, Any]
    schema_version: str = "path_c_formal_acting_benchmark_ledgers_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_formal_acting_benchmark_ledgers_v1":
            raise ValueError("Formal acting benchmark ledger schema version changed.")
        if self.design.split_role != "design":
            raise ValueError("Formal acting design ledger has the wrong split role.")
        if self.locked_audit.split_role != "locked_audit":
            raise ValueError("Formal acting locked ledger has the wrong split role.")
        design_names = {record.point.policy_name for record in self.design.points}
        locked_names = {record.point.policy_name for record in self.locked_audit.points}
        if design_names != set(FORMAL_ACTING_POLICY_NAMES):
            raise ValueError("Formal design ledger does not contain every registered policy.")
        if locked_names != set(FORMAL_LOCKED_ACTING_POLICY_NAMES):
            raise ValueError("Formal locked ledger policy set is invalid.")
        if "direct_information" in locked_names:
            raise ValueError("direct_information is design-only.")
        registry_sha256 = self.factory_registry_manifest.get("sha256")
        if (
            registry_sha256 != self.design.factory_registry_sha256
            or registry_sha256 != self.locked_audit.factory_registry_sha256
        ):
            raise ValueError("Formal ledgers changed the factory registry hash.")
        if self.design.environment_manifest_sha256 != (
            self.locked_audit.environment_manifest_sha256
        ):
            raise ValueError("Formal ledgers changed the environment manifest hash.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "factory_registry": dict(self.factory_registry_manifest),
            "design": self.design.to_mapping(),
            "locked_audit": self.locked_audit.to_mapping(),
        }


def run_formal_acting_benchmark(
    *,
    design_episode_specs: Sequence[ActingEpisodeSpec],
    locked_audit_episode_specs: Sequence[ActingEpisodeSpec],
    factory_registry: FormalActingPolicyFactoryRegistry,
    environment_factory: Callable[[ActingEpisodeSpec], PathCActingEnvironment],
    probe_cost_per_use: float,
    probe_budget_grid: Sequence[int],
    normalization_lower: float,
    normalization_upper: float,
    split_manifest: SplitManifestV1,
    environment_manifest_sha256: str,
) -> FormalActingBenchmarkLedgersV1:
    """Fail before launch unless every role-required concrete factory is available."""

    design_factories = factory_registry.factories_for_role("design")
    locked_factories = factory_registry.factories_for_role("locked_audit")
    contract = factory_registry.benchmark_contract
    registry_manifest = factory_registry.to_manifest()
    registry_sha256 = str(registry_manifest["sha256"])
    policy_hashes = {
        entry.policy_name: canonical_sha256(entry.to_manifest())
        for entry in factory_registry.entries
    }
    if split_manifest.sha256 != contract.split_manifest_sha256:
        raise ValueError("Formal acting registry changed the frozen split manifest.")
    if tuple(map(int, probe_budget_grid)) != contract.probe_budget_grid:
        raise ValueError("Formal acting runner changed the frozen probe-budget grid.")
    if float(probe_cost_per_use) != float(contract.probe_cost_per_use):
        raise ValueError("Formal acting runner changed the frozen per-probe cost.")
    design_specs = tuple(design_episode_specs)
    locked_specs = tuple(locked_audit_episode_specs)
    if not design_specs or not locked_specs:
        raise ValueError("Formal acting benchmark requires both split grids.")
    combined = design_specs + locked_specs
    for spec in combined:
        contract.validate_episode_spec(spec)
    training_steps = {spec.training_environment_steps for spec in combined}
    updates = {spec.gradient_updates for spec in combined}
    limits = {spec.evaluation_environment_step_limit for spec in combined}
    schedules = {spec.evaluation_schedule_id for spec in combined}
    if any(len(values) != 1 for values in (training_steps, updates, limits, schedules)):
        raise ValueError(
            "Design and locked policies must share training steps, updates, "
            "evaluation limit, and evaluation schedule."
        )
    design = run_acting_policy_grid(
        split_role="design",
        episode_specs=design_specs,
        policy_factories=design_factories,
        environment_factory=environment_factory,
        probe_cost_per_use=float(probe_cost_per_use),
        probe_budget_grid=probe_budget_grid,
        normalization_lower=float(normalization_lower),
        normalization_upper=float(normalization_upper),
        split_manifest=split_manifest,
        factory_registry_sha256=registry_sha256,
        environment_manifest_sha256=environment_manifest_sha256,
        policy_artifact_sha256_by_name={
            name: policy_hashes[name] for name in design_factories
        },
    )
    locked = run_acting_policy_grid(
        split_role="locked_audit",
        episode_specs=locked_specs,
        policy_factories=locked_factories,
        environment_factory=environment_factory,
        probe_cost_per_use=float(probe_cost_per_use),
        probe_budget_grid=probe_budget_grid,
        normalization_lower=float(normalization_lower),
        normalization_upper=float(normalization_upper),
        split_manifest=split_manifest,
        factory_registry_sha256=registry_sha256,
        environment_manifest_sha256=environment_manifest_sha256,
        policy_artifact_sha256_by_name={
            name: policy_hashes[name] for name in locked_factories
        },
    )
    return FormalActingBenchmarkLedgersV1(
        design=design,
        locked_audit=locked,
        factory_registry_manifest=factory_registry.to_manifest(),
    )


__all__ = [
    "RETURN_POINT_LEDGER_SCHEMA",
    "BoundReturnProbeBudgetPointV1",
    "ReturnPointLedgerV1",
    "FormalActingBenchmarkLedgersV1",
    "canonical_sha256",
    "run_acting_policy_grid",
    "run_formal_acting_benchmark",
]
