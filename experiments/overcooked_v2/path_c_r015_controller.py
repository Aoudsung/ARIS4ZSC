"""R015 audit-only controller over official local history.

This module implements the registered A1/A2-mask/A2-use decision rule.  It is
separate from adaptation training: importing or constructing this controller
does not make the learned 10M-step proposal controller available.

The planner is deliberately parameterized by a full-horizon branch executor.
The executor owns hidden simulator states; the decision layer receives only a
posterior produced from official history and recomputed branch returns.
"""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar


R015_ATOMIC_ACTIONS = ("up", "down", "right", "left", "stay", "interact")
R015_CONSULTATION_STEPS = tuple(range(1, 101, 5))
R015_CONTINUATION_CONTROLLER_ID = "map_prototype_committed_cook_v1"
R015_BRANCH_SAMPLING_ID = "sampled_hidden_state_branches_v1"
R015_BRANCH_BELIEF_ID = "frozen_belief_branch_continuation_v1"
R015_GRID_EVALUATION_ID = "lazy_ascending_first_pass"
R015_NESTED_BRANCH_COUNTS = (2, 4, 8, 16)
R015_FILTER_ALGORITHM_ID_V1 = "stratified_hidden_state_particles_v1"
R015_FILTER_ALGORITHM_ID_V2 = "official_history_fully_adapted_particle_filter_v2"
R015_RESAMPLING_ALGORITHM_ID_V1 = "systematic_per_prototype_v1"
R015_RESAMPLING_ALGORITHM_ID_V2 = "strict_systematic_per_prototype_v2"
R015_FUTURE_RANDOM_DERIVATION_ID = (
    "path_c_r015_controller_key_v1_future_environment_index_v1"
)
_UINT64_MODULUS = 1 << 64
R015_ALLOWED_CONTROLLER_FIELDS = (
    "official_local_observation",
    "ego_action_history",
    "ego_option_or_probe_history",
    "ego_option_or_probe_elapsed_time",
    "raw_team_reward_history",
    "episode_boundaries",
    "legal_action_mask_derived",
    "task_progress_derived",
    "response_summary_v1_derived",
)
R015_FORBIDDEN_CONTROLLER_FIELDS = (
    "evaluator_partner_action",
    "partner_identity",
    "partner_private_observation",
    "partner_recurrent_state",
    "complete_grid",
    "complete_hidden_state",
    "hidden_recipe",
    "evaluator_random_state",
    "training_family_label",
)
R015_PASSIVE_FILTER_UPDATE_FIELDS = frozenset(
    {
        "official_local_observation",
        "ego_action_history",
        "raw_team_reward_history",
        "episode_boundaries",
    }
)
R015_RESAMPLING_TIMINGS = (
    "adaptive_ess_below_half_v1",
    "every_environment_step_v1",
)
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("R015 canonical values must be finite.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("R015 canonical mappings require string keys.")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    # NumPy/JAX arrays and scalar values expose tolist/item without granting
    # the controller access to evaluator-only fields.
    if hasattr(value, "tolist"):
        return _canonical_value(value.tolist())
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise TypeError(f"Unsupported R015 canonical value {type(value).__name__}.")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_controller_key(root_key: str, *coordinates: object) -> str:
    if not _is_sha256(root_key):
        raise ValueError("R015 controller root key must be SHA-256.")
    return canonical_sha256(
        {
            "schema_version": "path_c_r015_controller_key_v1",
            "root_key": root_key,
            "coordinates": [str(item) for item in coordinates],
        }
    )


def _find_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in R015_FORBIDDEN_CONTROLLER_FIELDS:
                return str(key)
            nested = _find_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, (tuple, list)):
        for item in value:
            nested = _find_forbidden_key(item)
            if nested is not None:
                return nested
    return None


def _contains_canonical_value(container: Any, target: Any) -> bool:
    """Return whether a nested canonical value exactly contains ``target``."""

    normalized_container = _canonical_value(container)
    normalized_target = _canonical_value(target)
    if normalized_container == normalized_target:
        return True
    if isinstance(normalized_container, Mapping):
        return any(
            _contains_canonical_value(item, normalized_target)
            for item in normalized_container.values()
        )
    if isinstance(normalized_container, list):
        return any(
            _contains_canonical_value(item, normalized_target)
            for item in normalized_container
        )
    return False


@dataclass(frozen=True)
class OfficialHistoryV1:
    """Canonical sequence containing only the fields allowed by R015 §1.1."""

    records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("R015 official history must contain at least one record.")
        allowed = set(R015_ALLOWED_CONTROLLER_FIELDS)
        normalized: list[Mapping[str, Any]] = []
        for record in self.records:
            if not isinstance(record, Mapping):
                raise TypeError("R015 official-history records must be mappings.")
            unknown = set(record) - allowed
            if unknown:
                raise ValueError(
                    "R015 official history contains unknown field(s): "
                    + ", ".join(sorted(unknown))
                )
            forbidden = _find_forbidden_key(record)
            if forbidden is not None:
                raise ValueError(
                    f"R015 official history contains forbidden field {forbidden}."
                )
            if "official_local_observation" not in record:
                raise ValueError(
                    "Every R015 history record requires official_local_observation."
                )
            normalized.append(_canonical_value(record))
        object.__setattr__(self, "records", tuple(normalized))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> list[Mapping[str, Any]]:
        return [copy.deepcopy(dict(record)) for record in self.records]

    def append(self, record: Mapping[str, Any]) -> "OfficialHistoryV1":
        return OfficialHistoryV1((*self.records, record))


@dataclass(frozen=True)
class CurrentResponseProjectionV1:
    """Frozen deletion map for the current registered response and its aliases."""

    alias_paths: tuple[tuple[str, ...], ...]
    recurrent_state_write_paths: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        paths = (*self.alias_paths, *self.recurrent_state_write_paths)
        if any(not path or any(not part for part in path) for path in paths):
            raise ValueError("R015 response-projection paths must be non-empty.")
        if len(set(paths)) != len(paths):
            raise ValueError("R015 response-projection paths must be unique.")

    @staticmethod
    def _delete_path(record: dict[str, Any], path: tuple[str, ...]) -> None:
        cursor: Any = record
        for part in path[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                return
            cursor = cursor[part]
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)

    def project(
        self,
        history: OfficialHistoryV1,
        *,
        probe_id: str,
        probe_step: int,
    ) -> "ProjectedResponseV1":
        if not probe_id:
            raise ValueError("R015 response projection requires a probe id.")
        _positive_int(probe_step, name="probe_step")
        records = copy.deepcopy(history.to_payload())
        response: Mapping[str, Any] | None = None
        response_record_index: int | None = None
        for record_index, record in enumerate(records):
            candidate = record.get("response_summary_v1_derived")
            if isinstance(candidate, Mapping) and (
                candidate.get("source_probe_id") == probe_id
                and int(candidate.get("source_probe_step", -1)) == probe_step
            ):
                if response is not None:
                    raise ValueError("Current probe response appears more than once.")
                response = copy.deepcopy(dict(candidate))
                response_record_index = record_index
                record.pop("response_summary_v1_derived")
        if response is None:
            raise ValueError("Current probe response is absent from official history.")
        if response_record_index is None:  # pragma: no cover - guarded above
            raise RuntimeError("R015 response projection lost its source record.")
        for record in records[response_record_index:]:
            for path in (*self.alias_paths, *self.recurrent_state_write_paths):
                self._delete_path(record, path)
        masked = OfficialHistoryV1(tuple(records))
        if _contains_canonical_value(masked.to_payload(), response):
            raise ValueError("Masked history retains a serialized current response.")
        return ProjectedResponseV1(masked_history=masked, current_response=response)


@dataclass(frozen=True)
class ProjectedResponseV1:
    masked_history: OfficialHistoryV1
    current_response: Mapping[str, Any]


BeliefT = TypeVar("BeliefT")
MaskedBeliefUpdater = Callable[[BeliefT, OfficialHistoryV1], BeliefT]
ResponseBeliefUpdater = Callable[
    [BeliefT, OfficialHistoryV1, Mapping[str, Any]], BeliefT
]


@dataclass(frozen=True)
class PairedBeliefUpdateV1:
    """Apply the one registered interface difference between mask and use."""

    masked: BeliefT
    used: BeliefT
    masked_history_sha256: str
    current_response_sha256: str


def apply_paired_response_update(
    belief: BeliefT,
    projected: ProjectedResponseV1,
    *,
    mask_updater: MaskedBeliefUpdater[BeliefT],
    use_updater: ResponseBeliefUpdater[BeliefT],
) -> PairedBeliefUpdateV1:
    """Call ``B_mask(q,x)`` and ``B_use(q,x,y)`` with no other input change."""

    masked = mask_updater(copy.deepcopy(belief), projected.masked_history)
    used = use_updater(
        copy.deepcopy(belief),
        projected.masked_history,
        copy.deepcopy(projected.current_response),
    )
    return PairedBeliefUpdateV1(
        masked=masked,
        used=used,
        masked_history_sha256=projected.masked_history.sha256,
        current_response_sha256=canonical_sha256(projected.current_response),
    )


@dataclass(frozen=True)
class HiddenStateParticleV1:
    """One private compatible simulator state inside the official-history filter."""

    prototype_id: str
    state_sha256: str
    state: Any
    weight: float

    def __post_init__(self) -> None:
        if not self.prototype_id or not _is_sha256(self.state_sha256):
            raise ValueError("R015 particles require a prototype and state SHA-256.")
        weight = _finite(self.weight, name="particle weight")
        if weight < 0.0:
            raise ValueError("R015 particle weights must be non-negative.")
        object.__setattr__(self, "weight", weight)


ParticleFactory = Callable[[str, int, str], HiddenStateParticleV1]
ParticleTransition = Callable[
    [HiddenStateParticleV1, Mapping[str, Any], str],
    Sequence[HiddenStateParticleV1],
]
ParticleCompatibility = Callable[
    [HiddenStateParticleV1, Mapping[str, Any]], float
]


@dataclass(frozen=True)
class StratifiedParticleBeliefV1:
    """Four-prototype posterior with an explicit between/within factorization.

    ``prototype_masses`` is the four-element posterior ``q_k``.  The
    ``within_prototype_weights`` values are conditional weights that sum to one
    inside each prototype.  Particle ``weight`` values remain the globally
    normalized product of those two quantities so the planning sampler has one
    canonical sequence to consume.

    The historical v1 resampling identifier is retained only for diagnostic
    replay.  A belief can enter the formal planner only when it carries the v2
    filter and strict-resampling identifiers.
    """

    prototype_ids: tuple[str, ...]
    particles_per_prototype: int
    registered_prototype_prior: Mapping[str, float]
    resampling_algorithm: str
    resampling_interval_environment_steps: int
    particles: tuple[HiddenStateParticleV1, ...]
    resampling_timing: str = "every_environment_step_v1"
    resampling_ess_fraction_threshold: float | None = None
    last_pre_resample_ess_fraction_by_prototype: Mapping[str, float] = field(
        default_factory=dict
    )
    last_resampled_prototypes: tuple[str, ...] = ()
    update_count: int = 0
    filter_algorithm_id: str | None = None
    prototype_masses: Mapping[str, float] = field(default_factory=dict)
    within_prototype_weights: Mapping[str, tuple[float, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if len(self.prototype_ids) != 4 or len(set(self.prototype_ids)) != 4:
            raise ValueError("R015 filtering requires exactly four prototype ids.")
        _positive_int(
            self.particles_per_prototype,
            name="particles_per_prototype",
        )
        if set(self.registered_prototype_prior) != set(self.prototype_ids):
            raise ValueError("R015 filtering prior must cover the four prototypes.")
        normalized_prior = {
            prototype_id: _finite(
                self.registered_prototype_prior[prototype_id],
                name=f"prototype prior {prototype_id}",
            )
            for prototype_id in self.prototype_ids
        }
        if any(value <= 0.0 for value in normalized_prior.values()) or not math.isclose(
            sum(normalized_prior.values()),
            1.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("R015 filtering prior must be positive and sum to one.")
        object.__setattr__(self, "registered_prototype_prior", normalized_prior)
        algorithm_by_resampling = {
            R015_RESAMPLING_ALGORITHM_ID_V1: R015_FILTER_ALGORITHM_ID_V1,
            R015_RESAMPLING_ALGORITHM_ID_V2: R015_FILTER_ALGORITHM_ID_V2,
        }
        if self.resampling_algorithm not in algorithm_by_resampling:
            raise ValueError("R015 filter uses an unregistered resampling algorithm.")
        expected_filter_algorithm = algorithm_by_resampling[
            self.resampling_algorithm
        ]
        filter_algorithm_id = (
            expected_filter_algorithm
            if self.filter_algorithm_id is None
            else str(self.filter_algorithm_id)
        )
        if filter_algorithm_id != expected_filter_algorithm:
            raise ValueError(
                "R015 filter and resampling algorithm identifiers disagree."
            )
        object.__setattr__(self, "filter_algorithm_id", filter_algorithm_id)
        _positive_int(
            self.resampling_interval_environment_steps,
            name="resampling_interval_environment_steps",
        )
        if self.resampling_interval_environment_steps != 1:
            raise ValueError(
                "R015 registered resampling timing replaces fixed multi-step intervals."
            )
        if self.resampling_timing not in R015_RESAMPLING_TIMINGS:
            raise ValueError("R015 filter uses an unregistered resampling timing.")
        threshold = self.resampling_ess_fraction_threshold
        if self.resampling_timing == "adaptive_ess_below_half_v1":
            if threshold is None or not math.isclose(
                _finite(threshold, name="resampling ESS fraction threshold"),
                0.5,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Adaptive R015 resampling must use ESS < 0.5N.")
        elif threshold is not None:
            raise ValueError("Every-step R015 resampling does not use an ESS threshold.")
        if (
            isinstance(self.update_count, bool)
            or not isinstance(self.update_count, int)
            or self.update_count < 0
        ):
            raise ValueError("R015 filter update_count must be non-negative.")
        if not self.particles:
            raise ValueError("R015 particle belief cannot be empty.")
        if any(item.prototype_id not in self.prototype_ids for item in self.particles):
            raise ValueError("R015 particle uses an unregistered prototype.")
        total = sum(item.weight for item in self.particles)
        if not math.isclose(total, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("R015 particle weights must sum to one.")
        if any(
            sum(
                item.weight
                for item in self.particles
                if item.prototype_id == prototype_id
            )
            <= 0.0
            for prototype_id in self.prototype_ids
        ):
            raise ValueError("Every R015 prototype must retain positive posterior support.")
        particles_by_prototype = {
            prototype_id: tuple(
                item
                for item in self.particles
                if item.prototype_id == prototype_id
            )
            for prototype_id in self.prototype_ids
        }
        derived_masses = {
            prototype_id: sum(
                item.weight for item in particles_by_prototype[prototype_id]
            )
            for prototype_id in self.prototype_ids
        }
        supplied_masses = dict(self.prototype_masses)
        if supplied_masses:
            if set(supplied_masses) != set(self.prototype_ids):
                raise ValueError("R015 prototype masses must cover all four prototypes.")
            normalized_masses = {
                prototype_id: _finite(
                    supplied_masses[prototype_id],
                    name=f"prototype mass {prototype_id}",
                )
                for prototype_id in self.prototype_ids
            }
            if any(value <= 0.0 for value in normalized_masses.values()) or not (
                math.isclose(
                    sum(normalized_masses.values()),
                    1.0,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError("R015 prototype masses must be positive and sum to one.")
            for prototype_id in self.prototype_ids:
                if not math.isclose(
                    normalized_masses[prototype_id],
                    derived_masses[prototype_id],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "R015 prototype mass differs from its global particle weights."
                    )
        else:
            normalized_masses = derived_masses
        supplied_within = dict(self.within_prototype_weights)
        if supplied_within and set(supplied_within) != set(self.prototype_ids):
            raise ValueError(
                "R015 within-prototype weights must cover all four prototypes."
            )
        normalized_within: dict[str, tuple[float, ...]] = {}
        for prototype_id in self.prototype_ids:
            particles = particles_by_prototype[prototype_id]
            if filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2 and len(
                particles
            ) != self.particles_per_prototype:
                raise ValueError(
                    "R015 v2 belief must retain exactly P particles per prototype."
                )
            derived = tuple(
                item.weight / normalized_masses[prototype_id]
                for item in particles
            )
            if supplied_within:
                values = tuple(
                    _finite(value, name=f"within weight {prototype_id}")
                    for value in supplied_within[prototype_id]
                )
                if len(values) != len(particles) or any(
                    value < 0.0 for value in values
                ) or not math.isclose(
                    sum(values), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        "R015 within-prototype weights must match the particle count "
                        "and sum to one."
                    )
                if any(
                    not math.isclose(
                        value,
                        expected,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    for value, expected in zip(values, derived)
                ):
                    raise ValueError(
                        "R015 conditional weights differ from global particle weights."
                    )
                normalized_within[prototype_id] = values
            else:
                normalized_within[prototype_id] = derived
        if filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2:
            expected_order = tuple(
                prototype_id
                for prototype_id in self.prototype_ids
                for _ in range(self.particles_per_prototype)
            )
            if tuple(item.prototype_id for item in self.particles) != expected_order:
                raise ValueError(
                    "R015 v2 particles must use prototype-major canonical order."
                )
        object.__setattr__(self, "prototype_masses", normalized_masses)
        object.__setattr__(self, "within_prototype_weights", normalized_within)
        diagnostics = dict(self.last_pre_resample_ess_fraction_by_prototype)
        if diagnostics and set(diagnostics) != set(self.prototype_ids):
            raise ValueError("R015 ESS diagnostics must cover all four prototypes.")
        normalized_diagnostics = {
            prototype_id: _finite(value, name=f"ESS fraction {prototype_id}")
            for prototype_id, value in diagnostics.items()
        }
        if any(not 0.0 < value <= 1.0 + 1.0e-12 for value in normalized_diagnostics.values()):
            raise ValueError("R015 ESS fractions must lie in (0,1].")
        object.__setattr__(
            self,
            "last_pre_resample_ess_fraction_by_prototype",
            normalized_diagnostics,
        )
        if len(set(self.last_resampled_prototypes)) != len(
            self.last_resampled_prototypes
        ) or any(
            prototype_id not in self.prototype_ids
            for prototype_id in self.last_resampled_prototypes
        ):
            raise ValueError("R015 resampling diagnostics contain an unknown prototype.")

    @classmethod
    def initialize(
        cls,
        *,
        prototype_ids: Sequence[str],
        particles_per_prototype: int,
        registered_prototype_prior: Mapping[str, float],
        resampling_algorithm: str,
        resampling_interval_environment_steps: int,
        initialization_key: str,
        factory: ParticleFactory,
        resampling_timing: str = "every_environment_step_v1",
        resampling_ess_fraction_threshold: float | None = None,
    ) -> "StratifiedParticleBeliefV1":
        ids = tuple(str(item) for item in prototype_ids)
        if len(ids) != 4 or len(set(ids)) != 4 or any(not item for item in ids):
            raise ValueError("R015 filtering requires four distinct prototype ids.")
        count = _positive_int(
            particles_per_prototype,
            name="particles_per_prototype",
        )
        if not _is_sha256(initialization_key):
            raise ValueError("R015 filter initialization key must be SHA-256.")
        if not isinstance(registered_prototype_prior, Mapping) or set(
            registered_prototype_prior
        ) != set(ids):
            raise ValueError("R015 initialization requires the registered four-way prior.")
        prior = {
            prototype_id: _finite(
                registered_prototype_prior[prototype_id],
                name=f"prototype prior {prototype_id}",
            )
            for prototype_id in ids
        }
        if any(value <= 0.0 for value in prior.values()) or not math.isclose(
            sum(prior.values()),
            1.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("R015 registered prototype prior must be positive and normalized.")
        particles = []
        for prototype_id in ids:
            for index in range(count):
                key = derive_controller_key(
                    initialization_key,
                    "filter_initialization",
                    prototype_id,
                    index,
                )
                particle = factory(prototype_id, index, key)
                if particle.prototype_id != prototype_id:
                    raise ValueError("Particle factory changed its prototype stratum.")
                particles.append(
                    replace(particle, weight=prior[prototype_id] / count)
                )
        return cls(
            prototype_ids=ids,
            particles_per_prototype=count,
            registered_prototype_prior=prior,
            resampling_algorithm=str(resampling_algorithm),
            resampling_interval_environment_steps=_positive_int(
                resampling_interval_environment_steps,
                name="resampling_interval_environment_steps",
            ),
            particles=tuple(particles),
            resampling_timing=str(resampling_timing),
            resampling_ess_fraction_threshold=resampling_ess_fraction_threshold,
        )

    @property
    def prototype_weights(self) -> Mapping[str, float]:
        return dict(self.prototype_masses)

    @property
    def is_formal_v2(self) -> bool:
        return (
            self.filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2
            and self.resampling_algorithm == R015_RESAMPLING_ALGORITHM_ID_V2
        )

    @property
    def public_summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": (
                "path_c_r015_particle_belief_summary_v2"
                if self.is_formal_v2
                else "path_c_r015_particle_belief_summary_v1_diagnostic"
            ),
            "filter_algorithm_id": self.filter_algorithm_id,
            "formal_planning_eligible": self.is_formal_v2,
            "registered_prototype_prior": dict(self.registered_prototype_prior),
            "prototype_weights": dict(self.prototype_weights),
            "within_prototype_weights": {
                prototype_id: list(self.within_prototype_weights[prototype_id])
                for prototype_id in self.prototype_ids
            },
            "particle_counts": {
                prototype_id: sum(
                    item.prototype_id == prototype_id for item in self.particles
                )
                for prototype_id in self.prototype_ids
            },
            "resampling_algorithm": self.resampling_algorithm,
            "resampling_interval_environment_steps": (
                self.resampling_interval_environment_steps
            ),
            "resampling_timing": self.resampling_timing,
            "resampling_ess_fraction_threshold": (
                self.resampling_ess_fraction_threshold
            ),
            "pre_resample_ess_fraction_by_prototype": dict(
                self.last_pre_resample_ess_fraction_by_prototype
            ),
            "resampled_prototypes": list(self.last_resampled_prototypes),
            "update_count": self.update_count,
        }

    @staticmethod
    def _systematic_indices(
        weights: Sequence[float],
        *,
        count: int,
        key: str,
    ) -> tuple[int, ...]:
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("Cannot resample a zero-mass R015 stratum.")
        normalized = [value / total for value in weights]
        uniform = int(key[:16], 16) / float(16**16)
        start = uniform / count
        positions = [start + index / count for index in range(count)]
        cumulative: list[float] = []
        running = 0.0
        for value in normalized:
            running += value
            cumulative.append(running)
        cumulative[-1] = 1.0
        return tuple(
            min(bisect_left(cumulative, position), len(cumulative) - 1)
            for position in positions
        )

    def update(
        self,
        *,
        official_record: Mapping[str, Any],
        update_key: str,
        transition: ParticleTransition,
        compatibility: ParticleCompatibility,
    ) -> "StratifiedParticleBeliefV1":
        OfficialHistoryV1((official_record,))
        if set(official_record) != R015_PASSIVE_FILTER_UPDATE_FIELDS:
            raise ValueError(
                "R015 passive filtering accepts only observation, ego action, "
                "raw team reward, and episode boundary fields."
            )
        if not _is_sha256(update_key):
            raise ValueError("R015 particle update key must be SHA-256.")
        expanded: dict[str, list[HiddenStateParticleV1]] = {
            prototype_id: [] for prototype_id in self.prototype_ids
        }
        for particle_index, particle in enumerate(self.particles):
            transition_key = derive_controller_key(
                update_key,
                "filter_transition",
                particle.prototype_id,
                particle_index,
            )
            children = transition(particle, official_record, transition_key)
            if not children:
                continue
            for child in children:
                if child.prototype_id != particle.prototype_id:
                    raise ValueError("Particle transition crossed prototype strata.")
                likelihood = _finite(
                    compatibility(child, official_record),
                    name="particle compatibility",
                )
                if likelihood < 0.0:
                    raise ValueError("Particle compatibility must be non-negative.")
                expanded[particle.prototype_id].append(
                    replace(
                        child,
                        weight=particle.weight * child.weight * likelihood,
                    )
                )
        prototype_mass = {
            prototype_id: sum(item.weight for item in expanded[prototype_id])
            for prototype_id in self.prototype_ids
        }
        if any(value <= 0.0 for value in prototype_mass.values()):
            raise ValueError(
                "R015 official-history filter lost compatible support for a prototype."
            )
        total_mass = sum(prototype_mass.values())
        next_update_count = self.update_count + 1
        output_particles: list[HiddenStateParticleV1] = []
        ess_fraction_by_prototype = {}
        for prototype_id, items in expanded.items():
            weights = [item.weight for item in items]
            squared = sum(value * value for value in weights)
            effective_count = (
                0.0 if squared <= 0.0 else sum(weights) ** 2 / squared
            )
            ess_fraction_by_prototype[prototype_id] = effective_count / len(items)
        if self.resampling_timing == "every_environment_step_v1":
            resampling_due_by_prototype = {
                prototype_id: True for prototype_id in self.prototype_ids
            }
        else:
            threshold = float(self.resampling_ess_fraction_threshold)
            resampling_due_by_prototype = {
                prototype_id: ess_fraction_by_prototype[prototype_id] < threshold
                for prototype_id in self.prototype_ids
            }
        for prototype_id in self.prototype_ids:
            items = expanded[prototype_id]
            if resampling_due_by_prototype[prototype_id]:
                resample_key = derive_controller_key(
                    update_key,
                    self.resampling_algorithm,
                    prototype_id,
                )
                indices = self._systematic_indices(
                    [item.weight for item in items],
                    count=self.particles_per_prototype,
                    key=resample_key,
                )
                output_weight = (
                    prototype_mass[prototype_id]
                    / total_mass
                    / self.particles_per_prototype
                )
                output_particles.extend(
                    replace(items[index], weight=output_weight) for index in indices
                )
            else:
                output_particles.extend(
                    replace(item, weight=item.weight / total_mass) for item in items
                )
        return StratifiedParticleBeliefV1(
            prototype_ids=self.prototype_ids,
            particles_per_prototype=self.particles_per_prototype,
            registered_prototype_prior=self.registered_prototype_prior,
            resampling_algorithm=self.resampling_algorithm,
            resampling_interval_environment_steps=(
                self.resampling_interval_environment_steps
            ),
            particles=tuple(output_particles),
            resampling_timing=self.resampling_timing,
            resampling_ess_fraction_threshold=(
                self.resampling_ess_fraction_threshold
            ),
            last_pre_resample_ess_fraction_by_prototype=(
                ess_fraction_by_prototype
            ),
            last_resampled_prototypes=tuple(
                prototype_id
                for prototype_id in self.prototype_ids
                if resampling_due_by_prototype[prototype_id]
            ),
            update_count=next_update_count,
            filter_algorithm_id=self.filter_algorithm_id,
        )

    def use_current_response(
        self,
        current_response: Mapping[str, Any],
        *,
        likelihood: Callable[
            [HiddenStateParticleV1, Mapping[str, Any]], float
        ],
    ) -> "StratifiedParticleBeliefV1":
        if not isinstance(current_response, Mapping) or not current_response:
            raise ValueError("R015 response update requires one registered response.")
        weighted = []
        for particle in self.particles:
            response_likelihood = _finite(
                likelihood(particle, current_response),
                name="response likelihood",
            )
            if response_likelihood < 0.0:
                raise ValueError("Response likelihoods must be non-negative.")
            weighted.append(
                replace(
                    particle,
                    weight=particle.weight * response_likelihood,
                )
            )
        total = sum(item.weight for item in weighted)
        if total <= 0.0:
            raise ValueError("Current response has zero probability under all support.")
        normalized = tuple(replace(item, weight=item.weight / total) for item in weighted)
        return StratifiedParticleBeliefV1(
            prototype_ids=self.prototype_ids,
            particles_per_prototype=self.particles_per_prototype,
            registered_prototype_prior=self.registered_prototype_prior,
            resampling_algorithm=self.resampling_algorithm,
            resampling_interval_environment_steps=(
                self.resampling_interval_environment_steps
            ),
            particles=normalized,
            resampling_timing=self.resampling_timing,
            resampling_ess_fraction_threshold=(
                self.resampling_ess_fraction_threshold
            ),
            last_pre_resample_ess_fraction_by_prototype=(
                self.last_pre_resample_ess_fraction_by_prototype
            ),
            last_resampled_prototypes=self.last_resampled_prototypes,
            update_count=self.update_count,
            filter_algorithm_id=self.filter_algorithm_id,
        )


class R015ParticleBeliefView(Protocol):
    """Shared read interface for host and device-backed formal v2 beliefs."""

    prototype_ids: tuple[str, ...]
    particles_per_prototype: int
    particles: tuple[HiddenStateParticleV1, ...]
    filter_algorithm_id: str
    resampling_algorithm: str

    @property
    def prototype_weights(self) -> Mapping[str, float]:
        ...


def _require_formal_v2_belief_contract(belief: Any) -> None:
    """Reject a diagnostic v1 belief before formal routing or planning."""

    if getattr(belief, "filter_algorithm_id", None) != (
        R015_FILTER_ALGORITHM_ID_V2
    ) or getattr(belief, "resampling_algorithm", None) != (
        R015_RESAMPLING_ALGORITHM_ID_V2
    ):
        raise ValueError("R015 formal controller requires the v2 filter contract.")


@dataclass(frozen=True)
class PlanningBranchSampleV1:
    """One equal-weight hidden-state sample in the frozen nested schedule."""

    sample_count: int
    sample_slot: int
    canonical_slot_16: int
    paired_lower_slot: int | None
    systematic_phase_uint64: int
    systematic_position_numerator: int
    systematic_position_denominator: int
    source_particle_index: int
    source_particle: HiddenStateParticleV1
    source_belief_filter_algorithm_id: str
    source_belief_resampling_algorithm: str
    common_random_key: str

    def __post_init__(self) -> None:
        if self.sample_count not in R015_NESTED_BRANCH_COUNTS:
            raise ValueError("R015 planning sample count is outside 2,4,8,16.")
        if not 0 <= self.sample_slot < self.sample_count:
            raise ValueError("R015 planning sample slot is outside its schedule.")
        if not 0 <= self.canonical_slot_16 < 16:
            raise ValueError("R015 canonical planning slot must lie in 0,...,15.")
        if self.paired_lower_slot is not None and not (
            0 <= self.paired_lower_slot < self.sample_count // 2
        ):
            raise ValueError("R015 nested lower-grid slot is invalid.")
        if not 0 <= self.systematic_phase_uint64 < _UINT64_MODULUS:
            raise ValueError("R015 systematic phase must be an unsigned 64-bit value.")
        if self.systematic_position_denominator <= 0 or not (
            0 <= self.systematic_position_numerator
            < self.systematic_position_denominator
        ):
            raise ValueError("R015 systematic position is not in [0,1).")
        if self.source_particle_index < 0 or not _is_sha256(self.common_random_key):
            raise ValueError("R015 planning sample lacks a source index or random key.")
        if self.source_belief_filter_algorithm_id != R015_FILTER_ALGORITHM_ID_V2:
            raise ValueError("R015 formal planning cannot sample a v1 filter belief.")
        if self.source_belief_resampling_algorithm != (
            R015_RESAMPLING_ALGORITHM_ID_V2
        ):
            raise ValueError(
                "R015 formal planning requires strict v2 systematic resampling."
            )

    @property
    def estimator_weight(self) -> float:
        return 1.0 / self.sample_count

    def to_evidence(self) -> Mapping[str, Any]:
        return {
            "branch_sampling_rule_id": R015_BRANCH_SAMPLING_ID,
            "sample_count": self.sample_count,
            "sample_slot": self.sample_slot,
            "canonical_slot_16": self.canonical_slot_16,
            "paired_lower_slot": self.paired_lower_slot,
            "systematic_phase_uint64": self.systematic_phase_uint64,
            "systematic_position_numerator": self.systematic_position_numerator,
            "systematic_position_denominator": self.systematic_position_denominator,
            "source_particle_index": self.source_particle_index,
            "source_particle_prototype_id": self.source_particle.prototype_id,
            "source_particle_state_sha256": self.source_particle.state_sha256,
            "source_particle_weight": self.source_particle.weight,
            "source_belief_filter_algorithm_id": (
                self.source_belief_filter_algorithm_id
            ),
            "source_belief_resampling_algorithm": (
                self.source_belief_resampling_algorithm
            ),
            "estimator_weight": self.estimator_weight,
            "common_random_key": self.common_random_key,
        }


@dataclass(frozen=True)
class NestedHiddenStateBranchScheduleV1:
    """Systematic 2/4/8/16 schedule whose lower grids are exact subsets."""

    sample_count: int
    root_phase_uint64: int
    samples: tuple[PlanningBranchSampleV1, ...]
    particle_sequence_sha256: str
    source_belief_filter_algorithm_id: str = R015_FILTER_ALGORITHM_ID_V2
    source_belief_resampling_algorithm: str = R015_RESAMPLING_ALGORITHM_ID_V2

    def __post_init__(self) -> None:
        if self.sample_count not in R015_NESTED_BRANCH_COUNTS:
            raise ValueError("R015 nested schedule requires 2,4,8, or 16 samples.")
        if len(self.samples) != self.sample_count or tuple(
            sample.sample_slot for sample in self.samples
        ) != tuple(range(self.sample_count)):
            raise ValueError("R015 nested schedule slots are incomplete or reordered.")
        if any(sample.sample_count != self.sample_count for sample in self.samples):
            raise ValueError("R015 nested schedule mixed sample counts.")
        if len({sample.canonical_slot_16 for sample in self.samples}) != len(
            self.samples
        ):
            raise ValueError("R015 nested schedule duplicated a canonical slot.")
        if not _is_sha256(self.particle_sequence_sha256):
            raise ValueError("R015 nested schedule lacks its particle-sequence digest.")
        if self.source_belief_filter_algorithm_id != R015_FILTER_ALGORITHM_ID_V2:
            raise ValueError("R015 nested schedule cannot be labeled with a v1 filter.")
        if self.source_belief_resampling_algorithm != (
            R015_RESAMPLING_ALGORITHM_ID_V2
        ):
            raise ValueError("R015 nested schedule requires strict v2 resampling.")
        if any(
            sample.source_belief_filter_algorithm_id
            != self.source_belief_filter_algorithm_id
            or sample.source_belief_resampling_algorithm
            != self.source_belief_resampling_algorithm
            for sample in self.samples
        ):
            raise ValueError("R015 nested schedule mixed filter contracts.")

    @staticmethod
    def _formal_v2_particle_sequence(
        belief: Any,
    ) -> tuple[HiddenStateParticleV1, ...]:
        """Validate and return the globally normalized formal v2 particle order."""

        try:
            _require_formal_v2_belief_contract(belief)
        except ValueError as error:
            raise ValueError(
                "R015 formal planning refuses the diagnostic v1 filter container."
            ) from error
        prototype_ids = tuple(getattr(belief, "prototype_ids", ()))
        particles_per_prototype = getattr(
            belief, "particles_per_prototype", None
        )
        if (
            len(prototype_ids) != 4
            or len(set(prototype_ids)) != 4
            or isinstance(particles_per_prototype, bool)
            or not isinstance(particles_per_prototype, int)
            or particles_per_prototype <= 0
        ):
            raise ValueError("R015 formal planning requires four P-particle strata.")
        particles = tuple(getattr(belief, "particles", ()))
        if len(particles) != 4 * particles_per_prototype:
            raise ValueError("R015 formal planning received an incomplete particle cloud.")
        expected_order = tuple(
            prototype_id
            for prototype_id in prototype_ids
            for _ in range(particles_per_prototype)
        )
        if tuple(particle.prototype_id for particle in particles) != expected_order:
            raise ValueError(
                "R015 formal planning requires prototype-major particle order."
            )
        weights = tuple(
            _finite(particle.weight, name="planning particle weight")
            for particle in particles
        )
        if any(value < 0.0 for value in weights) or not math.isclose(
            sum(weights), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                "R015 formal planning requires globally normalized particle weights."
            )
        prototype_weights = dict(getattr(belief, "prototype_weights", {}))
        if set(prototype_weights) != set(prototype_ids) or any(
            _finite(
                prototype_weights[prototype_id], name="prototype mass"
            )
            <= 0.0
            for prototype_id in prototype_ids
        ):
            raise ValueError("R015 formal planning requires positive q_k values.")
        if not math.isclose(
            sum(float(value) for value in prototype_weights.values()),
            1.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("R015 formal planning requires normalized q_k values.")
        for prototype_index, prototype_id in enumerate(prototype_ids):
            start = prototype_index * particles_per_prototype
            mass = sum(weights[start : start + particles_per_prototype])
            if not math.isclose(
                mass,
                float(prototype_weights[prototype_id]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "R015 formal planning particle weights do not factor through q_k."
                )
        return particles

    @staticmethod
    def _source_particle_index(
        particles: Sequence[HiddenStateParticleV1],
        *,
        numerator: int,
        denominator: int,
    ) -> int:
        position = numerator / denominator
        cumulative = 0.0
        for index, particle in enumerate(particles):
            cumulative += particle.weight
            if position <= cumulative or index == len(particles) - 1:
                return index
        raise RuntimeError("R015 systematic sampling failed to select a particle.")

    @staticmethod
    def _canonical_slot_16(
        *,
        sample_count: int,
        sample_slot: int,
        phase_uint64: int,
    ) -> int:
        count = sample_count
        slot = sample_slot
        phase = phase_uint64
        while count < 16:
            doubled = phase * 2
            parity = doubled // _UINT64_MODULUS
            phase = doubled % _UINT64_MODULUS
            slot = 2 * slot + int(parity)
            count *= 2
        return slot

    @classmethod
    def build(
        cls,
        *,
        belief: Any,
        planning_key: str,
        sample_count: int,
    ) -> "NestedHiddenStateBranchScheduleV1":
        if sample_count not in R015_NESTED_BRANCH_COUNTS:
            raise ValueError("R015 branch count must be one of 2,4,8,16.")
        if not _is_sha256(planning_key):
            raise ValueError("R015 planning schedule requires a SHA-256 root key.")
        phase_key = derive_controller_key(
            planning_key,
            R015_BRANCH_SAMPLING_ID,
            "systematic_phase_uint64",
        )
        root_phase = int(phase_key[:16], 16)
        multiplier = sample_count // 2
        phase = (root_phase * multiplier) % _UINT64_MODULUS
        particles = cls._formal_v2_particle_sequence(belief)
        particle_digest = canonical_sha256(
            {
                "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
                "resampling_algorithm": R015_RESAMPLING_ALGORITHM_ID_V2,
                "prototype_weights": dict(belief.prototype_weights),
                "particles": [
                    {
                        "index": index,
                        "prototype_id": particle.prototype_id,
                        "state_sha256": particle.state_sha256,
                        "weight": particle.weight,
                    }
                    for index, particle in enumerate(particles)
                ],
            }
        )
        lower_phase = (
            None
            if sample_count == 2
            else (root_phase * (sample_count // 4)) % _UINT64_MODULUS
        )
        lower_parity = (
            None
            if lower_phase is None
            else int((lower_phase * 2) // _UINT64_MODULUS)
        )
        samples: list[PlanningBranchSampleV1] = []
        for slot in range(sample_count):
            numerator = phase + slot * _UINT64_MODULUS
            denominator = sample_count * _UINT64_MODULUS
            source_index = cls._source_particle_index(
                particles,
                numerator=numerator,
                denominator=denominator,
            )
            canonical_slot = cls._canonical_slot_16(
                sample_count=sample_count,
                sample_slot=slot,
                phase_uint64=phase,
            )
            paired_lower_slot = None
            if lower_parity is not None and slot % 2 == lower_parity:
                paired_lower_slot = (slot - lower_parity) // 2
            samples.append(
                PlanningBranchSampleV1(
                    sample_count=sample_count,
                    sample_slot=slot,
                    canonical_slot_16=canonical_slot,
                    paired_lower_slot=paired_lower_slot,
                    systematic_phase_uint64=phase,
                    systematic_position_numerator=numerator,
                    systematic_position_denominator=denominator,
                    source_particle_index=source_index,
                    source_particle=particles[source_index],
                    source_belief_filter_algorithm_id=(
                        R015_FILTER_ALGORITHM_ID_V2
                    ),
                    source_belief_resampling_algorithm=(
                        R015_RESAMPLING_ALGORITHM_ID_V2
                    ),
                    common_random_key=derive_controller_key(
                        planning_key,
                        R015_BRANCH_SAMPLING_ID,
                        "canonical_slot_16",
                        canonical_slot,
                    ),
                )
            )
        return cls(
            sample_count=sample_count,
            root_phase_uint64=root_phase,
            samples=tuple(samples),
            particle_sequence_sha256=particle_digest,
            source_belief_filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V2,
            source_belief_resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V2,
        )

    def lower_subset(self) -> tuple[PlanningBranchSampleV1, ...]:
        if self.sample_count == 2:
            raise ValueError("The R015 two-sample grid has no registered lower grid.")
        selected = tuple(
            sorted(
                (sample for sample in self.samples if sample.paired_lower_slot is not None),
                key=lambda sample: int(sample.paired_lower_slot),
            )
        )
        if len(selected) != self.sample_count // 2:
            raise RuntimeError("R015 nested schedule failed to expose its lower subset.")
        return selected


@dataclass(frozen=True)
class ProbeScriptV1:
    probe_id: str
    primitive_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.probe_id or not 1 <= len(self.primitive_actions) <= 2:
            raise ValueError("R015 probe scripts require one or two actions.")
        if any(action not in R015_ATOMIC_ACTIONS for action in self.primitive_actions):
            raise ValueError("R015 probe script contains an unknown primitive action.")


@dataclass(frozen=True)
class FullHorizonRolloutV1:
    raw_return: float
    primitive_steps: int
    trajectory_sha256: str
    task_transition_sha256: str
    future_random_root_key: str
    future_random_derivation_contract_id: str
    future_random_step_count: int
    future_random_sequence_sha256: str
    committed_member_id: str | None = None
    frozen_belief_sha256: str | None = None
    ego_action_sequence_sha256: str | None = None
    partner_action_sequence_sha256: str | None = None
    final_environment_state_sha256: str | None = None
    final_partner_state_sha256: str | None = None
    final_continuation_states_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_return",
            _finite(self.raw_return, name="full-horizon raw return"),
        )
        if self.primitive_steps < 0:
            raise ValueError("Full-horizon rollout step count must be non-negative.")
        if not _is_sha256(self.trajectory_sha256) or not _is_sha256(
            self.task_transition_sha256
        ):
            raise ValueError("Full-horizon rollout requires content hashes.")
        if not _is_sha256(self.future_random_root_key) or not _is_sha256(
            self.future_random_sequence_sha256
        ):
            raise ValueError("Full-horizon rollout random-key summaries require SHA-256.")
        if self.future_random_derivation_contract_id != (
            R015_FUTURE_RANDOM_DERIVATION_ID
        ):
            raise ValueError("Full-horizon rollout changed random-key derivation.")
        if (
            isinstance(self.future_random_step_count, bool)
            or not isinstance(self.future_random_step_count, int)
            or self.future_random_step_count != self.primitive_steps
        ):
            raise ValueError(
                "Full-horizon random-key step count must equal the rollout horizon."
            )
        if self.committed_member_id is not None and not self.committed_member_id:
            raise ValueError("R015 committed member id cannot be empty.")
        if self.frozen_belief_sha256 is not None and not _is_sha256(
            self.frozen_belief_sha256
        ):
            raise ValueError("R015 frozen branch belief requires SHA-256.")
        for value in (
            self.ego_action_sequence_sha256,
            self.partner_action_sequence_sha256,
            self.final_environment_state_sha256,
            self.final_partner_state_sha256,
            self.final_continuation_states_sha256,
        ):
            if value is not None and not _is_sha256(value):
                raise ValueError("R015 rollout state summary requires SHA-256.")


@dataclass(frozen=True)
class PairedProbeRolloutV1:
    masked: FullHorizonRolloutV1
    used: FullHorizonRolloutV1
    masked_branch_head_belief_sha256: str | None = None
    used_branch_head_belief_sha256: str | None = None
    branch_head_particle_transitions: int = 0

    def validate_pairing(self, *, expected_steps: int) -> None:
        if self.masked.primitive_steps != expected_steps or (
            self.used.primitive_steps != expected_steps
        ):
            raise ValueError("R015 planner branch did not cover the remaining episode.")
        if self.masked.task_transition_sha256 != self.used.task_transition_sha256:
            raise ValueError("R015 use and mask branches changed the probe transition.")
        masked_random_contract = (
            self.masked.future_random_root_key,
            self.masked.future_random_derivation_contract_id,
            self.masked.future_random_step_count,
            self.masked.future_random_sequence_sha256,
        )
        used_random_contract = (
            self.used.future_random_root_key,
            self.used.future_random_derivation_contract_id,
            self.used.future_random_step_count,
            self.used.future_random_sequence_sha256,
        )
        if masked_random_contract != used_random_contract:
            raise ValueError("R015 use and mask branches changed future random numbers.")
        for value in (
            self.masked_branch_head_belief_sha256,
            self.used_branch_head_belief_sha256,
        ):
            if value is not None and not _is_sha256(value):
                raise ValueError("R015 probe pair has an invalid branch-head belief hash.")
        if (
            isinstance(self.branch_head_particle_transitions, bool)
            or not isinstance(self.branch_head_particle_transitions, int)
            or self.branch_head_particle_transitions < 0
        ):
            raise ValueError("R015 branch-head particle-transition count is invalid.")


@dataclass(frozen=True)
class PlanningSampleRolloutsV1:
    """One sampled hidden state's base and six paired probe rollouts."""

    sample: PlanningBranchSampleV1
    base: FullHorizonRolloutV1
    probe_pairs: Mapping[str, PairedProbeRolloutV1]


@dataclass(frozen=True)
class PlanningBatchRolloutsV1:
    """All newly required hidden-state samples from one compiled batch schedule."""

    samples: tuple[PlanningSampleRolloutsV1, ...]
    compiled_batch_calls: int
    active_batch_sizes: tuple[int, ...]
    host_sync_inside_environment_loop: bool
    prepared_view: bool = False

    def __post_init__(self) -> None:
        if self.prepared_view:
            if self.compiled_batch_calls != 0 or self.active_batch_sizes:
                raise ValueError(
                    "A prepared R015 planning view may not claim device work."
                )
        elif self.compiled_batch_calls <= 0 or not self.active_batch_sizes:
            raise ValueError("R015 planning batch requires compiled calls and batch sizes.")
        if any(value <= 0 for value in self.active_batch_sizes):
            raise ValueError("R015 planning batch sizes must be positive.")
        if self.host_sync_inside_environment_loop:
            raise ValueError("R015 planning may not synchronize the host at each step.")


@dataclass(frozen=True)
class ContinuationMemberStepV1:
    """One library member's action and recurrent-state advance for one step."""

    action_id: str
    next_recurrent_state: Any

    def __post_init__(self) -> None:
        if self.action_id not in R015_ATOMIC_ACTIONS:
            raise ValueError("R015 continuation member returned an unknown action.")


ContinuationMemberActor = Callable[
    [str, OfficialHistoryV1, Any, Any], ContinuationMemberStepV1
]


@dataclass(frozen=True)
class ContinuationLibraryStatesV1:
    """Current recurrent states for the four prototypes and the ego baseline."""

    by_member_id: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuationActionV1:
    """Registered continuation action plus all five advanced recurrent states."""

    controller_id: str
    selected_member_id: str
    unique_map_prototype_id: str | None
    action_id: str
    next_states: ContinuationLibraryStatesV1
    member_action_ids: Mapping[str, str]


@dataclass(frozen=True)
class MAPPrototypeCommittedCookV1:
    """Route the cook to the unique MAP prototype, or to seed 100 on a tie.

    MAP means maximum a posteriori: the prototype with the largest current
    posterior probability.  Every member advances before routing, so switching
    adopts that member's current recurrent state instead of a zero reset.
    """

    prototype_ids: tuple[str, ...]
    baseline_member_id: str
    member_actor: ContinuationMemberActor
    controller_id: str = R015_CONTINUATION_CONTROLLER_ID

    def __post_init__(self) -> None:
        if self.controller_id != R015_CONTINUATION_CONTROLLER_ID:
            raise ValueError("R015 continuation-controller identity changed.")
        if len(self.prototype_ids) != 4 or len(set(self.prototype_ids)) != 4:
            raise ValueError("R015 continuation requires exactly four prototypes.")
        if not self.baseline_member_id or self.baseline_member_id in self.prototype_ids:
            raise ValueError("R015 continuation baseline must be distinct from prototypes.")
        if not callable(self.member_actor):
            raise TypeError("R015 continuation requires a callable policy library.")

    @property
    def member_ids(self) -> tuple[str, ...]:
        return (*self.prototype_ids, self.baseline_member_id)

    def unique_map_prototype(
        self,
        belief: R015ParticleBeliefView,
    ) -> str | None:
        _require_formal_v2_belief_contract(belief)
        if tuple(belief.prototype_ids) != self.prototype_ids:
            raise ValueError("R015 continuation belief changed its prototype order.")
        weights = dict(belief.prototype_weights)
        maximum = max(weights.values())
        winners = tuple(
            prototype_id
            for prototype_id in self.prototype_ids
            if weights[prototype_id] == maximum
        )
        return winners[0] if len(winners) == 1 else None

    def act(
        self,
        *,
        history: OfficialHistoryV1,
        belief: R015ParticleBeliefView,
        states: ContinuationLibraryStatesV1,
        random_keys_by_member: Mapping[str, Any],
    ) -> ContinuationActionV1:
        """Advance all five members, then route by the current posterior."""

        if not isinstance(history, OfficialHistoryV1):
            raise TypeError("R015 continuation requires projected official history.")
        if set(states.by_member_id) != set(self.member_ids):
            raise ValueError("R015 continuation recurrent-state library is incomplete.")
        if set(random_keys_by_member) != set(self.member_ids):
            raise ValueError("R015 continuation random-key library is incomplete.")
        steps: dict[str, ContinuationMemberStepV1] = {}
        for member_id in self.member_ids:
            step = self.member_actor(
                member_id,
                history,
                states.by_member_id[member_id],
                random_keys_by_member[member_id],
            )
            if not isinstance(step, ContinuationMemberStepV1):
                raise TypeError("R015 continuation member returned the wrong step type.")
            steps[member_id] = step
        map_prototype = self.unique_map_prototype(belief)
        selected_member = (
            self.baseline_member_id if map_prototype is None else map_prototype
        )
        return ContinuationActionV1(
            controller_id=self.controller_id,
            selected_member_id=selected_member,
            unique_map_prototype_id=map_prototype,
            action_id=steps[selected_member].action_id,
            next_states=ContinuationLibraryStatesV1(
                {member_id: steps[member_id].next_recurrent_state for member_id in self.member_ids}
            ),
            member_action_ids={
                member_id: steps[member_id].action_id for member_id in self.member_ids
            },
        )


class FullHorizonBranchExecutor(Protocol):
    continuation_controller_id: str

    def candidate_status(
        self,
        *,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        environment_step: int,
    ) -> tuple[bool, bool]:
        ...

    def rollout_base(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: R015ParticleBeliefView,
        history: OfficialHistoryV1,
        branch_key: str,
        remaining_steps: int,
    ) -> FullHorizonRolloutV1:
        ...

    def rollout_probe_pair(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: R015ParticleBeliefView,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        branch_key: str,
        remaining_steps: int,
    ) -> PairedProbeRolloutV1:
        ...

    def rollout_planning_batch(
        self,
        *,
        samples: Sequence[PlanningBranchSampleV1],
        belief: R015ParticleBeliefView,
        history: OfficialHistoryV1,
        scripts: Sequence[ProbeScriptV1],
        remaining_steps: int,
    ) -> PlanningBatchRolloutsV1:
        ...


@dataclass(frozen=True)
class CandidateValueV1:
    probe_id: str
    script_length: int
    eligible: bool
    static_safety_pass: bool
    j_use: float
    j_mask: float
    v_mask: float
    i_response: float
    c_task: float
    s_seq: float

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "probe_id": self.probe_id,
            "registered": True,
            "script_length": self.script_length,
            "eligible": self.eligible,
            "static_safety_pass": self.static_safety_pass,
            "j_use": self.j_use,
            "j_mask": self.j_mask,
            "v_mask": self.v_mask,
            "i_response": self.i_response,
            "c_task": self.c_task,
            "s_seq": self.s_seq,
        }


@dataclass(frozen=True)
class ConsultationResultV1:
    environment_step: int
    v_base: float
    v_mask: float
    masked_reference_action_id: str
    candidates: tuple[CandidateValueV1, ...]
    selected_for_safety_probe_id: str | None
    planning_branches: tuple[Mapping[str, Any], ...]
    official_history: tuple[Mapping[str, Any], ...]
    planning_random_key: str
    score_random_key: str
    branch_sampling_rule_id: str = R015_BRANCH_SAMPLING_ID
    branch_belief_rule_id: str = R015_BRANCH_BELIEF_ID
    cost_accounting: Mapping[str, Any] = field(default_factory=dict)
    device_execution: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "environment_step": self.environment_step,
            "v_base": self.v_base,
            "v_mask": self.v_mask,
            "masked_reference_action_id": self.masked_reference_action_id,
            "candidates": [item.to_payload() for item in self.candidates],
            "selected_for_safety_probe_id": self.selected_for_safety_probe_id,
            "branch_sampling_rule_id": self.branch_sampling_rule_id,
            "branch_belief_rule_id": self.branch_belief_rule_id,
            "cost_accounting": copy.deepcopy(dict(self.cost_accounting)),
            "planner_evidence": {
                "schema_version": "path_c_r015_consultation_planner_evidence_v2",
                "official_history": copy.deepcopy(list(self.official_history)),
                "planning_random_key": self.planning_random_key,
                "score_random_key": self.score_random_key,
                "planning_branches": copy.deepcopy(list(self.planning_branches)),
                "branch_sampling_rule_id": self.branch_sampling_rule_id,
                "branch_belief_rule_id": self.branch_belief_rule_id,
                "cost_accounting": copy.deepcopy(dict(self.cost_accounting)),
            },
        }


@dataclass(frozen=True)
class R015SequentialPlannerV1:
    """Full-remaining-episode common-random-number planner for R015."""

    probe_scripts: tuple[ProbeScriptV1, ...]
    branches_per_candidate: int
    maximum_environment_steps: int = 400
    filter_algorithm_id: str = R015_FILTER_ALGORITHM_ID_V2
    resampling_algorithm: str = R015_RESAMPLING_ALGORITHM_ID_V2

    def __post_init__(self) -> None:
        if tuple(script.probe_id for script in self.probe_scripts) != (
            R015_ATOMIC_ACTIONS
        ):
            raise ValueError("R015 planner probe order must match the frozen registry.")
        if self.branches_per_candidate not in R015_NESTED_BRANCH_COUNTS:
            raise ValueError("R015 planning branches must be one of 2,4,8,16.")
        if self.maximum_environment_steps != 400:
            raise ValueError("R015 planner requires the full 400-step episode.")
        if self.filter_algorithm_id != R015_FILTER_ALGORITHM_ID_V2 or (
            self.resampling_algorithm != R015_RESAMPLING_ALGORITHM_ID_V2
        ):
            raise ValueError("R015 formal planner requires the v2 filter contract.")

    @staticmethod
    def _branch_rollout_digest(branch: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {
                "canonical_slot_16": branch.get("canonical_slot_16"),
                "source_particle_index": branch.get("source_particle_index"),
                "source_particle_prototype_id": branch.get(
                    "source_particle_prototype_id"
                ),
                "source_particle_state_sha256": branch.get(
                    "source_particle_state_sha256"
                ),
                "source_belief_filter_algorithm_id": branch.get(
                    "source_belief_filter_algorithm_id"
                ),
                "source_belief_resampling_algorithm": branch.get(
                    "source_belief_resampling_algorithm"
                ),
                "common_random_key": branch.get("common_random_key"),
                "v_base_return": branch.get("v_base_return"),
                "base_trajectory_sha256": branch.get("base_trajectory_sha256"),
                "future_random_root_key": branch.get("future_random_root_key"),
                "future_random_derivation_contract_id": branch.get(
                    "future_random_derivation_contract_id"
                ),
                "future_random_step_count": branch.get(
                    "future_random_step_count"
                ),
                "future_random_sequence_sha256": branch.get(
                    "future_random_sequence_sha256"
                ),
                "base_committed_member_id": branch.get(
                    "base_committed_member_id"
                ),
                "base_frozen_belief_sha256": branch.get(
                    "base_frozen_belief_sha256"
                ),
                "base_ego_action_sequence_sha256": branch.get(
                    "base_ego_action_sequence_sha256"
                ),
                "base_partner_action_sequence_sha256": branch.get(
                    "base_partner_action_sequence_sha256"
                ),
                "base_final_environment_state_sha256": branch.get(
                    "base_final_environment_state_sha256"
                ),
                "base_final_partner_state_sha256": branch.get(
                    "base_final_partner_state_sha256"
                ),
                "base_final_continuation_states_sha256": branch.get(
                    "base_final_continuation_states_sha256"
                ),
                "candidates": branch.get("candidates"),
            }
        )

    @classmethod
    def _cached_by_canonical_slot(
        cls,
        cached: Sequence[Mapping[str, Any]],
    ) -> Mapping[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for raw_branch in cached:
            if not isinstance(raw_branch, Mapping):
                raise TypeError("R015 cached planning branch must be a mapping.")
            slot = raw_branch.get("canonical_slot_16")
            if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < 16:
                raise ValueError("R015 cached planning branch has an invalid slot.")
            if slot in result:
                raise ValueError("R015 cached planning branch duplicated a slot.")
            digest = raw_branch.get("branch_rollout_sha256")
            if digest != cls._branch_rollout_digest(raw_branch):
                raise ValueError("R015 cached planning branch digest changed.")
            if raw_branch.get("source_belief_filter_algorithm_id") != (
                R015_FILTER_ALGORITHM_ID_V2
            ) or raw_branch.get("source_belief_resampling_algorithm") != (
                R015_RESAMPLING_ALGORITHM_ID_V2
            ):
                raise ValueError("R015 cached planning branch uses a v1 filter contract.")
            result[slot] = copy.deepcopy(dict(raw_branch))
        return result

    def evaluate(
        self,
        *,
        history: OfficialHistoryV1,
        belief: R015ParticleBeliefView,
        environment_step: int,
        planning_key: str,
        score_key: str,
        executor: FullHorizonBranchExecutor,
        cached_planning_branches: Sequence[Mapping[str, Any]] = (),
    ) -> ConsultationResultV1:
        if environment_step not in R015_CONSULTATION_STEPS:
            raise ValueError("R015 consultation is outside steps 1,6,...,96.")
        if not _is_sha256(planning_key):
            raise ValueError("R015 planning key must be SHA-256.")
        if not _is_sha256(score_key) or score_key == planning_key:
            raise ValueError("R015 score key must be a distinct SHA-256 value.")
        if getattr(executor, "continuation_controller_id", None) != (
            R015_CONTINUATION_CONTROLLER_ID
        ):
            raise ValueError("R015 planner and execution must use the registered C(q,x).")
        # The formal planner consumes only the fully adapted v2 posterior.
        # NestedHiddenStateBranchScheduleV1 performs the complete structural
        # check and also supports the runtime's device-backed v2 belief by
        # interface rather than by a concrete Python class test.
        remaining_steps = self.maximum_environment_steps - environment_step
        schedule = NestedHiddenStateBranchScheduleV1.build(
            belief=belief,
            planning_key=planning_key,
            sample_count=self.branches_per_candidate,
        )
        if schedule.source_belief_filter_algorithm_id != self.filter_algorithm_id or (
            schedule.source_belief_resampling_algorithm != self.resampling_algorithm
        ):
            raise ValueError("R015 planner schedule changed its v2 filter contract.")
        cached_by_slot = self._cached_by_canonical_slot(cached_planning_branches)
        statuses = {
            script.probe_id: executor.candidate_status(
                history=history,
                script=script,
                environment_step=environment_step,
            )
            for script in self.probe_scripts
        }
        base_total = 0.0
        mask_totals = {script.probe_id: 0.0 for script in self.probe_scripts}
        use_totals = {script.probe_id: 0.0 for script in self.probe_scripts}
        branch_evidence: list[Mapping[str, Any]] = []
        new_sample_count = 0
        new_branch_head_particle_transitions = 0
        compiled_batch_calls = 0
        active_batch_sizes: list[int] = []
        missing_samples = tuple(
            sample
            for sample in schedule.samples
            if sample.canonical_slot_16 not in cached_by_slot
        )
        batch_by_slot: dict[int, PlanningSampleRolloutsV1] = {}
        if missing_samples:
            batch = executor.rollout_planning_batch(
                samples=missing_samples,
                belief=belief,
                history=history,
                scripts=self.probe_scripts,
                remaining_steps=remaining_steps,
            )
            if len(batch.samples) != len(missing_samples):
                raise ValueError("R015 planning batch omitted a requested sample.")
            compiled_batch_calls = batch.compiled_batch_calls
            active_batch_sizes.extend(batch.active_batch_sizes)
            for item in batch.samples:
                slot = item.sample.canonical_slot_16
                if slot in batch_by_slot or slot not in {
                    sample.canonical_slot_16 for sample in missing_samples
                }:
                    raise ValueError("R015 planning batch returned an unknown sample slot.")
                batch_by_slot[slot] = item
        for sample in schedule.samples:
                particle = sample.source_particle
                common_key = sample.common_random_key
                cached = cached_by_slot.get(sample.canonical_slot_16)
                if cached is not None:
                    if (
                        cached.get("source_particle_index")
                        != sample.source_particle_index
                        or cached.get("source_particle_prototype_id")
                        != particle.prototype_id
                        or cached.get("source_particle_state_sha256")
                        != particle.state_sha256
                        or cached.get("common_random_key") != common_key
                    ):
                        raise ValueError(
                            "R015 nested planning cache differs from the current particle cloud."
                        )
                    branch_payload = {
                        **copy.deepcopy(dict(cached)),
                        **sample.to_evidence(),
                        "reused_from_verified_lower_grid": True,
                    }
                    branch_payload["branch_rollout_sha256"] = (
                        self._branch_rollout_digest(branch_payload)
                    )
                    base_return = _finite(
                        branch_payload.get("v_base_return"),
                        name="cached R015 base return",
                    )
                    candidate_payloads = branch_payload.get("candidates")
                    if not isinstance(candidate_payloads, Mapping) or set(
                        candidate_payloads
                    ) != {script.probe_id for script in self.probe_scripts}:
                        raise ValueError("R015 cached branch changed the probe registry.")
                    base_random_contract = (
                        branch_payload.get("future_random_root_key"),
                        branch_payload.get("future_random_derivation_contract_id"),
                        branch_payload.get("future_random_step_count"),
                        branch_payload.get("future_random_sequence_sha256"),
                    )
                    if (
                        not _is_sha256(base_random_contract[0])
                        or base_random_contract[0] != common_key
                        or base_random_contract[1]
                        != R015_FUTURE_RANDOM_DERIVATION_ID
                        or base_random_contract[2] != remaining_steps
                        or not _is_sha256(base_random_contract[3])
                    ):
                        raise ValueError(
                            "R015 cached base changed its future-random summary."
                        )
                    for candidate in candidate_payloads.values():
                        if not isinstance(candidate, Mapping):
                            raise ValueError(
                                "R015 cached candidate is not a mapping."
                            )
                else:
                    new_sample_count += 1
                    batch_item = batch_by_slot[sample.canonical_slot_16]
                    if batch_item.sample != sample:
                        raise ValueError("R015 planning batch changed a sample descriptor.")
                    base = batch_item.base
                    if base.primitive_steps != remaining_steps:
                        raise ValueError(
                            "R015 base branch did not cover the remaining episode."
                        )
                    if base.future_random_root_key != common_key:
                        raise ValueError(
                            "R015 base branch changed its registered future-random root."
                        )
                    base_return = base.raw_return
                    candidate_payloads = {}
                    for script in self.probe_scripts:
                        pair = batch_item.probe_pairs.get(script.probe_id)
                        if not isinstance(pair, PairedProbeRolloutV1):
                            raise ValueError("R015 planning batch omitted a probe pair.")
                        pair.validate_pairing(expected_steps=remaining_steps)
                        pair_random_contract = (
                            pair.masked.future_random_root_key,
                            pair.masked.future_random_derivation_contract_id,
                            pair.masked.future_random_step_count,
                            pair.masked.future_random_sequence_sha256,
                        )
                        base_random_contract = (
                            base.future_random_root_key,
                            base.future_random_derivation_contract_id,
                            base.future_random_step_count,
                            base.future_random_sequence_sha256,
                        )
                        if pair_random_contract != base_random_contract:
                            raise ValueError(
                                "R015 base and candidate branches changed future random numbers."
                            )
                        new_branch_head_particle_transitions += (
                            pair.branch_head_particle_transitions
                        )
                        candidate_payloads[script.probe_id] = {
                            "j_mask_return": pair.masked.raw_return,
                            "j_use_return": pair.used.raw_return,
                            "task_transition_sha256": (
                                pair.masked.task_transition_sha256
                            ),
                            "mask_trajectory_sha256": (
                                pair.masked.trajectory_sha256
                            ),
                            "use_trajectory_sha256": pair.used.trajectory_sha256,
                            "mask_branch_head_belief_sha256": (
                                pair.masked_branch_head_belief_sha256
                            ),
                            "use_branch_head_belief_sha256": (
                                pair.used_branch_head_belief_sha256
                            ),
                            "mask_committed_member_id": (
                                pair.masked.committed_member_id
                            ),
                            "use_committed_member_id": (
                                pair.used.committed_member_id
                            ),
                            "branch_belief_rule_id": R015_BRANCH_BELIEF_ID,
                            "mask_ego_action_sequence_sha256": (
                                pair.masked.ego_action_sequence_sha256
                            ),
                            "use_ego_action_sequence_sha256": (
                                pair.used.ego_action_sequence_sha256
                            ),
                            "mask_partner_action_sequence_sha256": (
                                pair.masked.partner_action_sequence_sha256
                            ),
                            "use_partner_action_sequence_sha256": (
                                pair.used.partner_action_sequence_sha256
                            ),
                            "mask_final_environment_state_sha256": (
                                pair.masked.final_environment_state_sha256
                            ),
                            "use_final_environment_state_sha256": (
                                pair.used.final_environment_state_sha256
                            ),
                            "mask_final_partner_state_sha256": (
                                pair.masked.final_partner_state_sha256
                            ),
                            "use_final_partner_state_sha256": (
                                pair.used.final_partner_state_sha256
                            ),
                            "mask_final_continuation_states_sha256": (
                                pair.masked.final_continuation_states_sha256
                            ),
                            "use_final_continuation_states_sha256": (
                                pair.used.final_continuation_states_sha256
                            ),
                        }
                    branch_payload = {
                        **sample.to_evidence(),
                        "branch_belief_rule_id": R015_BRANCH_BELIEF_ID,
                        "reused_from_verified_lower_grid": False,
                        "score_evidence_key": derive_controller_key(
                            score_key,
                            "score_evidence",
                            environment_step,
                            sample.canonical_slot_16,
                        ),
                        "v_base_return": base.raw_return,
                        "base_trajectory_sha256": base.trajectory_sha256,
                        "future_random_root_key": base.future_random_root_key,
                        "future_random_derivation_contract_id": (
                            base.future_random_derivation_contract_id
                        ),
                        "future_random_step_count": (
                            base.future_random_step_count
                        ),
                        "future_random_sequence_sha256": (
                            base.future_random_sequence_sha256
                        ),
                        "base_committed_member_id": base.committed_member_id,
                        "base_frozen_belief_sha256": base.frozen_belief_sha256,
                        "base_ego_action_sequence_sha256": (
                            base.ego_action_sequence_sha256
                        ),
                        "base_partner_action_sequence_sha256": (
                            base.partner_action_sequence_sha256
                        ),
                        "base_final_environment_state_sha256": (
                            base.final_environment_state_sha256
                        ),
                        "base_final_partner_state_sha256": (
                            base.final_partner_state_sha256
                        ),
                        "base_final_continuation_states_sha256": (
                            base.final_continuation_states_sha256
                        ),
                        "candidates": candidate_payloads,
                    }
                    branch_payload["branch_rollout_sha256"] = (
                        self._branch_rollout_digest(branch_payload)
                    )
                branch_weight = sample.estimator_weight
                base_total += branch_weight * base_return
                for script in self.probe_scripts:
                    candidate = candidate_payloads[script.probe_id]
                    mask_totals[script.probe_id] += branch_weight * _finite(
                        candidate.get("j_mask_return"),
                        name="R015 J_mask branch return",
                    )
                    use_totals[script.probe_id] += branch_weight * _finite(
                        candidate.get("j_use_return"),
                        name="R015 J_use branch return",
                    )
                branch_evidence.append(branch_payload)
        eligible_mask_values = [base_total]
        for script in self.probe_scripts:
            eligible, static_safe = statuses[script.probe_id]
            if eligible and static_safe:
                eligible_mask_values.append(mask_totals[script.probe_id])
        v_mask = max(eligible_mask_values)
        masked_reference = "base"
        if not math.isclose(base_total, v_mask, rel_tol=1.0e-12, abs_tol=1.0e-12):
            for script in self.probe_scripts:
                eligible, static_safe = statuses[script.probe_id]
                if eligible and static_safe and math.isclose(
                    mask_totals[script.probe_id],
                    v_mask,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    masked_reference = script.probe_id
                    break
        candidates = []
        for script in self.probe_scripts:
            eligible, static_safe = statuses[script.probe_id]
            j_mask = mask_totals[script.probe_id]
            j_use = use_totals[script.probe_id]
            candidates.append(
                CandidateValueV1(
                    probe_id=script.probe_id,
                    script_length=len(script.primitive_actions),
                    eligible=eligible,
                    static_safety_pass=static_safe,
                    j_use=j_use,
                    j_mask=j_mask,
                    v_mask=v_mask,
                    i_response=j_use - j_mask,
                    c_task=v_mask - j_mask,
                    s_seq=j_use - v_mask,
                )
            )
        selected = next(
            (
                item.probe_id
                for item in candidates
                if item.eligible
                and item.static_safety_pass
                and item.s_seq > 0.0
                and math.isclose(
                    item.s_seq,
                    max(
                        candidate.s_seq
                        for candidate in candidates
                        if candidate.eligible
                        and candidate.static_safety_pass
                        and candidate.s_seq > 0.0
                    ),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ),
            None,
        )
        return ConsultationResultV1(
            environment_step=environment_step,
            v_base=base_total,
            v_mask=v_mask,
            masked_reference_action_id=masked_reference,
            candidates=tuple(candidates),
            selected_for_safety_probe_id=selected,
            planning_branches=tuple(branch_evidence),
            official_history=tuple(history.to_payload()),
            planning_random_key=planning_key,
            score_random_key=score_key,
            cost_accounting={
                "schema_version": "path_c_r015_planning_cost_v2",
                "sample_count": self.branches_per_candidate,
                "remaining_environment_steps": remaining_steps,
                "estimator_real_environment_transitions": (
                    self.branches_per_candidate * (13 * remaining_steps - 6)
                ),
                "estimator_paired_suffix_batched_lane_time_steps": (
                    self.branches_per_candidate * 7 * remaining_steps
                ),
                "new_sample_count": new_sample_count,
                "new_real_environment_transitions": (
                    new_sample_count * (13 * remaining_steps - 6)
                ),
                "new_paired_suffix_batched_lane_time_steps": (
                    new_sample_count * 7 * remaining_steps
                ),
                "new_branch_head_particle_transitions": (
                    new_branch_head_particle_transitions
                ),
                "particle_sequence_sha256": schedule.particle_sequence_sha256,
                "root_phase_uint64": schedule.root_phase_uint64,
                "source_belief_filter_algorithm_id": (
                    schedule.source_belief_filter_algorithm_id
                ),
                "source_belief_resampling_algorithm": (
                    schedule.source_belief_resampling_algorithm
                ),
            },
            device_execution={
                "compiled_batch_calls": compiled_batch_calls,
                "active_batch_sizes": tuple(active_batch_sizes),
                "host_sync_inside_environment_loop": False,
            },
        )


@dataclass(frozen=True)
class SafetyDecisionV1:
    passed: bool
    comparisons: tuple[Mapping[str, Any], ...]

    def validate(
        self,
        *,
        prototype_ids: Sequence[str],
        expected_repetitions: int,
    ) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("R015 safety verdict must be boolean.")
        ids = tuple(str(item) for item in prototype_ids)
        repetitions = _positive_int(
            expected_repetitions,
            name="expected_repetitions",
        )
        if len(self.comparisons) != len(ids):
            raise ValueError("R015 safety must compare all four support prototypes.")
        by_id: dict[str, Mapping[str, Any]] = {}
        for raw_comparison in self.comparisons:
            if not isinstance(raw_comparison, Mapping):
                raise TypeError("R015 safety comparisons must be mappings.")
            prototype_id = str(raw_comparison.get("prototype_id", ""))
            if prototype_id not in ids or prototype_id in by_id:
                raise ValueError("R015 safety comparison changed the support set.")
            if isinstance(raw_comparison.get("repetitions"), bool) or int(
                raw_comparison.get("repetitions", -1)
            ) != repetitions:
                raise ValueError("R015 safety comparison changed its repeat count.")
            wrong_count = raw_comparison.get("wrong_delivery_count")
            if isinstance(wrong_count, bool) or not isinstance(wrong_count, int) or (
                wrong_count < 0 or wrong_count > repetitions
            ):
                raise ValueError("R015 safety wrong-delivery count is invalid.")
            if not isinstance(
                raw_comparison.get("positive_posterior_support"), bool
            ) or not isinstance(
                raw_comparison.get("compatible_hidden_state_reconstructed"), bool
            ):
                raise ValueError("R015 safety comparison lacks support reconstruction.")
            by_id[prototype_id] = raw_comparison
        support_compatible = all(
            comparison["positive_posterior_support"] is True
            and comparison["compatible_hidden_state_reconstructed"] is True
            for comparison in by_id.values()
        )
        no_wrong_delivery = all(
            comparison["wrong_delivery_count"] == 0
            for comparison in by_id.values()
        )
        if self.passed is not (support_compatible and no_wrong_delivery):
            raise ValueError("R015 safety verdict differs from its four comparisons.")

    @property
    def support_compatible(self) -> bool:
        return bool(self.comparisons) and all(
            comparison.get("positive_posterior_support") is True
            and comparison.get("compatible_hidden_state_reconstructed") is True
            for comparison in self.comparisons
        )


SafetyEvaluator = Callable[
    [str, OfficialHistoryV1, R015ParticleBeliefView, str], SafetyDecisionV1
]


@dataclass(frozen=True)
class ControllerDecisionV1:
    consultation: ConsultationResultV1
    a1_action_id: str
    a2_probe_id: str | None
    shared_masked_action_id: str | None
    probe_fired: bool
    stop_consulting: bool
    no_probe_reason: str | None
    safety: SafetyDecisionV1 | None


@dataclass
class R015SequentialControllerV1:
    planner: R015SequentialPlannerV1
    continuation_controller: MAPPrototypeCommittedCookV1 | None = None
    safety_repeats_per_comparison: int = 279
    probe_used: bool = False
    consultation_closed: bool = False

    def act(
        self,
        *,
        history: OfficialHistoryV1,
        belief: R015ParticleBeliefView,
        states: ContinuationLibraryStatesV1,
        random_keys_by_member: Mapping[str, Any],
    ) -> ContinuationActionV1:
        """Execute the registered shared continuation controller C(q,x)."""

        if self.continuation_controller is None:
            raise ValueError("R015 continuation controller is not configured.")
        return self.continuation_controller.act(
            history=history,
            belief=belief,
            states=states,
            random_keys_by_member=random_keys_by_member,
        )

    def consult(
        self,
        *,
        history: OfficialHistoryV1,
        belief: R015ParticleBeliefView,
        environment_step: int,
        planning_key: str,
        score_key: str,
        safety_key: str,
        executor: FullHorizonBranchExecutor,
        safety_evaluator: SafetyEvaluator,
    ) -> ControllerDecisionV1:
        if self.probe_used or self.consultation_closed:
            raise ValueError("R015 controller cannot consult after its one-probe decision.")
        keys = (planning_key, score_key, safety_key)
        if any(not _is_sha256(key) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "R015 planning, score, and safety keys must be distinct SHA-256 values."
            )
        consultation = self.planner.evaluate(
            history=history,
            belief=belief,
            environment_step=environment_step,
            planning_key=planning_key,
            score_key=score_key,
            executor=executor,
        )
        selected = consultation.selected_for_safety_probe_id
        if selected is None:
            if environment_step == R015_CONSULTATION_STEPS[-1]:
                self.consultation_closed = True
            return ControllerDecisionV1(
                consultation=consultation,
                a1_action_id=consultation.masked_reference_action_id,
                a2_probe_id=None,
                shared_masked_action_id=consultation.masked_reference_action_id,
                probe_fired=False,
                stop_consulting=self.consultation_closed,
                no_probe_reason=(
                    "window_expired" if self.consultation_closed else "non_positive_score"
                ),
                safety=None,
            )
        derived_safety_key = derive_controller_key(
            safety_key,
            "safety_selection",
            environment_step,
            selected,
        )
        safety = safety_evaluator(selected, history, belief, derived_safety_key)
        if not isinstance(safety, SafetyDecisionV1):
            raise TypeError("R015 safety evaluator returned the wrong result type.")
        safety.validate(
            prototype_ids=belief.prototype_ids,
            expected_repetitions=self.safety_repeats_per_comparison,
        )
        self.consultation_closed = True
        if safety.passed:
            self.probe_used = True
            return ControllerDecisionV1(
                consultation=consultation,
                a1_action_id=consultation.masked_reference_action_id,
                a2_probe_id=selected,
                shared_masked_action_id=None,
                probe_fired=True,
                stop_consulting=True,
                no_probe_reason=None,
                safety=safety,
            )
        return ControllerDecisionV1(
            consultation=consultation,
            a1_action_id=consultation.masked_reference_action_id,
            a2_probe_id=None,
            shared_masked_action_id=consultation.masked_reference_action_id,
            probe_fired=False,
            stop_consulting=True,
            no_probe_reason=(
                "safety_rejected"
                if safety.support_compatible
                else "support_incompatible"
            ),
            safety=safety,
        )


def default_r015_probe_scripts() -> tuple[ProbeScriptV1, ...]:
    return tuple(ProbeScriptV1(action, (action,)) for action in R015_ATOMIC_ACTIONS)


def continuation_controller_manifest_payload(
    *,
    prototype_ids: Sequence[str],
    baseline_member_id: str,
    checkpoint_bindings: Mapping[str, Mapping[str, str]],
    implementation_path: str,
    implementation_sha256: str,
) -> Mapping[str, Any]:
    """Bind the five policy-library members to the registered continuation."""

    ordered_prototypes = tuple(str(value) for value in prototype_ids)
    expected_members = {*ordered_prototypes, str(baseline_member_id)}
    if len(ordered_prototypes) != 4 or len(set(ordered_prototypes)) != 4:
        raise ValueError("R015 continuation manifest requires four prototypes.")
    if set(checkpoint_bindings) != expected_members:
        raise ValueError("R015 continuation manifest requires five checkpoint bindings.")
    if not implementation_path or not _is_sha256(implementation_sha256):
        raise ValueError("R015 continuation implementation is not bound.")
    members: dict[str, Mapping[str, str]] = {}
    for member_id in (*ordered_prototypes, baseline_member_id):
        binding = checkpoint_bindings[member_id]
        if set(binding) != {"checkpoint_sha256", "model_weights_sha256"} or any(
            not _is_sha256(binding[field])
            for field in ("checkpoint_sha256", "model_weights_sha256")
        ):
            raise ValueError("R015 continuation checkpoint binding is incomplete.")
        members[member_id] = {
            "role": "baseline" if member_id == baseline_member_id else "prototype",
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "model_weights_sha256": binding["model_weights_sha256"],
        }
    return {
        "schema_version": "path_c_r015_continuation_controller_manifest_v1",
        "rule_id": R015_CONTINUATION_CONTROLLER_ID,
        "contract": {
            "action_rule": "official_flax_categorical_actor_v1",
            "routing_rule": "unique_exact_map_else_ego_seed100",
            "tie_fallback": "ego_seed100",
            "parallel_recurrent_member_count": 5,
            "switch_state_rule": "adopt_current_parallel_state_without_reset",
            "planning_branch_sampling": R015_BRANCH_SAMPLING_ID,
            "planning_branch_belief": R015_BRANCH_BELIEF_ID,
            "planning_filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "planning_resampling_algorithm": R015_RESAMPLING_ALGORITHM_ID_V2,
            "planning_routing_frequency": "commit_once_at_branch_head",
            "execution_routing_frequency": "update_online_each_environment_step",
            "shared_member_selection_rule": "unique_exact_map_else_ego_seed100",
            "controller_input": "q_and_projected_official_history_only",
        },
        "baseline_member_id": baseline_member_id,
        "prototype_member_ids": list(ordered_prototypes),
        "members": members,
        "implementation_path": implementation_path,
        "implementation_sha256": implementation_sha256,
    }


def sequential_controller_manifest_payload(
    *,
    implementation_path: str,
    implementation_sha256: str,
    response_projection_sha256: str,
    official_history_filter_sha256: str,
    probe_registry_semantic_sha256: str,
    random_key_derivation_sha256: str,
    continuation_controller_sha256: str,
    replay_backend_implementation_path: str,
    replay_backend_implementation_sha256: str,
) -> Mapping[str, Any]:
    """Build the existing adjudicator's formal controller manifest shape."""

    for value in (
        implementation_sha256,
        response_projection_sha256,
        official_history_filter_sha256,
        probe_registry_semantic_sha256,
        random_key_derivation_sha256,
        continuation_controller_sha256,
        replay_backend_implementation_sha256,
    ):
        if not _is_sha256(value):
            raise ValueError("R015 controller manifest requires SHA-256 bindings.")
    return {
        "schema_version": "path_c_r015_sequential_controller_manifest_v1",
        "controller_kind": "registered_response_sequential_branch_v1",
        "finite_prototype_two_action_surrogate_allowed": False,
        "paired_interfaces": {
            "belief_use": "B_use(q,x,y)",
            "belief_mask": "B_mask(q,x)",
            "continuation_controller": "C(q,x)",
            "common_projected_history": "x",
            "use_only_current_response": "y",
        },
        "continuation_controller_contract": {
            "rule_id": R015_CONTINUATION_CONTROLLER_ID,
            "action_rule": "official_flax_categorical_actor_v1",
            "routing_rule": "unique_exact_map_else_ego_seed100",
            "tie_fallback": "ego_seed100",
            "parallel_recurrent_member_count": 5,
            "switch_state_rule": "adopt_current_parallel_state_without_reset",
            "planning_branch_sampling": R015_BRANCH_SAMPLING_ID,
            "planning_branch_belief": R015_BRANCH_BELIEF_ID,
            "planning_filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "planning_resampling_algorithm": R015_RESAMPLING_ALGORITHM_ID_V2,
            "planning_routing_frequency": "commit_once_at_branch_head",
            "execution_routing_frequency": "update_online_each_environment_step",
            "shared_member_selection_rule": "unique_exact_map_else_ego_seed100",
            "controller_input": "q_and_projected_official_history_only",
        },
        "return_contract": {
            "reward": "undiscounted_raw_team_reward",
            "horizon": "full_remaining_episode",
            "discount_factor": 1.0,
        },
        "paired_branch_contract": {
            "same_hidden_execution_state": True,
            "same_task_transition": True,
            "same_future_random_branch": True,
            "current_response_is_only_interface_difference": True,
        },
        "lockstep_contract": {
            "same_frozen_planner": True,
            "same_passive_masked_belief_updates": True,
            "same_episode_random_numbers": True,
            "probe_rule_firing_is_only_branch_point": True,
        },
        "hidden_state_contract": {
            "complete_hidden_state_reconstruction": True,
            "positive_posterior_support_required": True,
            "branch_sampling_rule_id": R015_BRANCH_SAMPLING_ID,
            "nested_branch_counts": list(R015_NESTED_BRANCH_COUNTS),
            "equal_weight_sample_slots": True,
        },
        "score_rule": "S_seq=J_use-V_mask",
        "tie_break_rule": "base_then_probe_registry_order",
        "consultation_steps": list(R015_CONSULTATION_STEPS),
        "replay_backend": {
            "implementation_path": replay_backend_implementation_path,
            "implementation_sha256": replay_backend_implementation_sha256,
            "factory_name": "build_r015_replay_backend",
        },
        "ego_action_selection_uses_full_state": False,
        "implementation_path": implementation_path,
        "implementation_sha256": implementation_sha256,
        "response_projection_sha256": response_projection_sha256,
        "official_history_filter_sha256": official_history_filter_sha256,
        "probe_registry_semantic_sha256": probe_registry_semantic_sha256,
        "random_key_derivation_sha256": random_key_derivation_sha256,
        "continuation_controller_sha256": continuation_controller_sha256,
    }
