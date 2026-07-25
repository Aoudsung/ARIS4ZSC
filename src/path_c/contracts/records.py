"""Serializable records written by Path C training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One controller decision with enough detail to recompute its trigger."""

    episode_id: int
    environment_index: int
    episode_step: int
    base_action: int
    chosen_action: int
    probed: bool
    actor_owned_action: bool
    candidate: int | None
    score_components: Mapping[str, Any]
    budget_remaining: int
    controller: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "environment_index", "episode_step", "budget_remaining"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.controller, str) or not self.controller:
            raise ValueError("controller must be a non-empty string.")
        for name in ("base_action", "chosen_action"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.candidate is not None and (
            isinstance(self.candidate, bool)
            or not isinstance(self.candidate, Integral)
            or int(self.candidate) < 0
        ):
            raise ValueError("candidate must be null or a non-negative integer.")
        if not isinstance(self.probed, bool) or not isinstance(
            self.actor_owned_action, bool
        ):
            raise ValueError("Decision ownership flags must be boolean.")
        if self.actor_owned_action == self.probed:
            raise ValueError("Exactly one of actor ownership and controller probing must hold.")
        if self.probed:
            if self.candidate is None or self.chosen_action != self.candidate:
                raise ValueError("A probe must execute its recorded candidate action.")
            if self.chosen_action == self.base_action:
                raise ValueError("A probe candidate must differ from the sampled actor action.")
        elif self.chosen_action != self.base_action:
            raise ValueError("A non-probe decision must execute the sampled actor action.")
        if not isinstance(self.score_components, Mapping):
            raise TypeError("score_components must be a mapping.")

    def to_mapping(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One completed episode using the unmodified environment return."""

    episode_id: int
    partner_id: str
    ego_seat: int
    raw_return: float
    environment_steps: int
    probe_count: int
    safe_candidate_opportunity_count: int
    probe_trigger_rate: float
    maximum_probe_budget: int
    probe_budget_usage_rate: float
    correct_delivery_count: int = 0
    wrong_delivery_count: int = 0
    indicator_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "environment_steps",
            "probe_count",
            "safe_candidate_opportunity_count",
            "maximum_probe_budget",
            "correct_delivery_count",
            "wrong_delivery_count",
        ):
            value = getattr(self, name)
            minimum = 1 if name in {"environment_steps", "maximum_probe_budget"} else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if not isinstance(self.partner_id, str) or not self.partner_id:
            raise ValueError("partner_id must be a non-empty string.")
        if isinstance(self.ego_seat, bool) or not isinstance(self.ego_seat, Integral) or int(
            self.ego_seat
        ) not in {0, 1}:
            raise ValueError("ego_seat must be zero or one.")
        for name in ("raw_return", "indicator_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(
                float(value)
            ):
                raise ValueError(f"{name} must be finite and numeric.")
        if self.probe_count > self.safe_candidate_opportunity_count:
            raise ValueError("probe_count cannot exceed safe candidate opportunities.")
        if self.probe_count > self.maximum_probe_budget:
            raise ValueError("probe_count cannot exceed the maximum probe budget.")
        expected_trigger_rate = (
            self.probe_count / self.safe_candidate_opportunity_count
            if self.safe_candidate_opportunity_count
            else 0.0
        )
        expected_budget_rate = self.probe_count / self.maximum_probe_budget
        for name, expected in (
            ("probe_trigger_rate", expected_trigger_rate),
            ("probe_budget_usage_rate", expected_budget_rate),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"{name} must equal its registered count ratio.")

    def to_mapping(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One observation-only training metric row."""

    environment_steps: int
    completed_episodes: int
    losses: Mapping[str, float]
    raw_step_return: float
    completed_episode_return: float | None
    probe_count: int
    probe_rate: float
    gradient_norms: Mapping[str, float]
    parameters_finite: bool

    def to_mapping(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Content bindings required for a shared prefit or condition checkpoint."""

    schema_version: str
    stage: str
    condition_id: str | None
    controller: str | None
    shared_across_conditions: bool
    run_kind: str
    scientific_readout_allowed: bool
    environment_steps: int
    completed_episodes: int
    response_vocabulary_sha256: str
    backbone_origin_sha256: str
    config_sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_model_checkpoint_metadata_v1":
            raise ValueError("Checkpoint metadata schema version changed.")
        if self.stage not in {"prefit", "adaptation"}:
            raise ValueError("Checkpoint stage must be prefit or adaptation.")
        if self.run_kind not in {"development", "formal"}:
            raise ValueError("Checkpoint run_kind must be development or formal.")
        if self.run_kind == "development" and self.scientific_readout_allowed:
            raise ValueError("Development checkpoints cannot allow scientific readout.")
        if self.stage == "prefit":
            if self.condition_id is not None or self.controller is not None:
                raise ValueError("Shared prefit checkpoints cannot bind one condition.")
            if not self.shared_across_conditions:
                raise ValueError("Prefit checkpoints must be shared across conditions.")
        else:
            if not self.condition_id or not self.controller:
                raise ValueError("Adaptation checkpoints require condition and controller.")
            if self.shared_across_conditions:
                raise ValueError("Adaptation checkpoints are condition-specific.")
        for name in ("environment_steps", "completed_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in (
            "response_vocabulary_sha256",
            "backbone_origin_sha256",
            "config_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or not set(value).issubset(frozenset("0123456789abcdef"))
            ):
                raise ValueError(f"{name} must be a SHA-256 string.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("Checkpoint extra metadata must be a mapping.")

    def to_mapping(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckpointMetadata":
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint metadata must be a mapping.")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - allowed)
        missing = sorted((allowed - {"extra"}) - set(payload))
        if unknown or missing:
            raise ValueError(
                "Checkpoint metadata fields are invalid; "
                f"unknown={unknown}, missing={missing}."
            )
        return cls(**{name: payload[name] for name in allowed if name in payload})
