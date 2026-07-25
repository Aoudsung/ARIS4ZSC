"""Strict historical and formal Path C configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from .outer_units import (
    FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
    HISTORY_CHECKPOINT_COUNT,
    OuterTrainingUnit,
    OuterUnitsManifest,
)


SCHEMA_VERSION = "path_c_model_v1"
FORMAL_SCHEMA_VERSION = "path_c_model_v2"
FAMILY_POOL_SCHEMA_VERSION = "path_c_model_v3"
CONDITION_CONTROLLERS = {
    "decision_focused": "registered_response_sequential_branch_v1",
    "random_safe_probe": "registered_random_safe_probe_v1",
    "no_probe": "off",
    "generic_response_information": "generic_response_information",
}
PARTNER_FAMILIES = {
    "official_rnn_sp_ippo_v1",
    "official_rnn_op_other_play_v1",
}
FAMILY_PROTOTYPES = (
    "adapted_ego_family",
    "own_backbone_history_family",
    "foreign_self_play_history_family",
    "other_play_history_family",
)
FAMILY_PROTOTYPE_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
REGISTERED_OFFICIAL_CHECKPOINTS = {
    100: {
        "family_id": "official_rnn_sp_ippo_v1",
        "flax_weights_sha256": "5a5346221a74903bd1ef3f36e228bace486f0f8b80161d837764736749045f7e",
        "training_run_id": "05834fb2f550300ab4e05c5e593b3d89439b01a8a0aceffbd74326ec998cc5cc",
    },
    101: {
        "family_id": "official_rnn_sp_ippo_v1",
        "flax_weights_sha256": "8c4b37146a694ff97b56e6772460c4e7623621466993d11559853ecca74fa838",
        "training_run_id": "1a45685f76499c35bee9d967bbda1982d3c039c88e778ac2a83a07395672a28a",
    },
    102: {
        "family_id": "official_rnn_sp_ippo_v1",
        "flax_weights_sha256": "40206d91269d045d945592b4d8f61c6bf607779743ef4102e23bf328b1ab79e5",
        "training_run_id": "bc3601de469fe7897a4f68b3cd1a25448d4a3a2cde906fc7c677896f460abeb9",
    },
    201: {
        "family_id": "official_rnn_op_other_play_v1",
        "flax_weights_sha256": "87a77b5d693b8277af69480d3852940273b2c245d6e25cb276942ba40d2458b4",
        "training_run_id": "6181d5952042eca76ebc2d39b356b3f593554b0535e4bbbc76c9ebda46f0cb06",
    },
    202: {
        "family_id": "official_rnn_op_other_play_v1",
        "flax_weights_sha256": "26ffdb244cf016f41231e00c8fe63b19e52d493ac1db72239410376bb6f4c215",
        "training_run_id": "28c4827863464a23597f78e8d8628635b9e94282278e6173b9bd0ce936b51139",
    },
}
FORMAL_PREFIT_STEPS = 1_000_000
FORMAL_CALIBRATION_EPISODES = 500
FORMAL_ADAPTATION_STEPS = 10_000_000
FORMAL_NUM_ENVS = 250
FORMAL_EVALUATION_EPISODES = 500
FORMAL_DEPLOYMENT_CALIBRATION_EPISODES = 500
_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _fields(payload: Mapping[str, Any], *, allowed: set[str], required: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(f"{name} contains unknown key(s): {', '.join(unknown)}.")
    if missing:
        raise ValueError(f"{name} is missing key(s): {', '.join(missing)}.")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    number = float(value)
    if not (-float("inf") < number < float("inf")) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return number


def _probability(value: Any, name: str, *, open_interval: bool = False) -> float:
    number = _finite(value, name)
    valid = 0.0 < number < 1.0 if open_interval else 0.0 <= number <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}.")
    return number


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX):
        raise ValueError(f"{name} must be a lowercase SHA-256 string.")
    return value


def _path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path.")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _checkpoint_seed(path: Path) -> int:
    matches = [
        int(component[5:])
        for component in path.parts
        if component.startswith("seed_") and component[5:].isdigit()
    ]
    if len(matches) != 1:
        raise ValueError("An official checkpoint path must contain one seed_<number> directory.")
    return matches[0]


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    layout: str
    agent_view_size: int
    indicate_successful_delivery: bool
    episode_steps: int
    num_envs: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EnvironmentConfig":
        payload = _mapping(payload, "environment")
        allowed = {"layout", "agent_view_size", "indicate_successful_delivery", "episode_steps", "num_envs"}
        _fields(payload, allowed=allowed, required=allowed, name="environment")
        if payload["layout"] not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("The OvercookedV2 Test Time layout is not registered.")
        view_size = _positive_int(payload["agent_view_size"], "environment.agent_view_size")
        episode_steps = _positive_int(payload["episode_steps"], "environment.episode_steps")
        if view_size != 2 or episode_steps != 400:
            raise ValueError("The public protocol uses a local radius of two and 400 steps.")
        if payload["indicate_successful_delivery"] is not True:
            raise ValueError("Successful-delivery indication must be enabled.")
        return cls(
            layout=str(payload["layout"]),
            agent_view_size=2,
            indicate_successful_delivery=True,
            episode_steps=400,
            num_envs=_positive_int(payload["num_envs"], "environment.num_envs"),
        )


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_path: Path
    flax_weights_sha256: str
    training_run_id: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path, name: str) -> "CheckpointRef":
        payload = _mapping(payload, name)
        allowed = {"checkpoint_path", "flax_weights_sha256", "training_run_id"}
        _fields(payload, allowed=allowed, required=allowed, name=name)
        run_id = str(payload["training_run_id"])
        if not run_id:
            raise ValueError(f"{name}.training_run_id must be non-empty.")
        return cls(
            checkpoint_path=_path(payload["checkpoint_path"], base_dir=base_dir, name=f"{name}.checkpoint_path"),
            flax_weights_sha256=_sha256(payload["flax_weights_sha256"], f"{name}.flax_weights_sha256"),
            training_run_id=run_id,
        )


@dataclass(frozen=True, slots=True)
class PartnerMember(CheckpointRef):
    family_id: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path, index: int) -> "PartnerMember":
        name = f"partner_pool.members[{index}]"
        payload = _mapping(payload, name)
        allowed = {"checkpoint_path", "flax_weights_sha256", "training_run_id", "family_id"}
        _fields(payload, allowed=allowed, required=allowed, name=name)
        family = str(payload["family_id"])
        if family not in PARTNER_FAMILIES:
            raise ValueError(f"{name}.family_id is not registered.")
        base = CheckpointRef.from_mapping(
            {key: payload[key] for key in ("checkpoint_path", "flax_weights_sha256", "training_run_id")},
            base_dir=base_dir,
            name=name,
        )
        return cls(**asdict(base), family_id=family)


@dataclass(frozen=True, slots=True)
class FamilyPartnerMember(CheckpointRef):
    """One fixed checkpoint member routed to a family-level prototype."""

    member_id: str
    prototype_index: int
    family_id: str
    source_policy_family: str
    checkpoint_index: int
    update_step: int
    effective_environment_steps: int
    launch_config_path: Path
    launch_config_sha256: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: Path, index: int
    ) -> "FamilyPartnerMember":
        name = f"partner_pool.members[{index}]"
        payload = _mapping(payload, name)
        fields = {
            "checkpoint_path",
            "flax_weights_sha256",
            "training_run_id",
            "member_id",
            "prototype_index",
            "family_id",
            "source_policy_family",
            "checkpoint_index",
            "update_step",
            "effective_environment_steps",
            "launch_config_path",
            "launch_config_sha256",
        }
        _fields(payload, allowed=fields, required=fields, name=name)
        prototype_index = _nonnegative_int(
            payload["prototype_index"], f"{name}.prototype_index"
        )
        if prototype_index not in {1, 2, 3}:
            raise ValueError(f"{name}.prototype_index must identify a fixed family.")
        family_id = str(payload["family_id"])
        if family_id != FAMILY_PROTOTYPES[prototype_index]:
            raise ValueError(f"{name}.family_id does not match its prototype index.")
        source_policy_family = str(payload["source_policy_family"])
        expected_source = (
            "official_rnn_op_other_play_v1"
            if prototype_index == 3
            else "official_rnn_sp_ippo_v1"
        )
        if source_policy_family != expected_source:
            raise ValueError(f"{name}.source_policy_family is inconsistent.")
        member_id = str(payload["member_id"])
        if not member_id:
            raise ValueError(f"{name}.member_id must be non-empty.")
        base = CheckpointRef.from_mapping(
            {
                key: payload[key]
                for key in (
                    "checkpoint_path",
                    "flax_weights_sha256",
                    "training_run_id",
                )
            },
            base_dir=base_dir,
            name=name,
        )
        return cls(
            **asdict(base),
            member_id=member_id,
            prototype_index=prototype_index,
            family_id=family_id,
            source_policy_family=source_policy_family,
            checkpoint_index=_nonnegative_int(
                payload["checkpoint_index"], f"{name}.checkpoint_index"
            ),
            update_step=_nonnegative_int(
                payload["update_step"], f"{name}.update_step"
            ),
            effective_environment_steps=_nonnegative_int(
                payload["effective_environment_steps"],
                f"{name}.effective_environment_steps",
            ),
            launch_config_path=_path(
                payload["launch_config_path"],
                base_dir=base_dir,
                name=f"{name}.launch_config_path",
            ),
            launch_config_sha256=_sha256(
                payload["launch_config_sha256"], f"{name}.launch_config_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class FamilyPoolConfig:
    """Frozen contract for balanced family-first partner sampling."""

    schema_version: str
    family_ids: tuple[str, ...]
    family_weights: tuple[float, ...]
    schedule: str
    member_sampling: str
    current_policy_snapshot: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FamilyPoolConfig":
        payload = _mapping(payload, "partner_pool")
        fields = {
            "schema_version",
            "family_ids",
            "family_weights",
            "schedule",
            "member_sampling",
            "current_policy_snapshot",
            "members",
        }
        _fields(payload, allowed=fields, required=fields, name="partner_pool")
        family_ids = tuple(
            str(value)
            for value in _sequence(payload["family_ids"], "partner_pool.family_ids")
        )
        weights = tuple(
            _probability(value, "partner_pool.family_weights")
            for value in _sequence(
                payload["family_weights"], "partner_pool.family_weights"
            )
        )
        if family_ids != FAMILY_PROTOTYPES or weights != FAMILY_PROTOTYPE_WEIGHTS:
            raise ValueError("path_c_model_v3 requires the four registered equal-weight families.")
        schedule = str(payload["schedule"])
        member_sampling = str(payload["member_sampling"])
        snapshot = str(payload["current_policy_snapshot"])
        if str(payload["schema_version"]) != "path_c_family_pool_v1":
            raise ValueError("partner_pool.schema_version is not registered.")
        if schedule != "balanced_condition_independent_episode_v1":
            raise ValueError("The family schedule must be balanced and condition-independent.")
        if member_sampling != "uniform_within_family_v1":
            raise ValueError("Family members must be sampled uniformly within each family.")
        if snapshot != "freeze_complete_policy_at_rollout_start_v1":
            raise ValueError("The current-policy partner must be frozen at rollout start.")
        return cls(
            schema_version=str(payload["schema_version"]),
            family_ids=family_ids,
            family_weights=weights,
            schedule=schedule,
            member_sampling=member_sampling,
            current_policy_snapshot=snapshot,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    encoder_dim: int | None
    gru_hidden_dim: int | None
    response_hidden_dim: int
    value_hidden_dim: int
    transition_hidden_dim: int
    next_feature_summary_dim: int | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        payload = _mapping(payload, "model")
        allowed = {
            "encoder_dim", "gru_hidden_dim", "response_hidden_dim", "value_hidden_dim",
            "transition_hidden_dim", "next_feature_summary_dim",
        }
        required = {"response_hidden_dim", "value_hidden_dim", "transition_hidden_dim"}
        _fields(payload, allowed=allowed, required=required, name="model")
        optional = {}
        for name in ("encoder_dim", "gru_hidden_dim", "next_feature_summary_dim"):
            optional[name] = None if payload.get(name) is None else _positive_int(payload[name], f"model.{name}")
        return cls(
            **optional,
            response_hidden_dim=_positive_int(payload["response_hidden_dim"], "model.response_hidden_dim"),
            value_hidden_dim=_positive_int(payload["value_hidden_dim"], "model.value_hidden_dim"),
            transition_hidden_dim=_positive_int(payload["transition_hidden_dim"], "model.transition_hidden_dim"),
        )


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    candidate_window: int
    budget_per_episode: int
    safety_rule: str
    threshold_source: str
    decision_null_quantile: float
    information_quantile: float
    manual_threshold: float | None
    belief_probability_floor: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, run_kind: str) -> "ProbeConfig":
        payload = _mapping(payload, "probe")
        allowed = {
            "candidate_window", "budget_per_episode", "safety", "threshold",
            "belief_probability_floor",
        }
        _fields(payload, allowed=allowed, required=allowed, name="probe")
        safety = _mapping(payload["safety"], "probe.safety")
        _fields(safety, allowed={"rule"}, required={"rule"}, name="probe.safety")
        rule = str(safety["rule"])
        if rule != "block_interact_facing_visible_goal_v1":
            raise ValueError("probe.safety.rule must use the registered visible-goal rule.")
        threshold = _mapping(payload["threshold"], "probe.threshold")
        _fields(
            threshold,
            allowed={
                "source",
                "decision_null_quantile",
                "information_quantile",
                "manual_value",
            },
            required={"source", "decision_null_quantile", "information_quantile"},
            name="probe.threshold",
        )
        source = str(threshold["source"])
        if source not in {"calibration", "manual"}:
            raise ValueError("probe.threshold.source must be calibration or manual.")
        if run_kind == "formal" and source != "calibration":
            raise ValueError("Formal runs require a calibration threshold.")
        manual = threshold.get("manual_value")
        if source == "manual" and manual is None:
            raise ValueError("Manual thresholds require manual_value.")
        if source == "calibration" and manual is not None:
            raise ValueError("Calibration thresholds cannot also set manual_value.")
        return cls(
            candidate_window=_positive_int(payload["candidate_window"], "probe.candidate_window"),
            budget_per_episode=_positive_int(payload["budget_per_episode"], "probe.budget_per_episode"),
            safety_rule=rule,
            threshold_source=source,
            decision_null_quantile=_probability(
                threshold["decision_null_quantile"],
                "probe.threshold.decision_null_quantile",
                open_interval=True,
            ),
            information_quantile=_probability(
                threshold["information_quantile"],
                "probe.threshold.information_quantile",
                open_interval=True,
            ),
            manual_threshold=None if manual is None else _finite(manual, "probe.threshold.manual_value"),
            belief_probability_floor=_probability(
                payload["belief_probability_floor"], "probe.belief_probability_floor", open_interval=True
            ),
        )


@dataclass(frozen=True, slots=True)
class PrefitConfig:
    env_steps: int
    learning_rate: float
    batch_size: int
    loss_weights: Mapping[str, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrefitConfig":
        payload = _mapping(payload, "stages.prefit")
        allowed = {"env_steps", "learning_rate", "batch_size", "loss_weights"}
        _fields(payload, allowed=allowed, required=allowed, name="stages.prefit")
        weights = _mapping(payload["loss_weights"], "stages.prefit.loss_weights")
        expected = {"response", "prototype_value", "transition_response", "reward", "next_feature"}
        _fields(weights, allowed=expected, required=expected, name="stages.prefit.loss_weights")
        return cls(
            env_steps=_positive_int(payload["env_steps"], "stages.prefit.env_steps"),
            learning_rate=_finite(payload["learning_rate"], "stages.prefit.learning_rate", positive=True),
            batch_size=_positive_int(payload["batch_size"], "stages.prefit.batch_size"),
            loss_weights={name: _finite(weights[name], f"stages.prefit.loss_weights.{name}", positive=True) for name in expected},
        )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    episodes: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationConfig":
        payload = _mapping(payload, "stages.calibration")
        _fields(payload, allowed={"episodes"}, required={"episodes"}, name="stages.calibration")
        return cls(episodes=_positive_int(payload["episodes"], "stages.calibration.episodes"))


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    env_steps: int
    critic_learning_rate: float
    actor_learning_rate: float
    auxiliary_learning_rate: float
    ppo_clip: float
    gae_lambda: float
    gamma: float
    entropy_coefficient: float
    value_coefficient: float
    gradient_clip_norm: float
    update_epochs: int
    unroll_length: int
    num_minibatches: int
    learning_rate_warmup_fraction: float
    metrics_interval_env_steps: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AdaptationConfig":
        payload = _mapping(payload, "stages.adaptation")
        allowed = {
            "env_steps", "critic_learning_rate", "actor_learning_rate", "auxiliary_learning_rate",
            "ppo_clip", "gae_lambda", "gamma", "entropy_coefficient", "value_coefficient",
            "gradient_clip_norm", "update_epochs", "unroll_length", "num_minibatches",
            "learning_rate_warmup_fraction", "metrics_interval_env_steps",
        }
        _fields(payload, allowed=allowed, required=allowed, name="stages.adaptation")
        return cls(
            env_steps=_positive_int(payload["env_steps"], "stages.adaptation.env_steps"),
            critic_learning_rate=_finite(payload["critic_learning_rate"], "critic_learning_rate", positive=True),
            actor_learning_rate=_finite(payload["actor_learning_rate"], "actor_learning_rate", positive=True),
            auxiliary_learning_rate=_finite(payload["auxiliary_learning_rate"], "auxiliary_learning_rate", positive=True),
            ppo_clip=_probability(payload["ppo_clip"], "ppo_clip", open_interval=True),
            gae_lambda=_probability(payload["gae_lambda"], "gae_lambda", open_interval=True),
            gamma=_probability(payload["gamma"], "gamma", open_interval=True),
            entropy_coefficient=_finite(payload["entropy_coefficient"], "entropy_coefficient", positive=True),
            value_coefficient=_finite(payload["value_coefficient"], "value_coefficient", positive=True),
            gradient_clip_norm=_finite(payload["gradient_clip_norm"], "gradient_clip_norm", positive=True),
            update_epochs=_positive_int(payload["update_epochs"], "update_epochs"),
            unroll_length=_positive_int(payload["unroll_length"], "unroll_length"),
            num_minibatches=_positive_int(payload["num_minibatches"], "num_minibatches"),
            learning_rate_warmup_fraction=_probability(
                payload["learning_rate_warmup_fraction"], "learning_rate_warmup_fraction", open_interval=True
            ),
            metrics_interval_env_steps=_positive_int(payload["metrics_interval_env_steps"], "metrics_interval_env_steps"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schedule: str
    episodes_per_pairing: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvaluationConfig":
        payload = _mapping(payload, "stages.evaluation")
        allowed = {"schedule", "episodes_per_pairing"}
        _fields(payload, allowed=allowed, required=allowed, name="stages.evaluation")
        schedule = str(payload["schedule"])
        if schedule not in {
            "development_pool_bidirectional",
            "formal_manifest",
            "external_standard_matrix",
        }:
            raise ValueError("stages.evaluation.schedule is not registered.")
        return cls(schedule=schedule, episodes_per_pairing=_positive_int(payload["episodes_per_pairing"], "episodes_per_pairing"))


@dataclass(frozen=True, slots=True)
class SeedsConfig:
    model_seed: int
    environment_seed: int
    evaluation_seed: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SeedsConfig":
        payload = _mapping(payload, "seeds")
        allowed = {"model_seed", "environment_seed", "evaluation_seed"}
        _fields(payload, allowed=allowed, required=allowed, name="seeds")
        values = {name: _nonnegative_int(payload[name], f"seeds.{name}") for name in allowed}
        if len(set(values.values())) != 3:
            raise ValueError("The model, environment, and evaluation seeds must differ.")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FormalOuterUnitBinding:
    """Resolved provenance that binds one formal run to one outer unit."""

    outer_unit_id: int
    outer_units_manifest_path: Path
    outer_units_manifest_sha256: str
    backbone_launch_config: Path
    partner_launch_configs: tuple[Path, ...]

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: Path
    ) -> "FormalOuterUnitBinding":
        payload = _mapping(payload, "formal_outer_unit")
        fields = {
            "outer_unit_id",
            "outer_units_manifest_path",
            "outer_units_manifest_sha256",
            "backbone_launch_config",
            "partner_launch_configs",
        }
        _fields(payload, allowed=fields, required=fields, name="formal_outer_unit")
        unit_id = _nonnegative_int(
            payload["outer_unit_id"], "formal_outer_unit.outer_unit_id"
        )
        if unit_id >= 10:
            raise ValueError("formal_outer_unit.outer_unit_id must be between 0 and 9.")
        launches = tuple(
            _path(
                value,
                base_dir=base_dir,
                name=f"formal_outer_unit.partner_launch_configs[{index}]",
            )
            for index, value in enumerate(
                _sequence(
                    payload["partner_launch_configs"],
                    "formal_outer_unit.partner_launch_configs",
                )
            )
        )
        if len(launches) != 4 or len(set(launches)) != 4:
            raise ValueError("A formal outer unit requires four distinct partner launch configs.")
        return cls(
            outer_unit_id=unit_id,
            outer_units_manifest_path=_path(
                payload["outer_units_manifest_path"],
                base_dir=base_dir,
                name="formal_outer_unit.outer_units_manifest_path",
            ),
            outer_units_manifest_sha256=_sha256(
                payload["outer_units_manifest_sha256"],
                "formal_outer_unit.outer_units_manifest_sha256",
            ),
            backbone_launch_config=_path(
                payload["backbone_launch_config"],
                base_dir=base_dir,
                name="formal_outer_unit.backbone_launch_config",
            ),
            partner_launch_configs=launches,
        )


@dataclass(frozen=True, slots=True)
class PathCModelConfig:
    schema_version: str
    run_kind: str
    scientific_readout_allowed: bool
    condition_id: str
    controller: str
    environment: EnvironmentConfig
    backbone_init: CheckpointRef
    partner_pool: tuple[PartnerMember | FamilyPartnerMember, ...]
    family_pool: FamilyPoolConfig | None
    model: ModelConfig
    probe: ProbeConfig
    prefit: PrefitConfig
    calibration: CalibrationConfig
    adaptation: AdaptationConfig
    deployment_calibration: CalibrationConfig | None
    evaluation: EvaluationConfig
    seeds: SeedsConfig
    output_root: Path
    formal_outer_unit: FormalOuterUnitBinding | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: str | Path = ".") -> "PathCModelConfig":
        payload = _mapping(payload, "Path C model config")
        schema_version = str(payload.get("schema_version", ""))
        if schema_version not in {
            SCHEMA_VERSION,
            FORMAL_SCHEMA_VERSION,
            FAMILY_POOL_SCHEMA_VERSION,
        }:
            raise ValueError(
                "schema_version must identify a registered Path C model contract."
            )
        allowed = {
            "schema_version", "run_kind", "scientific_readout_allowed", "condition_id", "controller",
            "environment", "backbone_init", "partner_pool", "model", "probe", "stages", "seeds", "output_root",
            "formal_outer_unit",
        }
        required = allowed - {"formal_outer_unit"}
        if schema_version in {FORMAL_SCHEMA_VERSION, FAMILY_POOL_SCHEMA_VERSION}:
            required = allowed
        _fields(payload, allowed=allowed, required=required, name=f"{schema_version} config")
        run_kind = str(payload["run_kind"])
        if run_kind not in {"development", "formal"}:
            raise ValueError("run_kind must be development or formal.")
        scientific = payload["scientific_readout_allowed"]
        if not isinstance(scientific, bool):
            raise ValueError("scientific_readout_allowed must be a boolean.")
        if run_kind == "development" and scientific:
            raise ValueError("Development runs cannot allow scientific readout.")
        condition = str(payload["condition_id"])
        if condition not in CONDITION_CONTROLLERS:
            raise ValueError("condition_id is not registered.")
        controller = str(payload["controller"])
        if controller != CONDITION_CONTROLLERS[condition]:
            raise ValueError("controller does not match condition_id.")
        root = Path(base_dir).resolve()
        pool_payload = _mapping(payload["partner_pool"], "partner_pool")
        family_pool = (
            FamilyPoolConfig.from_mapping(pool_payload)
            if schema_version == FAMILY_POOL_SCHEMA_VERSION
            else None
        )
        if family_pool is None:
            _fields(
                pool_payload,
                allowed={"members"},
                required={"members"},
                name="partner_pool",
            )
            members = tuple(
                PartnerMember.from_mapping(item, base_dir=root, index=index)
                for index, item in enumerate(
                    _sequence(pool_payload["members"], "partner_pool.members")
                )
            )
            if len(members) != 4:
                raise ValueError("The historical complete model uses four partner checkpoints.")
            if len({item.training_run_id for item in members}) != 4:
                raise ValueError("Each partner checkpoint must come from a distinct training run.")
            if len({item.flax_weights_sha256 for item in members}) != 4:
                raise ValueError("Partner Flax weight hashes must be distinct.")
            family_counts = {
                family: sum(item.family_id == family for item in members)
                for family in PARTNER_FAMILIES
            }
            if set(family_counts.values()) != {2}:
                raise ValueError(
                    "The historical partner pool requires two checkpoints per family."
                )
        else:
            members = tuple(
                FamilyPartnerMember.from_mapping(
                    item, base_dir=root, index=index
                )
                for index, item in enumerate(
                    _sequence(pool_payload["members"], "partner_pool.members")
                )
            )
            expected_counts = {1: 3, 2: 6, 3: 6}
            observed_counts = {
                prototype: sum(
                    item.prototype_index == prototype for item in members
                )
                for prototype in expected_counts
            }
            if observed_counts != expected_counts or len(members) != 15:
                raise ValueError(
                    "The family pool requires 3 own-history, 6 foreign self-play, "
                    "and 6 Other-Play checkpoints."
                )
            if len({item.member_id for item in members}) != 15:
                raise ValueError("Family-pool member identifiers must be unique.")
            if len({item.checkpoint_path for item in members}) != 15:
                raise ValueError("Family-pool checkpoint paths must be unique.")
            if len({item.flax_weights_sha256 for item in members}) != 15:
                raise ValueError("Family-pool checkpoint hashes must be unique.")
            for prototype, family_members in (
                (
                    value,
                    tuple(item for item in members if item.prototype_index == value),
                )
                for value in (1, 2, 3)
            ):
                run_counts: dict[str, int] = {}
                for item in family_members:
                    run_counts[item.training_run_id] = (
                        run_counts.get(item.training_run_id, 0) + 1
                    )
                expected_run_count = 1 if prototype == 1 else 2
                if (
                    len(run_counts) != expected_run_count
                    or set(run_counts.values()) != {HISTORY_CHECKPOINT_COUNT}
                ):
                    raise ValueError(
                        "Every admitted official run must contribute exactly three "
                        "mechanically scheduled checkpoints."
                    )
        stages = _mapping(payload["stages"], "stages")
        if schema_version == FAMILY_POOL_SCHEMA_VERSION:
            stage_fields = {
                "prefit",
                "training_calibration",
                "adaptation",
                "deployment_calibration",
                "evaluation",
            }
        else:
            stage_fields = {"prefit", "calibration", "adaptation", "evaluation"}
        _fields(stages, allowed=stage_fields, required=stage_fields, name="stages")
        training_calibration_key = (
            "training_calibration"
            if schema_version == FAMILY_POOL_SCHEMA_VERSION
            else "calibration"
        )
        config = cls(
            schema_version=schema_version,
            run_kind=run_kind,
            scientific_readout_allowed=scientific,
            condition_id=condition,
            controller=controller,
            environment=EnvironmentConfig.from_mapping(payload["environment"]),
            backbone_init=CheckpointRef.from_mapping(payload["backbone_init"], base_dir=root, name="backbone_init"),
            partner_pool=members,
            family_pool=family_pool,
            model=ModelConfig.from_mapping(payload["model"]),
            probe=ProbeConfig.from_mapping(payload["probe"], run_kind=run_kind),
            prefit=PrefitConfig.from_mapping(stages["prefit"]),
            calibration=CalibrationConfig.from_mapping(
                stages[training_calibration_key]
            ),
            adaptation=AdaptationConfig.from_mapping(stages["adaptation"]),
            deployment_calibration=(
                CalibrationConfig.from_mapping(stages["deployment_calibration"])
                if schema_version == FAMILY_POOL_SCHEMA_VERSION
                else None
            ),
            evaluation=EvaluationConfig.from_mapping(stages["evaluation"]),
            seeds=SeedsConfig.from_mapping(payload["seeds"]),
            output_root=_path(payload["output_root"], base_dir=root, name="output_root"),
            formal_outer_unit=(
                None
                if payload.get("formal_outer_unit") is None
                else FormalOuterUnitBinding.from_mapping(
                    payload["formal_outer_unit"], base_dir=root
                )
            ),
        )
        config._validate_cross_fields()
        return config

    def _validate_cross_fields(self) -> None:
        if self.schema_version == SCHEMA_VERSION:
            backbone_seed = _checkpoint_seed(self.backbone_init.checkpoint_path)
            backbone_registration = REGISTERED_OFFICIAL_CHECKPOINTS.get(backbone_seed)
            if (
                backbone_seed != 100
                or backbone_registration is None
                or self.backbone_init.flax_weights_sha256
                != backbone_registration["flax_weights_sha256"]
                or self.backbone_init.training_run_id
                != backbone_registration["training_run_id"]
            ):
                raise ValueError("The backbone must be the registered official seed-100 checkpoint.")
            partner_seeds = {
                _checkpoint_seed(member.checkpoint_path) for member in self.partner_pool
            }
            if partner_seeds != {101, 102, 201, 202}:
                raise ValueError(
                    "The partner pool must use registered seeds 101, 102, 201, and 202."
                )
            for member in self.partner_pool:
                registration = REGISTERED_OFFICIAL_CHECKPOINTS[
                    _checkpoint_seed(member.checkpoint_path)
                ]
                if (
                    member.family_id != registration["family_id"]
                    or member.flax_weights_sha256
                    != registration["flax_weights_sha256"]
                    or member.training_run_id != registration["training_run_id"]
                ):
                    raise ValueError(
                        "A partner checkpoint differs from its registered source identity."
                    )
        elif self.schema_version == FORMAL_SCHEMA_VERSION:
            if self.run_kind != "formal" or self.formal_outer_unit is None:
                raise ValueError("path_c_model_v2 is restricted to resolved formal outer units.")
            source_paths = {
                self.backbone_init.checkpoint_path,
                *(member.checkpoint_path for member in self.partner_pool),
            }
            source_runs = {
                self.backbone_init.training_run_id,
                *(member.training_run_id for member in self.partner_pool),
            }
            source_hashes = {
                self.backbone_init.flax_weights_sha256,
                *(member.flax_weights_sha256 for member in self.partner_pool),
            }
            if len(source_paths) != 5 or len(source_runs) != 5 or len(source_hashes) != 5:
                raise ValueError("A formal outer unit cannot reuse a backbone or partner source.")
        else:
            if (
                self.run_kind != "formal"
                or self.formal_outer_unit is None
                or self.family_pool is None
                or self.deployment_calibration is None
            ):
                raise ValueError(
                    "path_c_model_v3 is restricted to resolved formal family-pool units."
                )
            if (
                self.formal_outer_unit.backbone_launch_config
                in self.formal_outer_unit.partner_launch_configs
            ):
                raise ValueError("The backbone launch cannot also identify a foreign run.")
            source_runs = {
                self.backbone_init.training_run_id,
                *(member.training_run_id for member in self.partner_pool),
            }
            if len(source_runs) != 5:
                raise ValueError(
                    "A family-pool unit must bind one backbone and four foreign official runs."
                )
            if any(
                item.checkpoint_index not in range(HISTORY_CHECKPOINT_COUNT)
                for item in self.partner_pool
                if isinstance(item, FamilyPartnerMember)
            ):
                raise ValueError("A family history contains an unregistered checkpoint index.")
        if self.probe.decision_null_quantile != 0.95:
            raise ValueError("The registered decision null quantile is 0.95.")
        if self.probe.information_quantile != 0.80:
            raise ValueError("The registered information quantile is 0.80.")
        if self.probe.candidate_window > self.environment.episode_steps:
            raise ValueError("The probe window cannot exceed one episode.")
        if self.probe.budget_per_episode > self.probe.candidate_window:
            raise ValueError("The probe budget cannot exceed the candidate window.")
        if self.adaptation.unroll_length != self.environment.episode_steps:
            raise ValueError("Path C uses one complete 400-step episode per recurrent rollout.")
        if self.calibration.episodes % self.environment.num_envs:
            raise ValueError(
                "Training calibration episodes must divide into complete vector batches."
            )
        if (
            self.deployment_calibration is not None
            and self.deployment_calibration.episodes % self.environment.num_envs
        ):
            raise ValueError(
                "Deployment calibration episodes must divide into complete vector batches."
            )
        if self.environment.num_envs * self.adaptation.unroll_length <= 0:
            raise ValueError("The vector rollout size must be positive.")
        rollout_size = self.environment.num_envs * self.adaptation.unroll_length
        if self.prefit.env_steps % rollout_size or self.adaptation.env_steps % rollout_size:
            raise ValueError("Prefit and adaptation budgets must contain complete vector rollouts.")
        if (
            self.prefit.batch_size > rollout_size
            or rollout_size % self.prefit.batch_size
            or self.prefit.batch_size % self.adaptation.unroll_length
        ):
            raise ValueError("Prefit batches must contain complete recurrent environment lanes.")
        if self.adaptation.metrics_interval_env_steps % rollout_size:
            raise ValueError("The metrics interval must align with complete vector rollouts.")
        if rollout_size % self.adaptation.num_minibatches:
            raise ValueError("A rollout must divide evenly into adaptation minibatches.")
        if self.environment.num_envs % self.adaptation.num_minibatches:
            raise ValueError("Environment lanes must divide evenly into recurrent minibatches.")
        if self.run_kind == "development" and self.evaluation.schedule != "development_pool_bidirectional":
            raise ValueError("Development runs require the registered development pairing schedule.")
        if self.run_kind == "formal":
            observed = (
                self.environment.num_envs,
                self.prefit.env_steps,
                self.calibration.episodes,
                self.adaptation.env_steps,
                self.evaluation.episodes_per_pairing,
            )
            expected = (
                FORMAL_NUM_ENVS,
                FORMAL_PREFIT_STEPS,
                FORMAL_CALIBRATION_EPISODES,
                FORMAL_ADAPTATION_STEPS,
                FORMAL_EVALUATION_EPISODES,
            )
            expected_schedule = (
                "formal_manifest"
                if self.schema_version == SCHEMA_VERSION
                else "external_standard_matrix"
            )
            if observed != expected or self.evaluation.schedule != expected_schedule:
                raise ValueError("Formal runs must use the registered environment, stage, and evaluation budgets.")
            if (
                self.schema_version == FAMILY_POOL_SCHEMA_VERSION
                and (
                    self.deployment_calibration is None
                    or self.deployment_calibration.episodes
                    != FORMAL_DEPLOYMENT_CALIBRATION_EPISODES
                )
            ):
                raise ValueError(
                    "Formal family-pool runs require 500 deployment-calibration episodes."
                )
        if self.schema_version == SCHEMA_VERSION and self.formal_outer_unit is not None:
            raise ValueError("Historical path_c_model_v1 configs cannot bind a formal outer unit.")

    def to_mapping(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): convert(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(child) for child in value]
            return value

        raw = asdict(self)
        if raw.get("formal_outer_unit") is None:
            raw.pop("formal_outer_unit", None)
        probe = raw["probe"]
        threshold = {
            "source": probe["threshold_source"],
            "decision_null_quantile": probe["decision_null_quantile"],
            "information_quantile": probe["information_quantile"],
        }
        if probe["manual_threshold"] is not None:
            threshold["manual_value"] = probe["manual_threshold"]
        raw["probe"] = {
            "candidate_window": probe["candidate_window"],
            "budget_per_episode": probe["budget_per_episode"],
            "safety": {"rule": probe["safety_rule"]},
            "threshold": threshold,
            "belief_probability_floor": probe["belief_probability_floor"],
        }
        family_pool = raw.pop("family_pool", None)
        deployment = raw.pop("deployment_calibration", None)
        if self.schema_version == FAMILY_POOL_SCHEMA_VERSION:
            raw["stages"] = {
                "prefit": raw.pop("prefit"),
                "training_calibration": raw.pop("calibration"),
                "adaptation": raw.pop("adaptation"),
                "deployment_calibration": deployment,
                "evaluation": raw.pop("evaluation"),
            }
            raw["partner_pool"] = {
                **family_pool,
                "members": raw["partner_pool"],
            }
        else:
            raw["stages"] = {
                "prefit": raw.pop("prefit"),
                "calibration": raw.pop("calibration"),
                "adaptation": raw.pop("adaptation"),
                "evaluation": raw.pop("evaluation"),
            }
            raw["partner_pool"] = {"members": raw["partner_pool"]}
        return convert(raw)

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def shared_stage_mapping(self) -> dict[str, Any]:
        payload = self.to_mapping()
        payload.pop("condition_id")
        payload.pop("controller")
        return payload

    @property
    def num_prototypes(self) -> int:
        return len(FAMILY_PROTOTYPES) if self.family_pool is not None else len(self.partner_pool)

    @property
    def is_family_pool(self) -> bool:
        return self.schema_version == FAMILY_POOL_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PathCFormalTemplateV2:
    """Source-free formal template resolved only through an outer-unit manifest."""

    condition_id: str
    controller: str
    scientific_readout_allowed: bool
    environment: EnvironmentConfig
    model: ModelConfig
    probe: ProbeConfig
    prefit: PrefitConfig
    calibration: CalibrationConfig
    adaptation: AdaptationConfig
    evaluation: EvaluationConfig
    output_root: Path

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> "PathCFormalTemplateV2":
        payload = _mapping(payload, "path_c_model_v2 formal template")
        allowed = {
            "schema_version",
            "run_kind",
            "scientific_readout_allowed",
            "condition_id",
            "controller",
            "environment",
            "model",
            "probe",
            "stages",
            "output_root",
        }
        _fields(payload, allowed=allowed, required=allowed, name="path_c_model_v2 formal template")
        if payload["schema_version"] != FORMAL_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {FORMAL_SCHEMA_VERSION}.")
        if payload["run_kind"] != "formal":
            raise ValueError("path_c_model_v2 templates are formal only.")
        scientific = payload["scientific_readout_allowed"]
        if scientific is not False:
            raise ValueError(
                "Formal templates must keep scientific_readout_allowed false until manifest freeze."
            )
        condition = str(payload["condition_id"])
        controller = str(payload["controller"])
        if condition not in CONDITION_CONTROLLERS or controller != CONDITION_CONTROLLERS[condition]:
            raise ValueError("controller does not match the registered formal condition.")
        stages = _mapping(payload["stages"], "stages")
        stage_fields = {"prefit", "calibration", "adaptation", "evaluation"}
        _fields(stages, allowed=stage_fields, required=stage_fields, name="stages")
        template = cls(
            condition_id=condition,
            controller=controller,
            scientific_readout_allowed=False,
            environment=EnvironmentConfig.from_mapping(payload["environment"]),
            model=ModelConfig.from_mapping(payload["model"]),
            probe=ProbeConfig.from_mapping(payload["probe"], run_kind="formal"),
            prefit=PrefitConfig.from_mapping(stages["prefit"]),
            calibration=CalibrationConfig.from_mapping(stages["calibration"]),
            adaptation=AdaptationConfig.from_mapping(stages["adaptation"]),
            evaluation=EvaluationConfig.from_mapping(stages["evaluation"]),
            output_root=_path(
                payload["output_root"],
                base_dir=Path(base_dir).resolve(),
                name="output_root",
            ),
        )
        observed = (
            template.environment.num_envs,
            template.prefit.env_steps,
            template.calibration.episodes,
            template.adaptation.env_steps,
            template.evaluation.episodes_per_pairing,
            template.evaluation.schedule,
        )
        expected = (
            FORMAL_NUM_ENVS,
            FORMAL_PREFIT_STEPS,
            FORMAL_CALIBRATION_EPISODES,
            FORMAL_ADAPTATION_STEPS,
            FORMAL_EVALUATION_EPISODES,
            "external_standard_matrix",
        )
        if observed != expected:
            raise ValueError(
                "Formal runs must use the registered training and matrix budgets."
            )
        return template

    def resolve(
        self, manifest: OuterUnitsManifest, *, outer_unit_id: int
    ) -> PathCModelConfig:
        unit: OuterTrainingUnit = manifest.unit(outer_unit_id)
        if self.environment.layout != manifest.layout:
            raise ValueError("Formal template and outer-unit manifest layouts differ.")
        binding = FormalOuterUnitBinding(
            outer_unit_id=unit.outer_unit_id,
            outer_units_manifest_path=manifest.path,
            outer_units_manifest_sha256=manifest.sha256,
            backbone_launch_config=unit.backbone.launch_config_path,
            partner_launch_configs=tuple(
                member.launch_config_path for member in unit.partners
            ),
        )
        config = PathCModelConfig(
            schema_version=FORMAL_SCHEMA_VERSION,
            run_kind="formal",
            scientific_readout_allowed=self.scientific_readout_allowed,
            condition_id=self.condition_id,
            controller=self.controller,
            environment=self.environment,
            backbone_init=CheckpointRef(
                checkpoint_path=unit.backbone.checkpoint_path,
                flax_weights_sha256=unit.backbone.flax_weights_sha256,
                training_run_id=unit.backbone.training_run_id,
            ),
            partner_pool=tuple(
                PartnerMember(
                    checkpoint_path=member.checkpoint_path,
                    flax_weights_sha256=member.flax_weights_sha256,
                    training_run_id=member.training_run_id,
                    family_id=member.family_id,
                )
                for member in unit.partners
            ),
            family_pool=None,
            model=self.model,
            probe=self.probe,
            prefit=self.prefit,
            calibration=self.calibration,
            adaptation=self.adaptation,
            deployment_calibration=None,
            evaluation=self.evaluation,
            seeds=SeedsConfig(**unit.seeds.to_mapping()),
            output_root=self.output_root / f"outer_unit_{unit.outer_unit_id:02d}",
            formal_outer_unit=binding,
        )
        config._validate_cross_fields()
        return config


@dataclass(frozen=True, slots=True)
class PathCFormalTemplateV3:
    """Source-free family-pool template resolved through a version-two unit manifest."""

    condition_id: str
    controller: str
    scientific_readout_allowed: bool
    environment: EnvironmentConfig
    family_pool: FamilyPoolConfig
    model: ModelConfig
    probe: ProbeConfig
    prefit: PrefitConfig
    training_calibration: CalibrationConfig
    adaptation: AdaptationConfig
    deployment_calibration: CalibrationConfig
    evaluation: EvaluationConfig
    output_root: Path

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> "PathCFormalTemplateV3":
        payload = _mapping(payload, "path_c_model_v3 formal template")
        fields = {
            "schema_version",
            "run_kind",
            "scientific_readout_allowed",
            "condition_id",
            "controller",
            "environment",
            "partner_pool",
            "model",
            "probe",
            "stages",
            "output_root",
        }
        _fields(payload, allowed=fields, required=fields, name="path_c_model_v3 formal template")
        if payload["schema_version"] != FAMILY_POOL_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {FAMILY_POOL_SCHEMA_VERSION}.")
        if payload["run_kind"] != "formal":
            raise ValueError("path_c_model_v3 templates are formal only.")
        if payload["scientific_readout_allowed"] is not False:
            raise ValueError(
                "Formal templates keep scientific_readout_allowed false until final review."
            )
        condition = str(payload["condition_id"])
        controller = str(payload["controller"])
        if (
            condition not in CONDITION_CONTROLLERS
            or controller != CONDITION_CONTROLLERS[condition]
        ):
            raise ValueError("controller does not match the registered formal condition.")
        pool_payload = dict(_mapping(payload["partner_pool"], "partner_pool"))
        if "members" in pool_payload:
            raise ValueError("A source-free v3 template cannot enumerate checkpoint members.")
        pool_payload["members"] = []
        family_pool = FamilyPoolConfig.from_mapping(pool_payload)
        stages = _mapping(payload["stages"], "stages")
        stage_fields = {
            "prefit",
            "training_calibration",
            "adaptation",
            "deployment_calibration",
            "evaluation",
        }
        _fields(stages, allowed=stage_fields, required=stage_fields, name="stages")
        template = cls(
            condition_id=condition,
            controller=controller,
            scientific_readout_allowed=False,
            environment=EnvironmentConfig.from_mapping(payload["environment"]),
            family_pool=family_pool,
            model=ModelConfig.from_mapping(payload["model"]),
            probe=ProbeConfig.from_mapping(payload["probe"], run_kind="formal"),
            prefit=PrefitConfig.from_mapping(stages["prefit"]),
            training_calibration=CalibrationConfig.from_mapping(
                stages["training_calibration"]
            ),
            adaptation=AdaptationConfig.from_mapping(stages["adaptation"]),
            deployment_calibration=CalibrationConfig.from_mapping(
                stages["deployment_calibration"]
            ),
            evaluation=EvaluationConfig.from_mapping(stages["evaluation"]),
            output_root=_path(
                payload["output_root"],
                base_dir=Path(base_dir).resolve(),
                name="output_root",
            ),
        )
        observed = (
            template.environment.num_envs,
            template.prefit.env_steps,
            template.training_calibration.episodes,
            template.adaptation.env_steps,
            template.deployment_calibration.episodes,
            template.evaluation.episodes_per_pairing,
            template.evaluation.schedule,
        )
        expected = (
            FORMAL_NUM_ENVS,
            FORMAL_PREFIT_STEPS,
            FORMAL_CALIBRATION_EPISODES,
            FORMAL_ADAPTATION_STEPS,
            FORMAL_DEPLOYMENT_CALIBRATION_EPISODES,
            FORMAL_EVALUATION_EPISODES,
            "external_standard_matrix",
        )
        if observed != expected:
            raise ValueError(
                "Formal family-pool runs must use the registered production budgets."
            )
        return template

    def resolve(
        self, manifest: OuterUnitsManifest, *, outer_unit_id: int
    ) -> PathCModelConfig:
        if manifest.schema_version != FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION:
            raise ValueError("path_c_model_v3 requires path_c_outer_units_v2.")
        if self.environment.layout != manifest.layout:
            raise ValueError("Formal template and outer-unit manifest layouts differ.")
        unit = manifest.unit(outer_unit_id)
        binding = FormalOuterUnitBinding(
            outer_unit_id=unit.outer_unit_id,
            outer_units_manifest_path=manifest.path,
            outer_units_manifest_sha256=manifest.sha256,
            backbone_launch_config=unit.backbone.launch_config_path,
            partner_launch_configs=tuple(
                source.launch_config_path for source in unit.partners
            ),
        )
        members: list[FamilyPartnerMember] = []

        def append_history(
            *,
            source: Any,
            prototype_index: int,
            source_label: str,
        ) -> None:
            for snapshot in source.checkpoint_history:
                members.append(
                    FamilyPartnerMember(
                        checkpoint_path=snapshot.checkpoint_path,
                        flax_weights_sha256=snapshot.flax_weights_sha256,
                        training_run_id=source.training_run_id,
                        member_id=(
                            f"{source_label}_checkpoint_{snapshot.checkpoint_index}"
                        ),
                        prototype_index=prototype_index,
                        family_id=FAMILY_PROTOTYPES[prototype_index],
                        source_policy_family=source.family_id,
                        checkpoint_index=snapshot.checkpoint_index,
                        update_step=snapshot.update_step,
                        effective_environment_steps=(
                            snapshot.effective_environment_steps
                        ),
                        launch_config_path=source.launch_config_path,
                        launch_config_sha256=source.launch_config_sha256,
                    )
                )

        append_history(
            source=unit.backbone,
            prototype_index=1,
            source_label="own_backbone",
        )
        for source in unit.partners[:2]:
            append_history(
                source=source,
                prototype_index=2,
                source_label=source.member_id,
            )
        for source in unit.partners[2:]:
            append_history(
                source=source,
                prototype_index=3,
                source_label=source.member_id,
            )
        config = PathCModelConfig(
            schema_version=FAMILY_POOL_SCHEMA_VERSION,
            run_kind="formal",
            scientific_readout_allowed=False,
            condition_id=self.condition_id,
            controller=self.controller,
            environment=self.environment,
            backbone_init=CheckpointRef(
                checkpoint_path=unit.backbone.checkpoint_path,
                flax_weights_sha256=unit.backbone.flax_weights_sha256,
                training_run_id=unit.backbone.training_run_id,
            ),
            partner_pool=tuple(members),
            family_pool=self.family_pool,
            model=self.model,
            probe=self.probe,
            prefit=self.prefit,
            calibration=self.training_calibration,
            adaptation=self.adaptation,
            deployment_calibration=self.deployment_calibration,
            evaluation=self.evaluation,
            seeds=SeedsConfig(**unit.seeds.to_mapping()),
            output_root=self.output_root / f"outer_unit_{unit.outer_unit_id:02d}",
            formal_outer_unit=binding,
        )
        config._validate_cross_fields()
        return config
