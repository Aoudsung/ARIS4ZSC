"""Strict configuration contract for the fourth Path C model.

The value-quotient belief-conditioned controller is intentionally isolated from
the historical Path C configuration objects.  This prevents a fourth-version
run from silently accepting actor, probe, calibration, or labelled-prototype
fields that belong only to versions one through three.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.outer_units import (
    FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
    OuterTrainingUnit,
    OuterUnitsManifest,
)


VQBC_SCHEMA_VERSION = "path_c_model_v4_1"
VQBC_FORMAL_ENVIRONMENT_STEPS = 11_000_000
VQBC_DEVELOPMENT_ENVIRONMENT_STEPS = 1_228_800
VQBC_FORMAL_NUM_ENVS = 250
VQBC_DEVELOPMENT_NUM_ENVS = 32
VQBC_FORMAL_MINIBATCHES = 50
VQBC_DEVELOPMENT_MINIBATCHES = 8
VQBC_EPISODE_STEPS = 400
VQBC_ACTION_COUNT = 6
VQBC_SLOT_COUNT = 8
VQBC_RESPONSE_COUNT = 16
VQBC_TERMINAL_RESPONSE = 15
VQBC_DEPLOYMENT_MODES = (
    "posterior_use",
    "prior_only",
    "reference_only",
    "generic_response_information",
)
VQBC_PARTNER_SCHEDULE = "uniform_member_episode_v1"
_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _fields(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown or missing:
        raise ValueError(
            f"{name} fields are invalid; unknown={unknown}, missing={missing}."
        )


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
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _probability(value: Any, name: str, *, open_interval: bool = False) -> float:
    result = _finite(value, name)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}.")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX):
        raise ValueError(f"{name} must be a lowercase SHA-256 string.")
    return value


def _path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else base_dir / path).resolve()


@dataclass(frozen=True, slots=True)
class VQBCEnvironmentConfig:
    layout: str
    agent_view_size: int
    indicate_successful_delivery: bool
    episode_steps: int
    num_envs: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCEnvironmentConfig":
        payload = _mapping(payload, "environment")
        fields = {
            "layout",
            "agent_view_size",
            "indicate_successful_delivery",
            "episode_steps",
            "num_envs",
        }
        _fields(payload, allowed=fields, required=fields, name="environment")
        layout = str(payload["layout"])
        if layout not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("The fourth model supports only Test Time Simple and Wide.")
        if (
            _positive_int(payload["agent_view_size"], "environment.agent_view_size")
            != 2
            or _positive_int(payload["episode_steps"], "environment.episode_steps")
            != VQBC_EPISODE_STEPS
            or payload["indicate_successful_delivery"] is not True
        ):
            raise ValueError(
                "The public protocol requires view radius two, delivery indication, "
                "and 400-step episodes."
            )
        return cls(
            layout=layout,
            agent_view_size=2,
            indicate_successful_delivery=True,
            episode_steps=VQBC_EPISODE_STEPS,
            num_envs=_positive_int(payload["num_envs"], "environment.num_envs"),
        )


@dataclass(frozen=True, slots=True)
class VQBCCheckpointRef:
    checkpoint_path: Path
    flax_weights_sha256: str
    training_run_id: str
    launch_config_path: Path
    launch_config_sha256: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
        name: str,
    ) -> "VQBCCheckpointRef":
        payload = _mapping(payload, name)
        fields = {
            "checkpoint_path",
            "flax_weights_sha256",
            "training_run_id",
            "launch_config_path",
            "launch_config_sha256",
        }
        _fields(payload, allowed=fields, required=fields, name=name)
        run_id = str(payload["training_run_id"])
        if not run_id:
            raise ValueError(f"{name}.training_run_id must be non-empty.")
        return cls(
            checkpoint_path=_path(
                payload["checkpoint_path"],
                base_dir=base_dir,
                name=f"{name}.checkpoint_path",
            ),
            flax_weights_sha256=_sha256(
                payload["flax_weights_sha256"], f"{name}.flax_weights_sha256"
            ),
            training_run_id=run_id,
            launch_config_path=_path(
                payload["launch_config_path"],
                base_dir=base_dir,
                name=f"{name}.launch_config_path",
            ),
            launch_config_sha256=_sha256(
                payload["launch_config_sha256"], f"{name}.launch_config_sha256"
            ),
        )

    @classmethod
    def from_official_source(cls, source: Any) -> "VQBCCheckpointRef":
        return cls(
            checkpoint_path=Path(source.checkpoint_path).resolve(),
            flax_weights_sha256=str(source.flax_weights_sha256),
            training_run_id=str(source.training_run_id),
            launch_config_path=Path(source.launch_config_path).resolve(),
            launch_config_sha256=str(source.launch_config_sha256),
        )


@dataclass(frozen=True, slots=True)
class VQBCPartnerMember:
    """One immutable training partner with audit-only source metadata."""

    member_id: str
    checkpoint: VQBCCheckpointRef
    family_id: str
    training_seed: int
    checkpoint_index: int
    update_step: int
    effective_environment_steps: int

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
        index: int,
    ) -> "VQBCPartnerMember":
        name = f"partner_sampling.members[{index}]"
        payload = _mapping(payload, name)
        checkpoint_fields = {
            "checkpoint_path",
            "flax_weights_sha256",
            "training_run_id",
            "launch_config_path",
            "launch_config_sha256",
        }
        fields = checkpoint_fields | {
            "member_id",
            "family_id",
            "training_seed",
            "checkpoint_index",
            "update_step",
            "effective_environment_steps",
        }
        _fields(payload, allowed=fields, required=fields, name=name)
        member_id = str(payload["member_id"])
        family_id = str(payload["family_id"])
        if not member_id or not family_id:
            raise ValueError(f"{name} requires non-empty member and family identifiers.")
        checkpoint = VQBCCheckpointRef.from_mapping(
            {field: payload[field] for field in checkpoint_fields},
            base_dir=base_dir,
            name=name,
        )
        return cls(
            member_id=member_id,
            checkpoint=checkpoint,
            family_id=family_id,
            training_seed=_nonnegative_int(
                payload["training_seed"], f"{name}.training_seed"
            ),
            checkpoint_index=_nonnegative_int(
                payload["checkpoint_index"], f"{name}.checkpoint_index"
            ),
            update_step=_nonnegative_int(payload["update_step"], f"{name}.update_step"),
            effective_environment_steps=_nonnegative_int(
                payload["effective_environment_steps"],
                f"{name}.effective_environment_steps",
            ),
        )

    @classmethod
    def from_snapshot(
        cls, *, source: Any, snapshot: Any, member_id: str
    ) -> "VQBCPartnerMember":
        return cls(
            member_id=member_id,
            checkpoint=VQBCCheckpointRef(
                checkpoint_path=Path(snapshot.checkpoint_path).resolve(),
                flax_weights_sha256=str(snapshot.flax_weights_sha256),
                training_run_id=str(source.training_run_id),
                launch_config_path=Path(source.launch_config_path).resolve(),
                launch_config_sha256=str(source.launch_config_sha256),
            ),
            family_id=str(source.family_id),
            training_seed=int(source.training_seed),
            checkpoint_index=int(snapshot.checkpoint_index),
            update_step=int(snapshot.update_step),
            effective_environment_steps=int(snapshot.effective_environment_steps),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **_checkpoint_mapping(self.checkpoint),
            "member_id": self.member_id,
            "family_id": self.family_id,
            "training_seed": self.training_seed,
            "checkpoint_index": self.checkpoint_index,
            "update_step": self.update_step,
            "effective_environment_steps": self.effective_environment_steps,
        }


@dataclass(frozen=True, slots=True)
class VQBCPartnerSamplingConfig:
    schedule: str
    include_frozen_current_policy: bool
    members: tuple[VQBCPartnerMember, ...]

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: Path
    ) -> "VQBCPartnerSamplingConfig":
        payload = _mapping(payload, "partner_sampling")
        fields = {"schedule", "include_frozen_current_policy", "members"}
        _fields(payload, allowed=fields, required=fields, name="partner_sampling")
        if payload["schedule"] != VQBC_PARTNER_SCHEDULE:
            raise ValueError("The fourth model samples training partners uniformly by member.")
        if not isinstance(payload["include_frozen_current_policy"], bool):
            raise ValueError("include_frozen_current_policy must be boolean.")
        members = tuple(
            VQBCPartnerMember.from_mapping(item, base_dir=base_dir, index=index)
            for index, item in enumerate(
                _sequence(payload["members"], "partner_sampling.members")
            )
        )
        if not members:
            raise ValueError("The partner sampling plan cannot be empty.")
        if len({member.member_id for member in members}) != len(members):
            raise ValueError("Partner member identifiers must be unique.")
        identities = {
            (
                member.checkpoint.training_run_id,
                member.checkpoint.checkpoint_path,
                member.checkpoint_index,
            )
            for member in members
        }
        if len(identities) != len(members):
            raise ValueError("The partner sampling plan repeats a checkpoint.")
        return cls(
            schedule=VQBC_PARTNER_SCHEDULE,
            include_frozen_current_policy=payload["include_frozen_current_policy"],
            members=members,
        )


@dataclass(frozen=True, slots=True)
class VQBCModelConfig:
    hidden_dim: int
    head_hidden_dim: int
    action_embedding_dim: int
    slot_count: int
    response_count: int
    action_count: int
    prior_scale: float
    log_standard_deviation_minimum: float
    log_standard_deviation_maximum: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCModelConfig":
        payload = _mapping(payload, "model")
        fields = {
            "hidden_dim",
            "head_hidden_dim",
            "action_embedding_dim",
            "slot_count",
            "response_count",
            "action_count",
            "prior_scale",
            "log_standard_deviation_minimum",
            "log_standard_deviation_maximum",
        }
        _fields(payload, allowed=fields, required=fields, name="model")
        config = cls(
            hidden_dim=_positive_int(payload["hidden_dim"], "model.hidden_dim"),
            head_hidden_dim=_positive_int(
                payload["head_hidden_dim"], "model.head_hidden_dim"
            ),
            action_embedding_dim=_positive_int(
                payload["action_embedding_dim"], "model.action_embedding_dim"
            ),
            slot_count=_positive_int(payload["slot_count"], "model.slot_count"),
            response_count=_positive_int(
                payload["response_count"], "model.response_count"
            ),
            action_count=_positive_int(payload["action_count"], "model.action_count"),
            prior_scale=_finite(payload["prior_scale"], "model.prior_scale"),
            log_standard_deviation_minimum=_finite(
                payload["log_standard_deviation_minimum"],
                "model.log_standard_deviation_minimum",
            ),
            log_standard_deviation_maximum=_finite(
                payload["log_standard_deviation_maximum"],
                "model.log_standard_deviation_maximum",
            ),
        )
        if (
            config.hidden_dim != 128
            or config.head_hidden_dim != 128
            or config.action_embedding_dim != 16
            or config.slot_count != VQBC_SLOT_COUNT
            or config.response_count != VQBC_RESPONSE_COUNT
            or config.action_count != VQBC_ACTION_COUNT
            or config.prior_scale != 0.01
            or config.log_standard_deviation_minimum != -5.0
            or config.log_standard_deviation_maximum != 2.0
        ):
            raise ValueError("The fourth-model architecture differs from its fixed contract.")
        return config


@dataclass(frozen=True, slots=True)
class VQBCTrainingConfig:
    environment_steps: int
    unroll_length: int
    update_epochs: int
    minibatches_per_epoch: int
    bellman_learning_rate: float
    outcome_learning_rate: float
    gradient_clip_norm: float
    gamma: float
    responsibility_temperature: float
    bootstrap_probability: float
    polyak_coefficient: float
    codebook_decay: float
    code_replacement_rollouts: int
    metrics_interval_environment_steps: int
    checkpoint_interval_environment_steps: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCTrainingConfig":
        payload = _mapping(payload, "training")
        fields = {
            "environment_steps",
            "unroll_length",
            "update_epochs",
            "minibatches_per_epoch",
            "bellman_learning_rate",
            "outcome_learning_rate",
            "gradient_clip_norm",
            "gamma",
            "responsibility_temperature",
            "bootstrap_probability",
            "polyak_coefficient",
            "codebook_decay",
            "code_replacement_rollouts",
            "metrics_interval_environment_steps",
            "checkpoint_interval_environment_steps",
        }
        _fields(payload, allowed=fields, required=fields, name="training")
        return cls(
            environment_steps=_positive_int(
                payload["environment_steps"], "training.environment_steps"
            ),
            unroll_length=_positive_int(
                payload["unroll_length"], "training.unroll_length"
            ),
            update_epochs=_positive_int(
                payload["update_epochs"], "training.update_epochs"
            ),
            minibatches_per_epoch=_positive_int(
                payload["minibatches_per_epoch"],
                "training.minibatches_per_epoch",
            ),
            bellman_learning_rate=_finite(
                payload["bellman_learning_rate"],
                "training.bellman_learning_rate",
                positive=True,
            ),
            outcome_learning_rate=_finite(
                payload["outcome_learning_rate"],
                "training.outcome_learning_rate",
                positive=True,
            ),
            gradient_clip_norm=_finite(
                payload["gradient_clip_norm"],
                "training.gradient_clip_norm",
                positive=True,
            ),
            gamma=_probability(payload["gamma"], "training.gamma", open_interval=True),
            responsibility_temperature=_finite(
                payload["responsibility_temperature"],
                "training.responsibility_temperature",
                positive=True,
            ),
            bootstrap_probability=_probability(
                payload["bootstrap_probability"],
                "training.bootstrap_probability",
                open_interval=True,
            ),
            polyak_coefficient=_probability(
                payload["polyak_coefficient"],
                "training.polyak_coefficient",
                open_interval=True,
            ),
            codebook_decay=_probability(
                payload["codebook_decay"],
                "training.codebook_decay",
                open_interval=True,
            ),
            code_replacement_rollouts=_positive_int(
                payload["code_replacement_rollouts"],
                "training.code_replacement_rollouts",
            ),
            metrics_interval_environment_steps=_positive_int(
                payload["metrics_interval_environment_steps"],
                "training.metrics_interval_environment_steps",
            ),
            checkpoint_interval_environment_steps=_positive_int(
                payload["checkpoint_interval_environment_steps"],
                "training.checkpoint_interval_environment_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class VQBCKLConfig:
    target_per_step: float
    initial_temperature: float
    dual_learning_rate: float
    minimum_temperature: float
    maximum_temperature: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCKLConfig":
        payload = _mapping(payload, "kl_control")
        fields = {
            "target_per_step",
            "initial_temperature",
            "dual_learning_rate",
            "minimum_temperature",
            "maximum_temperature",
        }
        _fields(payload, allowed=fields, required=fields, name="kl_control")
        config = cls(
            target_per_step=_finite(
                payload["target_per_step"], "kl_control.target_per_step", positive=True
            ),
            initial_temperature=_finite(
                payload["initial_temperature"],
                "kl_control.initial_temperature",
                positive=True,
            ),
            dual_learning_rate=_finite(
                payload["dual_learning_rate"],
                "kl_control.dual_learning_rate",
                positive=True,
            ),
            minimum_temperature=_finite(
                payload["minimum_temperature"],
                "kl_control.minimum_temperature",
                positive=True,
            ),
            maximum_temperature=_finite(
                payload["maximum_temperature"],
                "kl_control.maximum_temperature",
                positive=True,
            ),
        )
        if (
            config.target_per_step != 0.02
            or config.initial_temperature != 1.0
            or config.dual_learning_rate != 1.0e-3
            or config.minimum_temperature != 0.05
            or config.maximum_temperature != 20.0
        ):
            raise ValueError("KL control differs from the fixed fourth-model contract.")
        return config


@dataclass(frozen=True, slots=True)
class VQBCEvaluationConfig:
    deployment_modes: tuple[str, ...]
    episodes_per_pairing: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCEvaluationConfig":
        payload = _mapping(payload, "evaluation")
        fields = {"deployment_modes", "episodes_per_pairing"}
        _fields(payload, allowed=fields, required=fields, name="evaluation")
        modes = tuple(
            str(value)
            for value in _sequence(
                payload["deployment_modes"], "evaluation.deployment_modes"
            )
        )
        if modes != VQBC_DEPLOYMENT_MODES:
            raise ValueError("Evaluation must list all four registered deployment modes.")
        episodes = _positive_int(
            payload["episodes_per_pairing"], "evaluation.episodes_per_pairing"
        )
        if episodes != 500:
            raise ValueError("The standard matrix uses 500 episodes per pairing.")
        return cls(deployment_modes=modes, episodes_per_pairing=episodes)


@dataclass(frozen=True, slots=True)
class VQBCSeeds:
    model_seed: int
    environment_seed: int
    evaluation_seed: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VQBCSeeds":
        payload = _mapping(payload, "seeds")
        fields = {"model_seed", "environment_seed", "evaluation_seed"}
        _fields(payload, allowed=fields, required=fields, name="seeds")
        values = {
            name: _nonnegative_int(payload[name], f"seeds.{name}") for name in fields
        }
        if len(set(values.values())) != 3:
            raise ValueError("Model, environment, and evaluation random seeds must differ.")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class VQBCOuterUnitBinding:
    outer_unit_id: int
    outer_units_manifest_path: Path
    outer_units_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class VQBCConfig:
    schema_version: str
    run_kind: str
    scientific_readout_allowed: bool
    environment: VQBCEnvironmentConfig
    backbone_init: VQBCCheckpointRef
    partner_sampling: VQBCPartnerSamplingConfig
    model: VQBCModelConfig
    training: VQBCTrainingConfig
    kl_control: VQBCKLConfig
    evaluation: VQBCEvaluationConfig
    seeds: VQBCSeeds
    output_root: Path
    outer_unit: VQBCOuterUnitBinding | None = None

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> "VQBCConfig":
        payload = _mapping(payload, VQBC_SCHEMA_VERSION)
        fields = {
            "schema_version",
            "run_kind",
            "scientific_readout_allowed",
            "environment",
            "backbone_init",
            "partner_sampling",
            "model",
            "training",
            "kl_control",
            "evaluation",
            "seeds",
            "output_root",
            "outer_unit",
        }
        _fields(
            payload,
            allowed=fields,
            required=fields - {"outer_unit"},
            name=VQBC_SCHEMA_VERSION,
        )
        if payload["schema_version"] != VQBC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {VQBC_SCHEMA_VERSION}.")
        run_kind = str(payload["run_kind"])
        if run_kind not in {"development", "formal"}:
            raise ValueError("run_kind must be development or formal.")
        if payload["scientific_readout_allowed"] is not False:
            raise ValueError(
                "Fourth-model configs remain non-claim until a reviewed formal readout."
            )
        root = Path(base_dir).resolve()
        config = cls(
            schema_version=VQBC_SCHEMA_VERSION,
            run_kind=run_kind,
            scientific_readout_allowed=False,
            environment=VQBCEnvironmentConfig.from_mapping(payload["environment"]),
            backbone_init=VQBCCheckpointRef.from_mapping(
                payload["backbone_init"], base_dir=root, name="backbone_init"
            ),
            partner_sampling=VQBCPartnerSamplingConfig.from_mapping(
                payload["partner_sampling"], base_dir=root
            ),
            model=VQBCModelConfig.from_mapping(payload["model"]),
            training=VQBCTrainingConfig.from_mapping(payload["training"]),
            kl_control=VQBCKLConfig.from_mapping(payload["kl_control"]),
            evaluation=VQBCEvaluationConfig.from_mapping(payload["evaluation"]),
            seeds=VQBCSeeds.from_mapping(payload["seeds"]),
            output_root=_path(payload["output_root"], base_dir=root, name="output_root"),
            outer_unit=(
                None
                if payload.get("outer_unit") is None
                else _outer_unit_binding(payload["outer_unit"], base_dir=root)
            ),
        )
        config._validate_cross_fields()
        return config

    def _validate_cross_fields(self) -> None:
        expected_envs = (
            VQBC_FORMAL_NUM_ENVS
            if self.run_kind == "formal"
            else VQBC_DEVELOPMENT_NUM_ENVS
        )
        expected_steps = (
            VQBC_FORMAL_ENVIRONMENT_STEPS
            if self.run_kind == "formal"
            else VQBC_DEVELOPMENT_ENVIRONMENT_STEPS
        )
        expected_minibatches = (
            VQBC_FORMAL_MINIBATCHES
            if self.run_kind == "formal"
            else VQBC_DEVELOPMENT_MINIBATCHES
        )
        if (
            self.environment.num_envs != expected_envs
            or self.training.environment_steps != expected_steps
            or self.training.unroll_length != VQBC_EPISODE_STEPS
            or self.training.update_epochs != 4
            or self.training.minibatches_per_epoch != expected_minibatches
            or self.training.bellman_learning_rate != 1.0e-4
            or self.training.outcome_learning_rate != 2.5e-4
            or self.training.gradient_clip_norm != 0.25
            or self.training.gamma != 0.99
            or self.training.responsibility_temperature != 1.0
            or self.training.bootstrap_probability != 0.8
            or self.training.polyak_coefficient != 0.005
            or self.training.codebook_decay != 0.99
            or self.training.code_replacement_rollouts != 10
        ):
            raise ValueError("The fourth-model training schedule differs from its fixed budget.")
        rollout_steps = self.environment.num_envs * self.training.unroll_length
        if self.training.environment_steps % rollout_steps:
            raise ValueError("Training must consist only of complete vectorized episodes.")
        if (
            self.training.metrics_interval_environment_steps % rollout_steps
            or self.training.checkpoint_interval_environment_steps % rollout_steps
        ):
            raise ValueError("Metrics and checkpoints must align to complete rollouts.")
        if self.run_kind == "formal" and (
            self.training.metrics_interval_environment_steps != 100_000
            or self.training.checkpoint_interval_environment_steps != 100_000
        ):
            raise ValueError("Formal metrics and checkpoints are written every 100,000 steps.")
        if not self.partner_sampling.include_frozen_current_policy:
            raise ValueError(
                "The training pool must include a rollout-frozen current-policy partner."
            )
        if (
            self.backbone_init.training_run_id
            in {
                member.checkpoint.training_run_id
                for member in self.partner_sampling.members
                if member.family_id != "official_rnn_sp_ippo_v1"
            }
        ):
            raise ValueError("The reference run cannot masquerade as a foreign partner family.")

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_kind": self.run_kind,
            "scientific_readout_allowed": self.scientific_readout_allowed,
            "environment": asdict(self.environment),
            "backbone_init": _checkpoint_mapping(self.backbone_init),
            "partner_sampling": {
                "schedule": self.partner_sampling.schedule,
                "include_frozen_current_policy": (
                    self.partner_sampling.include_frozen_current_policy
                ),
                "members": [
                    member.to_mapping() for member in self.partner_sampling.members
                ],
            },
            "model": asdict(self.model),
            "training": asdict(self.training),
            "kl_control": asdict(self.kl_control),
            "evaluation": {
                "deployment_modes": list(self.evaluation.deployment_modes),
                "episodes_per_pairing": self.evaluation.episodes_per_pairing,
            },
            "seeds": asdict(self.seeds),
            "output_root": str(self.output_root),
        }
        if self.outer_unit is not None:
            payload["outer_unit"] = {
                "outer_unit_id": self.outer_unit.outer_unit_id,
                "outer_units_manifest_path": str(
                    self.outer_unit.outer_units_manifest_path
                ),
                "outer_units_manifest_sha256": (
                    self.outer_unit.outer_units_manifest_sha256
                ),
            }
        return payload

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def rollout_count(self) -> int:
        return self.training.environment_steps // (
            self.environment.num_envs * self.training.unroll_length
        )

    def assert_reference_matches_outer_unit(self, unit: OuterTrainingUnit) -> None:
        expected = VQBCCheckpointRef.from_official_source(unit.backbone)
        if self.backbone_init != expected:
            raise ValueError(
                "The reference policy must be this outer unit's own official backbone."
            )
        if self.outer_unit is None or self.outer_unit.outer_unit_id != unit.outer_unit_id:
            raise ValueError("The resolved config is bound to another outer training unit.")


@dataclass(frozen=True, slots=True)
class VQBCFormalTemplate:
    """Source-free formal template resolved through path_c_outer_units_v2."""

    environment: VQBCEnvironmentConfig
    model: VQBCModelConfig
    training: VQBCTrainingConfig
    kl_control: VQBCKLConfig
    evaluation: VQBCEvaluationConfig
    output_root: Path

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> "VQBCFormalTemplate":
        payload = _mapping(payload, "path_c_model_v4_1 formal template")
        fields = {
            "schema_version",
            "run_kind",
            "scientific_readout_allowed",
            "environment",
            "partner_sampling",
            "model",
            "training",
            "kl_control",
            "evaluation",
            "output_root",
        }
        _fields(payload, allowed=fields, required=fields, name="v4 formal template")
        if (
            payload["schema_version"] != VQBC_SCHEMA_VERSION
            or payload["run_kind"] != "formal"
            or payload["scientific_readout_allowed"] is not False
        ):
            raise ValueError("The source-free fourth-model template must be non-claim formal.")
        sampling = _mapping(payload["partner_sampling"], "partner_sampling")
        _fields(
            sampling,
            allowed={"schedule", "include_frozen_current_policy"},
            required={"schedule", "include_frozen_current_policy"},
            name="partner_sampling",
        )
        if (
            sampling["schedule"] != VQBC_PARTNER_SCHEDULE
            or sampling["include_frozen_current_policy"] is not True
        ):
            raise ValueError("The formal partner schedule changed.")
        root = Path(base_dir).resolve()
        template = cls(
            environment=VQBCEnvironmentConfig.from_mapping(payload["environment"]),
            model=VQBCModelConfig.from_mapping(payload["model"]),
            training=VQBCTrainingConfig.from_mapping(payload["training"]),
            kl_control=VQBCKLConfig.from_mapping(payload["kl_control"]),
            evaluation=VQBCEvaluationConfig.from_mapping(payload["evaluation"]),
            output_root=_path(payload["output_root"], base_dir=root, name="output_root"),
        )
        if (
            template.environment.num_envs != VQBC_FORMAL_NUM_ENVS
            or template.training.environment_steps
            != VQBC_FORMAL_ENVIRONMENT_STEPS
        ):
            raise ValueError("The source-free formal template has a non-formal budget.")
        return template

    def resolve(
        self, manifest: OuterUnitsManifest, *, outer_unit_id: int
    ) -> VQBCConfig:
        if manifest.schema_version != FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION:
            raise ValueError("The fourth model requires path_c_outer_units_v2.")
        if manifest.layout != self.environment.layout:
            raise ValueError("The template and outer-unit manifest layouts differ.")
        unit = manifest.unit(outer_unit_id)
        members = []
        for source in (unit.backbone, *unit.partners):
            for snapshot in source.checkpoint_history:
                members.append(
                    VQBCPartnerMember.from_snapshot(
                        source=source,
                        snapshot=snapshot,
                        member_id=(
                            f"{source.member_id}_checkpoint_"
                            f"{snapshot.checkpoint_index}"
                        ),
                    )
                )
        config = VQBCConfig(
            schema_version=VQBC_SCHEMA_VERSION,
            run_kind="formal",
            scientific_readout_allowed=False,
            environment=self.environment,
            backbone_init=VQBCCheckpointRef.from_official_source(unit.backbone),
            partner_sampling=VQBCPartnerSamplingConfig(
                schedule=VQBC_PARTNER_SCHEDULE,
                include_frozen_current_policy=True,
                members=tuple(members),
            ),
            model=self.model,
            training=self.training,
            kl_control=self.kl_control,
            evaluation=self.evaluation,
            seeds=VQBCSeeds(**unit.seeds.to_mapping()),
            output_root=self.output_root / f"outer_unit_{unit.outer_unit_id:02d}",
            outer_unit=VQBCOuterUnitBinding(
                outer_unit_id=unit.outer_unit_id,
                outer_units_manifest_path=manifest.path,
                outer_units_manifest_sha256=manifest.sha256,
            ),
        )
        config._validate_cross_fields()
        config.assert_reference_matches_outer_unit(unit)
        return config


def _checkpoint_mapping(reference: VQBCCheckpointRef) -> dict[str, Any]:
    return {
        "checkpoint_path": str(reference.checkpoint_path),
        "flax_weights_sha256": reference.flax_weights_sha256,
        "training_run_id": reference.training_run_id,
        "launch_config_path": str(reference.launch_config_path),
        "launch_config_sha256": reference.launch_config_sha256,
    }


def _outer_unit_binding(
    payload: Mapping[str, Any], *, base_dir: Path
) -> VQBCOuterUnitBinding:
    payload = _mapping(payload, "outer_unit")
    fields = {
        "outer_unit_id",
        "outer_units_manifest_path",
        "outer_units_manifest_sha256",
    }
    _fields(payload, allowed=fields, required=fields, name="outer_unit")
    return VQBCOuterUnitBinding(
        outer_unit_id=_nonnegative_int(
            payload["outer_unit_id"], "outer_unit.outer_unit_id"
        ),
        outer_units_manifest_path=_path(
            payload["outer_units_manifest_path"],
            base_dir=base_dir,
            name="outer_unit.outer_units_manifest_path",
        ),
        outer_units_manifest_sha256=_sha256(
            payload["outer_units_manifest_sha256"],
            "outer_unit.outer_units_manifest_sha256",
        ),
    )


def load_vqbc_config(path: str | Path) -> VQBCConfig:
    import yaml

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return VQBCConfig.from_mapping(payload, base_dir=config_path.parent)


def load_vqbc_formal_template(path: str | Path) -> VQBCFormalTemplate:
    import yaml

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return VQBCFormalTemplate.from_mapping(payload, base_dir=config_path.parent)


__all__ = [
    "VQBC_ACTION_COUNT",
    "VQBC_DEPLOYMENT_MODES",
    "VQBC_DEVELOPMENT_ENVIRONMENT_STEPS",
    "VQBC_DEVELOPMENT_MINIBATCHES",
    "VQBC_FORMAL_ENVIRONMENT_STEPS",
    "VQBC_FORMAL_MINIBATCHES",
    "VQBC_RESPONSE_COUNT",
    "VQBC_SCHEMA_VERSION",
    "VQBC_SLOT_COUNT",
    "VQBC_TERMINAL_RESPONSE",
    "VQBCCheckpointRef",
    "VQBCConfig",
    "VQBCEvaluationConfig",
    "VQBCFormalTemplate",
    "VQBCKLConfig",
    "VQBCModelConfig",
    "VQBCPartnerMember",
    "VQBCPartnerSamplingConfig",
    "VQBCSeeds",
    "VQBCTrainingConfig",
    "load_vqbc_config",
    "load_vqbc_formal_template",
]
