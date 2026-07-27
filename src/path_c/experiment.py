"""Experiment specification for the active Path C implementation.

The project has one explicit experiment description.  It is used by training,
checkpoint resume, deployment, and evaluation, so those paths cannot silently
interpret the same files differently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .method import DEPLOYMENT_MODES

CONFIG_VERSION = 2
METHOD_VERSION = "path_c_v4_2_control_memory_r1"
MANIFEST_VERSION = 1
POPULATION_VERSION = 1
RUN_KINDS = ("development", "formal")
LAYOUTS = ("test_time_simple", "test_time_wide")


@dataclass(frozen=True, slots=True)
class RunBudget:
    num_envs: int
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int


RUN_BUDGETS: Mapping[str, RunBudget] = {
    "development": RunBudget(
        num_envs=32,
        environment_steps=1_228_800,
        minibatches_per_epoch=8,
        checkpoint_interval_environment_steps=102_400,
    ),
    "formal": RunBudget(
        num_envs=250,
        environment_steps=11_000_000,
        minibatches_per_epoch=50,
        checkpoint_interval_environment_steps=100_000,
    ),
}


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    layout: str
    agent_view_size: int
    indicate_successful_delivery: bool
    episode_steps: int
    num_envs: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int
    action_embedding_dim: int
    slot_count: int
    response_count: int
    action_count: int
    prior_scale: float
    log_standard_deviation_minimum: float
    log_standard_deviation_maximum: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    environment_steps: int
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
    checkpoint_interval_environment_steps: int


@dataclass(frozen=True, slots=True)
class KLConfig:
    target_per_step: float
    initial_temperature: float
    dual_learning_rate: float
    minimum_temperature: float
    maximum_temperature: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    deployment_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    total_timesteps: int
    reward_shaping_horizon: int
    checkpoint_progress: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_kind: str
    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    kl: KLConfig
    evaluation: EvaluationConfig
    upstream: UpstreamConfig

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(
            json.dumps({"version": CONFIG_VERSION, **asdict(self)}, sort_keys=True)
        )


@dataclass(frozen=True, slots=True)
class TrainingUnit:
    outer_unit_id: int
    reference_checkpoint: Path
    partner_checkpoints: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class TrainingUnitManifest:
    layout: str
    units: tuple[TrainingUnit, ...]

    def unit(self, outer_unit_id: int) -> TrainingUnit:
        matches = [unit for unit in self.units if unit.outer_unit_id == outer_unit_id]
        if len(matches) != 1:
            raise ValueError(f"Training unit {outer_unit_id} is absent or duplicated.")
        return matches[0]


@dataclass(frozen=True, slots=True)
class PopulationEntry:
    outer_unit_id: int
    run_directory: Path


@dataclass(frozen=True, slots=True)
class Population:
    name: str
    layout: str
    evaluation_kind: str
    entries: tuple[PopulationEntry, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": POPULATION_VERSION,
            "name": self.name,
            "layout": self.layout,
            "evaluation_kind": self.evaluation_kind,
            "policies": [
                {
                    "outer_unit_id": entry.outer_unit_id,
                    "run_directory": str(entry.run_directory),
                }
                for entry in self.entries
            ],
        }


def _exact_fields(payload: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{location} must be a mapping.")
    present = set(payload)
    if present != expected:
        raise ValueError(
            f"{location} fields differ: missing={sorted(expected - present)}, "
            f"unknown={sorted(present - expected)}."
        )
    return payload


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: str | Path, *, run_kind: str) -> RunConfig:
    """Load one layout config and apply the registered development/formal budget."""

    import yaml

    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {RUN_KINDS}.")
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = _exact_fields(
        payload,
        {"version", "environment", "model", "training", "kl", "evaluation", "upstream"},
        "configuration",
    )
    if int(payload["version"]) != CONFIG_VERSION:
        raise ValueError(f"The active Path C configuration version is {CONFIG_VERSION}.")

    environment_payload = _exact_fields(
        payload["environment"],
        {"layout", "agent_view_size", "indicate_successful_delivery", "episode_steps"},
        "environment",
    )
    model_payload = _exact_fields(
        payload["model"], set(ModelConfig.__dataclass_fields__), "model"
    )
    training_payload = _exact_fields(
        payload["training"],
        {
            "update_epochs",
            "bellman_learning_rate",
            "outcome_learning_rate",
            "gradient_clip_norm",
            "gamma",
            "responsibility_temperature",
            "bootstrap_probability",
            "polyak_coefficient",
            "codebook_decay",
            "code_replacement_rollouts",
        },
        "training",
    )
    kl_payload = _exact_fields(payload["kl"], set(KLConfig.__dataclass_fields__), "kl")
    evaluation_payload = _exact_fields(
        payload["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation"
    )
    upstream_payload = _exact_fields(
        payload["upstream"], set(UpstreamConfig.__dataclass_fields__), "upstream"
    )

    budget = RUN_BUDGETS[run_kind]
    config = RunConfig(
        run_kind=run_kind,
        environment=EnvironmentConfig(
            **environment_payload,
            num_envs=budget.num_envs,
        ),
        model=ModelConfig(**model_payload),
        training=TrainingConfig(
            **training_payload,
            environment_steps=budget.environment_steps,
            minibatches_per_epoch=budget.minibatches_per_epoch,
            checkpoint_interval_environment_steps=(
                budget.checkpoint_interval_environment_steps
            ),
        ),
        kl=KLConfig(**kl_payload),
        evaluation=EvaluationConfig(
            episodes_per_pairing=int(evaluation_payload["episodes_per_pairing"]),
            deployment_modes=tuple(evaluation_payload["deployment_modes"]),
        ),
        upstream=UpstreamConfig(
            total_timesteps=int(upstream_payload["total_timesteps"]),
            reward_shaping_horizon=int(upstream_payload["reward_shaping_horizon"]),
            checkpoint_progress=tuple(upstream_payload["checkpoint_progress"]),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    if config.run_kind not in RUN_KINDS:
        raise ValueError("Unknown run kind.")
    if config.environment.layout not in LAYOUTS:
        raise ValueError(f"Unknown OvercookedV2 layout: {config.environment.layout}")
    if (
        config.environment.agent_view_size != 2
        or not config.environment.indicate_successful_delivery
        or config.environment.episode_steps != 400
    ):
        raise ValueError("The registered environment uses view radius two and 400-step episodes.")
    if config.model.hidden_dim != 128:
        raise ValueError("Path C control memory must match the official 128-wide recurrent feature.")
    if config.model.action_count != 6:
        raise ValueError("JaxMARL OvercookedV2 exposes six actions.")
    if config.model.slot_count < 2 or config.model.response_count < 2:
        raise ValueError("The method needs multiple slots and response codes.")
    if config.model.log_standard_deviation_minimum >= config.model.log_standard_deviation_maximum:
        raise ValueError("The outcome standard-deviation bounds are reversed.")
    if config.training.update_epochs <= 0:
        raise ValueError("Training epochs must be positive.")
    if min(
        config.training.bellman_learning_rate,
        config.training.outcome_learning_rate,
        config.training.gradient_clip_norm,
        config.training.responsibility_temperature,
    ) <= 0.0:
        raise ValueError("Training rates, clipping, and responsibility temperature must be positive.")
    if not 0.0 < config.training.gamma <= 1.0:
        raise ValueError("The return discount must lie in (0, 1].")
    for name, value in (
        ("bootstrap_probability", config.training.bootstrap_probability),
        ("polyak_coefficient", config.training.polyak_coefficient),
        ("codebook_decay", config.training.codebook_decay),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    rollout_steps = config.environment.num_envs * config.environment.episode_steps
    if config.training.environment_steps % rollout_steps:
        raise ValueError("Training must contain whole vectorized episodes.")
    if config.environment.num_envs % config.training.minibatches_per_epoch:
        raise ValueError("Environment lanes must divide into whole minibatches.")
    if config.training.checkpoint_interval_environment_steps % rollout_steps:
        raise ValueError("Checkpoint intervals must align with complete vector rollouts.")
    if config.evaluation.episodes_per_pairing != 500:
        raise ValueError("Standard evaluation uses 500 episodes per pairing.")
    if config.evaluation.deployment_modes != DEPLOYMENT_MODES:
        raise ValueError("Evaluation must contain the four registered deployment modes.")
    if (
        config.kl.target_per_step < 0.0
        or config.kl.initial_temperature <= 0.0
        or config.kl.dual_learning_rate <= 0.0
        or config.kl.minimum_temperature <= 0.0
        or config.kl.maximum_temperature < config.kl.minimum_temperature
    ):
        raise ValueError("Kullback–Leibler temperature settings are invalid.")
    if (
        config.upstream.total_timesteps != 30_000_000
        or config.upstream.reward_shaping_horizon != 15_000_000
        or config.upstream.checkpoint_progress != (0.0, 0.5, 1.0)
    ):
        raise ValueError("The official upstream schedule differs from the registered protocol.")


def load_training_unit_manifest(
    path: str | Path, *, expected_layout: str, run_kind: str
) -> TrainingUnitManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = _exact_fields(payload, {"version", "layout", "units"}, "unit manifest")
    if int(payload["version"]) != MANIFEST_VERSION:
        raise ValueError("Unknown unit-manifest version.")
    if payload["layout"] != expected_layout:
        raise ValueError("Unit manifest and configuration layouts differ.")
    if not isinstance(payload["units"], Sequence):
        raise ValueError("Unit manifest units must be a sequence.")

    units = []
    for index, raw in enumerate(payload["units"]):
        raw = _exact_fields(
            raw,
            {"outer_unit_id", "reference_checkpoint", "partner_checkpoints"},
            f"unit[{index}]",
        )
        partners_raw = raw["partner_checkpoints"]
        if not isinstance(partners_raw, Sequence) or isinstance(partners_raw, (str, bytes)):
            raise ValueError("partner_checkpoints must be a sequence.")
        reference = _resolve(manifest_path.parent, raw["reference_checkpoint"])
        partners = tuple(_resolve(manifest_path.parent, item) for item in partners_raw)
        if not partners or len(set(partners)) != len(partners) or reference in partners:
            raise ValueError("Each unit needs distinct partners and a separate reference.")
        units.append(
            TrainingUnit(
                outer_unit_id=int(raw["outer_unit_id"]),
                reference_checkpoint=reference,
                partner_checkpoints=partners,
            )
        )
    identifiers = [unit.outer_unit_id for unit in units]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Outer-unit identifiers must be unique.")
    if run_kind == "formal" and identifiers != list(range(10)):
        raise ValueError("Formal training requires ordered outer units 0 through 9.")
    if run_kind == "formal":
        all_references = [unit.reference_checkpoint for unit in units]
        all_partners = [path for unit in units for path in unit.partner_checkpoints]
        if len(set(all_references)) != len(all_references):
            raise ValueError("Formal outer units cannot share reference checkpoints.")
        if len(set(all_partners)) != len(all_partners):
            raise ValueError("Formal outer units cannot silently reuse partner checkpoints.")
        if set(all_references) & set(all_partners):
            raise ValueError(
                "A formal reference checkpoint cannot reappear as another unit's partner."
            )
    return TrainingUnitManifest(layout=expected_layout, units=tuple(units))


def load_population(path: str | Path) -> Population:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = _exact_fields(
        payload,
        {"version", "name", "layout", "evaluation_kind", "policies"},
        "population manifest",
    )
    if int(payload["version"]) != POPULATION_VERSION:
        raise ValueError("Unknown population-manifest version.")
    if payload["layout"] not in LAYOUTS:
        raise ValueError("Population manifest has an unknown layout.")
    if payload["evaluation_kind"] not in {
        "standard_matrix",
        "response_contrast",
        "development_diagnostic",
    }:
        raise ValueError("Unknown evaluation kind.")
    if not isinstance(payload["policies"], Sequence):
        raise ValueError("Population policies must be a sequence.")
    entries = []
    for index, raw in enumerate(payload["policies"]):
        raw = _exact_fields(raw, {"outer_unit_id", "run_directory"}, f"policy[{index}]")
        entries.append(
            PopulationEntry(
                outer_unit_id=int(raw["outer_unit_id"]),
                run_directory=_resolve(manifest_path.parent, raw["run_directory"]),
            )
        )
    if payload["evaluation_kind"] == "development_diagnostic":
        if len(entries) != 1 or entries[0].outer_unit_id != 0:
            raise ValueError(
                "A development diagnostic contains only outer unit zero."
            )
    else:
        if (
            len(entries) != 10
            or [entry.outer_unit_id for entry in entries] != list(range(10))
        ):
            raise ValueError(
                "A standard population contains ordered outer units 0 through 9."
            )
        if len({entry.run_directory for entry in entries}) != 10:
            raise ValueError(
                "Each outer unit must use a distinct training run directory."
            )
    return Population(
        name=str(payload["name"]),
        layout=str(payload["layout"]),
        evaluation_kind=str(payload["evaluation_kind"]),
        entries=tuple(entries),
    )


def write_population(path: str | Path, population: Population) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(population.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "CONFIG_VERSION",
    "EnvironmentConfig",
    "EvaluationConfig",
    "KLConfig",
    "MANIFEST_VERSION",
    "METHOD_VERSION",
    "ModelConfig",
    "POPULATION_VERSION",
    "Population",
    "PopulationEntry",
    "RUN_BUDGETS",
    "RUN_KINDS",
    "RunConfig",
    "TrainingConfig",
    "TrainingUnit",
    "TrainingUnitManifest",
    "UpstreamConfig",
    "load_config",
    "load_population",
    "load_training_unit_manifest",
    "validate_config",
    "write_population",
]
