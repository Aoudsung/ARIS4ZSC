"""Single configuration authority for unified DELTA-ZSC.

Only three values define the scientific method:

* ``latent_components``: capacity of the exchangeable coordination-mode model;
* ``continuation_horizon``: the registered decision-equivalence estimand;
* ``adaptation_kl_budget``: the maximum deployment policy deviation from the
  base policy.

Network widths, optimizer settings, sample counts, and reporting budgets are
engineering or statistical-design fields.  They are deliberately separated
from the method section and never enter the paper as independent mechanisms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_VERSION = 1
METHOD_VERSION = "delta_joint_response_decision_bayes_v1"
CHECKPOINT_SCHEMA_VERSION = 1
DEPLOYMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    layout: str
    agent_view_size: int
    episode_steps: int
    indicate_successful_delivery: bool
    negative_rewards: bool
    random_agent_positions: bool
    sample_recipe_on_delivery: bool


@dataclass(frozen=True, slots=True)
class MethodConfig:
    latent_components: int
    continuation_horizon: int
    adaptation_kl_budget: float


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    task_hidden_dim: int
    instant_partner_dim: int
    latent_hidden_dim: int
    response_hidden_dim: int
    action_embedding_dim: int


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float
    update_epochs: int
    minibatches: int
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_weight: float
    entropy_weight: float
    gradient_clip_norm: float
    adam_epsilon: float
    normalize_advantages: bool


@dataclass(frozen=True, slots=True)
class LatentOptimizerConfig:
    learning_rate: float
    gradient_clip_norm: float
    adam_epsilon: float
    updates_per_rollout: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    num_envs: int
    rollout_length: int
    total_environment_steps: int
    anchor_interval_updates: int
    anchor_states: int
    fit_replicas: int
    evaluation_replicas: int


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    evaluate_both_roles: bool
    bootstrap_replicates: int
    minimum_effect: float


@dataclass(frozen=True, slots=True)
class UnifiedConfig:
    version: int
    environment: EnvironmentConfig
    method: MethodConfig
    architecture: ArchitectureConfig
    ppo: PPOConfig
    latent_optimizer: LatentOptimizerConfig
    data: DataConfig
    evaluation: EvaluationConfig

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()



def _exact_fields(
    payload: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} fields differ; missing={missing}, extra={extra}."
        )
    return payload


def validate_config(config: UnifiedConfig) -> None:
    if config.version != CONFIG_VERSION:
        raise ValueError(f"Unified DELTA config version must be {CONFIG_VERSION}.")
    if config.environment.layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError("Unified DELTA supports the two Official test-time layouts.")
    if config.environment.agent_view_size != 2:
        raise ValueError("The registered benchmark uses view radius two.")
    if config.environment.episode_steps != 400:
        raise ValueError("The registered benchmark uses 400-step episodes.")
    if not all(
        (
            config.environment.indicate_successful_delivery,
            config.environment.negative_rewards,
            config.environment.random_agent_positions,
            config.environment.sample_recipe_on_delivery,
        )
    ):
        raise ValueError("Environment flags differ from the Official protocol.")

    method = config.method
    if method.latent_components not in {2, 4, 8}:
        raise ValueError("Latent components are restricted to K in {2,4,8}.")
    if not 1 <= method.continuation_horizon <= config.environment.episode_steps:
        raise ValueError("Continuation horizon is outside the episode.")
    if not 0.0 < method.adaptation_kl_budget <= 0.25:
        raise ValueError("Adaptation KL budget must lie in (0,0.25].")

    architecture = config.architecture
    for name in architecture.__dataclass_fields__:
        if int(getattr(architecture, name)) <= 0:
            raise ValueError(f"Architecture field {name} must be positive.")

    ppo = config.ppo
    if not 0.0 < ppo.learning_rate:
        raise ValueError("PPO learning rate must be positive.")
    if ppo.update_epochs <= 0 or ppo.minibatches <= 0:
        raise ValueError("PPO epochs and minibatches must be positive.")
    if not 0.0 < ppo.gamma <= 1.0 or not 0.0 <= ppo.gae_lambda <= 1.0:
        raise ValueError("PPO discount parameters are invalid.")
    if not 0.0 < ppo.clip_epsilon < 1.0:
        raise ValueError("PPO clip epsilon must lie in (0,1).")
    if min(
        ppo.value_weight,
        ppo.entropy_weight,
        ppo.gradient_clip_norm,
        ppo.adam_epsilon,
    ) < 0.0:
        raise ValueError("PPO coefficients must be non-negative.")

    latent = config.latent_optimizer
    if min(
        latent.learning_rate,
        latent.gradient_clip_norm,
        latent.adam_epsilon,
        latent.updates_per_rollout,
    ) <= 0:
        raise ValueError("Latent optimizer fields must be positive.")

    data = config.data
    if min(
        data.num_envs,
        data.rollout_length,
        data.total_environment_steps,
        data.anchor_interval_updates,
        data.anchor_states,
        data.fit_replicas,
        data.evaluation_replicas,
    ) <= 0:
        raise ValueError("Data and anchor budgets must be positive.")
    rollout_steps = data.num_envs * data.rollout_length
    if data.total_environment_steps % rollout_steps:
        raise ValueError("Training budget must contain whole vector rollouts.")
    if data.num_envs % ppo.minibatches:
        raise ValueError("Environment lanes must divide exactly into PPO minibatches.")

    evaluation = config.evaluation
    if min(
        evaluation.episodes_per_pairing,
        evaluation.bootstrap_replicates,
        evaluation.minimum_effect,
    ) <= 0:
        raise ValueError("Evaluation fields must be positive.")


def load_config(path: str | Path) -> UnifiedConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    top = _exact_fields(
        raw,
        {
            "version",
            "environment",
            "method",
            "architecture",
            "ppo",
            "latent_optimizer",
            "data",
            "evaluation",
        },
        "unified DELTA configuration",
    )
    config = UnifiedConfig(
        version=int(top["version"]),
        environment=EnvironmentConfig(
            **_exact_fields(
                top["environment"],
                set(EnvironmentConfig.__dataclass_fields__),
                "environment",
            )
        ),
        method=MethodConfig(
            **_exact_fields(
                top["method"], set(MethodConfig.__dataclass_fields__), "method"
            )
        ),
        architecture=ArchitectureConfig(
            **_exact_fields(
                top["architecture"],
                set(ArchitectureConfig.__dataclass_fields__),
                "architecture",
            )
        ),
        ppo=PPOConfig(
            **_exact_fields(
                top["ppo"], set(PPOConfig.__dataclass_fields__), "ppo"
            )
        ),
        latent_optimizer=LatentOptimizerConfig(
            **_exact_fields(
                top["latent_optimizer"],
                set(LatentOptimizerConfig.__dataclass_fields__),
                "latent_optimizer",
            )
        ),
        data=DataConfig(
            **_exact_fields(
                top["data"], set(DataConfig.__dataclass_fields__), "data"
            )
        ),
        evaluation=EvaluationConfig(
            **_exact_fields(
                top["evaluation"],
                set(EvaluationConfig.__dataclass_fields__),
                "evaluation",
            )
        ),
    )
    validate_config(config)
    return config


__all__ = [
    "ArchitectureConfig",
    "CHECKPOINT_SCHEMA_VERSION",
    "CONFIG_VERSION",
    "DEPLOYMENT_SCHEMA_VERSION",
    "DataConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LatentOptimizerConfig",
    "METHOD_VERSION",
    "MethodConfig",
    "PPOConfig",
    "UnifiedConfig",
    "load_config",
    "validate_config",
]
