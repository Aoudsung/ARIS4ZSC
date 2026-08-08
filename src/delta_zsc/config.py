"""Single configuration authority for the unified DELTA-ZSC method.

Only three fields are scientific method choices:

* ``latent_components`` K;
* ``continuation_horizon`` H;
* ``adaptation_kl_budget`` delta.

Network widths, optimizers, data budgets, and statistical sample sizes remain
engineering/evaluation settings and are deliberately separated from the method
contract.  There are no auxiliary-loss weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_VERSION = 3
METHOD_VERSION = "delta_episode_static_centered_residual_delayed_exact_voi_v4"
# Schema version 2 dropped every checksum/fingerprint field: artifacts are
# identified by run id and path, never by a digest.
CHECKPOINT_SCHEMA_VERSION = 4
MANIFEST_VERSION = 2
FORMAL_METHOD_LABEL = "delta-active"
OFFICIAL_BASELINE_METHODS = (
    "sp",
    "state-augmented",
    "op",
    "fcp",
    "ippo-large",
)
OFFICIAL_SOURCE_COMMIT = "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e"
OFFICIAL_PROTOCOL_VERSION = "overcooked_v2_iclr2025_5ce1707_v1"
OFFICIAL_CORRECT_DELIVERY_REWARD = 20.0
OFFICIAL_ACTION_COUNT = 6
OFFICIAL_EPISODE_STEPS = 400
OFFICIAL_EVALUATION_ROOT_SEED = 0
OFFICIAL_TRAINING_ROOT_SEED = 42
OFFICIAL_TRAINING_RUN_COUNT = 10
OFFICIAL_ROLLOUT_LENGTH = 256
OFFICIAL_SP_TOTAL_TIMESTEPS = 30_000_000
OFFICIAL_OP_TOTAL_TIMESTEPS = 50_000_000
OFFICIAL_SP_NUM_ENVS = 256
OFFICIAL_OP_NUM_ENVS = 64
OFFICIAL_NUM_MINIBATCHES = 64
OFFICIAL_UPDATE_EPOCHS = 4

RUN_KINDS = ("mechanical", "development", "formal")
LAYOUTS = ("test_time_simple", "test_time_wide")
METHOD_VARIANTS = (
    "history_rnn",
    "base",
    "response_only",
    "delta_passive",
    "delta_active",
)


@dataclass(frozen=True, slots=True)
class RunBudget:
    num_envs: int
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int


FORMAL_NUM_ENVS = 128
"""Formal vector width, reduced from the registered 256.

At 256 the CRN anchor kernel asks CUDA for 131072 bytes of shared memory per
block and the L40 this runs on offers 101376, so the anchor update fails to
compile -- reproduced by cuda-preflight and unaffected by
``--xla_gpu_enable_triton_gemm=false`` or ``--xla_gpu_autotune_level=0``.  A
bisection over the anchor kernel put the ceiling between 192 and 256.

128 is the largest width that clears both that ceiling and every registered
divisibility rule: 29_949_952 total steps and the 1_048_576 anchor interval are
whole multiples of 128*256, and 128 divides the 64 minibatches per epoch.  Every
other registered number is therefore unchanged; only the width moves.

This is a deviation from the registered protocol, not a neutral engineering
knob.  Halving the width doubles the update count (457 -> 914) and halves the
per-update sample, so the optimisation trajectory differs even at identical
total steps.  Results produced this way must not be reported as the registered
Official protocol.  Restore 256 on hardware with >=128 KiB of shared memory per
block (Hopper and later).
"""

RUN_BUDGETS: Mapping[str, RunBudget] = {
    "mechanical": RunBudget(4, 1_024, 1, 1_024),
    "development": RunBudget(32, 1_228_800, 8, 98_304),
    "formal": RunBudget(FORMAL_NUM_ENVS, 29_949_952, 64, 29_949_952),
}


@dataclass(frozen=True, slots=True)
class MethodConfig:
    latent_components: int
    continuation_horizon: int
    adaptation_kl_budget: float


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    layout: str
    agent_view_size: int
    indicate_successful_delivery: bool
    negative_rewards: bool
    random_agent_positions: bool
    sample_recipe_on_delivery: bool
    episode_steps: int
    num_envs: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    task_hidden_dim: int
    task_embedding_dim: int
    instant_partner_dim: int
    latent_hidden_dim: int
    latent_embedding_dim: int
    action_embedding_dim: int


@dataclass(frozen=True, slots=True)
class PPOConfig:
    update_epochs: int
    learning_rate: float
    gradient_clip_norm: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_clip_epsilon: float
    entropy_weight: float
    value_weight: float
    normalize_advantages: bool
    lr_warmup_fraction: float
    anneal_learning_rate: bool
    adam_epsilon: float


@dataclass(frozen=True, slots=True)
class LatentOptimizerConfig:
    learning_rate: float
    gradient_clip_norm: float
    adam_epsilon: float


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    enabled: bool
    interval_environment_steps: int
    states_per_trigger: int
    fit_replicas: int
    evaluation_replicas: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    rollout_length: int
    extra_ppo_environment_steps: int
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int


@dataclass(frozen=True, slots=True)
class PartnerPoolConfig:
    checkpoint_stages: tuple[float, ...]
    mechanism_uniform_sampling: bool
    heuristic_family_test_only: bool


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    bootstrap_replicates: int
    evaluation_seed: int
    evaluate_both_roles: bool
    minimum_ego_runs: int
    minimum_partner_runs_per_mechanism: int
    minimum_effect: float
    one_sided_alpha: float


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    total_timesteps: int
    reward_shaping_horizon: int
    checkpoint_progress: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OfficialProtocolConfig:
    protocol_version: str
    source_commit: str
    training_root_seed: int
    training_run_count: int
    evaluation_root_seed: int
    evaluation_episodes_per_pairing: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    version: int
    run_kind: str
    method_variant: str
    method: MethodConfig
    environment: EnvironmentConfig
    model: ModelConfig
    ppo: PPOConfig
    latent_optimizer: LatentOptimizerConfig
    anchors: AnchorConfig
    training: TrainingConfig
    partner_pool: PartnerPoolConfig
    evaluation: EvaluationConfig
    upstream: UpstreamConfig
    official_protocol: OfficialProtocolConfig

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)


_TOP_LEVEL_FIELDS = {
    "version",
    "method_variant",
    "method",
    "environment",
    "model",
    "ppo",
    "latent_optimizer",
    "anchors",
    "training",
    "partner_pool",
    "evaluation",
    "upstream",
    "official_protocol",
}


def _exact_fields(payload: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        observed = set(payload) if isinstance(payload, Mapping) else type(payload).__name__
        raise ValueError(
            f"{label} fields differ. expected={sorted(fields)!r}, observed={observed!r}"
        )
    return dict(payload)


def load_config(path: str | Path, *, run_kind: str) -> RunConfig:
    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {RUN_KINDS}.")
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload = _exact_fields(raw, _TOP_LEVEL_FIELDS, "configuration")
    if int(payload["version"]) != CONFIG_VERSION:
        raise ValueError(f"Active DELTA config version is {CONFIG_VERSION}.")
    budget = RUN_BUDGETS[run_kind]
    training_raw = _exact_fields(
        payload["training"], {"rollout_length", "extra_ppo_environment_steps"}, "training"
    )
    method_raw = _exact_fields(
        payload["method"], set(MethodConfig.__dataclass_fields__), "method"
    )
    environment_raw = _exact_fields(
        payload["environment"],
        set(EnvironmentConfig.__dataclass_fields__) - {"num_envs"},
        "environment",
    )
    model_raw = _exact_fields(payload["model"], set(ModelConfig.__dataclass_fields__), "model")
    ppo_raw = _exact_fields(payload["ppo"], set(PPOConfig.__dataclass_fields__), "ppo")
    latent_raw = _exact_fields(
        payload["latent_optimizer"],
        set(LatentOptimizerConfig.__dataclass_fields__),
        "latent_optimizer",
    )
    anchor_raw = _exact_fields(
        payload["anchors"], set(AnchorConfig.__dataclass_fields__), "anchors"
    )
    pool_raw = _exact_fields(
        payload["partner_pool"], set(PartnerPoolConfig.__dataclass_fields__), "partner_pool"
    )
    evaluation_raw = _exact_fields(
        payload["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation"
    )
    upstream_raw = _exact_fields(
        payload["upstream"], set(UpstreamConfig.__dataclass_fields__), "upstream"
    )
    protocol_raw = _exact_fields(
        payload["official_protocol"],
        set(OfficialProtocolConfig.__dataclass_fields__),
        "official_protocol",
    )
    config = RunConfig(
        version=CONFIG_VERSION,
        run_kind=run_kind,
        method_variant=str(payload["method_variant"]).lower(),
        method=MethodConfig(**method_raw),
        environment=EnvironmentConfig(**environment_raw, num_envs=budget.num_envs),
        model=ModelConfig(**model_raw),
        ppo=PPOConfig(**ppo_raw),
        latent_optimizer=LatentOptimizerConfig(**latent_raw),
        anchors=AnchorConfig(**anchor_raw),
        training=TrainingConfig(
            rollout_length=int(training_raw["rollout_length"]),
            extra_ppo_environment_steps=int(training_raw["extra_ppo_environment_steps"]),
            environment_steps=(
                budget.environment_steps
                + int(training_raw["extra_ppo_environment_steps"])
            ),
            minibatches_per_epoch=budget.minibatches_per_epoch,
            checkpoint_interval_environment_steps=budget.checkpoint_interval_environment_steps,
        ),
        partner_pool=PartnerPoolConfig(
            checkpoint_stages=tuple(float(v) for v in pool_raw["checkpoint_stages"]),
            mechanism_uniform_sampling=bool(pool_raw["mechanism_uniform_sampling"]),
            heuristic_family_test_only=bool(pool_raw["heuristic_family_test_only"]),
        ),
        evaluation=EvaluationConfig(**evaluation_raw),
        upstream=UpstreamConfig(
            total_timesteps=int(upstream_raw["total_timesteps"]),
            reward_shaping_horizon=int(upstream_raw["reward_shaping_horizon"]),
            checkpoint_progress=tuple(float(v) for v in upstream_raw["checkpoint_progress"]),
        ),
        official_protocol=OfficialProtocolConfig(**protocol_raw),
    )
    validate_config(config)
    return config


def run_config_from_mapping(payload: Mapping[str, Any]) -> RunConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("Resolved config must be a mapping.")
    config = RunConfig(
        version=int(payload["version"]),
        run_kind=str(payload["run_kind"]),
        method_variant=str(payload["method_variant"]),
        method=MethodConfig(**payload["method"]),
        environment=EnvironmentConfig(**payload["environment"]),
        model=ModelConfig(**payload["model"]),
        ppo=PPOConfig(**payload["ppo"]),
        latent_optimizer=LatentOptimizerConfig(**payload["latent_optimizer"]),
        anchors=AnchorConfig(**payload["anchors"]),
        training=TrainingConfig(**payload["training"]),
        partner_pool=PartnerPoolConfig(
            **{
                **payload["partner_pool"],
                "checkpoint_stages": tuple(payload["partner_pool"]["checkpoint_stages"]),
            }
        ),
        evaluation=EvaluationConfig(**payload["evaluation"]),
        upstream=UpstreamConfig(**{**payload["upstream"], "checkpoint_progress": tuple(payload["upstream"]["checkpoint_progress"])}),
        official_protocol=OfficialProtocolConfig(**payload["official_protocol"]),
    )
    validate_config(config)
    normalized = json.loads(json.dumps(config.to_mapping(), sort_keys=True))
    if normalized != dict(payload):
        raise ValueError("Resolved config does not round-trip exactly.")
    return config


def validate_config(config: RunConfig) -> None:
    if config.version != CONFIG_VERSION:
        raise ValueError("Config version differs.")
    if config.run_kind not in RUN_KINDS:
        raise ValueError("Unknown run kind.")
    if config.method_variant not in METHOD_VARIANTS:
        raise ValueError(f"method_variant must be one of {METHOD_VARIANTS}.")
    if config.run_kind == "formal" and config.method_variant != "delta_active":
        raise ValueError("The confirmatory method is DELTA-active.")
    if config.environment.layout not in LAYOUTS:
        raise ValueError("Unknown OvercookedV2 layout.")
    if config.environment.agent_view_size != 2:
        raise ValueError("Official protocol uses view radius two.")
    if config.environment.episode_steps != OFFICIAL_EPISODE_STEPS:
        raise ValueError("Official protocol uses 400-step episodes.")
    if not all(
        (
            config.environment.indicate_successful_delivery,
            config.environment.negative_rewards,
            config.environment.random_agent_positions,
            config.environment.sample_recipe_on_delivery,
        )
    ):
        raise ValueError("Environment flags differ from Official.")
    if config.method.latent_components not in (2, 4, 8):
        raise ValueError("K sensitivity is restricted to {2,4,8}.")
    if config.run_kind == "formal" and config.method.latent_components != 4:
        raise ValueError("Formal DELTA fixes K=4.")
    if not 1 <= config.method.continuation_horizon <= config.environment.episode_steps:
        raise ValueError("Continuation horizon is outside the episode.")
    if config.run_kind != "mechanical" and config.method.continuation_horizon != 128:
        raise ValueError("Registered development/formal continuation horizon is 128.")
    if not 0.0 <= config.method.adaptation_kl_budget <= 1.0:
        raise ValueError("Adaptation KL budget must lie in [0,1].")
    # K/H/delta remain in every variant's immutable identity even when a
    # control does not execute all three mechanisms.  This preserves exact
    # capacity/config matching without adding variant-specific fields.
    for name in (
        "task_hidden_dim",
        "task_embedding_dim",
        "instant_partner_dim",
        "latent_hidden_dim",
        "latent_embedding_dim",
        "action_embedding_dim",
    ):
        if int(getattr(config.model, name)) <= 0:
            raise ValueError(f"Model field {name} must be positive.")
    if config.ppo.update_epochs <= 0:
        raise ValueError("PPO update epochs must be positive.")
    if not 0.0 < config.ppo.gamma <= 1.0:
        raise ValueError("PPO gamma must lie in (0,1].")
    if not 0.0 <= config.ppo.gae_lambda <= 1.0:
        raise ValueError("GAE lambda must lie in [0,1].")
    if min(config.ppo.learning_rate, config.latent_optimizer.learning_rate) <= 0.0:
        raise ValueError("Optimizer learning rates must be positive.")
    if not 0.0 <= config.ppo.lr_warmup_fraction < 1.0:
        raise ValueError("PPO warm-up fraction must lie in [0,1).")
    if min(config.ppo.adam_epsilon, config.latent_optimizer.adam_epsilon) <= 0.0:
        raise ValueError("Adam epsilon values must be positive.")
    if min(config.ppo.clip_epsilon, config.ppo.value_clip_epsilon) <= 0.0:
        raise ValueError("PPO clipping radii must be positive.")
    if min(config.ppo.entropy_weight, config.ppo.value_weight) < 0.0:
        raise ValueError("PPO entropy/value weights must be non-negative.")
    if min(config.ppo.gradient_clip_norm, config.latent_optimizer.gradient_clip_norm) <= 0.0:
        raise ValueError("Gradient clip norms must be positive.")
    if config.training.rollout_length <= 0:
        raise ValueError("Rollout length must be positive.")
    if config.environment.num_envs % config.training.minibatches_per_epoch:
        raise ValueError("Environment lanes must divide into minibatches.")
    base_steps = config.training.environment_steps - config.training.extra_ppo_environment_steps
    rollout_steps = config.environment.num_envs * config.training.rollout_length
    if base_steps % rollout_steps:
        raise ValueError("Base training budget must contain whole vector rollouts.")
    if config.training.environment_steps % rollout_steps:
        raise ValueError("Total training budget must contain whole vector rollouts.")
    if config.training.checkpoint_interval_environment_steps % rollout_steps:
        raise ValueError("Checkpoint intervals must align with vector rollouts.")
    if config.anchors.enabled:
        if min(
            config.anchors.interval_environment_steps,
            config.anchors.states_per_trigger,
            config.anchors.fit_replicas,
            config.anchors.evaluation_replicas,
        ) <= 0:
            raise ValueError("Anchor sampling values must be positive.")
        if config.anchors.fit_replicas < OFFICIAL_ACTION_COUNT:
            raise ValueError(
                "Full-rank five-dimensional CRN covariance requires at least "
                f"{OFFICIAL_ACTION_COUNT} fit replicas."
            )
        if config.anchors.interval_environment_steps % rollout_steps:
            raise ValueError("Anchor interval must align with rollouts.")
    if tuple(config.partner_pool.checkpoint_stages) != (0.0, 0.5, 1.0):
        raise ValueError("Partner checkpoint stages are fixed at 0/0.5/1.")
    if not config.partner_pool.mechanism_uniform_sampling:
        raise ValueError("Training partner mechanisms must be sampled uniformly.")
    if not config.partner_pool.heuristic_family_test_only:
        raise ValueError("Heuristics remain test-only.")
    if min(
        config.evaluation.episodes_per_pairing,
        config.evaluation.bootstrap_replicates,
        config.evaluation.minimum_ego_runs,
        config.evaluation.minimum_partner_runs_per_mechanism,
    ) <= 0:
        raise ValueError("Evaluation sample sizes must be positive.")
    if not 0.0 < config.evaluation.one_sided_alpha < 1.0:
        raise ValueError("Evaluation one-sided alpha must lie in (0,1).")
    if config.evaluation.minimum_effect != OFFICIAL_CORRECT_DELIVERY_REWARD:
        raise ValueError("Material effect is one correct delivery (20 points).")
    if config.upstream.total_timesteps <= 0 or config.upstream.reward_shaping_horizon < 0:
        raise ValueError("Upstream budgets are invalid.")
    if tuple(config.upstream.checkpoint_progress) != (0.0, 0.5, 1.0):
        raise ValueError("Upstream checkpoint progress is fixed at 0/0.5/1.")
    if config.run_kind == "formal":
        formal_method = {
            "latent_components": 4,
            "continuation_horizon": 128,
            "adaptation_kl_budget": 0.04,
        }
        for name, expected in formal_method.items():
            if getattr(config.method, name) != expected:
                raise ValueError(f"Formal method field {name} must equal {expected!r}.")
        formal_ppo = {
            "update_epochs": 4,
            "learning_rate": 0.00025,
            "gradient_clip_norm": 0.25,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_clip_epsilon": 0.2,
            "entropy_weight": 0.01,
            "value_weight": 0.5,
            "lr_warmup_fraction": 0.05,
            "anneal_learning_rate": True,
            "adam_epsilon": 1.0e-5,
        }
        for name, expected in formal_ppo.items():
            if getattr(config.ppo, name) != expected:
                raise ValueError(f"Formal PPO field {name} must equal {expected!r}.")
        if (
            config.environment.num_envs != FORMAL_NUM_ENVS
            or config.training.rollout_length != OFFICIAL_ROLLOUT_LENGTH
            or config.training.environment_steps != 29_949_952
            or config.training.minibatches_per_epoch != OFFICIAL_NUM_MINIBATCHES
        ):
            raise ValueError("Formal vectorized training budget differs from registration.")
        if (
            not config.anchors.enabled
            or config.anchors.interval_environment_steps != 1_048_576
            or config.anchors.states_per_trigger != 16
            or config.anchors.fit_replicas != 8
            or config.anchors.evaluation_replicas != 8
        ):
            raise ValueError("Formal sparse decision-observation budget differs.")
        if (
            config.evaluation.episodes_per_pairing != 500
            or config.evaluation.bootstrap_replicates != 9_999
            or not config.evaluation.evaluate_both_roles
            or config.evaluation.minimum_ego_runs != 10
            or config.evaluation.minimum_partner_runs_per_mechanism != 4
            or config.evaluation.evaluation_seed != OFFICIAL_EVALUATION_ROOT_SEED
            or config.evaluation.one_sided_alpha != 0.05
        ):
            raise ValueError("Formal evaluation registration differs.")

    expected_protocol = {
        "protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "training_root_seed": OFFICIAL_TRAINING_ROOT_SEED,
        "training_run_count": OFFICIAL_TRAINING_RUN_COUNT,
        "evaluation_root_seed": OFFICIAL_EVALUATION_ROOT_SEED,
        "evaluation_episodes_per_pairing": config.evaluation.episodes_per_pairing,
    }
    for name, expected in expected_protocol.items():
        if getattr(config.official_protocol, name) != expected:
            raise ValueError(f"Official protocol field {name} differs.")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONFIG_VERSION",
    "LAYOUTS",
    "MANIFEST_VERSION",
    "METHOD_VARIANTS",
    "METHOD_VERSION",
    "OFFICIAL_ACTION_COUNT",
    "OFFICIAL_CORRECT_DELIVERY_REWARD",
    "OFFICIAL_EPISODE_STEPS",
    "OFFICIAL_EVALUATION_ROOT_SEED",
    "OFFICIAL_PROTOCOL_VERSION",
    "OFFICIAL_SOURCE_COMMIT",
    "OFFICIAL_TRAINING_ROOT_SEED",
    "OFFICIAL_TRAINING_RUN_COUNT",
    "RUN_BUDGETS",
    "RUN_KINDS",
    "RunConfig",
    "load_config",
    "run_config_from_mapping",
    "validate_config",
]
