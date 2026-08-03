"""Single configuration and manifest authority for DEPI (DELTA-ZSC foundation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_VERSION = 11
METHOD_VERSION = "delta_zsc_depi_three_object_bayes_coordination"
MANIFEST_VERSION = 2
OFFICIAL_PROTOCOL_VERSION = "overcooked_v2_iclr2025_5ce1707_v1"
OFFICIAL_SOURCE_COMMIT = "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e"
OFFICIAL_TRAINING_ROOT_SEED = 42
OFFICIAL_EVALUATION_ROOT_SEED = 0
OFFICIAL_TRAINING_RUN_COUNT = 10
ENGINEERING_SEED_INDEX = -1
OFFICIAL_EPISODES_PER_PAIRING = 500
OFFICIAL_ACTION_COUNT = 6
OFFICIAL_EPISODE_STEPS = 400
OFFICIAL_CORRECT_DELIVERY_REWARD = 20.0
OFFICIAL_ROLLOUT_LENGTH = 256
OFFICIAL_SP_TOTAL_TIMESTEPS = 30_000_000
OFFICIAL_OP_TOTAL_TIMESTEPS = 50_000_000
OFFICIAL_SP_NUM_ENVS = 256
OFFICIAL_OP_NUM_ENVS = 64
OFFICIAL_NUM_MINIBATCHES = 64
OFFICIAL_UPDATE_EPOCHS = 4
OFFICIAL_TRAINING_KEYS = (
    (1039196627, 2465224267),
    (1885534764, 892988520),
    (1592073730, 4208621316),
    (4195573804, 1556624894),
    (1378340893, 2340506400),
    (1683848162, 2527321690),
    (2857519579, 594117140),
    (3206027959, 287420602),
    (2042750619, 3456201790),
    (1462505072, 2580034575),
)
RUN_KINDS = ("mechanical", "development", "formal")
LAYOUTS = ("test_time_simple", "test_time_wide")
PARTNER_ROLES = (
    "owner_source",
    "generator_init_source",
    "development_support",
    "calibration",
    "confirmatory",
)


@dataclass(frozen=True, slots=True)
class RunBudget:
    num_envs: int
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int


RUN_BUDGETS: Mapping[str, RunBudget] = {
    "mechanical": RunBudget(4, 1_024, 1, 1_024),
    "development": RunBudget(32, 1_228_800, 8, 98_304),
    "formal": RunBudget(256, 29_949_952, 64, 29_949_952),
}


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
    capability_hidden_dim: int
    protocol_hidden_dim: int
    capability_dim: int
    protocol_components: int
    component_embedding_dim: int
    actor_hidden_dim: int
    critic_hidden_dim: int
    response_hidden_dim: int
    modulation_rank: int
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
    polyak_coefficient: float
    lr_warmup_fraction: float
    anneal_learning_rate: bool
    adam_epsilon: float


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Deprecated V6 loss section; retained for exactly one release cycle."""

    response_weight: float
    raw_q_weight: float
    counterfactual_weight: float
    decision_equivalence_weight: float
    q_policy_weight: float
    q_policy_temperature: float
    q_policy_gap_midpoint: float
    q_policy_gap_temperature: float
    q_policy_disagreement_temperature: float
    information_bottleneck_weight: float
    information_bottleneck_free_bits_per_dimension: float
    robust_generalist_weight: float
    policy_belief_gradient_scale: float
    belief_gradient_ema_decay: float
    decision_regret_weight_maximum: float
    decision_regret_schedule_midpoint: float
    decision_regret_schedule_temperature: float
    response_updates_per_outer_update: int
    raw_q_updates_per_outer_update: int


@dataclass(frozen=True, slots=True)
class LossV2Config:
    """Active DEPI objective weights and gates (METHOD_SPEC §3.1/§3.2/§3.4)."""

    signature_weight: float
    response_weight: float
    separation_weight: float
    combined_policy_kl_threshold: float
    rank_hinge_margin: float
    rank_hinge_advantage_gap: float
    separation_margin_scale: float


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    enabled: bool
    interval_updates: int
    ordinary_states: int
    matched_code_pairs: int
    fit_replicas: int
    continuation_horizon: int
    probe_steps: int
    replay_capacity: int
    replay_minibatch_size: int
    policy_kl_decay: float
    age_decay_updates: float
    minimum_replay_weight: float
    return_lower_bound: float
    return_upper_bound: float
    # §5.3 frozen comparator thresholds and the §6 extended M1 gate.
    observable_equivalent_accuracy_max: float
    decision_distinct_accuracy_min: float
    signature_distance_threshold: float
    m1_spearman_threshold: float
    m1_minimum_anchor_fraction: float

    # Read-only aliases used by optional audit utilities.
    @property
    def states_per_interval(self) -> int:
        return self.ordinary_states + 2 * self.matched_code_pairs

    @property
    def sampling_time_bins(self) -> int:
        return 1

    @property
    def sampling_regret_bins(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class PartnerGeneratorConfig:
    enabled: bool
    code_dim: int
    hidden_dim: int
    modulation_rank: int
    episodes_per_update: int
    episode_steps: int
    update_epochs: int
    environment_minibatches: int
    cvar_level: float
    target_polyak_coefficient: float
    maximum_generator_probability: float
    mixture_ramp_fraction: float
    competence_temperature: float
    cvar_ema_decay: float
    imitation_initial_weight: float
    brdiv_maximum_weight: float
    smoothness_weight: float
    lagrangian_learning_rate: float
    competence_multiplier_maximum: float
    kernel_bandwidth: float
    kernel_jitter: float
    codes_per_update: int
    # §7.2 shared-state diversity bank, code archive and backbone freeze.
    diversity_bank_size: int
    diversity_codes_per_update: int
    code_archive_capacity: int
    freeze_after_imitation: bool


@dataclass(frozen=True, slots=True)
class PartnerPoolConfig:
    """Static wide partner pool, the default B0–B2 partner distribution
    (METHOD_SPEC §7.3): SP/OP multi-seed runs x checkpoint stages, OP width
    variants, and a heuristic family held out exclusively for testing."""

    enabled: bool
    checkpoint_stages: tuple[float, ...]
    heuristic_family_test_only: bool
    family_uniform_sampling: bool

@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    enabled: bool
    alpha: float
    support_quantile: float
    minimum_run_count: int
    block_unit: str
    episodes_per_run: int
    anchors_per_run: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int
    rollout_length: int
    context_dropout_initial: float
    context_dropout_final: float
    owner_sources_per_seed: int
    generator_init_sources_per_mechanism: int
    development_runs_per_mechanism: int
    calibration_runs_per_mechanism: int

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    bootstrap_replicates: int
    evaluation_seed: int
    report_br_prox: bool
    br_prox_anchors_per_pairing: int
    br_prox_fit_replicas: int
    br_prox_evaluation_replicas: int
    br_prox_continuation_horizon: int
    evaluate_both_roles: bool
    one_sided_alpha: float
    minimum_ego_runs: int
    minimum_partner_runs_per_mechanism: int
    minimum_mechanisms: int


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
    run_kind: str
    environment: EnvironmentConfig
    model: ModelConfig
    ppo: PPOConfig
    loss: LossConfig
    loss_v2: LossV2Config
    anchors: AnchorConfig
    partner_generator: PartnerGeneratorConfig
    partner_pool: PartnerPoolConfig
    calibration: CalibrationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    upstream: UpstreamConfig
    official_protocol: OfficialProtocolConfig

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(
            json.dumps({"version": CONFIG_VERSION, **asdict(self)}, sort_keys=True)
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PartnerRun:
    run_id: str
    role: str
    checkpoint: Path
    checkpoint_sha256: str
    parent_training_run_id: str
    generation_mechanism: str
    seed: int
    seed_index: int | None
    jax_prng_key: tuple[int, int] | None
    owner_seed_index: int | None
    co_training_group_id: str | None
    partner_type_id: str | None


@dataclass(frozen=True, slots=True)
class PartnerManifest:
    layout: str
    runs: tuple[PartnerRun, ...]

    def by_role(self, role: str) -> tuple[PartnerRun, ...]:
        if role not in PARTNER_ROLES:
            raise ValueError(f"Unknown partner role: {role}")
        return tuple(run for run in self.runs if run.role == role)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "layout": self.layout,
            "runs": [
                {
                    **asdict(run),
                    "checkpoint": str(run.checkpoint),
                }
                for run in self.runs
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



def _sha256(path: Path) -> str:
    """Use the one canonical file/directory fingerprint implementation."""

    # Local import avoids the experiment/storage module initialization cycle.
    from .storage import sha256_path

    return sha256_path(path)


def official_training_key(seed_index: int) -> tuple[int, int]:
    """Return the exact indexed key from split(PRNGKey(42), 10)."""

    index = int(seed_index)
    if not 0 <= index < OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Official seed_index must lie in 0..9.")
    return OFFICIAL_TRAINING_KEYS[index]


def engineering_training_key() -> tuple[int, int]:
    """Named non-scientific key that cannot alias Official seed indices 0..9."""

    digest = hashlib.sha256(
        b"delta-zsc-v6/end-to-end-bayes-coordination/engineering-seed"
    ).digest()
    return (
        int.from_bytes(digest[:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    )


def official_training_domain_keys(seed_index: int) -> Mapping[str, tuple[int, int]]:
    """Derive registered independent domains from one Official outer-run key.

    Domain integers are not hand-selected hyperparameters: each is the first
    unsigned 32-bit word of SHA-256(``delta-zsc-v6/<domain>``), recorded in the
    run identity together with the resulting JAX key.
    """

    import jax
    import numpy as np

    root_key = (
        engineering_training_key()
        if int(seed_index) == ENGINEERING_SEED_INDEX
        else official_training_key(seed_index)
    )
    root = np.asarray(root_key, dtype=np.uint32)
    result: dict[str, tuple[int, int]] = {}
    for name in (
        "ego",
        "generator",
        "training_anchor",
        "audit_anchor",
        "context_dropout",
        "anchor_replay",
        "response",
        "calibration",
    ):
        tag = int.from_bytes(
            hashlib.sha256(f"delta-zsc-v6/{name}".encode("utf-8")).digest()[:4],
            "big",
        )
        key = np.asarray(jax.random.fold_in(root, tag), dtype=np.uint32)
        result[name] = (int(key[0]), int(key[1]))
    return result



def load_config(path: str | Path, *, run_kind: str) -> RunConfig:
    """Load config v11 and apply the registered run budget."""

    import yaml

    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {RUN_KINDS}.")
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload = _exact_fields(
        payload,
        {
            "version",
            "environment",
            "model",
            "ppo",
            "loss",
            "loss_v2",
            "anchors",
            "partner_generator",
            "partner_pool",
            "calibration",
            "training",
            "evaluation",
            "upstream",
            "official_protocol",
        },
        "configuration",
    )
    if int(payload["version"]) != CONFIG_VERSION:
        raise ValueError(f"The active DELTA-ZSC config version is {CONFIG_VERSION}.")

    environment_payload = _exact_fields(
        payload["environment"],
        {
            "layout",
            "agent_view_size",
            "indicate_successful_delivery",
            "negative_rewards",
            "random_agent_positions",
            "sample_recipe_on_delivery",
            "episode_steps",
        },
        "environment",
    )
    model_payload = _exact_fields(payload["model"], set(ModelConfig.__dataclass_fields__), "model")
    ppo_payload = _exact_fields(payload["ppo"], set(PPOConfig.__dataclass_fields__), "ppo")
    loss_payload = _exact_fields(payload["loss"], set(LossConfig.__dataclass_fields__), "loss")
    loss_v2_payload = _exact_fields(
        payload["loss_v2"], set(LossV2Config.__dataclass_fields__), "loss_v2"
    )
    anchor_payload = _exact_fields(payload["anchors"], set(AnchorConfig.__dataclass_fields__), "anchors")
    generator_payload = _exact_fields(
        payload["partner_generator"],
        set(PartnerGeneratorConfig.__dataclass_fields__),
        "partner_generator",
    )
    pool_payload = _exact_fields(
        payload["partner_pool"], set(PartnerPoolConfig.__dataclass_fields__), "partner_pool"
    )
    calibration_payload = _exact_fields(
        payload["calibration"], set(CalibrationConfig.__dataclass_fields__), "calibration"
    )
    training_payload = _exact_fields(
        payload["training"], set(TrainingConfig.__dataclass_fields__) - {
            "environment_steps",
            "minibatches_per_epoch",
            "checkpoint_interval_environment_steps",
        }, "training"
    )
    evaluation_payload = _exact_fields(
        payload["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation"
    )
    upstream_payload = _exact_fields(
        payload["upstream"], set(UpstreamConfig.__dataclass_fields__), "upstream"
    )
    protocol_payload = _exact_fields(
        payload["official_protocol"],
        set(OfficialProtocolConfig.__dataclass_fields__),
        "official_protocol",
    )

    budget = RUN_BUDGETS[run_kind]
    config = RunConfig(
        run_kind=run_kind,
        environment=EnvironmentConfig(
            **environment_payload,
            num_envs=budget.num_envs,
        ),
        model=ModelConfig(**model_payload),
        ppo=PPOConfig(**ppo_payload),
        loss=LossConfig(**loss_payload),
        loss_v2=LossV2Config(**loss_v2_payload),
        anchors=AnchorConfig(**anchor_payload),
        partner_generator=PartnerGeneratorConfig(**generator_payload),
        partner_pool=PartnerPoolConfig(
            enabled=bool(pool_payload["enabled"]),
            checkpoint_stages=tuple(float(value) for value in pool_payload["checkpoint_stages"]),
            heuristic_family_test_only=bool(pool_payload["heuristic_family_test_only"]),
            family_uniform_sampling=bool(pool_payload["family_uniform_sampling"]),
        ),
        calibration=CalibrationConfig(**calibration_payload),
        training=TrainingConfig(
            **training_payload,
            environment_steps=budget.environment_steps,
            minibatches_per_epoch=budget.minibatches_per_epoch,
            checkpoint_interval_environment_steps=(
                budget.checkpoint_interval_environment_steps
            ),
        ),
        evaluation=EvaluationConfig(**evaluation_payload),
        upstream=UpstreamConfig(
            total_timesteps=int(upstream_payload["total_timesteps"]),
            reward_shaping_horizon=int(upstream_payload["reward_shaping_horizon"]),
            checkpoint_progress=tuple(upstream_payload["checkpoint_progress"]),
        ),
        official_protocol=OfficialProtocolConfig(**protocol_payload),
    )
    validate_config(config)
    return config



def validate_config(config: RunConfig) -> None:
    if config.environment.layout not in LAYOUTS:
        raise ValueError(f"Unknown OvercookedV2 layout: {config.environment.layout}")
    if config.environment.agent_view_size != 2:
        raise ValueError("The registered public protocol uses view radius two.")
    if config.environment.episode_steps != OFFICIAL_EPISODE_STEPS:
        raise ValueError("The registered public protocol uses 400-step episodes.")
    if not (
        config.environment.negative_rewards
        and config.environment.random_agent_positions
        and config.environment.sample_recipe_on_delivery
        and config.environment.indicate_successful_delivery
    ):
        raise ValueError("Formal OvercookedV2 environment flags must match Official.")
    if config.model.protocol_components != 4:
        raise ValueError("DEPI fixes protocol_components=4 (METHOD_SPEC §1.1).")
    if config.model.component_embedding_dim != 16:
        raise ValueError("DEPI fixes component_embedding_dim=16 (METHOD_SPEC §1.1).")
    if config.model.capability_dim != 16:
        raise ValueError("DEPI fixes capability_dim=16 (METHOD_SPEC §1.1).")
    if config.model.capability_hidden_dim != 64:
        raise ValueError("DEPI fixes capability_hidden_dim=64 (METHOD_SPEC §1.1).")
    if config.model.protocol_hidden_dim != 128:
        raise ValueError("DEPI fixes protocol_hidden_dim=128 (METHOD_SPEC §1.1).")
    if config.model.modulation_rank <= 0:
        raise ValueError("Low-rank actor modulation needs positive rank.")

    if not 0.0 < config.ppo.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if not 0.0 <= config.ppo.gae_lambda <= 1.0:
        raise ValueError("GAE lambda must lie in [0, 1].")
    if config.ppo.update_epochs <= 0 or config.ppo.learning_rate <= 0.0:
        raise ValueError("PPO update count and learning rate must be positive.")
    if not 0.0 < config.ppo.polyak_coefficient <= 1.0:
        raise ValueError("Polyak coefficient must lie in (0, 1].")
    if config.ppo.polyak_coefficient != 0.005:
        raise ValueError("V6 fixes the ego EMA target coefficient at 0.005.")
    if not 0.0 <= config.ppo.lr_warmup_fraction < 1.0:
        raise ValueError("LR warmup fraction must lie in [0, 1).")
    if config.ppo.adam_epsilon <= 0.0:
        raise ValueError("Adam epsilon must be positive.")
    frozen_losses = {
        "response_weight": 1.0,
        "raw_q_weight": 1.0,
        "counterfactual_weight": 1.0,
        "decision_equivalence_weight": 0.10,
        "q_policy_weight": 0.25,
        "q_policy_temperature": 1.0,
        "q_policy_gap_midpoint": 1.0,
        "q_policy_gap_temperature": 1.0,
        "q_policy_disagreement_temperature": 5.0,
        "information_bottleneck_weight": 0.001,
        "information_bottleneck_free_bits_per_dimension": 0.1,
        "robust_generalist_weight": 0.10,
        "policy_belief_gradient_scale": 0.10,
        "belief_gradient_ema_decay": 0.99,
        "decision_regret_weight_maximum": 0.10,
        "decision_regret_schedule_midpoint": 0.20,
        "decision_regret_schedule_temperature": 0.05,
        "response_updates_per_outer_update": 2,
        "raw_q_updates_per_outer_update": 4,
    }
    for name, expected in frozen_losses.items():
        if getattr(config.loss, name) != expected:
            raise ValueError(f"V6 loss field {name} must equal {expected!r}.")

    frozen_loss_v2 = {
        "signature_weight": 1.0,
        "response_weight": 1.0,
        "separation_weight": 0.1,
        "combined_policy_kl_threshold": 0.04,
        "rank_hinge_margin": 0.1,
        "rank_hinge_advantage_gap": 2.0,
        "separation_margin_scale": 0.25,
    }
    for name, expected in frozen_loss_v2.items():
        if getattr(config.loss_v2, name) != expected:
            raise ValueError(f"DEPI loss_v2 field {name} must equal {expected!r}.")

    if config.partner_generator.enabled:
        raise ValueError(
            "The learned partner generator stays disabled: the current tree "
            "registers only SP/OP seeds, SA/FCP upstream sources are missing, "
            "and the §7.3 static wide partner pool is the default "
            "(METHOD_SPEC §7.3)."
        )
    if config.partner_generator.code_dim != 3:
        raise ValueError(
            "§7.1 fixes the generator code space to the 3-dimensional "
            "tetrahedral simplex; code_dim must be 3."
        )
    if config.partner_generator.cvar_level != 0.20:
        raise ValueError("V6 fixes generator competence to CVaR20.")
    registered_shape = config.run_kind != "mechanical"
    if registered_shape and config.partner_generator.episodes_per_update != 32:
        raise ValueError("V6 generator requires 32 complete episodes per outer update.")
    if config.partner_generator.episode_steps != OFFICIAL_EPISODE_STEPS:
        raise ValueError("Generator episodes must use all 400 Official steps.")
    if registered_shape and config.partner_generator.update_epochs != 4:
        raise ValueError("Generator PPO uses four Official update epochs.")
    if registered_shape and config.partner_generator.environment_minibatches != 8:
        raise ValueError("Generator PPO uses eight environment minibatches.")
    if config.partner_generator.target_polyak_coefficient != 0.005:
        raise ValueError("Generator EMA coefficient must remain 0.005.")
    if config.partner_generator.maximum_generator_probability != 0.75:
        raise ValueError("Generator mixture probability is capped at 0.75.")
    if config.partner_generator.mixture_ramp_fraction != 0.30:
        raise ValueError("Generator mixture ramp must finish at 30% of training.")
    if config.partner_generator.competence_temperature != 20.0:
        raise ValueError("Generator competence temperature equals one delivery reward.")
    if not 0.0 < config.partner_generator.cvar_ema_decay < 1.0:
        raise ValueError("Generator CVaR EMA decay must lie in (0,1).")
    frozen_generator = {
        "cvar_ema_decay": 0.95,
        "imitation_initial_weight": 1.0,
        "brdiv_maximum_weight": 1.0,
        "smoothness_weight": 0.05,
        "lagrangian_learning_rate": 0.01,
        "competence_multiplier_maximum": 10.0,
        "diversity_bank_size": 64,
        "diversity_codes_per_update": 8,
        "code_archive_capacity": 256,
        "freeze_after_imitation": True,
    }
    for name, expected in frozen_generator.items():
        if getattr(config.partner_generator, name) != expected:
            raise ValueError(
                f"V6 partner-generator field {name} must equal {expected!r}."
            )
    if registered_shape and config.partner_generator.codes_per_update != 32:
        raise ValueError("V6 generator uses 32 continuous codes per outer update.")

    if registered_shape and config.anchors.interval_updates != 16:
        raise ValueError("V6 training anchors run every 16 outer updates.")
    if registered_shape and (
        config.anchors.ordinary_states != 32
        or config.anchors.matched_code_pairs != 16
        or config.anchors.fit_replicas != 4
    ):
        raise ValueError("V6 anchors require 32 ordinary, 16 matched pairs and 4 replicas.")
    if not 1 <= config.anchors.continuation_horizon <= config.environment.episode_steps:
        raise ValueError("Anchor continuation horizon is outside the episode.")
    if registered_shape and (
        config.anchors.continuation_horizon != 128 or config.anchors.probe_steps != 16
    ):
        raise ValueError("V6 anchors require horizon 128 and 16 evidence steps.")
    if registered_shape and (
        config.anchors.replay_capacity != 512
        or config.anchors.replay_minibatch_size != 64
        or config.anchors.policy_kl_decay != 0.05
        or config.anchors.age_decay_updates != 64.0
        or config.anchors.minimum_replay_weight != 0.001
    ):
        raise ValueError("V6 replay capacity and continuous decay constants are frozen.")
    if config.anchors.return_lower_bound >= config.anchors.return_upper_bound:
        raise ValueError("Anchor return bounds are reversed.")
    frozen_anchor_comparator = {
        "observable_equivalent_accuracy_max": 0.55,
        "decision_distinct_accuracy_min": 0.70,
        "signature_distance_threshold": 1.0,
    }
    for name, expected in frozen_anchor_comparator.items():
        if getattr(config.anchors, name) != expected:
            raise ValueError(
                f"§5.3 frozen comparator field {name} must equal {expected!r}."
            )
    frozen_m1_gate = {
        "m1_spearman_threshold": 0.8,
        "m1_minimum_anchor_fraction": 0.9,
    }
    for name, expected in frozen_m1_gate.items():
        if getattr(config.anchors, name) != expected:
            raise ValueError(
                f"§6 extended M1 gate field {name} must equal {expected!r}."
            )

    if not config.partner_pool.enabled:
        raise ValueError(
            "The §7.3 static wide partner pool is the default B0–B2 partner "
            "distribution and stays enabled."
        )
    if tuple(config.partner_pool.checkpoint_stages) != (0.0, 0.5, 1.0):
        raise ValueError(
            "Partner-pool checkpoint stages are frozen at (0.0, 0.5, 1.0)."
        )
    if not config.partner_pool.heuristic_family_test_only:
        raise ValueError(
            "The heuristic partner family stays held out as the test-only "
            "algorithm family (family-disjoint declaration)."
        )
    if not config.partner_pool.family_uniform_sampling:
        raise ValueError("Partner-pool sampling is uniform across families.")

    if not 0.0 < config.calibration.alpha < 1.0:
        raise ValueError("Conformal alpha must lie in (0, 1).")
    if not 0.0 <= config.calibration.support_quantile < 1.0:
        raise ValueError("Calibration support_quantile must lie in [0, 1).")
    if config.calibration.block_unit != "partner_run":
        raise ValueError("The registered calibration block unit is partner_run.")
    minimum_for_finite_quantile = int(math.ceil(1.0 / config.calibration.alpha) - 1)
    if config.calibration.minimum_run_count < minimum_for_finite_quantile:
        raise ValueError(
            "Calibration minimum_run_count cannot yield a finite split-conformal "
            f"quantile at alpha={config.calibration.alpha}; need at least "
            f"{minimum_for_finite_quantile}."
        )
    if config.calibration.episodes_per_run <= 0 or config.calibration.anchors_per_run <= 0:
        raise ValueError("Calibration episodes and anchors per run must be positive.")
    if config.run_kind == "formal" and config.calibration.enabled:
        raise ValueError("Primary DELTA-ZSC-E2E formal runs cannot enable the safety wrapper.")

    if not (
        0.0 <= config.training.context_dropout_final
        <= config.training.context_dropout_initial < 1.0
    ):
        raise ValueError("Context dropout must anneal within [0,1).")
    if registered_shape and (
        config.training.context_dropout_initial != 0.30
        or config.training.context_dropout_final != 0.10
    ):
        raise ValueError("V6 context dropout is frozen at 0.30 -> 0.10.")
    if (
        config.training.owner_sources_per_seed != 1
        or config.training.generator_init_sources_per_mechanism != 1
        or config.training.development_runs_per_mechanism != 4
        or config.training.calibration_runs_per_mechanism != 5
    ):
        raise ValueError("Per-seed V6 partner resource counts changed.")
    if config.training.rollout_length <= 0:
        raise ValueError("Training rollout length must be positive.")
    if config.evaluation.episodes_per_pairing <= 0:
        raise ValueError("Evaluation episodes_per_pairing must be positive.")
    if config.evaluation.bootstrap_replicates <= 0:
        raise ValueError("Evaluation bootstrap_replicates must be positive.")
    if min(
        config.evaluation.br_prox_anchors_per_pairing,
        config.evaluation.br_prox_fit_replicas,
        config.evaluation.br_prox_evaluation_replicas,
        config.evaluation.br_prox_continuation_horizon,
    ) <= 0:
        raise ValueError("Empirical BR-Prox audit budgets must be positive.")
    if (
        config.evaluation.br_prox_anchors_per_pairing
        > config.environment.episode_steps * config.evaluation.episodes_per_pairing
    ):
        raise ValueError("BR-Prox anchors exceed available evaluation states.")
    if (
        config.evaluation.br_prox_continuation_horizon
        > config.environment.episode_steps
    ):
        raise ValueError("BR-Prox continuation horizon exceeds the episode.")
    if not 0.0 < config.evaluation.one_sided_alpha < 1.0:
        raise ValueError("Evaluation one_sided_alpha must lie in (0, 1).")
    if min(
        config.evaluation.minimum_ego_runs,
        config.evaluation.minimum_partner_runs_per_mechanism,
        config.evaluation.minimum_mechanisms,
    ) <= 0:
        raise ValueError("Evaluation independent-run minima must be positive.")
    if config.run_kind == "formal":
        if not config.evaluation.report_br_prox:
            raise ValueError(
                "The DELTA-ZSC design requires empirical BR-Prox in the formal "
                "mechanism-disjoint confirmatory evaluation."
            )
        if config.training.rollout_length != OFFICIAL_ROLLOUT_LENGTH:
            raise ValueError("Official formal recurrent rollouts contain 256 steps.")
        official_ppo = {
            "update_epochs": OFFICIAL_UPDATE_EPOCHS,
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
        for name, expected in official_ppo.items():
            if getattr(config.ppo, name) != expected:
                raise ValueError(f"Formal PPO field {name} must equal {expected!r}.")
        if config.environment.num_envs != OFFICIAL_SP_NUM_ENVS:
            raise ValueError("Official DELTA formal training uses 256 environments.")
        if config.training.environment_steps != 29_949_952:
            raise ValueError("Official DELTA formal training executes 29,949,952 steps.")
        if config.training.minibatches_per_epoch != OFFICIAL_NUM_MINIBATCHES:
            raise ValueError("Official DELTA formal training uses 64 minibatches.")
        if config.evaluation.episodes_per_pairing != OFFICIAL_EPISODES_PER_PAIRING:
            raise ValueError("Official evaluation uses 500 episodes per pairing.")
        if config.evaluation.evaluation_seed != OFFICIAL_EVALUATION_ROOT_SEED:
            raise ValueError("Repository-defined Official evaluation root key is zero.")
        if config.evaluation.one_sided_alpha != 0.05:
            raise ValueError("Formal scoreboards use the registered one-sided 95% bound.")
        if config.evaluation.bootstrap_replicates != 9_999:
            raise ValueError("Formal scoreboards use exactly 9,999 node bootstraps.")
        if (
            config.evaluation.minimum_ego_runs != 10
            or config.evaluation.minimum_partner_runs_per_mechanism != 4
            or config.evaluation.minimum_mechanisms != 4
            or not config.evaluation.evaluate_both_roles
        ):
            raise ValueError("Formal Common-Partner support is fixed at 10 egos and 4x4 partners in both roles.")

    protocol = config.official_protocol
    expected_protocol = {
        "protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "training_root_seed": OFFICIAL_TRAINING_ROOT_SEED,
        "training_run_count": OFFICIAL_TRAINING_RUN_COUNT,
        "evaluation_root_seed": OFFICIAL_EVALUATION_ROOT_SEED,
        "evaluation_episodes_per_pairing": OFFICIAL_EPISODES_PER_PAIRING,
    }
    for name, expected in expected_protocol.items():
        if getattr(protocol, name) != expected:
            raise ValueError(f"Official protocol field {name} must equal {expected!r}.")

    rollout_steps = config.environment.num_envs * config.training.rollout_length
    if config.training.environment_steps % rollout_steps:
        raise ValueError("Training budget must contain whole vectorized rollouts.")
    if config.environment.num_envs % config.training.minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    if config.training.checkpoint_interval_environment_steps % rollout_steps:
        raise ValueError("Checkpoint intervals must align with vectorized rollouts.")



def load_partner_manifest(
    path: str | Path,
    *,
    expected_layout: str,
    verify_files: bool = True,
) -> PartnerManifest:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload = _exact_fields(payload, {"version", "layout", "runs"}, "partner manifest")
    if int(payload["version"]) != MANIFEST_VERSION:
        raise ValueError(f"Partner manifest version must be {MANIFEST_VERSION}.")
    if str(payload["layout"]) != expected_layout:
        raise ValueError("Partner manifest layout differs from the config.")
    if not isinstance(payload["runs"], Sequence):
        raise ValueError("Partner manifest runs must be a sequence.")

    runs: list[PartnerRun] = []
    for index, raw in enumerate(payload["runs"]):
        raw = _exact_fields(
            raw,
            {
                "run_id",
                "role",
                "checkpoint",
                "checkpoint_sha256",
                "parent_training_run_id",
                "generation_mechanism",
                "seed",
                "seed_index",
                "jax_prng_key",
                "owner_seed_index",
                "co_training_group_id",
                "partner_type_id",
            },
            f"partner manifest run[{index}]",
        )
        role = str(raw["role"])
        if role not in PARTNER_ROLES:
            raise ValueError(f"Unknown partner role at run[{index}]: {role}")
        checkpoint = Path(str(raw["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = (source.parent / checkpoint).resolve()
        run = PartnerRun(
            run_id=str(raw["run_id"]),
            role=role,
            checkpoint=checkpoint,
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            parent_training_run_id=str(raw["parent_training_run_id"]),
            generation_mechanism=str(raw["generation_mechanism"]),
            seed=int(raw["seed"]),
            seed_index=(
                None if raw["seed_index"] is None else int(raw["seed_index"])
            ),
            jax_prng_key=(
                None
                if raw["jax_prng_key"] is None
                else tuple(int(value) for value in raw["jax_prng_key"])
            ),
            owner_seed_index=(
                None
                if raw["owner_seed_index"] is None
                else int(raw["owner_seed_index"])
            ),
            co_training_group_id=(
                None
                if raw["co_training_group_id"] is None
                else str(raw["co_training_group_id"])
            ),
            partner_type_id=(
                None if raw["partner_type_id"] is None else str(raw["partner_type_id"])
            ),
        )
        if run.jax_prng_key is not None and len(run.jax_prng_key) != 2:
            raise ValueError(f"JAX PRNG key must have two words for run {run.run_id}.")
        if verify_files:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Partner checkpoint is missing: {checkpoint}")
            observed = _sha256(checkpoint)
            if observed != run.checkpoint_sha256:
                raise ValueError(f"Checkpoint SHA-256 differs for run {run.run_id}.")
        runs.append(run)

    manifest = PartnerManifest(str(payload["layout"]), tuple(runs))
    validate_partner_manifest(manifest)
    return manifest



def validate_partner_manifest(manifest: PartnerManifest) -> None:
    if not manifest.runs:
        raise ValueError("Partner manifest cannot be empty.")
    run_ids = [run.run_id for run in manifest.runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Partner run IDs must be unique.")
    for run in manifest.runs:
        if run.owner_seed_index is not None and not (
            0 <= run.owner_seed_index < OFFICIAL_TRAINING_RUN_COUNT
        ):
            raise ValueError(f"Partner owner_seed_index is outside 0..9: {run.run_id}")
        if run.jax_prng_key is None:
            raise ValueError(
                f"Every V6 partner run must record its real two-word JAX key: "
                f"{run.run_id}"
            )
        if run.role == "confirmatory" and run.owner_seed_index is not None:
            raise ValueError(
                f"Common confirmatory partner cannot belong to one ego run: {run.run_id}"
            )
        official_parent = (
            run.role in {
                "owner_source",
                "generator_init_source",
                "development_support",
            }
            and run.generation_mechanism in {"rnn-sp", "rnn-op"}
        )
        if official_parent:
            if run.seed_index is None or run.jax_prng_key is None:
                raise ValueError(
                    f"Official SP/OP partner lacks indexed PRNG provenance: {run.run_id}"
                )
            if not 0 <= run.seed_index < OFFICIAL_TRAINING_RUN_COUNT:
                raise ValueError(f"Partner seed_index is outside 0..9: {run.run_id}")
            if tuple(run.jax_prng_key) != official_training_key(run.seed_index):
                raise ValueError(
                    f"Official partner PRNG key differs from split(PRNGKey(42), 10): "
                    f"{run.run_id}"
                )
        if run.role in {"calibration", "confirmatory"}:
            if run.seed_index is not None:
                raise ValueError(
                    "Fresh calibration/confirmatory partners cannot reuse Official "
                    f"seed indexes: {run.run_id}"
                )
            official_keys = {
                tuple(official_training_key(index))
                for index in range(OFFICIAL_TRAINING_RUN_COUNT)
            }
            if tuple(run.jax_prng_key or ()) in official_keys:
                raise ValueError(
                    "Fresh calibration/confirmatory partner reuses a formal "
                    f"training key: {run.run_id}"
                )
        if run.jax_prng_key is not None and len(run.jax_prng_key) != 2:
            raise ValueError(f"Partner JAX PRNG key is malformed: {run.run_id}")
    hashes = [run.checkpoint_sha256 for run in manifest.runs]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Exact checkpoint overlap is forbidden across manifest entries.")
    evaluation_keys = [
        tuple(run.jax_prng_key or ())
        for run in manifest.runs
        if run.role in {"calibration", "confirmatory"}
    ]
    if len(evaluation_keys) != len(set(evaluation_keys)):
        raise ValueError(
            "Fresh calibration/confirmatory partners must use distinct JAX keys."
        )

    for role in (
        "owner_source",
        "generator_init_source",
        "development_support",
        "calibration",
    ):
        owners_by_parent: dict[str, set[int | None]] = {}
        for run in manifest.by_role(role):
            owners_by_parent.setdefault(run.parent_training_run_id, set()).add(
                run.owner_seed_index
            )
        shared_parents = {
            parent: owners
            for parent, owners in owners_by_parent.items()
            if len(owners) > 1
        }
        if shared_parents:
            raise ValueError(
                f"{role} parent runs cannot be shared across DELTA outer runs: "
                f"{shared_parents}"
            )

    train_roles = {
        "owner_source",
        "generator_init_source",
        "development_support",
    }
    eval_roles = {"calibration", "confirmatory"}
    train_parents = {
        run.parent_training_run_id for run in manifest.runs if run.role in train_roles
    }
    eval_parents = {
        run.parent_training_run_id for run in manifest.runs if run.role in eval_roles
    }
    overlap = train_parents & eval_parents
    if overlap:
        raise ValueError(
            "Training and evaluation partner parent runs overlap: "
            f"{sorted(overlap)}"
        )

    train_groups = {
        run.co_training_group_id
        for run in manifest.runs
        if run.role in train_roles and run.co_training_group_id is not None
    }
    eval_groups = {
        run.co_training_group_id
        for run in manifest.runs
        if run.role in eval_roles and run.co_training_group_id is not None
    }
    group_overlap = train_groups & eval_groups
    if group_overlap:
        raise ValueError(
            "Training and evaluation partners share a co-training group: "
            f"{sorted(group_overlap)}"
        )

    calibration_parents = {
        run.parent_training_run_id for run in manifest.runs if run.role == "calibration"
    }
    confirmatory_parents = {
        run.parent_training_run_id for run in manifest.runs if run.role == "confirmatory"
    }
    calibration_confirmatory_overlap = calibration_parents & confirmatory_parents
    if calibration_confirmatory_overlap:
        raise ValueError(
            "Calibration and confirmatory parent runs overlap: "
            f"{sorted(calibration_confirmatory_overlap)}"
        )

    development_support = manifest.by_role("development_support")
    calibration = manifest.by_role("calibration")
    confirmatory = manifest.by_role("confirmatory")
    if development_support and len({run.parent_training_run_id for run in development_support}) < 2:
        raise ValueError("Training support requires at least two independent frozen external runs.")
    if calibration and len({run.parent_training_run_id for run in calibration}) < 2:
        raise ValueError("Calibration requires at least two independent parent runs.")
    if confirmatory and len({run.parent_training_run_id for run in confirmatory}) < 2:
        raise ValueError("Confirmatory evaluation requires at least two independent parent runs.")


def validate_seed_training_manifest(
    manifest: PartnerManifest,
    *,
    owner_seed_index: int,
    formal: bool,
) -> None:
    """Enforce lineage-exclusive V6 initialization and training support."""

    validate_partner_manifest(manifest)
    owner = int(owner_seed_index)
    if formal and not 0 <= owner < OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Formal DELTA owner seed must lie in 0..9.")
    if not formal and owner not in {ENGINEERING_SEED_INDEX, *range(10)}:
        raise ValueError("Mechanical/development owner seed is invalid.")
    manifest_owner = None if owner == ENGINEERING_SEED_INDEX else owner
    owned = tuple(run for run in manifest.runs if run.owner_seed_index == manifest_owner)
    foreign = tuple(
        run
        for run in manifest.runs
        if owner != ENGINEERING_SEED_INDEX
        and run.owner_seed_index is not None
        and run.owner_seed_index != owner
    )
    if foreign:
        raise ValueError("A per-seed V6 manifest contains another seed's resource.")

    owners = tuple(run for run in owned if run.role == "owner_source")
    if len(owners) != 1 or owners[0].generation_mechanism not in {"rnn-sp", "sp"}:
        raise ValueError("Each DELTA seed needs exactly one independent owner-SP source.")

    mechanism_alias = {
        "rnn-sp": "sp",
        "sp": "sp",
        "rnn-op": "op",
        "op": "op",
        "state-augmented": "sa",
        "sa": "sa",
        "fcp": "fcp",
    }

    def counts(role: str) -> Mapping[str, int]:
        result = {name: 0 for name in ("sp", "op", "sa", "fcp")}
        for run in owned:
            if run.role != role:
                continue
            mechanism = mechanism_alias.get(run.generation_mechanism)
            if mechanism is None:
                raise ValueError(f"Unknown V6 partner mechanism: {run.generation_mechanism}")
            result[mechanism] += 1
        return result

    expected = {
        "generator_init_source": 1,
        "development_support": 4,
    }
    if formal:
        for role, per_mechanism in expected.items():
            observed = counts(role)
            if any(value != per_mechanism for value in observed.values()):
                raise ValueError(
                    f"{role} must contain {per_mechanism} independent runs per mechanism; "
                    f"observed={dict(observed)}."
                )

    role_groups = {
        role: {
            (run.checkpoint_sha256, run.parent_training_run_id, run.co_training_group_id)
            for run in owned
            if run.role == role
        }
        for role in (
            "owner_source",
            "generator_init_source",
            "development_support",
        )
    }
    roles = tuple(role_groups)
    for index, first in enumerate(roles):
        for second in roles[index + 1 :]:
            first_parents = {row[1] for row in role_groups[first]}
            second_parents = {row[1] for row in role_groups[second]}
            if first_parents & second_parents:
                raise ValueError(f"V6 resource groups {first}/{second} share a parent run.")
            first_groups = {row[2] for row in role_groups[first] if row[2] is not None}
            second_groups = {row[2] for row in role_groups[second] if row[2] is not None}
            if first_groups & second_groups:
                raise ValueError(f"V6 resource groups {first}/{second} share co-training lineage.")


__all__ = [
    "AnchorConfig",
    "CONFIG_VERSION",
    "ENGINEERING_SEED_INDEX",
    "CalibrationConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LAYOUTS",
    "LossConfig",
    "LossV2Config",
    "MANIFEST_VERSION",
    "METHOD_VERSION",
    "OFFICIAL_ACTION_COUNT",
    "OFFICIAL_EPISODE_STEPS",
    "OFFICIAL_CORRECT_DELIVERY_REWARD",
    "OFFICIAL_EPISODES_PER_PAIRING",
    "OFFICIAL_EVALUATION_ROOT_SEED",
    "OFFICIAL_NUM_MINIBATCHES",
    "OFFICIAL_OP_NUM_ENVS",
    "OFFICIAL_OP_TOTAL_TIMESTEPS",
    "OFFICIAL_PROTOCOL_VERSION",
    "OFFICIAL_ROLLOUT_LENGTH",
    "OFFICIAL_SOURCE_COMMIT",
    "OFFICIAL_SP_NUM_ENVS",
    "OFFICIAL_SP_TOTAL_TIMESTEPS",
    "OFFICIAL_TRAINING_ROOT_SEED",
    "OFFICIAL_TRAINING_KEYS",
    "OFFICIAL_TRAINING_RUN_COUNT",
    "OFFICIAL_UPDATE_EPOCHS",
    "OfficialProtocolConfig",
    "ModelConfig",
    "PARTNER_ROLES",
    "PPOConfig",
    "PartnerGeneratorConfig",
    "PartnerManifest",
    "PartnerPoolConfig",
    "PartnerRun",
    "RUN_BUDGETS",
    "RUN_KINDS",
    "RunConfig",
    "TrainingConfig",
    "UpstreamConfig",
    "load_config",
    "load_partner_manifest",
    "official_training_key",
    "official_training_domain_keys",
    "engineering_training_key",
    "validate_config",
    "validate_partner_manifest",
    "validate_seed_training_manifest",
]
