"""Single configuration and manifest authority for DEPI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_VERSION = 17
METHOD_VERSION = "depi_instant_partner_exact_filter_decision_supervision_v7"
CHECKPOINT_SCHEMA_VERSION = 7
MECHANISM_ABLATION_VARIANTS = (
    "deterministic_context",
    "decision_only",
    "q_only",
    "actor_only",
    "no_separation",
    "no_capability",
)
METHOD_VARIANTS = ("r0", "b0", "b1", "b2", *MECHANISM_ABLATION_VARIANTS, "b3")
MANIFEST_VERSION = 4
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
    "development_support",
    "comparator_fit",
    "comparator_validation",
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
    instant_partner_dim: int
    capability_hidden_dim: int
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
class LossV2Config:
    """Active DEPI objective weights and gates (METHOD_SPEC §3.1/§3.2/§3.4)."""

    signature_weight: float
    response_weight: float
    separation_weight: float
    combined_policy_kl_threshold: float
    rank_hinge_margin: float
    rank_hinge_advantage_gap: float
    separation_margin_scale: float
    decision_policy_weight: float
    decision_policy_temperature: float
    capability_consistency_weight: float
    capability_prediction_weight: float
    capability_variance_weight: float
    capability_variance_floor: float
    component_signature_weight: float


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    enabled: bool
    interval_updates: int
    ordinary_states: int
    matched_history_pairs: int
    fit_replicas: int
    evaluation_replicas: int
    continuation_horizon: int
    probe_steps: int
    return_lower_bound: float
    return_upper_bound: float
    # §5.3 frozen comparator thresholds and the §6 extended M1 gate.
    observable_equivalent_probability_max: float
    decision_distinct_probability_min: float
    signature_distance_threshold: float
    m1_spearman_threshold: float
    m1_minimum_anchor_fraction: float

@dataclass(frozen=True, slots=True)
class PartnerPoolConfig:
    """Static wide partner pool, the default R0/B0–B2 partner distribution
    (METHOD_SPEC §7.3): SP/OP multi-seed runs x checkpoint stages, OP width
    variants, and a heuristic family held out exclusively for testing."""

    enabled: bool
    checkpoint_stages: tuple[float, ...]
    heuristic_family_test_only: bool
    family_uniform_sampling: bool

@dataclass(frozen=True, slots=True)
class PosteriorCalibrationConfig:
    """Held-out posterior-predictive calibration preregistration."""

    minimum_run_count: int
    episodes_per_run: int
    bootstrap_replicates: int
    bootstrap_seed: int
    root_seed_offset: int
    credibility: float
    log_score_margin: float
    coverage_low: float
    coverage_high: float
    brier_ratio: float
    primary_unit: str
    secondary_unit: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    extra_ppo_environment_steps: int
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int
    rollout_length: int
    context_dropout_initial: float
    context_dropout_final: float
    owner_sources_per_seed: int
    support_parent_runs_per_family: int
    support_checkpoints_per_parent: int
    support_width_variants: int
    calibration_runs_per_family: int

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
    mechanism_anchors_per_pairing: int
    mechanism_fit_replicas: int
    mechanism_evaluation_replicas: int
    mechanism_continuation_horizon: int
    recoverable_signal_threshold: float
    evaluate_both_roles: bool
    one_sided_alpha: float
    minimum_ego_runs: int
    minimum_partner_runs_per_mechanism: int
    minimum_mechanisms: int
    inference_mode: str
    superiority_lcb_threshold: float
    minimum_effect: float
    minimum_effect_rule: str


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
    method_variant: str
    environment: EnvironmentConfig
    model: ModelConfig
    ppo: PPOConfig
    loss_v2: LossV2Config
    anchors: AnchorConfig
    partner_pool: PartnerPoolConfig
    posterior_calibration: PosteriorCalibrationConfig
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
    checkpoint_stage: float
    hyperparameter_family: str
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
        b"depi/exact-filter-decision-supervision/engineering-seed"
    ).digest()
    return (
        int.from_bytes(digest[:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    )


def official_training_domain_keys(seed_index: int) -> Mapping[str, tuple[int, int]]:
    """Derive registered independent domains from one Official outer-run key.

    Domain integers are not hand-selected hyperparameters: each is the first
    unsigned 32-bit word of SHA-256(``depi/<domain>``), recorded in the
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
        "training_anchor",
        "audit_anchor",
        "context_dropout",
        "posterior_calibration",
        "identifiability",
        "recoverable_value",
    ):
        tag = int.from_bytes(
            hashlib.sha256(f"depi/{name}".encode("utf-8")).digest()[:4],
            "big",
        )
        key = np.asarray(jax.random.fold_in(root, tag), dtype=np.uint32)
        result[name] = (int(key[0]), int(key[1]))
    return result



def load_config(path: str | Path, *, run_kind: str) -> RunConfig:
    """Load the active DEPI config and apply the registered run budget."""

    import yaml

    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {RUN_KINDS}.")
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload = _exact_fields(
        payload,
        {
            "version",
            "method_variant",
            "environment",
            "model",
            "ppo",
            "loss_v2",
            "anchors",
            "partner_pool",
            "posterior_calibration",
            "training",
            "evaluation",
            "upstream",
            "official_protocol",
        },
        "configuration",
    )
    if int(payload["version"]) != CONFIG_VERSION:
        raise ValueError(f"The active DEPI config version is {CONFIG_VERSION}.")

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
    loss_v2_payload = _exact_fields(
        payload["loss_v2"], set(LossV2Config.__dataclass_fields__), "loss_v2"
    )
    anchor_payload = _exact_fields(payload["anchors"], set(AnchorConfig.__dataclass_fields__), "anchors")
    pool_payload = _exact_fields(
        payload["partner_pool"], set(PartnerPoolConfig.__dataclass_fields__), "partner_pool"
    )
    calibration_payload = _exact_fields(
        payload["posterior_calibration"],
        set(PosteriorCalibrationConfig.__dataclass_fields__),
        "posterior_calibration",
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
        method_variant=str(payload["method_variant"]).lower(),
        environment=EnvironmentConfig(
            **environment_payload,
            num_envs=budget.num_envs,
        ),
        model=ModelConfig(**model_payload),
        ppo=PPOConfig(**ppo_payload),
        loss_v2=LossV2Config(**loss_v2_payload),
        anchors=AnchorConfig(**anchor_payload),
        partner_pool=PartnerPoolConfig(
            enabled=bool(pool_payload["enabled"]),
            checkpoint_stages=tuple(float(value) for value in pool_payload["checkpoint_stages"]),
            heuristic_family_test_only=bool(pool_payload["heuristic_family_test_only"]),
            family_uniform_sampling=bool(pool_payload["family_uniform_sampling"]),
        ),
        posterior_calibration=PosteriorCalibrationConfig(**calibration_payload),
        training=TrainingConfig(
            **training_payload,
            environment_steps=(
                budget.environment_steps
                + int(training_payload["extra_ppo_environment_steps"])
            ),
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
    if config.method_variant not in METHOD_VARIANTS:
        raise ValueError(f"method_variant must be one of {METHOD_VARIANTS}.")
    if config.method_variant == "b3":
        raise NotImplementedError(
            "B3 action-conditioned VOI is explicitly not implemented; it cannot be run."
        )
    if config.run_kind == "formal" and config.method_variant != "b2":
        raise ValueError("The confirmatory formal method is frozen to B2.")
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
    if config.model.protocol_components not in (2, 4, 8):
        raise ValueError("DEPI development sensitivity supports K in {2,4,8}.")
    if config.run_kind == "formal" and config.model.protocol_components != 4:
        raise ValueError("The preregistered formal DEPI configuration fixes K=4.")
    if config.model.instant_partner_dim != 32:
        raise ValueError("DEPI fixes instant_partner_dim=32 (METHOD_SPEC §1.1).")
    if config.model.component_embedding_dim != 16:
        raise ValueError("DEPI fixes component_embedding_dim=16 (METHOD_SPEC §1.1).")
    if config.model.capability_dim != 16:
        raise ValueError("DEPI fixes capability_dim=16 (METHOD_SPEC §1.1).")
    if config.model.capability_hidden_dim != 64:
        raise ValueError("DEPI fixes capability_hidden_dim=64 (METHOD_SPEC §1.1).")
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
        raise ValueError("DEPI fixes the ego EMA target coefficient at 0.005.")
    if not 0.0 <= config.ppo.lr_warmup_fraction < 1.0:
        raise ValueError("LR warmup fraction must lie in [0, 1).")
    if config.ppo.adam_epsilon <= 0.0:
        raise ValueError("Adam epsilon must be positive.")
    frozen_loss_v2 = {
        "signature_weight": 1.0,
        "response_weight": 1.0,
        "separation_weight": 0.1,
        "combined_policy_kl_threshold": 0.04,
        "rank_hinge_margin": 0.1,
        "rank_hinge_advantage_gap": 2.0,
        "separation_margin_scale": 0.25,
        "decision_policy_weight": 0.25,
        "decision_policy_temperature": 1.0,
        "capability_consistency_weight": 0.01,
        "capability_prediction_weight": 0.1,
        "capability_variance_weight": 0.01,
        "capability_variance_floor": 0.05,
        "component_signature_weight": 0.25,
    }
    for name, expected in frozen_loss_v2.items():
        if getattr(config.loss_v2, name) != expected:
            raise ValueError(f"DEPI loss_v2 field {name} must equal {expected!r}.")

    registered_shape = config.run_kind != "mechanical"
    if registered_shape and config.anchors.interval_updates != 16:
        raise ValueError("DEPI training anchors run every 16 outer updates.")
    expected_anchor_shape = (
        (4, 2) if config.run_kind == "development" else (32, 16)
    )
    if registered_shape and (
        config.anchors.ordinary_states != expected_anchor_shape[0]
        or config.anchors.matched_history_pairs != expected_anchor_shape[1]
        or config.anchors.fit_replicas != 4
    ):
        raise ValueError(
            "DEPI anchor states must match the registered per-million-PPO-transition density."
        )
    if registered_shape and config.anchors.evaluation_replicas != 8:
        raise ValueError("DEPI M1 evaluation requires 8 independent replicas.")
    if not 1 <= config.anchors.continuation_horizon <= config.environment.episode_steps:
        raise ValueError("Anchor continuation horizon is outside the episode.")
    if registered_shape and (
        config.anchors.continuation_horizon != 128 or config.anchors.probe_steps != 16
    ):
        raise ValueError("DEPI anchors require horizon 128 and 16 evidence steps.")
    if config.anchors.return_lower_bound >= config.anchors.return_upper_bound:
        raise ValueError("Anchor return bounds are reversed.")
    frozen_anchor_comparator = {
        "observable_equivalent_probability_max": 0.55,
        "decision_distinct_probability_min": 0.70,
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
            "The registered static wide partner pool is the default R0/B0–B2 partner "
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
        raise ValueError("Partner-pool registered hierarchical sampling must stay enabled.")

    calibration = config.posterior_calibration
    if calibration.minimum_run_count < 2 or calibration.episodes_per_run <= 0:
        raise ValueError(
            "Posterior calibration requires at least two partner-run blocks and positive episodes."
        )
    if calibration.bootstrap_replicates <= 0:
        raise ValueError("Posterior-calibration bootstrap count must be positive.")
    if calibration.root_seed_offset != 2_000:
        raise ValueError("Posterior calibration uses the registered root-key offset 2000.")
    if calibration.credibility != 0.90:
        raise ValueError("Posterior calibration uses 90% highest-probability sets.")
    if not 0.0 <= calibration.coverage_low <= calibration.coverage_high <= 1.0:
        raise ValueError("Posterior-calibration coverage band is invalid.")
    if calibration.log_score_margin != 0.02 or calibration.brier_ratio != 0.90:
        raise ValueError("Posterior-calibration score thresholds differ from registration.")
    if (calibration.coverage_low, calibration.coverage_high) != (0.85, 0.95):
        raise ValueError("Posterior-calibration coverage band is fixed at [0.85, 0.95].")
    if calibration.primary_unit != "partner_run" or calibration.secondary_unit != "episode":
        raise ValueError("Posterior calibration uses partner-run/episode hierarchy.")
    if config.run_kind == "formal" and (
        calibration.minimum_run_count != 20
        or calibration.episodes_per_run != 64
        or calibration.bootstrap_replicates != 9_999
    ):
        raise ValueError(
            "Formal posterior calibration requires 20 runs, 64 episodes per run, "
            "and 9,999 hierarchical bootstrap replicates."
        )

    if not (
        0.0 <= config.training.context_dropout_final
        <= config.training.context_dropout_initial < 1.0
    ):
        raise ValueError("Context dropout must anneal within [0,1).")
    if registered_shape and (
        config.training.context_dropout_initial != 0.30
        or config.training.context_dropout_final != 0.10
    ):
        raise ValueError("DEPI context dropout is frozen at 0.30 -> 0.10.")
    if (
        config.training.owner_sources_per_seed != 1
        or config.training.support_parent_runs_per_family != 10
        or config.training.support_checkpoints_per_parent != 3
        or config.training.support_width_variants != 2
        or config.training.calibration_runs_per_family != 5
    ):
        raise ValueError("Registered DEPI static partner-resource counts changed.")
    if config.training.rollout_length <= 0:
        raise ValueError("Training rollout length must be positive.")
    extra_steps = int(config.training.extra_ppo_environment_steps)
    if extra_steps < 0 or extra_steps % config.environment.num_envs:
        raise ValueError("Extra PPO steps must be non-negative whole vector steps.")
    if extra_steps:
        if config.method_variant not in {"r0", "b0", "b1"}:
            raise ValueError("Only R0/B0/B1 total-budget controls may add PPO steps.")
        base = RUN_BUDGETS[config.run_kind]
        base_updates = base.environment_steps // (
            base.num_envs * config.training.rollout_length
        )
        trigger_count = (
            base_updates + int(config.anchors.interval_updates) - 1
        ) // int(config.anchors.interval_updates)
        expected_extra = trigger_count * (
            (
                int(config.anchors.ordinary_states)
                + 2 * int(config.anchors.matched_history_pairs)
            )
            * OFFICIAL_ACTION_COUNT
            * (
                int(config.anchors.fit_replicas)
                + int(config.anchors.evaluation_replicas)
            )
            * int(config.anchors.continuation_horizon)
            + 2
            * int(config.anchors.matched_history_pairs)
            * int(config.anchors.probe_steps)
        )
        if extra_steps != expected_extra:
            raise ValueError(
                "Total-budget control PPO steps must equal the B2 "
                "continuation/probe transition cost exactly."
            )
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
    if min(
        config.evaluation.mechanism_anchors_per_pairing,
        config.evaluation.mechanism_fit_replicas,
        config.evaluation.mechanism_evaluation_replicas,
        config.evaluation.mechanism_continuation_horizon,
    ) <= 0:
        raise ValueError("Mechanism-control continuation budgets must be positive.")
    if config.evaluation.recoverable_signal_threshold != OFFICIAL_CORRECT_DELIVERY_REWARD:
        raise ValueError(
            "Recoverable-value signal states are preregistered at one delivery (20 points)."
        )
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
    if (
        config.evaluation.mechanism_continuation_horizon
        > config.environment.episode_steps
    ):
        raise ValueError("Mechanism continuation horizon exceeds the episode.")
    if not 0.0 < config.evaluation.one_sided_alpha < 1.0:
        raise ValueError("Evaluation one_sided_alpha must lie in (0, 1).")
    if min(
        config.evaluation.minimum_ego_runs,
        config.evaluation.minimum_partner_runs_per_mechanism,
        config.evaluation.minimum_mechanisms,
    ) <= 0:
        raise ValueError("Evaluation independent-run minima must be positive.")
    if config.evaluation.inference_mode not in {"independent_run", "paired_run"}:
        raise ValueError("Evaluation inference_mode must be independent_run or paired_run.")
    if config.evaluation.superiority_lcb_threshold != 0.0:
        raise ValueError("The preregistered superiority LCB threshold is 0.0.")
    if config.evaluation.minimum_effect != OFFICIAL_CORRECT_DELIVERY_REWARD:
        raise ValueError("The preregistered material effect is one delivery (20 points).")
    if config.evaluation.minimum_effect_rule not in {
        "point_estimate",
        "lower_confidence_bound",
    }:
        raise ValueError("Unknown minimum_effect_rule.")
    if config.run_kind == "formal":
        if not config.evaluation.report_br_prox:
            raise ValueError(
                "The DEPI design requires empirical BR-Prox in the formal "
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
            raise ValueError("Official DEPI formal training uses 256 environments.")
        if config.training.environment_steps != 29_949_952:
            raise ValueError("Official DEPI formal training executes 29,949,952 steps.")
        if config.training.minibatches_per_epoch != OFFICIAL_NUM_MINIBATCHES:
            raise ValueError("Official DEPI formal training uses 64 minibatches.")
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
    base_environment_steps = (
        config.training.environment_steps
        - config.training.extra_ppo_environment_steps
    )
    if base_environment_steps % rollout_steps:
        raise ValueError("Base training budget must contain whole vectorized rollouts.")
    if config.training.environment_steps % config.environment.num_envs:
        raise ValueError("Training budget must contain whole vector steps.")
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
                "checkpoint_stage",
                "hyperparameter_family",
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
            checkpoint_stage=float(raw["checkpoint_stage"]),
            hyperparameter_family=str(raw["hyperparameter_family"]),
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
                f"Every DEPI partner run must record its real two-word JAX key: "
                f"{run.run_id}"
            )
        if float(run.checkpoint_stage) not in {0.0, 0.5, 1.0}:
            raise ValueError(
                f"Partner checkpoint stage must be one of 0/0.5/1: {run.run_id}"
            )
        if run.role != "development_support" and run.checkpoint_stage != 1.0:
            raise ValueError(
                f"Only development support may use intermediate stages: {run.run_id}"
            )
        if not run.hyperparameter_family.strip():
            raise ValueError(f"Partner hyperparameter family is empty: {run.run_id}")
        if (
            run.role != "development_support"
            and run.hyperparameter_family != "default"
        ):
            raise ValueError(
                f"Non-support partners must use hyperparameter_family=default: {run.run_id}"
            )
        if run.role == "confirmatory" and run.owner_seed_index is not None:
            raise ValueError(
                f"Common confirmatory partner cannot belong to one ego run: {run.run_id}"
            )
        official_parent = (
            run.role in {
                "owner_source",
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
        if run.role in {
            "comparator_fit",
            "comparator_validation",
            "calibration",
            "confirmatory",
        }:
            if run.seed_index is not None:
                raise ValueError(
                    "Fresh comparator/calibration/confirmatory partners cannot reuse Official "
                    f"seed indexes: {run.run_id}"
                )
            official_keys = {
                tuple(official_training_key(index))
                for index in range(OFFICIAL_TRAINING_RUN_COUNT)
            }
            if tuple(run.jax_prng_key or ()) in official_keys:
                raise ValueError(
                    "Fresh comparator/calibration/confirmatory partner reuses a formal "
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
        if run.role in {
            "comparator_fit",
            "comparator_validation",
            "calibration",
            "confirmatory",
        }
    ]
    if len(evaluation_keys) != len(set(evaluation_keys)):
        raise ValueError(
            "Fresh comparator/calibration/confirmatory partners must use distinct JAX keys."
        )

    for role in (
        "owner_source",
        "development_support",
        "comparator_fit",
        "comparator_validation",
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
                f"{role} parent runs cannot be shared across DEPI outer runs: "
                f"{shared_parents}"
            )

    train_roles = {
        "owner_source",
        "development_support",
        "comparator_fit",
        "comparator_validation",
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

    isolated_roles = (
        "development_support",
        "comparator_fit",
        "comparator_validation",
        "calibration",
        "confirmatory",
    )
    parents_by_role = {
        role: {
            run.parent_training_run_id
            for run in manifest.runs
            if run.role == role
        }
        for role in isolated_roles
    }
    groups_by_role = {
        role: {
            run.co_training_group_id
            for run in manifest.runs
            if run.role == role and run.co_training_group_id is not None
        }
        for role in isolated_roles
    }
    for left_index, left in enumerate(isolated_roles):
        for right in isolated_roles[left_index + 1 :]:
            parent_overlap = parents_by_role[left] & parents_by_role[right]
            group_overlap = groups_by_role[left] & groups_by_role[right]
            if parent_overlap or group_overlap:
                raise ValueError(
                    "Partner data roles are not lineage-disjoint: "
                    f"{left}/{right}, parents={sorted(parent_overlap)}, "
                    f"groups={sorted(group_overlap)}."
                )

    development_support = manifest.by_role("development_support")
    comparator_fit = manifest.by_role("comparator_fit")
    comparator_validation = manifest.by_role("comparator_validation")
    calibration = manifest.by_role("calibration")
    confirmatory = manifest.by_role("confirmatory")
    if development_support and len({run.parent_training_run_id for run in development_support}) < 2:
        raise ValueError("Training support requires at least two independent frozen external runs.")
    if comparator_fit and len({run.parent_training_run_id for run in comparator_fit}) < 2:
        raise ValueError("Comparator fitting requires at least two independent parent runs.")
    if comparator_validation and len({run.parent_training_run_id for run in comparator_validation}) < 2:
        raise ValueError("Comparator validation requires at least two independent parent runs.")
    if calibration and len({run.parent_training_run_id for run in calibration}) < 2:
        raise ValueError("Calibration requires at least two independent parent runs.")
    if confirmatory and len({run.parent_training_run_id for run in confirmatory}) < 2:
        raise ValueError("Confirmatory evaluation requires at least two independent parent runs.")


def validate_seed_training_manifest(
    manifest: PartnerManifest,
    *,
    config: RunConfig,
    owner_seed_index: int,
    formal: bool,
) -> None:
    """Enforce the lineage-disjoint, static DEPI training distribution."""

    validate_partner_manifest(manifest)
    if config.environment.layout != manifest.layout:
        raise ValueError("Seed training manifest layout differs from the run config.")
    owner = int(owner_seed_index)
    if formal and not 0 <= owner < OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Formal DEPI owner seed must lie in 0..9.")
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
        raise ValueError("A per-seed DEPI manifest contains another seed's resource.")

    owners = tuple(run for run in owned if run.role == "owner_source")
    if len(owners) != 1 or owners[0].generation_mechanism not in {"rnn-sp", "sp"}:
        raise ValueError("Each DEPI seed needs exactly one independent owner-SP source.")

    mechanism_alias = {
        "rnn-sp": "sp",
        "sp": "sp",
        "rnn-op": "op",
        "op": "op",
        "state-augmented": "sa",
        "sa": "sa",
        "fcp": "fcp",
    }
    support = manifest.by_role("development_support")
    if any(run.owner_seed_index is not None for run in support):
        raise ValueError(
            "The registered static support pool is shared and must not be owned by an ego seed."
        )
    if {mechanism_alias.get(run.generation_mechanism) for run in support} - {
        "sp",
        "op",
    }:
        raise ValueError("Static training support may contain only SP and OP mechanisms.")

    owner_parent = owners[0].parent_training_run_id
    support_parents = {run.parent_training_run_id for run in support}
    if owner_parent in support_parents:
        raise ValueError("Owner initialization and static support share a parent run.")
    owner_group = owners[0].co_training_group_id
    support_groups = {
        run.co_training_group_id
        for run in support
        if run.co_training_group_id is not None
    }
    if owner_group is not None and owner_group in support_groups:
        raise ValueError("Owner initialization and static support share co-training lineage.")

    if formal:
        comparator_fit = manifest.by_role("comparator_fit")
        comparator_validation = manifest.by_role("comparator_validation")
        for role, runs in (
            ("comparator_fit", comparator_fit),
            ("comparator_validation", comparator_validation),
        ):
            if any(run.owner_seed_index is not None for run in runs):
                raise ValueError(f"Formal {role} runs must be shared and ego-owner independent.")
            if len({run.parent_training_run_id for run in runs}) < 8:
                raise ValueError(
                    f"Formal {role} needs at least eight independent fresh parent runs."
                )
            if len(runs) > config.environment.num_envs // 8:
                raise ValueError(
                    f"Formal {role} exceeds its deterministic reserved-lane capacity."
                )
            if any(
                mechanism_alias.get(run.generation_mechanism) not in {"sp", "op"}
                or run.checkpoint_stage != 1.0
                or run.hyperparameter_family != "default"
                for run in runs
            ):
                raise ValueError(
                    f"Formal {role} accepts only fresh final-stage default SP/OP runs."
                )
        default_support = tuple(
            run for run in support if run.hyperparameter_family == "default"
        )
        width_support = tuple(
            run for run in support if run.hyperparameter_family != "default"
        )
        for mechanism in ("sp", "op"):
            rows = tuple(
                run
                for run in default_support
                if mechanism_alias[run.generation_mechanism] == mechanism
            )
            parents = {run.parent_training_run_id for run in rows}
            if len(parents) != 10:
                raise ValueError(
                    f"Formal static support needs 10 independent {mechanism.upper()} parents."
                )
            for parent in parents:
                stages = {
                    run.checkpoint_stage
                    for run in rows
                    if run.parent_training_run_id == parent
                }
                if stages != {0.0, 0.5, 1.0}:
                    raise ValueError(
                        f"Static support parent {parent} must provide stages 0/0.5/1."
                    )
                if sum(run.parent_training_run_id == parent for run in rows) != 3:
                    raise ValueError(
                        f"Static support parent {parent} has duplicate stage entries."
                    )
        if (
            len(width_support) != 2
            or any(
                mechanism_alias[run.generation_mechanism] != "op"
                or run.checkpoint_stage != 1.0
                for run in width_support
            )
            or len({run.parent_training_run_id for run in width_support}) != 2
        ):
            raise ValueError(
                "Formal static support needs exactly two independent final-stage OP width variants."
            )

        calibration = tuple(manifest.by_role("calibration"))
        if calibration:
            if any(run.owner_seed_index is not None for run in calibration):
                raise ValueError(
                    "The formal calibration panel is shared across ego seeds."
                )
            observed = {name: 0 for name in ("sp", "op", "sa", "fcp")}
            for run in calibration:
                mechanism = mechanism_alias.get(run.generation_mechanism)
                if mechanism is None:
                    raise ValueError(
                        f"Unknown calibration mechanism: {run.generation_mechanism}"
                    )
                observed[mechanism] += 1
            if any(value != 5 for value in observed.values()):
                raise ValueError(
                    "Formal posterior calibration needs five fresh runs per mechanism; "
                    f"observed={observed}."
                )


__all__ = [
    "AnchorConfig",
    "CONFIG_VERSION",
    "ENGINEERING_SEED_INDEX",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LAYOUTS",
    "LossV2Config",
    "MANIFEST_VERSION",
    "METHOD_VERSION",
    "METHOD_VARIANTS",
    "MECHANISM_ABLATION_VARIANTS",
    "CHECKPOINT_SCHEMA_VERSION",
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
    "PartnerManifest",
    "PartnerPoolConfig",
    "PartnerRun",
    "PosteriorCalibrationConfig",
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
