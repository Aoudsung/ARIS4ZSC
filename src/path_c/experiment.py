"""Single configuration and manifest authority for DELTA-ZSC v5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_VERSION = 7
METHOD_VERSION = "delta_zsc_v5_decision_equivalent_bayes_r3_signal_contract"
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
    belief_hidden_dim: int
    latent_dim: int
    mixture_components: int
    posterior_particles: int
    belief_embedding_dim: int
    actor_hidden_dim: int
    critic_hidden_dim: int
    response_hidden_dim: int
    modulation_rank: int
    action_embedding_dim: int
    log_variance_minimum: float
    log_variance_maximum: float


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
    max_approx_kl: float
    normalize_advantages: bool
    polyak_coefficient: float
    lr_warmup_fraction: float
    anneal_learning_rate: bool
    adam_epsilon: float


@dataclass(frozen=True, slots=True)
class LossConfig:
    response_weight: float
    raw_q_weight: float
    q_policy_weight: float
    q_policy_temperature: float
    information_bottleneck_weight: float
    information_bottleneck_free_bits: float
    decision_regret_weight_maximum: float
    conditional_entropy_weight: float
    conditional_kl_weight: float
    residual_norm_weight: float
    response_updates_per_outer_update: int
    raw_q_updates_per_outer_update: int

    @property
    def decision_regret_weight(self) -> float:
        """Unqualified default; the r3 state machine supplies 0.1 only after C5."""

        return 0.0


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    interval_updates: int
    pilot_ordinary_candidates: int
    pilot_matched_code_candidates: int
    pilot_replicas: int
    selected_ordinary: int
    selected_matched_code: int
    fit_replicas: int
    continuation_horizon: int
    probe_steps: int
    replay_capacity_per_epoch: int
    audit_milestones: tuple[str, ...]
    audit_ordinary_states: int
    audit_matched_code_states: int
    audit_fit_replicas: int
    audit_evaluation_replicas: int
    audit_continuation_horizon: int
    audit_uniform_fraction: float
    return_lower_bound: float
    return_upper_bound: float

    # Transitional read-only aliases used by audit/calibration utilities.  The
    # r3 training lifecycle uses the explicit training/audit fields above.
    @property
    def states_per_interval(self) -> int:
        return self.selected_ordinary + self.selected_matched_code

    @property
    def evaluation_replicas(self) -> int:
        return self.audit_evaluation_replicas

    @property
    def sampling_time_bins(self) -> int:
        return 1

    @property
    def sampling_regret_bins(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class PartnerGeneratorConfig:
    code_dim: int
    hidden_dim: int
    modulation_rank: int
    initial_current_probability: float
    initial_snapshot_probability: float
    initial_frozen_external_probability: float
    episodes_per_update: int
    episode_steps: int
    update_epochs: int
    environment_minibatches: int
    snapshot_interval: int
    cvar_level: float
    source_distillation_kl_maximum: float
    trust_region_kl_maximum: float
    interpolation_uniform_kl_minimum: float
    delivery_noninferiority_margin: float
    brdiv_weight: float
    smoothness_weight: float
    lagrangian_learning_rate: float
    kernel_bandwidth: float
    kernel_jitter: float
    codes_per_update: int

    @property
    def current_probability(self) -> float:
        return self.initial_current_probability

    @property
    def snapshot_probability(self) -> float:
        return self.initial_snapshot_probability

    @property
    def frozen_external_probability(self) -> float:
        return self.initial_frozen_external_probability

    @property
    def update_interval(self) -> int:
        return 1

    @property
    def competence_threshold(self) -> float:
        raise AttributeError("r3 uses paired noninferiority, not an absolute threshold")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    alpha: float
    support_quantile: float
    minimum_run_count: int
    block_unit: str
    enable_hard_gate_at_evaluation: bool
    episodes_per_run: int
    anchors_per_run: int
    always_on_ablation: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    environment_steps: int
    minibatches_per_epoch: int
    checkpoint_interval_environment_steps: int
    rollout_length: int
    base_policy_lane_probability: float
    conditional_policy_lane_probability: float
    target_policy_epoch_updates: int
    base_qualification_deadline_steps: int
    owner_sources_per_seed: int
    generator_init_sources_per_mechanism: int
    development_runs_per_mechanism: int
    calibration_runs_per_mechanism: int

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    bootstrap_replicates: int
    evaluation_seed: int
    enable_adaptation: bool
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
    anchors: AnchorConfig
    partner_generator: PartnerGeneratorConfig
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
        b"delta-zsc-v5-r3/signal-contract/engineering-seed"
    ).digest()
    return (
        int.from_bytes(digest[:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    )


def official_training_domain_keys(seed_index: int) -> Mapping[str, tuple[int, int]]:
    """Derive registered independent domains from one Official outer-run key.

    Domain integers are not hand-selected hyperparameters: each is the first
    unsigned 32-bit word of SHA-256(``delta-zsc-v5/<domain>``), recorded in the
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
        "snapshot",
        "training_anchor",
        "audit_anchor",
        "qualification",
        "calibration",
    ):
        tag = int.from_bytes(
            hashlib.sha256(f"delta-zsc-v5/{name}".encode("utf-8")).digest()[:4],
            "big",
        )
        key = np.asarray(jax.random.fold_in(root, tag), dtype=np.uint32)
        result[name] = (int(key[0]), int(key[1]))
    return result



def load_config(path: str | Path, *, run_kind: str) -> RunConfig:
    """Load config v7 and apply the registered run budget."""

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
            "anchors",
            "partner_generator",
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
    anchor_payload = _exact_fields(payload["anchors"], set(AnchorConfig.__dataclass_fields__), "anchors")
    generator_payload = _exact_fields(
        payload["partner_generator"],
        set(PartnerGeneratorConfig.__dataclass_fields__),
        "partner_generator",
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
        anchors=AnchorConfig(
            **{
                **anchor_payload,
                "audit_milestones": tuple(anchor_payload["audit_milestones"]),
            }
        ),
        partner_generator=PartnerGeneratorConfig(**generator_payload),
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
    if config.model.latent_dim <= 0 or config.model.mixture_components <= 0:
        raise ValueError("Continuous belief dimensions must be positive.")
    if config.model.posterior_particles < config.model.mixture_components:
        raise ValueError("Posterior particles must cover every mixture component.")
    if config.model.modulation_rank <= 0:
        raise ValueError("Low-rank actor modulation needs positive rank.")
    if config.model.log_variance_minimum >= config.model.log_variance_maximum:
        raise ValueError("Posterior log-variance bounds are reversed.")

    if not 0.0 < config.ppo.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if not 0.0 <= config.ppo.gae_lambda <= 1.0:
        raise ValueError("GAE lambda must lie in [0, 1].")
    if config.ppo.update_epochs <= 0 or config.ppo.learning_rate <= 0.0:
        raise ValueError("PPO update count and learning rate must be positive.")
    if not 0.0 < config.ppo.polyak_coefficient <= 1.0:
        raise ValueError("Polyak coefficient must lie in (0, 1].")
    if not 0.0 <= config.ppo.lr_warmup_fraction < 1.0:
        raise ValueError("LR warmup fraction must lie in [0, 1).")
    if config.ppo.adam_epsilon <= 0.0:
        raise ValueError("Adam epsilon must be positive.")

    probabilities = (
        config.partner_generator.initial_current_probability,
        config.partner_generator.initial_snapshot_probability,
        config.partner_generator.initial_frozen_external_probability,
    )
    if any(value < 0.0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError("Partner-source probabilities must be non-negative and sum to one.")
    if config.partner_generator.code_dim <= 0:
        raise ValueError("Partner generator code dimension must be positive.")
    if not 0.0 < config.partner_generator.cvar_level <= 1.0:
        raise ValueError("Generator CVaR level must lie in (0, 1].")
    registered_shape = config.run_kind != "mechanical"
    if registered_shape and config.partner_generator.episodes_per_update != 32:
        raise ValueError("r3 generator requires 32 complete episodes per outer update.")
    if config.partner_generator.episode_steps != OFFICIAL_EPISODE_STEPS:
        raise ValueError("Generator episodes must use all 400 Official steps.")
    if registered_shape and config.partner_generator.update_epochs != 4:
        raise ValueError("Generator PPO uses four Official update epochs.")
    if registered_shape and config.partner_generator.environment_minibatches != 8:
        raise ValueError("Generator PPO uses eight environment minibatches.")
    if abs(config.partner_generator.trust_region_kl_maximum - 0.03) > 1e-12:
        raise ValueError("Generator trust region must remain 0.03.")
    if config.partner_generator.interpolation_uniform_kl_minimum <= 0.0:
        raise ValueError(
            "Generator interpolation must be registered as measurably non-uniform."
        )
    if abs(
        config.partner_generator.delivery_noninferiority_margin
        - OFFICIAL_CORRECT_DELIVERY_REWARD
    ) > 1.0e-12:
        raise ValueError(
            "The competence noninferiority margin must equal one Official "
            "correct-delivery reward."
        )

    if registered_shape and config.anchors.interval_updates != 16:
        raise ValueError("Training anchors must align to 16-update target-policy epochs.")
    if registered_shape and (
        config.anchors.pilot_replicas != 2 or config.anchors.fit_replicas != 8
    ):
        raise ValueError("Training anchor pilot/fit replicas must remain 2/8.")
    if registered_shape and (
        config.anchors.selected_ordinary != 24
        or config.anchors.selected_matched_code != 24
    ):
        raise ValueError("Training anchor selected strata must remain 24/24.")
    if registered_shape and (
        config.anchors.pilot_ordinary_candidates != 128
        or config.anchors.pilot_matched_code_candidates != 64
    ):
        raise ValueError("Training anchor pilot candidate strata must remain 128/64.")
    if not 1 <= config.anchors.continuation_horizon <= config.environment.episode_steps:
        raise ValueError("Anchor continuation horizon is outside the episode.")
    if registered_shape and (
        config.anchors.continuation_horizon != 128 or config.anchors.probe_steps != 16
    ):
        raise ValueError("r3 training anchors require horizon 128 and 16 evidence steps.")
    if tuple(config.anchors.audit_milestones) != ("C0", "15M", "22.5M", "final"):
        raise ValueError("Audit anchor milestones differ from the preregistration.")
    if registered_shape and (
        config.anchors.audit_ordinary_states != 32
        or config.anchors.audit_matched_code_states != 32
        or config.anchors.audit_fit_replicas != 32
        or config.anchors.audit_evaluation_replicas != 64
        or config.anchors.audit_continuation_horizon != 400
        or abs(config.anchors.audit_uniform_fraction - 0.25) > 1e-12
    ):
        raise ValueError("Audit anchor split/horizon/uniform fraction changed.")
    if config.anchors.return_lower_bound >= config.anchors.return_upper_bound:
        raise ValueError("Anchor return bounds are reversed.")

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
    if config.run_kind == "formal" and config.calibration.always_on_ablation:
        raise ValueError("Formal Full DELTA cannot enable the always-on ablation.")

    lane_sum = (
        config.training.base_policy_lane_probability
        + config.training.conditional_policy_lane_probability
    )
    if abs(lane_sum - 1.0) > 1e-8:
        raise ValueError("Exclusive base and conditional lane probabilities must sum to one.")
    if registered_shape and config.training.target_policy_epoch_updates != 16:
        raise ValueError("Target-policy epoch must remain 16 updates.")
    if registered_shape and config.training.base_qualification_deadline_steps != 15_000_000:
        raise ValueError("C0 deadline must remain 15M ego steps.")
    if (
        config.training.owner_sources_per_seed != 1
        or config.training.generator_init_sources_per_mechanism != 1
        or config.training.development_runs_per_mechanism != 4
        or config.training.calibration_runs_per_mechanism != 5
    ):
        raise ValueError("Per-seed r3 partner resource counts changed.")
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
        if not config.calibration.enable_hard_gate_at_evaluation:
            raise ValueError(
                "The DELTA-ZSC design requires the frozen run-block conformal "
                "hard gate for formal confirmatory evaluation."
            )
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
        elif run.seed_index is not None and run.jax_prng_key is None:
            raise ValueError(f"Indexed partner lacks its explicit JAX key: {run.run_id}")
        if run.role in {"calibration", "confirmatory"} and run.jax_prng_key is None:
            raise ValueError(
                f"Run-disjoint evaluation partner lacks JAX key provenance: {run.run_id}"
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


def validate_seed_signal_contract_manifest(
    manifest: PartnerManifest,
    *,
    owner_seed_index: int,
    formal: bool,
) -> None:
    """Enforce the per-seed, lineage-exclusive r3 resource contract."""

    validate_partner_manifest(manifest)
    owner = int(owner_seed_index)
    if formal and not 0 <= owner < OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Formal signal-contract owner seed must lie in 0..9.")
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
        raise ValueError("A per-seed signal-contract manifest contains another seed's resource.")

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
                raise ValueError(f"Unknown r3 partner mechanism: {run.generation_mechanism}")
            result[mechanism] += 1
        return result

    expected = {
        "generator_init_source": 1,
        "development_support": 4,
        "calibration": 5,
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
        for role in ("generator_init_source", "development_support", "calibration")
    }
    roles = tuple(role_groups)
    for index, first in enumerate(roles):
        for second in roles[index + 1 :]:
            first_parents = {row[1] for row in role_groups[first]}
            second_parents = {row[1] for row in role_groups[second]}
            if first_parents & second_parents:
                raise ValueError(f"r3 resource groups {first}/{second} share a parent run.")
            first_groups = {row[2] for row in role_groups[first] if row[2] is not None}
            second_groups = {row[2] for row in role_groups[second] if row[2] is not None}
            if first_groups & second_groups:
                raise ValueError(f"r3 resource groups {first}/{second} share co-training lineage.")


__all__ = [
    "AnchorConfig",
    "CONFIG_VERSION",
    "ENGINEERING_SEED_INDEX",
    "CalibrationConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LAYOUTS",
    "LossConfig",
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
    "validate_seed_signal_contract_manifest",
]
