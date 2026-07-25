"""机会审计实验编号 R015 的静态机器合同。

本模块验证冻结设计，并裁决已经产生的配对回报。它不运行环境、不选择动作，也不构造
伙伴后验。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from experiments.overcooked_v2.path_c_pool_admission import (
    R015PartnerSupportSpec,
    load_r015_partner_support_config,
    validate_r015_support_report_evidence,
)
from experiments.overcooked_v2.path_c_standard_training import (
    partner_training_run_id,
)


R015_PREREGISTRATION_SCHEMA = "path_c_r015_preregistration_v2"
R015_PAIRED_BLOCK_SCHEMA = "path_c_r015_paired_block_v2"
R015_FORMAL_DATASET_SCHEMA = "path_c_r015_formal_dataset_v3"
R015_FIRING_COUNT_SCHEMA = "path_c_r015_firing_count_checkpoint_v1"
R015_SUPPORT_REPORT_SCHEMA = "path_c_r015_partner_support_report_v2"
R015_EXPERIMENT_ID = "R015"
R015_FORMAL_GROUPS = ("A1", "A2-mask", "A2-use")
R015_DIAGNOSTIC_GROUP = "A0"
R015_CONTINUATION_CONTROLLER_ID = "map_prototype_committed_cook_v1"
R015_BRANCH_SAMPLING_ID = "sampled_hidden_state_branches_v1"
R015_BRANCH_BELIEF_ID = "frozen_belief_branch_continuation_v1"
R015_GRID_EVALUATION_ID = "lazy_ascending_first_pass"
R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA = "path_c_r015_formal_sampling_schedule_v4"
R015_FORMAL_ROUND_SAMPLING_CONTRACT = (
    "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
)
R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID = (
    "sha256_registered_coordinate_low32_allow_collisions_v1"
)
R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT = 32
R015_NESTED_BRANCH_COUNTS = (2, 4, 8, 16)
R015_FILTER_ALGORITHM_ID_V2 = "official_history_fully_adapted_particle_filter_v2"
R015_RESAMPLING_ALGORITHM_ID_V2 = "strict_systematic_per_prototype_v2"
R015_FUTURE_RANDOM_DERIVATION_ID = (
    "path_c_r015_controller_key_v1_future_environment_index_v1"
)
R015_EGO_BASELINE_MEMBER_ID = "official_rnn_sp_ippo_v1_seed100_step29949952"
_HEX_DIGITS = frozenset("0123456789abcdef")
_NO_PROBE_REASONS = frozenset(
    {
        "non_positive_score",
        "window_expired",
        "support_incompatible",
        "safety_rejected",
        "no_safe_candidate",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_pending(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower().startswith("pending")
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_DIGITS)
    )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence.")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    return number


def _optional_finite_float(value: Any, *, field: str) -> float | None:
    if _is_pending(value):
        return None
    return _finite_float(value, field=field)


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if _is_pending(value):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer or pending.")
    return int(value)


def _optional_positive_probability_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, float] | None:
    if _is_pending(value):
        return None
    payload = _require_mapping(value, field=field)
    if len(payload) != 4 or any(
        not isinstance(key, str) or not key for key in payload
    ):
        raise ValueError(f"{field} must contain exactly four prototype ids.")
    result = {
        str(key): _finite_float(probability, field=f"{field}.{key}")
        for key, probability in payload.items()
    }
    if any(probability <= 0.0 for probability in result.values()) or not math.isclose(
        sum(result.values()),
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{field} values must be positive and sum to one.")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return int(value)


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean.")
    return value


def _resolve_path(base: Path, value: Any) -> Path | None:
    if _is_pending(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_path_sequence(base: Path, value: Any) -> tuple[Path, ...] | None:
    if _is_pending(value):
        return None
    return tuple(
        path
        for item in _require_sequence(value, field="firing_count_checkpoint_paths")
        if (path := _resolve_path(base, item)) is not None
    )


@dataclass(frozen=True)
class ArtifactBinding:
    """One pre-run object bound by a path and a SHA-256 content digest."""

    name: str
    path: Path | None
    sha256: str | None

    @classmethod
    def from_mapping(
        cls,
        name: str,
        payload: Mapping[str, Any],
        *,
        base: Path,
    ) -> "ArtifactBinding":
        path = _resolve_path(base, payload.get("path"))
        raw_sha = payload.get("sha256")
        sha256 = None if _is_pending(raw_sha) else str(raw_sha)
        if sha256 is not None and not _is_sha256(sha256):
            raise ValueError(f"artifacts.{name}.sha256 must be a SHA-256 digest.")
        return cls(name=name, path=path, sha256=sha256)

    def require_ready(self) -> None:
        if self.path is None or self.sha256 is None:
            raise ValueError(f"Artifact binding {self.name!r} is still pending.")
        if not self.path.is_file():
            raise ValueError(f"Artifact binding {self.name!r} does not exist.")
        if _file_sha256(self.path) != self.sha256:
            raise ValueError(f"Artifact binding {self.name!r} changed after freeze.")


@dataclass(frozen=True)
class R015Statistics:
    """Frozen round-level empirical-Bernstein, firing-count, and safety rules."""

    partner_prototype_count: int
    pilot_blocks_per_prototype: int
    n_rounds_max: int
    formal_paired_block_budget: int
    formal_arm_episode_budget: int
    formal_arm_environment_step_budget: int
    maximum_safety_comparisons: int
    planning_fork_steps_included: bool
    delivery_reward: float
    practical_margin_fraction: float
    return_lower_bound: float
    return_upper_bound: float
    paired_difference_absolute_bound: float
    paired_difference_width: float
    empirical_bernstein_formula: str
    round_sampling_contract: str
    formal_sampling_schedule_schema: str
    mechanical_replacement_attempts_per_coordinate: int
    mechanical_replacement_attempt_indices: str
    replacement_episode_seed_derivation_id: str
    replacement_episode_seed_collision_policy: str
    audit_unit_changes_across_replacement_attempts: bool
    replacement_seed_selection_uses_outcomes: bool
    mechanical_attempt_exhaustion_rule: str
    ego_agent_id: str
    zero_count_independence_contract: str
    alpha_total: float
    alpha_safety: float
    alpha_eb_net: float
    alpha_eb_response: float
    alpha_rho: float
    kill_checkpoint_rounds: tuple[int, ...]
    alpha_rho_per_checkpoint: float
    alpha_rho_per_prototype_cp: float
    safety_risk_limit: float
    safety_repeats_per_comparison: int
    formal_effect_look_count: int
    formal_effect_look_round: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "R015Statistics":
        retired_seat_fields = {
            "ego_position_one_probability",
            "ego_position_schedule_conditioned_on_balance",
        }
        if retired_seat_fields.intersection(payload):
            raise ValueError(
                "R015 statistics retain a retired random-seat field; the ego cook "
                "must stay at agent_1."
            )
        result = cls(
            partner_prototype_count=_positive_int(
                payload["partner_prototype_count"],
                field="statistics.partner_prototype_count",
            ),
            pilot_blocks_per_prototype=_positive_int(
                payload["pilot_blocks_per_prototype"],
                field="statistics.pilot_blocks_per_prototype",
            ),
            n_rounds_max=_positive_int(
                payload["n_rounds_max"],
                field="statistics.n_rounds_max",
            ),
            formal_paired_block_budget=_positive_int(
                payload["formal_paired_block_budget"],
                field="statistics.formal_paired_block_budget",
            ),
            formal_arm_episode_budget=_positive_int(
                payload["formal_arm_episode_budget"],
                field="statistics.formal_arm_episode_budget",
            ),
            formal_arm_environment_step_budget=_positive_int(
                payload["formal_arm_environment_step_budget"],
                field="statistics.formal_arm_environment_step_budget",
            ),
            maximum_safety_comparisons=_positive_int(
                payload["maximum_safety_comparisons"],
                field="statistics.maximum_safety_comparisons",
            ),
            planning_fork_steps_included=_boolean(
                payload.get("planning_fork_steps_included"),
                field="statistics.planning_fork_steps_included",
            ),
            delivery_reward=_finite_float(
                payload["delivery_reward"],
                field="statistics.delivery_reward",
            ),
            practical_margin_fraction=_finite_float(
                payload["practical_margin_fraction"],
                field="statistics.practical_margin_fraction",
            ),
            return_lower_bound=_finite_float(
                payload["return_lower_bound"],
                field="statistics.return_lower_bound",
            ),
            return_upper_bound=_finite_float(
                payload["return_upper_bound"],
                field="statistics.return_upper_bound",
            ),
            paired_difference_absolute_bound=_finite_float(
                payload["paired_difference_absolute_bound"],
                field="statistics.paired_difference_absolute_bound",
            ),
            paired_difference_width=_finite_float(
                payload["paired_difference_width"],
                field="statistics.paired_difference_width",
            ),
            empirical_bernstein_formula=str(
                payload.get("empirical_bernstein_formula", "")
            ),
            round_sampling_contract=str(payload.get("round_sampling_contract", "")),
            formal_sampling_schedule_schema=str(
                payload.get("formal_sampling_schedule_schema", "")
            ),
            mechanical_replacement_attempts_per_coordinate=_positive_int(
                payload.get("mechanical_replacement_attempts_per_coordinate"),
                field="statistics.mechanical_replacement_attempts_per_coordinate",
            ),
            mechanical_replacement_attempt_indices=str(
                payload.get("mechanical_replacement_attempt_indices", "")
            ),
            replacement_episode_seed_derivation_id=str(
                payload.get("replacement_episode_seed_derivation_id", "")
            ),
            replacement_episode_seed_collision_policy=str(
                payload.get("replacement_episode_seed_collision_policy", "")
            ),
            audit_unit_changes_across_replacement_attempts=_boolean(
                payload.get("audit_unit_changes_across_replacement_attempts"),
                field="statistics.audit_unit_changes_across_replacement_attempts",
            ),
            replacement_seed_selection_uses_outcomes=_boolean(
                payload.get("replacement_seed_selection_uses_outcomes"),
                field="statistics.replacement_seed_selection_uses_outcomes",
            ),
            mechanical_attempt_exhaustion_rule=str(
                payload.get("mechanical_attempt_exhaustion_rule", "")
            ),
            ego_agent_id=str(payload.get("ego_agent_id", "")),
            zero_count_independence_contract=str(
                payload.get("zero_count_independence_contract", "")
            ),
            alpha_total=_finite_float(
                payload["alpha_total"], field="statistics.alpha_total"
            ),
            alpha_safety=_finite_float(
                payload["alpha_safety"], field="statistics.alpha_safety"
            ),
            alpha_eb_net=_finite_float(
                payload["alpha_eb_net"], field="statistics.alpha_eb_net"
            ),
            alpha_eb_response=_finite_float(
                payload["alpha_eb_response"], field="statistics.alpha_eb_response"
            ),
            alpha_rho=_finite_float(
                payload["alpha_rho"], field="statistics.alpha_rho"
            ),
            kill_checkpoint_rounds=tuple(
                _positive_int(item, field="statistics.kill_checkpoint_rounds")
                for item in _require_sequence(
                    payload.get("kill_checkpoint_rounds"),
                    field="statistics.kill_checkpoint_rounds",
                )
            ),
            alpha_rho_per_checkpoint=_finite_float(
                payload["alpha_rho_per_checkpoint"],
                field="statistics.alpha_rho_per_checkpoint",
            ),
            alpha_rho_per_prototype_cp=_finite_float(
                payload["alpha_rho_per_prototype_cp"],
                field="statistics.alpha_rho_per_prototype_cp",
            ),
            safety_risk_limit=_finite_float(
                payload["safety_risk_limit"],
                field="statistics.safety_risk_limit",
            ),
            safety_repeats_per_comparison=_positive_int(
                payload["safety_repeats_per_comparison"],
                field="statistics.safety_repeats_per_comparison",
            ),
            formal_effect_look_count=_positive_int(
                payload["formal_effect_look_count"],
                field="statistics.formal_effect_look_count",
            ),
            formal_effect_look_round=_positive_int(
                payload["formal_effect_look_round"],
                field="statistics.formal_effect_look_round",
            ),
        )
        result._validate_fixed_design()
        return result

    def _validate_fixed_design(self) -> None:
        fixed_values = {
            "partner_prototype_count": (self.partner_prototype_count, 4),
            "pilot_blocks_per_prototype": (self.pilot_blocks_per_prototype, 20),
            "n_rounds_max": (self.n_rounds_max, 2500),
            "formal_paired_block_budget": (self.formal_paired_block_budget, 10000),
            "formal_arm_episode_budget": (self.formal_arm_episode_budget, 30000),
            "formal_arm_environment_step_budget": (
                self.formal_arm_environment_step_budget,
                12000000,
            ),
            "maximum_safety_comparisons": (
                self.maximum_safety_comparisons,
                40000,
            ),
            "mechanical_replacement_attempts_per_coordinate": (
                self.mechanical_replacement_attempts_per_coordinate,
                R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT,
            ),
            "audit_unit_changes_across_replacement_attempts": (
                self.audit_unit_changes_across_replacement_attempts,
                False,
            ),
            "replacement_seed_selection_uses_outcomes": (
                self.replacement_seed_selection_uses_outcomes,
                False,
            ),
            "formal_effect_look_count": (self.formal_effect_look_count, 1),
            "formal_effect_look_round": (self.formal_effect_look_round, 2500),
            "safety_repeats_per_comparison": (
                self.safety_repeats_per_comparison,
                279,
            ),
        }
        for name, (actual, expected) in fixed_values.items():
            if actual != expected:
                raise ValueError(f"statistics.{name} must equal {expected}.")
        fixed_probabilities = {
            "practical_margin_fraction": (self.practical_margin_fraction, 0.25),
            "delivery_reward": (self.delivery_reward, 20.0),
            "return_lower_bound": (self.return_lower_bound, -400.0),
            "return_upper_bound": (self.return_upper_bound, 400.0),
            "paired_difference_absolute_bound": (
                self.paired_difference_absolute_bound,
                800.0,
            ),
            "paired_difference_width": (self.paired_difference_width, 1600.0),
            "alpha_total": (self.alpha_total, 0.05),
            "alpha_safety": (self.alpha_safety, 0.025),
            "alpha_eb_net": (self.alpha_eb_net, 0.010),
            "alpha_eb_response": (self.alpha_eb_response, 0.010),
            "alpha_rho": (self.alpha_rho, 0.005),
            "alpha_rho_per_checkpoint": (
                self.alpha_rho_per_checkpoint,
                0.001,
            ),
            "alpha_rho_per_prototype_cp": (
                self.alpha_rho_per_prototype_cp,
                0.00025,
            ),
            "safety_risk_limit": (self.safety_risk_limit, 0.05),
        }
        for name, (actual, expected) in fixed_probabilities.items():
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"statistics.{name} must equal {expected}.")
        if not math.isclose(
            self.alpha_safety
            + self.alpha_eb_net
            + self.alpha_eb_response
            + self.alpha_rho,
            self.alpha_total,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("The safety, two effect, and firing-count budgets must sum to 0.05.")
        if not math.isclose(
            len(self.kill_checkpoint_rounds) * self.alpha_rho_per_checkpoint,
            self.alpha_rho,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("The firing-count budget must split across checkpoints.")
        if not math.isclose(
            self.partner_prototype_count * self.alpha_rho_per_prototype_cp,
            self.alpha_rho_per_checkpoint,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Each checkpoint must split its error across prototypes.")
        if self.kill_checkpoint_rounds != (200, 400, 800, 1600, 2500):
            raise ValueError("The firing-count checkpoint schedule changed.")
        if self.kill_checkpoint_rounds[-1] != self.n_rounds_max:
            raise ValueError("The final firing-count checkpoint must equal N_rounds_max.")
        if self.empirical_bernstein_formula != (
            "maurer_pontil_empirical_bernstein_bounded_v1"
        ):
            raise ValueError("The empirical-Bernstein formula identifier changed.")
        if self.round_sampling_contract != R015_FORMAL_ROUND_SAMPLING_CONTRACT:
            raise ValueError("The independent round-sampling contract changed.")
        if self.formal_sampling_schedule_schema != (
            R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA
        ):
            raise ValueError("The formal sampling-schedule schema changed.")
        if self.mechanical_replacement_attempt_indices != "0_through_31":
            raise ValueError("The formal replacement-attempt indices changed.")
        if self.replacement_episode_seed_derivation_id != (
            R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID
        ):
            raise ValueError("The formal replacement-seed derivation changed.")
        if self.replacement_episode_seed_collision_policy != (
            "allow_value_collisions_without_redraw"
        ):
            raise ValueError("The formal replacement-seed collision policy changed.")
        if self.mechanical_attempt_exhaustion_rule != (
            "terminate_entire_formal_audit"
        ):
            raise ValueError("The formal replacement exhaustion rule changed.")
        if self.ego_agent_id != "agent_1":
            raise ValueError("R015 fixes the ego cook at agent_1.")
        if self.zero_count_independence_contract != (
            "independent_firing_prefixes_across_prototypes_v1"
        ):
            raise ValueError("The zero-count independence contract changed.")
        if self.planning_fork_steps_included:
            raise ValueError("Planning-fork steps must be excluded and reported separately.")
        if self.delivery_reward <= 0.0:
            raise ValueError("statistics.delivery_reward must be positive.")
        if self.return_lower_bound >= self.return_upper_bound:
            raise ValueError("The deterministic return bounds are not ordered.")
        if not self.return_lower_bound <= 0.0 <= self.return_upper_bound:
            raise ValueError("The deterministic return bounds must contain zero.")
        if self.return_upper_bound < self.delivery_reward:
            raise ValueError("The return upper bound cannot be below one delivery reward.")
        if not math.isclose(
            self.paired_difference_absolute_bound,
            self.return_upper_bound - self.return_lower_bound,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            self.paired_difference_width,
            2.0 * self.paired_difference_absolute_bound,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("The structural paired-difference constants are inconsistent.")
        if self.maximum_safety_comparisons != (
            self.partner_prototype_count**2 * self.n_rounds_max
        ):
            raise ValueError("The safety-comparison count must equal K squared times N_rounds_max.")
        if (
            self.formal_paired_block_budget
            != self.partner_prototype_count * self.n_rounds_max
            or self.formal_arm_episode_budget != 3 * self.formal_paired_block_budget
            or self.formal_arm_environment_step_budget
            != 400 * self.formal_arm_episode_budget
        ):
            raise ValueError("The registered formal block and environment-step budgets disagree.")
        if self.required_safety_repeats() != self.safety_repeats_per_comparison:
            raise ValueError("The safety repeat count does not equal the union-bound rule.")

    @property
    def practical_margin(self) -> float:
        return self.practical_margin_fraction * self.delivery_reward

    def required_safety_repeats(self) -> int:
        comparisons = self.maximum_safety_comparisons
        return math.ceil(
            math.log(self.alpha_safety / comparisons)
            / math.log(1.0 - self.safety_risk_limit)
        )

    def require_formal_ready(self) -> None:
        self._validate_fixed_design()


@dataclass(frozen=True)
class InformationContract:
    controller_input: str
    allowed_controller_fields: tuple[str, ...]
    forbidden_controller_fields: tuple[str, ...]
    projection_name: str
    projection_mask_targets: tuple[str, ...]
    projection_preserved_fields: tuple[str, ...]
    use_only_route: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InformationContract":
        projection = _require_mapping(
            payload.get("current_probe_response_projection"),
            field="information.current_probe_response_projection",
        )
        result = cls(
            controller_input=str(payload.get("controller_input")),
            allowed_controller_fields=tuple(
                str(item)
                for item in _require_sequence(
                    payload.get("allowed_controller_fields"),
                    field="information.allowed_controller_fields",
                )
            ),
            forbidden_controller_fields=tuple(
                str(item)
                for item in _require_sequence(
                    payload.get("forbidden_controller_fields"),
                    field="information.forbidden_controller_fields",
                )
            ),
            projection_name=str(projection.get("name")),
            projection_mask_targets=tuple(
                str(item)
                for item in _require_sequence(
                    projection.get("mask_targets"),
                    field="information.current_probe_response_projection.mask_targets",
                )
            ),
            projection_preserved_fields=tuple(
                str(item)
                for item in _require_sequence(
                    projection.get("preserved_fields"),
                    field=(
                        "information.current_probe_response_projection."
                        "preserved_fields"
                    ),
                )
            ),
            use_only_route=str(projection.get("use_only_route")),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.controller_input != "official_local_history_only":
            raise ValueError("R015 formal controllers may use official local history only.")
        required_allowed = {
            "official_local_observation",
            "ego_action_history",
            "ego_option_or_probe_history",
            "ego_option_or_probe_elapsed_time",
            "raw_team_reward_history",
            "episode_boundaries",
            "legal_action_mask_derived",
            "task_progress_derived",
            "response_summary_v1_derived",
        }
        if set(self.allowed_controller_fields) != required_allowed:
            raise ValueError("The R015 official local-history field list changed.")
        required_forbidden = {
            "evaluator_partner_action",
            "partner_identity",
            "partner_private_observation",
            "partner_recurrent_state",
            "complete_grid",
            "complete_hidden_state",
            "hidden_recipe",
            "evaluator_random_state",
            "training_family_label",
        }
        if set(self.forbidden_controller_fields) != required_forbidden:
            raise ValueError("The R015 forbidden controller-field list changed.")
        required_targets = {
            "partner_belief_update_input",
            "continuation_controller_serialized_input",
            "recurrent_state_writes",
        }
        if set(self.projection_mask_targets) != required_targets:
            raise ValueError("The current-response projection mask targets changed.")
        required_preserved = {
            "official_local_observation",
            "ego_actions",
            "raw_team_rewards",
            "episode_boundaries",
            "future_passive_response_tokens",
        }
        if set(self.projection_preserved_fields) != required_preserved:
            raise ValueError("The current-response projection preserved fields changed.")
        if self.use_only_route != "current_response_token_to_partner_belief_update":
            raise ValueError("A2-use may add the current response only to belief update.")


@dataclass(frozen=True)
class R015ControllerSlots:
    """Static controller choices, including numeric slots that remain pending."""

    consultation_steps: tuple[int, ...]
    score_rule: str
    score_tie_break: str
    masked_reference_tie_break: str
    filter_mode: str
    filter_prototype_count: int
    prototype_prior: Mapping[str, float] | None
    particles_per_prototype: int | None
    filter_initialization_rule: str
    initialization_key_source: str
    filter_transition_update_rule: str
    filter_ess_rule: str
    filter_zero_support_condition: str
    resampling_algorithm: str | None
    resampling_interval_environment_steps: int | None
    resampling_timing: str | None
    resampling_ess_fraction_threshold: float | None
    zero_support_action: str
    filter_device_execution_id: str
    filter_device_key_contract: str
    filter_microbatch_schedule_id: str
    filter_parent_particle_slot_target: int
    filter_group_partner_network_by_prototype: bool
    filter_continuation_member_forwards: int
    planning_branches_per_candidate: int | None
    planning_branch_sampling: str
    planning_branch_belief: str
    planning_grid_evaluation: str
    planning_nested_branch_counts: tuple[int, ...]
    planning_horizon: str
    planning_stability_threshold: float
    continuation_rule_id: str
    continuation_action_rule: str
    continuation_routing_rule: str
    continuation_tie_fallback: str
    continuation_parallel_recurrent_member_count: int
    continuation_switch_state_rule: str
    continuation_planning_routing_frequency: str
    continuation_execution_routing_frequency: str
    continuation_shared_member_selection_rule: str
    continuation_baseline_member_id: str
    continuation_prototype_member_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "R015ControllerSlots":
        filtering = _require_mapping(
            payload.get("hidden_state_filter"), field="controller.hidden_state_filter"
        )
        planning = _require_mapping(
            payload.get("planning"), field="controller.planning"
        )
        continuation = _require_mapping(
            payload.get("continuation"), field="controller.continuation"
        )
        result = cls(
            consultation_steps=tuple(
                _positive_int(item, field="controller.consultation_steps")
                for item in _require_sequence(
                    payload.get("consultation_steps"),
                    field="controller.consultation_steps",
                )
            ),
            score_rule=str(payload.get("score_rule", "")),
            score_tie_break=str(payload.get("score_tie_break", "")),
            masked_reference_tie_break=str(
                payload.get("masked_reference_tie_break", "")
            ),
            filter_mode=str(filtering.get("mode", "")),
            filter_prototype_count=_positive_int(
                filtering.get("prototype_count"),
                field="controller.hidden_state_filter.prototype_count",
            ),
            prototype_prior=_optional_positive_probability_mapping(
                filtering.get("prototype_prior"),
                field="controller.hidden_state_filter.prototype_prior",
            ),
            particles_per_prototype=_optional_positive_int(
                filtering.get("particles_per_prototype"),
                field="controller.hidden_state_filter.particles_per_prototype",
            ),
            filter_initialization_rule=str(filtering.get("initialization_rule", "")),
            initialization_key_source=str(
                filtering.get("initialization_key_source", "")
            ),
            filter_transition_update_rule=str(
                filtering.get("transition_update_rule", "")
            ),
            filter_ess_rule=str(filtering.get("ess_rule", "")),
            filter_zero_support_condition=str(
                filtering.get("zero_support_condition", "")
            ),
            resampling_algorithm=(
                None
                if _is_pending(filtering.get("resampling_algorithm"))
                else str(filtering.get("resampling_algorithm", ""))
            ),
            resampling_interval_environment_steps=_optional_positive_int(
                filtering.get("resampling_interval_environment_steps"),
                field=(
                    "controller.hidden_state_filter."
                    "resampling_interval_environment_steps"
                ),
            ),
            resampling_timing=(
                None
                if _is_pending(filtering.get("resampling_timing"))
                else str(filtering.get("resampling_timing", ""))
            ),
            resampling_ess_fraction_threshold=(
                None
                if _is_pending(filtering.get("resampling_ess_fraction_threshold"))
                else _finite_float(
                    filtering.get("resampling_ess_fraction_threshold"),
                    field=(
                        "controller.hidden_state_filter."
                        "resampling_ess_fraction_threshold"
                    ),
                )
            ),
            zero_support_action=str(filtering.get("zero_support_action", "")),
            filter_device_execution_id=str(
                filtering.get("device_execution_id", "")
            ),
            filter_device_key_contract=str(filtering.get("device_key_contract", "")),
            filter_microbatch_schedule_id=str(
                filtering.get("microbatch_schedule_id", "")
            ),
            filter_parent_particle_slot_target=_positive_int(
                filtering.get("parent_particle_slot_target_per_microbatch"),
                field=(
                    "controller.hidden_state_filter."
                    "parent_particle_slot_target_per_microbatch"
                ),
            ),
            filter_group_partner_network_by_prototype=(
                filtering.get("group_partner_network_by_prototype") is True
            ),
            filter_continuation_member_forwards=_nonnegative_int(
                filtering.get("continuation_member_forwards_in_filter_candidate_stage"),
                field=(
                    "controller.hidden_state_filter."
                    "continuation_member_forwards_in_filter_candidate_stage"
                ),
            ),
            planning_branches_per_candidate=_optional_positive_int(
                planning.get("branches_per_candidate"),
                field="controller.planning.branches_per_candidate",
            ),
            planning_branch_sampling=str(planning.get("branch_sampling", "")),
            planning_branch_belief=str(planning.get("branch_belief", "")),
            planning_grid_evaluation=str(planning.get("grid_evaluation", "")),
            planning_nested_branch_counts=tuple(
                _positive_int(
                    item,
                    field="controller.planning.nested_branch_counts",
                )
                for item in _require_sequence(
                    planning.get("nested_branch_counts"),
                    field="controller.planning.nested_branch_counts",
                )
            ),
            planning_horizon=str(planning.get("horizon", "")),
            planning_stability_threshold=_finite_float(
                planning.get("stability_threshold"),
                field="controller.planning.stability_threshold",
            ),
            continuation_rule_id=str(continuation.get("rule_id", "")),
            continuation_action_rule=str(continuation.get("action_rule", "")),
            continuation_routing_rule=str(continuation.get("routing_rule", "")),
            continuation_tie_fallback=str(continuation.get("tie_fallback", "")),
            continuation_parallel_recurrent_member_count=_positive_int(
                continuation.get("parallel_recurrent_member_count"),
                field="controller.continuation.parallel_recurrent_member_count",
            ),
            continuation_switch_state_rule=str(
                continuation.get("switch_state_rule", "")
            ),
            continuation_planning_routing_frequency=str(
                continuation.get("planning_routing_frequency", "")
            ),
            continuation_execution_routing_frequency=str(
                continuation.get("execution_routing_frequency", "")
            ),
            continuation_shared_member_selection_rule=str(
                continuation.get("shared_member_selection_rule", "")
            ),
            continuation_baseline_member_id=str(
                continuation.get("baseline_member_id", "")
            ),
            continuation_prototype_member_ids=tuple(
                str(value)
                for value in _require_sequence(
                    continuation.get("prototype_member_ids"),
                    field="controller.continuation.prototype_member_ids",
                )
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.consultation_steps != tuple(range(1, 101, 5)):
            raise ValueError("R015 consultation boundaries must be 1,6,...,96.")
        if self.score_rule != "S_seq=J_use-V_mask":
            raise ValueError("R015 sequential score rule changed.")
        if self.score_tie_break != "registered_probe_order":
            raise ValueError("R015 positive-score ties must use probe registry order.")
        if self.masked_reference_tie_break != "base_then_registered_probe_order":
            raise ValueError("R015 masked-reference ties must choose base first.")
        if self.filter_mode != "official_history_fully_adapted_particle_filter_v2" or (
            self.filter_prototype_count != 4
        ):
            raise ValueError("R015 filtering must retain four prototype strata.")
        if self.resampling_algorithm not in {
            None,
            "strict_systematic_per_prototype_v2",
        }:
            raise ValueError("R015 static filtering does not support that resampling rule.")
        if self.resampling_timing not in {
            None,
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 filtering uses an unregistered resampling timing.")
        if self.resampling_interval_environment_steps not in {None, 1}:
            raise ValueError("R015 design replaces fixed multi-step resampling intervals.")
        if self.resampling_timing == "adaptive_ess_below_half_v1":
            if self.resampling_ess_fraction_threshold is None or not math.isclose(
                self.resampling_ess_fraction_threshold,
                0.5,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Adaptive R015 resampling must use ESS < 0.5N.")
        elif self.resampling_timing == "every_environment_step_v1" and (
            self.resampling_ess_fraction_threshold is not None
        ):
            raise ValueError("Every-step R015 resampling has no ESS threshold.")
        if self.filter_initialization_rule != (
            "independent_prior_draws_conditioned_on_opening_official_observation_v1"
        ) or self.initialization_key_source != "r015_filter_device_fold_in_keys_v2":
            raise ValueError("R015 particle initialization may not use episode seed.")
        if self.filter_transition_update_rule != (
            "exact_six_action_marginal_then_conditional_successor_v1"
        ) or self.filter_ess_rule != "pre_resample_parent_predictive_weight_ess_v1" or (
            self.filter_zero_support_condition
            != "sum_parent_weight_times_predictive_likelihood_equals_zero"
        ):
            raise ValueError("R015 fully adapted filter semantics changed.")
        if self.filter_device_execution_id != "r015_filter_jit_scan_vmap_v2" or (
            self.filter_device_key_contract != "r015_filter_device_fold_in_keys_v2"
        ) or self.filter_microbatch_schedule_id != (
            "fixed_4096_parent_particle_slots_v1"
        ) or self.filter_parent_particle_slot_target != 4096 or (
            not self.filter_group_partner_network_by_prototype
        ) or self.filter_continuation_member_forwards != 0:
            raise ValueError("R015 filter device execution contract changed.")
        if self.zero_support_action != "invalidate_and_replace_whole_block":
            raise ValueError("R015 zero particle support must invalidate the whole block.")
        if self.planning_horizon != "full_remaining_episode_to_step_400":
            raise ValueError("R015 planning must cover the complete remaining episode.")
        if self.planning_branch_sampling != "sampled_hidden_state_branches_v1":
            raise ValueError("R015 planning branch-sampling rule changed.")
        if self.planning_branch_belief != "frozen_belief_branch_continuation_v1":
            raise ValueError("R015 planning branch-belief rule changed.")
        if self.planning_grid_evaluation != "lazy_ascending_first_pass" or (
            self.planning_nested_branch_counts != (2, 4, 8, 16)
        ):
            raise ValueError("R015 planning nested grid rule changed.")
        if not math.isclose(
            self.planning_stability_threshold,
            0.95,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("R015 planning stability threshold must remain 0.95.")
        if self.continuation_rule_id != R015_CONTINUATION_CONTROLLER_ID:
            raise ValueError("R015 continuation-controller identity changed.")
        if self.continuation_action_rule != "official_flax_categorical_actor_v1":
            raise ValueError("R015 continuation action rule changed.")
        if self.continuation_routing_rule != "unique_exact_map_else_ego_seed100" or (
            self.continuation_tie_fallback != "ego_seed100"
        ):
            raise ValueError("R015 continuation MAP routing or tie fallback changed.")
        if self.continuation_parallel_recurrent_member_count != 5 or (
            self.continuation_switch_state_rule
            != "adopt_current_parallel_state_without_reset"
        ):
            raise ValueError("R015 continuation recurrent-state rule changed.")
        if self.continuation_planning_routing_frequency != (
            "commit_once_at_branch_head"
        ) or self.continuation_execution_routing_frequency != (
            "update_online_each_environment_step"
        ) or self.continuation_shared_member_selection_rule != (
            "unique_exact_map_else_ego_seed100"
        ):
            raise ValueError("R015 planning and execution routing frequencies changed.")
        if self.continuation_baseline_member_id != R015_EGO_BASELINE_MEMBER_ID:
            raise ValueError("R015 continuation baseline must be official seed 100.")
        if len(self.continuation_prototype_member_ids) != 4 or len(
            set(self.continuation_prototype_member_ids)
        ) != 4:
            raise ValueError("R015 continuation library requires four prototypes.")
        if self.prototype_prior is not None and set(
            self.continuation_prototype_member_ids
        ) != set(self.prototype_prior):
            raise ValueError("R015 continuation library differs from the belief support.")

    def require_formal_ready(self) -> None:
        if self.prototype_prior is None:
            raise ValueError("R015 prototype_prior is still pending.")
        if self.particles_per_prototype is None:
            raise ValueError("R015 particles_per_prototype is still pending.")
        if self.resampling_algorithm is None:
            raise ValueError("R015 resampling_algorithm is still pending.")
        if self.resampling_interval_environment_steps is None:
            raise ValueError("R015 resampling compatibility interval is still pending.")
        if self.resampling_timing is None:
            raise ValueError("R015 resampling timing is still pending.")
        if self.resampling_timing == "adaptive_ess_below_half_v1" and (
            self.resampling_ess_fraction_threshold is None
        ):
            raise ValueError("R015 resampling ESS threshold is still pending.")
        if self.planning_branches_per_candidate is None:
            raise ValueError("R015 planning branches_per_candidate is still pending.")


@dataclass(frozen=True)
class ProbeScriptSpec:
    probe_id: str
    primitive_actions: tuple[str, ...]


def _canonical_probe_registry_sha256(
    scripts: Mapping[str, ProbeScriptSpec],
) -> str:
    """Hash script semantics independently of YAML formatting and order."""

    payload = {
        "schema_version": "path_c_r015_probe_registry_semantics_v1",
        "probe_scripts": [
            {
                "probe_id": probe_id,
                "primitive_actions": list(scripts[probe_id].primitive_actions),
            }
            for probe_id in sorted(scripts)
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _bytes_sha256(encoded)


def _probe_scripts_from_experiment(
    experiment: Mapping[str, Any],
) -> Mapping[str, ProbeScriptSpec]:
    raw_scripts = _require_sequence(
        experiment.get("registered_probe_scripts"),
        field="experiment.registered_probe_scripts",
    )
    atomic_actions = {"up", "down", "right", "left", "stay", "interact"}
    scripts: dict[str, ProbeScriptSpec] = {}
    for raw_script in raw_scripts:
        script = _require_mapping(raw_script, field="registered probe script")
        probe_id = str(script.get("probe_id", ""))
        actions = tuple(
            str(item)
            for item in _require_sequence(
                script.get("primitive_actions"),
                field=f"registered_probe_scripts.{probe_id}.primitive_actions",
            )
        )
        if not probe_id or probe_id in scripts:
            raise ValueError("Registered probe identifiers must be non-empty and unique.")
        if not 1 <= len(actions) <= 2 or any(
            action not in atomic_actions for action in actions
        ):
            raise ValueError("Registered probes use one or two of the six primitive actions.")
        scripts[probe_id] = ProbeScriptSpec(probe_id, actions)
    singleton_actions = {
        script.primitive_actions[0]
        for script in scripts.values()
        if len(script.primitive_actions) == 1
    }
    if singleton_actions != atomic_actions:
        raise ValueError("The registered probes must include all six primitive actions.")
    return scripts


@dataclass(frozen=True)
class R015Preregistration:
    source_path: Path
    source_sha256: str
    document_status: str
    scientific_readout_allowed: bool
    freeze_status: str
    statistics: R015Statistics
    controller: R015ControllerSlots
    information: InformationContract
    probe_scripts: Mapping[str, ProbeScriptSpec]
    probe_registry_semantic_sha256: str
    artifacts: Mapping[str, ArtifactBinding]
    ego_model_weights_sha256: str | None
    support_registration_path: Path
    support_registration_sha256: str | None
    support_report_path: Path | None
    formal_dataset_path: Path | None
    firing_count_checkpoint_paths: tuple[Path, ...] | None
    formal_view_record_path: Path | None
    support_spec: R015PartnerSupportSpec
    full_state_upper_bound_status: str

    def _require_frozen_support(self) -> Mapping[str, Mapping[str, Any]]:
        if (
            self.freeze_status != "frozen"
            or self.document_status != "frozen"
            or not self.scientific_readout_allowed
        ):
            raise ValueError("R015 preregistration is not frozen for a formal readout.")
        self.statistics.require_formal_ready()
        self.controller.require_formal_ready()
        for binding in self.artifacts.values():
            binding.require_ready()
        if not _is_sha256(self.ego_model_weights_sha256):
            raise ValueError("The R015 ego model-weight digest is pending.")
        if self.support_registration_sha256 is None:
            raise ValueError("The partner-support registration digest is pending.")
        if _file_sha256(self.support_registration_path) != (
            self.support_registration_sha256
        ):
            raise ValueError("The partner-support registration changed after freeze.")
        if self.support_report_path is None or not self.support_report_path.is_file():
            raise ValueError("The admitted partner-support report is pending.")
        report = _load_json_or_yaml(self.support_report_path)
        members = validate_r015_support_report(report, self.support_spec)
        if set(self.controller.prototype_prior or {}) != set(members):
            raise ValueError("R015 registered prior differs from the admitted prototypes.")
        environment_sha = self.artifacts["environment_config"].sha256
        if {member.get("environment_config_sha256") for member in members.values()} != {
            environment_sha
        }:
            raise ValueError("Partner support and R015 bind different environments.")
        _validate_frozen_scientific_artifacts(self, members)
        return members

    def require_firing_checkpoint_ready(
        self, checkpoint_round: int
    ) -> tuple[Mapping[str, Mapping[str, Any]], tuple[Path, ...]]:
        members = self._require_frozen_support()
        if checkpoint_round not in self.statistics.kill_checkpoint_rounds:
            raise ValueError("The firing-count checkpoint is outside the frozen schedule.")
        if self.firing_count_checkpoint_paths is None or len(
            self.firing_count_checkpoint_paths
        ) != len(self.statistics.kill_checkpoint_rounds):
            raise ValueError("The firing-count checkpoint path list is incomplete.")
        checkpoint_index = self.statistics.kill_checkpoint_rounds.index(checkpoint_round)
        completed_paths = self.firing_count_checkpoint_paths[: checkpoint_index + 1]
        if any(not path.is_file() for path in completed_paths):
            raise ValueError("A completed firing-count checkpoint record is missing.")
        if self.formal_view_record_path is None:
            raise ValueError("The formal-view consumption record path is pending.")
        if self.formal_view_record_path.exists():
            raise ValueError("Effect values were already consumed before this checkpoint.")
        return members, completed_paths

    def require_formal_ready(self) -> Mapping[str, Mapping[str, Any]]:
        members = self._require_frozen_support()
        if self.firing_count_checkpoint_paths is None or len(
            self.firing_count_checkpoint_paths
        ) != len(self.statistics.kill_checkpoint_rounds) or any(
            not path.is_file() for path in self.firing_count_checkpoint_paths
        ):
            raise ValueError("The five firing-count checkpoint records are incomplete.")
        if self.formal_view_record_path is None:
            raise ValueError("The formal-view consumption record path is pending.")
        if not self.formal_view_record_path.parent.is_dir():
            raise ValueError("The formal-view record directory does not exist.")
        if self.formal_view_record_path.exists():
            raise ValueError("The single R015 formal view has already been consumed.")
        bound_paths = {
            binding.path.resolve()
            for binding in self.artifacts.values()
            if binding.path is not None
        }
        bound_paths.update(
            {
                self.support_registration_path.resolve(),
                self.support_report_path.resolve(),
                *(path.resolve() for path in self.firing_count_checkpoint_paths),
            }
        )
        if self.formal_dataset_path is not None:
            bound_paths.add(self.formal_dataset_path.resolve())
        if self.formal_view_record_path.resolve() in bound_paths:
            raise ValueError("The formal-view record must have its own path.")
        return members


def load_r015_preregistration(
    path: str | Path,
    *,
    for_formal_decision: bool = False,
) -> R015Preregistration:
    """Load the R015 design; formal mode rejects every pending binding."""

    source_path = Path(path).resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload = _require_mapping(payload, field="R015 preregistration")
    if payload.get("schema_version") != R015_PREREGISTRATION_SCHEMA:
        raise ValueError("Unsupported R015 preregistration schema.")
    document_status = str(payload.get("status", ""))
    scientific_readout_allowed = _boolean(
        payload.get("scientific_readout_allowed"),
        field="scientific_readout_allowed",
    )
    experiment = _require_mapping(payload.get("experiment"), field="experiment")
    if experiment.get("experiment_id") != R015_EXPERIMENT_ID:
        raise ValueError("This contract accepts opportunity-audit experiment R015 only.")
    if experiment.get("ego_agent_id") != "agent_1" or experiment.get(
        "partner_agent_id"
    ) != "agent_0" or experiment.get("ego_role") != "cook" or experiment.get(
        "partner_role"
    ) != "signaler":
        raise ValueError("R015 fixes the ego cook at agent_1 and partner at agent_0.")
    freeze_status = str(experiment.get("freeze_status", ""))
    if freeze_status == "static_registered_not_frozen":
        if document_status != "template_not_valid_for_runs" or (
            scientific_readout_allowed
        ):
            raise ValueError("The static R015 preregistration must remain a non-run template.")
    elif freeze_status == "frozen":
        if document_status != "frozen" or not scientific_readout_allowed:
            raise ValueError("A formal R015 preregistration must be explicitly enabled.")
    else:
        raise ValueError("The R015 freeze status is unsupported.")
    if tuple(experiment.get("formal_groups", ())) != R015_FORMAL_GROUPS:
        raise ValueError("R015 formal groups must be A1, A2-mask, and A2-use.")
    if tuple(experiment.get("diagnostic_groups", ())) != (R015_DIAGNOSTIC_GROUP,):
        raise ValueError("A0 must remain diagnostic and outside formal adjudication.")
    if _positive_int(
        experiment.get("maximum_probes_per_episode"),
        field="experiment.maximum_probes_per_episode",
    ) != 1:
        raise ValueError("R015 permits at most one probe per episode.")
    if _positive_int(
        experiment.get("probe_window_environment_steps"),
        field="experiment.probe_window_environment_steps",
    ) != 100:
        raise ValueError("R015 probe window must be the first 100 steps.")
    if _positive_int(
        experiment.get("consultation_interval_environment_steps"),
        field="experiment.consultation_interval_environment_steps",
    ) != 5:
        raise ValueError("R015 consultation interval must be five steps.")

    base = source_path.parent
    controller = R015ControllerSlots.from_mapping(
        _require_mapping(payload.get("controller"), field="controller")
    )
    support = _require_mapping(payload.get("partner_support"), field="partner_support")
    registration_path = _resolve_path(base, support.get("registration_path"))
    if registration_path is None or not registration_path.is_file():
        raise ValueError("The static two-family support registration is missing.")
    registration_payload = load_r015_partner_support_config(registration_path)
    support_spec = R015PartnerSupportSpec.from_mapping(registration_payload)
    raw_registration_sha = support.get("registration_sha256")
    registration_sha = (
        None if _is_pending(raw_registration_sha) else str(raw_registration_sha)
    )
    if registration_sha is not None:
        if not _is_sha256(registration_sha):
            raise ValueError("partner_support.registration_sha256 is invalid.")
        if _file_sha256(registration_path) != registration_sha:
            raise ValueError("The partner-support registration digest does not match.")

    probe_scripts = _probe_scripts_from_experiment(experiment)
    raw_artifacts = _require_mapping(payload.get("artifacts"), field="artifacts")
    required_artifacts = {
        "environment_config",
        "environment_source",
        "ego_checkpoint",
        "ego_evidence_contract",
        "formal_sampling_schedule",
        "response_projection",
        "response_vocabulary",
        "probe_registry",
        "official_history_filter",
        "continuation_planner",
        "continuation_controller",
        "decision_evidence_verifier",
        "return_bound_derivation",
        "wrong_delivery_detector",
        "planning_stability_report",
        "pilot_wiring_report",
        "random_key_derivation",
        "safety_branch_evidence_verifier",
        "trace_replay_verifier",
    }
    if set(raw_artifacts) != required_artifacts:
        raise ValueError("The R015 frozen artifact list changed.")
    continuation_binding = _require_mapping(
        raw_artifacts.get("continuation_controller"),
        field="artifacts.continuation_controller",
    )
    expected_weight_references = {
        R015_EGO_BASELINE_MEMBER_ID: "artifacts.ego_checkpoint.model_weights_sha256",
        **{
            prototype_id: (
                "partner_support.report.candidates."
                f"{prototype_id}.model_weights_sha256"
            )
            for prototype_id in controller.continuation_prototype_member_ids
        },
    }
    if continuation_binding.get("rule_id") != R015_CONTINUATION_CONTROLLER_ID or (
        continuation_binding.get("action_rule")
        != "official_flax_categorical_actor_v1"
    ) or continuation_binding.get("tie_fallback") != "ego_seed100" or (
        continuation_binding.get("switch_state_rule")
        != "adopt_current_parallel_state_without_reset"
    ) or _require_mapping(
        continuation_binding.get("checkpoint_weight_sha256_references"),
        field=(
            "artifacts.continuation_controller."
            "checkpoint_weight_sha256_references"
        ),
    ) != expected_weight_references:
        raise ValueError("The R015 continuation-controller artifact contract changed.")
    artifacts = {
        str(name): ArtifactBinding.from_mapping(
            str(name),
            _require_mapping(item, field=f"artifacts.{name}"),
            base=base,
        )
        for name, item in raw_artifacts.items()
    }
    raw_ego_weights_sha256 = _require_mapping(
        raw_artifacts.get("ego_checkpoint"), field="artifacts.ego_checkpoint"
    ).get("model_weights_sha256")
    if not _is_pending(raw_ego_weights_sha256) and not _is_sha256(
        raw_ego_weights_sha256
    ):
        raise ValueError("artifacts.ego_checkpoint.model_weights_sha256 is invalid.")

    upper = _require_mapping(
        payload.get("full_state_upper_bound"), field="full_state_upper_bound"
    )
    if upper.get("status") != "not_provided":
        raise ValueError(
            "R015 has no proved full-state upper bound; status must be not_provided."
        )
    if any(
        upper.get(field) is not None
        for field in ("value", "proof_artifact_path", "proof_artifact_sha256")
    ):
        raise ValueError("An absent full-state upper bound cannot carry proof fields.")

    formal_data = _require_mapping(payload.get("formal_data"), field="formal_data")
    preregistration = R015Preregistration(
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
        document_status=document_status,
        scientific_readout_allowed=scientific_readout_allowed,
        freeze_status=freeze_status,
        statistics=R015Statistics.from_mapping(
            _require_mapping(payload.get("statistics"), field="statistics")
        ),
        controller=controller,
        information=InformationContract.from_mapping(
            _require_mapping(payload.get("information"), field="information")
        ),
        probe_scripts=probe_scripts,
        probe_registry_semantic_sha256=_canonical_probe_registry_sha256(
            probe_scripts
        ),
        artifacts=artifacts,
        ego_model_weights_sha256=(
            None
            if _is_pending(raw_ego_weights_sha256)
            else str(raw_ego_weights_sha256)
        ),
        support_registration_path=registration_path,
        support_registration_sha256=registration_sha,
        support_report_path=_resolve_path(base, support.get("report_path")),
        formal_dataset_path=_resolve_path(
            base,
            formal_data.get("dataset_path"),
        ),
        firing_count_checkpoint_paths=_resolve_path_sequence(
            base,
            formal_data.get("firing_count_checkpoint_paths"),
        ),
        formal_view_record_path=_resolve_path(
            base, formal_data.get("view_consumption_record_path")
        ),
        support_spec=support_spec,
        full_state_upper_bound_status="not_provided",
    )
    if for_formal_decision:
        preregistration.require_formal_ready()
    return preregistration


def _load_json_or_yaml(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = (
        json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    )
    return _require_mapping(payload, field=str(path))


def _parse_json_or_yaml_bytes(path: Path, payload: bytes) -> Mapping[str, Any]:
    text = payload.decode("utf-8")
    parsed = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return _require_mapping(parsed, field=str(path))


def _canonical_value_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _bytes_sha256(encoded)


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    return _canonical_value_sha256(payload)


_R015_RANDOM_KEY_COORDINATES = (
    "audit_unit_id",
    "partner_prototype_id",
    "episode_seed",
    "purpose",
    "environment_step",
    "branch_index",
)


def _derive_r015_random_key(
    *,
    audit_unit_id: str,
    partner_prototype_id: str,
    episode_seed: int,
    purpose: str,
    environment_step: int,
    branch_index: int,
) -> str:
    """Derive one registered random key from all six frozen coordinates."""

    coordinates = {
        "audit_unit_id": str(audit_unit_id),
        "partner_prototype_id": str(partner_prototype_id),
        "episode_seed": _nonnegative_int(episode_seed, field="episode_seed"),
        "purpose": str(purpose),
        "environment_step": _nonnegative_int(
            environment_step, field="environment_step"
        ),
        "branch_index": _nonnegative_int(branch_index, field="branch_index"),
    }
    return _canonical_mapping_sha256(
        {
            "schema_version": "path_c_r015_random_key_v1",
            "coordinates": coordinates,
        }
    )


def _derive_r015_controller_key(root_key: str, *coordinates: object) -> str:
    """重建控制器从一个已登记根键派生的确定性子键。"""

    if not _is_sha256(root_key):
        raise ValueError("R015 controller root key must be SHA-256.")
    return _canonical_mapping_sha256(
        {
            "schema_version": "path_c_r015_controller_key_v1",
            "root_key": root_key,
            "coordinates": [str(item) for item in coordinates],
        }
    )


def _future_random_sequence_summary(
    *,
    root_key: str,
    sequence_length: int,
) -> Mapping[str, Any]:
    """从根键重建完整未来随机序列，但只返回冻结的紧凑证据。"""

    length = _positive_int(sequence_length, field="future_random_step_count")
    keys = [
        _derive_r015_controller_key(root_key, "future_environment", index)
        for index in range(length)
    ]
    return {
        "future_random_root_key": root_key,
        "future_random_derivation_contract_id": (
            R015_FUTURE_RANDOM_DERIVATION_ID
        ),
        "future_random_step_count": length,
        "future_random_sequence_sha256": _canonical_value_sha256(keys),
    }


def _validate_sequential_controller_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
) -> None:
    """Reject short-horizon surrogates from the formal sequential controller slot."""

    if manifest.get(
        "schema_version"
    ) != "path_c_r015_sequential_controller_manifest_v1":
        raise ValueError("The sequential-controller manifest has the wrong schema.")
    if manifest.get("controller_kind") != "registered_response_sequential_branch_v1":
        raise ValueError("Formal R015 requires the registered sequential branch controller.")
    if manifest.get("finite_prototype_two_action_surrogate_allowed") is not False:
        raise ValueError("The finite-prototype two-action surrogate is forbidden.")
    interfaces = _require_mapping(
        manifest.get("paired_interfaces"), field="controller.paired_interfaces"
    )
    if interfaces != {
        "belief_use": "B_use(q,x,y)",
        "belief_mask": "B_mask(q,x)",
        "continuation_controller": "C(q,x)",
        "common_projected_history": "x",
        "use_only_current_response": "y",
    }:
        raise ValueError("The formal paired belief and continuation interfaces changed.")
    continuation_contract = _require_mapping(
        manifest.get("continuation_controller_contract"),
        field="controller.continuation_controller_contract",
    )
    if continuation_contract != {
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
    }:
        raise ValueError("The formal continuation-controller contract changed.")
    return_contract = _require_mapping(
        manifest.get("return_contract"), field="controller.return_contract"
    )
    if return_contract != {
        "reward": "undiscounted_raw_team_reward",
        "horizon": "full_remaining_episode",
        "discount_factor": 1.0,
    }:
        raise ValueError("The formal planner must estimate raw full-episode return.")
    branch_contract = _require_mapping(
        manifest.get("paired_branch_contract"),
        field="controller.paired_branch_contract",
    )
    if branch_contract != {
        "same_hidden_execution_state": True,
        "same_task_transition": True,
        "same_future_random_branch": True,
        "current_response_is_only_interface_difference": True,
    }:
        raise ValueError("The formal use and mask branches are not paired correctly.")
    lockstep_contract = _require_mapping(
        manifest.get("lockstep_contract"), field="controller.lockstep_contract"
    )
    if lockstep_contract != {
        "same_frozen_planner": True,
        "same_passive_masked_belief_updates": True,
        "same_episode_random_numbers": True,
        "probe_rule_firing_is_only_branch_point": True,
    }:
        raise ValueError("The three formal groups lack the frozen lockstep contract.")
    hidden_state_contract = _require_mapping(
        manifest.get("hidden_state_contract"),
        field="controller.hidden_state_contract",
    )
    if hidden_state_contract != {
        "complete_hidden_state_reconstruction": True,
        "positive_posterior_support_required": True,
        "branch_sampling_rule_id": R015_BRANCH_SAMPLING_ID,
        "nested_branch_counts": list(R015_NESTED_BRANCH_COUNTS),
        "equal_weight_sample_slots": True,
    }:
        raise ValueError("The formal planner lacks complete supported-state integration.")
    if manifest.get("score_rule") != "S_seq=J_use-V_mask":
        raise ValueError("The formal sequential score rule changed.")
    if manifest.get("tie_break_rule") != "base_then_probe_registry_order":
        raise ValueError("The formal sequential tie-break rule changed.")
    if tuple(manifest.get("consultation_steps", ())) != tuple(range(1, 101, 5)):
        raise ValueError("The formal consultation boundaries changed.")
    replay_backend = _require_mapping(
        manifest.get("replay_backend"), field="controller.replay_backend"
    )
    if set(replay_backend) != {
        "implementation_path",
        "implementation_sha256",
        "factory_name",
    } or replay_backend.get("factory_name") != "build_r015_replay_backend" or (
        not str(replay_backend.get("implementation_path", ""))
    ) or not _is_sha256(replay_backend.get("implementation_sha256")):
        raise ValueError("The formal controller lacks a bound replay backend.")
    if manifest.get("ego_action_selection_uses_full_state") is not False:
        raise ValueError("The formal ego controller cannot select from full state.")
    if not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The sequential controller implementation is not bound.")
    expected_bindings = {
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "continuation_controller_sha256": preregistration.artifacts[
            "continuation_controller"
        ].sha256,
    }
    if any(manifest.get(field) != value for field, value in expected_bindings.items()):
        raise ValueError("The sequential controller binds different frozen machinery.")


def _validate_continuation_controller_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind the five official checkpoints to the registered MAP continuation."""

    if manifest.get(
        "schema_version"
    ) != "path_c_r015_continuation_controller_manifest_v1" or manifest.get(
        "rule_id"
    ) != R015_CONTINUATION_CONTROLLER_ID:
        raise ValueError("The continuation-controller manifest has the wrong identity.")
    expected_contract = {
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
    }
    if _require_mapping(
        manifest.get("contract"), field="continuation_controller.contract"
    ) != expected_contract:
        raise ValueError("The continuation-controller behavior changed.")
    if manifest.get("baseline_member_id") != R015_EGO_BASELINE_MEMBER_ID or tuple(
        manifest.get("prototype_member_ids", ())
    ) != preregistration.controller.continuation_prototype_member_ids:
        raise ValueError("The continuation-controller policy library changed.")
    members = _require_mapping(
        manifest.get("members"), field="continuation_controller.members"
    )
    expected_member_ids = {
        R015_EGO_BASELINE_MEMBER_ID,
        *preregistration.controller.continuation_prototype_member_ids,
    }
    if set(members) != expected_member_ids:
        raise ValueError("The continuation-controller member set changed.")
    baseline = _require_mapping(
        members[R015_EGO_BASELINE_MEMBER_ID],
        field="continuation_controller.members.ego_seed100",
    )
    if baseline.get("role") != "baseline" or baseline.get(
        "checkpoint_sha256"
    ) != preregistration.artifacts["ego_checkpoint"].sha256 or baseline.get(
        "model_weights_sha256"
    ) != preregistration.ego_model_weights_sha256:
        raise ValueError("The continuation baseline checkpoint changed.")
    for prototype_id in preregistration.controller.continuation_prototype_member_ids:
        registered = _require_mapping(
            members[prototype_id],
            field=f"continuation_controller.members.{prototype_id}",
        )
        support = _require_mapping(
            support_members[prototype_id], field=f"support member {prototype_id}"
        )
        if registered.get("role") != "prototype" or registered.get(
            "checkpoint_sha256"
        ) != support.get("checkpoint_sha256") or registered.get(
            "model_weights_sha256"
        ) != support.get("model_weights_sha256"):
            raise ValueError("A continuation prototype checkpoint changed.")
    if not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The continuation-controller implementation is not bound.")


def _validate_official_history_filter_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
) -> None:
    if manifest.get("schema_version") != "path_c_r015_official_history_filter_v2":
        raise ValueError("The official-history filter manifest has the wrong schema.")
    if manifest.get("controller_input") != "official_local_history_only":
        raise ValueError("The formal filter must expose official local history only.")
    if set(manifest.get("allowed_controller_fields", ())) != set(
        preregistration.information.allowed_controller_fields
    ) or set(manifest.get("forbidden_controller_fields", ())) != set(
        preregistration.information.forbidden_controller_fields
    ):
        raise ValueError("The official-history filter field list changed.")
    if manifest.get("full_state_fields_exposed") is not False or manifest.get(
        "partner_identity_exposed"
    ) is not False:
        raise ValueError("The official-history filter exposes evaluator information.")
    expected_filter_parameters = {
        "prototype_prior": dict(preregistration.controller.prototype_prior or {}),
        "particles_per_prototype": preregistration.controller.particles_per_prototype,
        "filter_mode": preregistration.controller.filter_mode,
        "initialization_rule": (
            preregistration.controller.filter_initialization_rule
        ),
        "initialization_key_source": preregistration.controller.initialization_key_source,
        "transition_update_rule": (
            preregistration.controller.filter_transition_update_rule
        ),
        "ess_rule": preregistration.controller.filter_ess_rule,
        "zero_support_condition": (
            preregistration.controller.filter_zero_support_condition
        ),
        "resampling_algorithm": preregistration.controller.resampling_algorithm,
        "resampling_interval_environment_steps": (
            preregistration.controller.resampling_interval_environment_steps
        ),
        "resampling_timing": preregistration.controller.resampling_timing,
        "resampling_ess_fraction_threshold": (
            preregistration.controller.resampling_ess_fraction_threshold
        ),
        "zero_support_action": preregistration.controller.zero_support_action,
        "device_execution_id": (
            preregistration.controller.filter_device_execution_id
        ),
        "device_key_contract": preregistration.controller.filter_device_key_contract,
        "microbatch_schedule_id": (
            preregistration.controller.filter_microbatch_schedule_id
        ),
        "parent_particle_slot_target_per_microbatch": (
            preregistration.controller.filter_parent_particle_slot_target
        ),
        "group_partner_network_by_prototype": (
            preregistration.controller.filter_group_partner_network_by_prototype
        ),
        "continuation_member_forwards_in_filter_candidate_stage": (
            preregistration.controller.filter_continuation_member_forwards
        ),
    }
    if _require_mapping(
        manifest.get("filter_parameters"),
        field="official_history_filter.filter_parameters",
    ) != expected_filter_parameters:
        raise ValueError("The official-history filter parameters changed after freeze.")
    if manifest.get("response_vocabulary_sha256") != preregistration.artifacts[
        "response_vocabulary"
    ].sha256 or not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The official-history filter is not fully bound.")


def _validate_manifest_implementation(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    label: str,
) -> None:
    implementation_value = manifest.get("implementation_path")
    if not isinstance(implementation_value, str) or not implementation_value:
        raise ValueError(f"The {label} implementation path is missing.")
    implementation_path = Path(implementation_value)
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    if not implementation_path.is_file():
        raise ValueError(f"The {label} implementation file does not exist.")
    if _file_sha256(implementation_path) != manifest.get("implementation_sha256"):
        raise ValueError(f"The {label} implementation changed after freeze.")


def _validate_manifest_implementation_sources(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    label: str,
    expected_filenames: frozenset[str],
) -> None:
    """重算清单内的实现依赖；外层清单摘要不能替代源码摘要。"""

    raw_sources = _require_sequence(
        manifest.get("implementation_sources"),
        field=f"{label}.implementation_sources",
    )
    if len(raw_sources) != len(expected_filenames):
        raise ValueError(f"The {label} implementation-source set is incomplete.")
    observed_filenames: set[str] = set()
    observed_paths: set[Path] = set()
    for index, raw_binding in enumerate(raw_sources):
        binding = _require_mapping(
            raw_binding,
            field=f"{label}.implementation_sources[{index}]",
        )
        if set(binding) != {"path", "sha256"} or not _is_sha256(
            binding.get("sha256")
        ):
            raise ValueError(f"The {label} implementation-source binding is invalid.")
        implementation_path = Path(str(binding["path"]))
        if not implementation_path.is_absolute():
            implementation_path = (
                manifest_path.parent / implementation_path
            ).resolve()
        if implementation_path in observed_paths or not implementation_path.is_file():
            raise ValueError(f"The {label} implementation-source path is invalid.")
        if _file_sha256(implementation_path) != binding["sha256"]:
            raise ValueError(f"The {label} implementation source changed after freeze.")
        observed_paths.add(implementation_path)
        observed_filenames.add(implementation_path.name)
    if observed_filenames != expected_filenames:
        raise ValueError(f"The {label} binds different implementation sources.")


def _validate_manifest_file_binding_mapping(
    manifest: Mapping[str, Any],
    *,
    field: str,
    label: str,
    expected_names: frozenset[str],
) -> Mapping[str, Mapping[str, str]]:
    """重算命名源码闭包，并返回规范化的绝对路径绑定。"""

    raw_bindings = _require_mapping(manifest.get(field), field=f"{label}.{field}")
    if set(raw_bindings) != expected_names:
        raise ValueError(f"The {label} source closure is incomplete.")
    result: dict[str, Mapping[str, str]] = {}
    for name, raw_binding in raw_bindings.items():
        binding = _require_mapping(raw_binding, field=f"{label}.{field}.{name}")
        if set(binding) != {"path", "sha256"} or not _is_sha256(
            binding.get("sha256")
        ):
            raise ValueError(f"The {label} source binding is invalid.")
        path = Path(str(binding["path"])).resolve()
        if not path.is_file() or _file_sha256(path) != binding["sha256"]:
            raise ValueError(f"The {label} source changed after freeze: {name}")
        result[str(name)] = {"path": str(path), "sha256": str(binding["sha256"])}
    return result


def _validate_controller_replay_backend_implementation(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> None:
    replay_backend = _require_mapping(
        manifest.get("replay_backend"), field="controller.replay_backend"
    )
    implementation_path = Path(str(replay_backend["implementation_path"]))
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    if not implementation_path.is_file():
        raise ValueError("The sequential-controller replay backend does not exist.")
    if _file_sha256(implementation_path) != replay_backend.get(
        "implementation_sha256"
    ):
        raise ValueError("The sequential-controller replay backend changed after freeze.")


def _validate_trace_replay_verifier_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
) -> None:
    if manifest.get(
        "schema_version"
    ) != "path_c_r015_trace_replay_verifier_manifest_v2":
        raise ValueError("The trace-replay verifier manifest has the wrong schema.")
    if manifest.get("callable_name") != "verify_r015_trace_manifest":
        raise ValueError("The trace-replay verifier callable changed.")
    contract = _require_mapping(
        manifest.get("contract"), field="trace_replay_verifier.contract"
    )
    if contract != {
        "full_environment_replay": True,
        "environment_steps": 400,
        "validate_actions": True,
        "validate_official_observations": True,
        "validate_projected_controller_inputs": True,
        "validate_raw_rewards": True,
        "validate_done_boundary": True,
        "replay_frozen_controller_every_step": True,
        "recompute_every_ego_action": True,
        "recompute_every_partner_action": True,
        "validate_recurrent_state_writes": True,
        "validate_partner_recurrent_state_writes": True,
        "validate_a1_selected_action_and_registered_script": True,
        "validate_a2_pre_probe_action_identity": True,
        "validate_a2_mask_and_use_belief_routes": True,
        "validate_a2_continuation_controller_actions": True,
    }:
        raise ValueError("The trace-replay verifier contract is incomplete.")
    support_report_sha256 = (
        None
        if preregistration.support_report_path is None
        else _file_sha256(preregistration.support_report_path)
    )
    expected_dependencies = {
        "environment_config_sha256": preregistration.artifacts[
            "environment_config"
        ].sha256,
        "environment_source_sha256": preregistration.artifacts[
            "environment_source"
        ].sha256,
        "ego_checkpoint_sha256": preregistration.artifacts["ego_checkpoint"].sha256,
        "continuation_planner_sha256": preregistration.artifacts[
            "continuation_planner"
        ].sha256,
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "ego_evidence_contract_sha256": preregistration.artifacts[
            "ego_evidence_contract"
        ].sha256,
        "response_vocabulary_sha256": preregistration.artifacts[
            "response_vocabulary"
        ].sha256,
        "probe_registry_sha256": preregistration.artifacts["probe_registry"].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": support_report_sha256,
    }
    if any(manifest.get(field) != value for field, value in expected_dependencies.items()):
        raise ValueError("The trace-replay verifier binds different controller machinery.")
    if not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The trace-replay verifier implementation is not bound.")


TraceReplayVerifier = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


def _load_trace_replay_verifier(
    preregistration: R015Preregistration,
) -> TraceReplayVerifier:
    manifest_path = preregistration.artifacts["trace_replay_verifier"].path
    if manifest_path is None:
        raise ValueError("The trace-replay verifier manifest is pending.")
    manifest = _load_json_or_yaml(manifest_path)
    _validate_trace_replay_verifier_manifest(manifest, preregistration)
    _validate_manifest_implementation(
        manifest,
        manifest_path=manifest_path,
        label="trace-replay verifier",
    )
    implementation_path = Path(str(manifest["implementation_path"]))
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    module_name = f"path_c_r015_trace_replay_{manifest['implementation_sha256']}"
    module_spec = importlib.util.spec_from_file_location(module_name, implementation_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("The trace-replay verifier cannot be loaded.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    verifier = getattr(module, "verify_r015_trace_manifest", None)
    if not callable(verifier):
        sys.modules.pop(module_name, None)
        raise ValueError("The trace-replay verifier callable is missing.")
    return verifier


SafetyBranchEvidenceVerifier = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]

DecisionEvidenceVerifier = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


def _validate_decision_evidence_verifier_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
) -> None:
    if manifest.get(
        "schema_version"
    ) != "path_c_r015_decision_evidence_verifier_v1" or manifest.get(
        "callable"
    ) != "verify_r015_probe_decision":
        raise ValueError("The decision-evidence verifier has the wrong interface.")
    expected_contract = {
        "replay_every_consultation": True,
        "use_official_local_history_only": True,
        "recompute_complete_probe_registry": True,
        "recompute_eligibility_and_static_safety": True,
        "recompute_j_use_j_mask_v_base_v_mask": True,
        "recompute_score_and_selected_probe": True,
        "validate_first_positive_stop": True,
        "validate_registered_random_keys": True,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("The decision-evidence verifier contract is incomplete.")
    support_report_sha256 = (
        None
        if preregistration.support_report_path is None
        else _file_sha256(preregistration.support_report_path)
    )
    expected_dependencies = {
        "environment_config_sha256": preregistration.artifacts[
            "environment_config"
        ].sha256,
        "environment_source_sha256": preregistration.artifacts[
            "environment_source"
        ].sha256,
        "continuation_planner_sha256": preregistration.artifacts[
            "continuation_planner"
        ].sha256,
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "ego_evidence_contract_sha256": preregistration.artifacts[
            "ego_evidence_contract"
        ].sha256,
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "wrong_delivery_detector_sha256": preregistration.artifacts[
            "wrong_delivery_detector"
        ].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "probe_registry_sha256": preregistration.artifacts["probe_registry"].sha256,
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": support_report_sha256,
    }
    if any(manifest.get(field) != value for field, value in expected_dependencies.items()):
        raise ValueError("The decision verifier binds different frozen machinery.")
    if not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The decision-evidence verifier implementation is not bound.")


def _load_decision_evidence_verifier(
    preregistration: R015Preregistration,
) -> DecisionEvidenceVerifier:
    manifest_path = preregistration.artifacts["decision_evidence_verifier"].path
    if manifest_path is None:
        raise ValueError("The decision-evidence verifier manifest is pending.")
    manifest = _load_json_or_yaml(manifest_path)
    _validate_decision_evidence_verifier_manifest(manifest, preregistration)
    _validate_manifest_implementation(
        manifest,
        manifest_path=manifest_path,
        label="decision-evidence verifier",
    )
    implementation_path = Path(str(manifest["implementation_path"]))
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    module_name = f"path_c_r015_decision_replay_{manifest['implementation_sha256']}"
    module_spec = importlib.util.spec_from_file_location(module_name, implementation_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("The decision-evidence verifier cannot be loaded.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    verifier = getattr(module, "verify_r015_probe_decision", None)
    if not callable(verifier):
        sys.modules.pop(module_name, None)
        raise ValueError("The decision-evidence verifier callable is missing.")
    return verifier


def _validate_safety_branch_evidence_verifier_manifest(
    manifest: Mapping[str, Any],
    preregistration: R015Preregistration,
) -> None:
    if manifest.get(
        "schema_version"
    ) != "path_c_r015_safety_branch_evidence_verifier_v2" or manifest.get(
        "callable"
    ) != "verify_r015_safety_comparison":
        raise ValueError("The safety-branch verifier has the wrong interface.")
    expected_contract = {
        "replay_every_branch_in_frozen_environment": True,
        "recompute_wrong_delivery_per_branch": True,
        "recompute_branch_count": True,
        "validate_registered_random_keys": True,
        "validate_positive_posterior_support": True,
        "validate_compatible_hidden_state": True,
        "reconstruct_from_official_history_and_support_checkpoint": True,
        "execute_selected_registered_probe_script": True,
        "return_branch_evidence_and_result_hashes": True,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("The safety-branch verifier contract is incomplete.")
    support_report_sha256 = (
        None
        if preregistration.support_report_path is None
        else _file_sha256(preregistration.support_report_path)
    )
    expected_dependencies = {
        "environment_config_sha256": preregistration.artifacts[
            "environment_config"
        ].sha256,
        "environment_source_sha256": preregistration.artifacts[
            "environment_source"
        ].sha256,
        "wrong_delivery_detector_sha256": preregistration.artifacts[
            "wrong_delivery_detector"
        ].sha256,
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "ego_evidence_contract_sha256": preregistration.artifacts[
            "ego_evidence_contract"
        ].sha256,
        "response_vocabulary_sha256": preregistration.artifacts[
            "response_vocabulary"
        ].sha256,
        "probe_registry_sha256": preregistration.artifacts["probe_registry"].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": support_report_sha256,
    }
    if any(manifest.get(field) != value for field, value in expected_dependencies.items()):
        raise ValueError("The safety-branch verifier binds different frozen machinery.")
    if not str(manifest.get("implementation_path", "")) or not _is_sha256(
        manifest.get("implementation_sha256")
    ):
        raise ValueError("The safety-branch verifier implementation is not bound.")


def _load_safety_branch_evidence_verifier(
    preregistration: R015Preregistration,
) -> SafetyBranchEvidenceVerifier:
    manifest_path = preregistration.artifacts[
        "safety_branch_evidence_verifier"
    ].path
    if manifest_path is None:
        raise ValueError("The safety-branch evidence verifier manifest is pending.")
    manifest = _load_json_or_yaml(manifest_path)
    _validate_safety_branch_evidence_verifier_manifest(manifest, preregistration)
    _validate_manifest_implementation(
        manifest,
        manifest_path=manifest_path,
        label="safety-branch evidence verifier",
    )
    implementation_path = Path(str(manifest["implementation_path"]))
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    module_name = f"path_c_r015_safety_replay_{manifest['implementation_sha256']}"
    module_spec = importlib.util.spec_from_file_location(module_name, implementation_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("The safety-branch evidence verifier cannot be loaded.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    verifier = getattr(module, "verify_r015_safety_comparison", None)
    if not callable(verifier):
        sys.modules.pop(module_name, None)
        raise ValueError("The safety-branch evidence verifier callable is missing.")
    return verifier


FormalSamplingSchedule = frozenset[
    tuple[int, str, str, tuple[int, ...], int]
]


def _load_formal_sampling_schedule(
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
) -> FormalSamplingSchedule:
    schedule_path = preregistration.artifacts["formal_sampling_schedule"].path
    if schedule_path is None:
        raise ValueError("The formal sampling schedule is pending.")
    manifest = _load_json_or_yaml(schedule_path)
    if manifest.get("schema_version") != (
        R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA
    ) or manifest.get(
        "experiment_id"
    ) != R015_EXPERIMENT_ID:
        raise ValueError("The formal sampling schedule has the wrong schema.")
    if manifest.get("frozen_before_formal_data") is not True or manifest.get(
        "selection_uses_outcomes"
    ) is not False or manifest.get(
        "replacement_seed_selection_uses_outcomes"
    ) is not False:
        raise ValueError("The formal sampling schedule permits outcome-based selection.")
    formal_root_seed = _nonnegative_int(
        manifest.get("formal_root_seed"),
        field="formal_sampling_schedule.formal_root_seed",
    )
    if _positive_int(
        manifest.get("mechanical_replacement_attempts_per_coordinate"),
        field="formal_sampling_schedule.mechanical_replacement_attempts_per_coordinate",
    ) != preregistration.statistics.mechanical_replacement_attempts_per_coordinate or (
        manifest.get("mechanical_replacement_attempt_indices")
        != preregistration.statistics.mechanical_replacement_attempt_indices
    ) or manifest.get("replacement_episode_seed_derivation_id") != (
        preregistration.statistics.replacement_episode_seed_derivation_id
    ) or manifest.get("replacement_episode_seed_collision_policy") != (
        preregistration.statistics.replacement_episode_seed_collision_policy
    ) or (
        manifest.get("audit_unit_changes_across_replacement_attempts") is not (
            preregistration.statistics.audit_unit_changes_across_replacement_attempts
        )
    ) or manifest.get("mechanical_attempt_exhaustion_rule") != (
        preregistration.statistics.mechanical_attempt_exhaustion_rule
    ):
        raise ValueError("The formal sampling schedule changed its replacement contract.")
    if manifest.get("zero_count_independence_contract") != (
        preregistration.statistics.zero_count_independence_contract
    ) or manifest.get("firing_prefixes_independent_across_prototypes") is not True or (
        manifest.get("cross_prototype_shared_randomness_before_firing") is not False
    ):
        raise ValueError(
            "The formal sampling schedule does not establish independent firing prefixes."
        )
    if any(
        legacy_field in manifest
        for legacy_field in (
            "ego_position_one_probability",
            "ego_position_schedule_conditioned_on_balance",
        )
    ):
        raise ValueError("The formal sampling schedule retains a retired random-seat field.")
    if manifest.get("round_sampling_contract") != (
        preregistration.statistics.round_sampling_contract
    ) or manifest.get("round_vectors_generated_independently") is not True or (
        manifest.get("ego_agent_id") != "agent_1"
    ) or manifest.get("partner_agent_id") != "agent_0":
        raise ValueError(
            "The formal sampling schedule does not establish independent cook-seat rounds."
        )
    expected_rounds = preregistration.statistics.n_rounds_max
    if _positive_int(
        manifest.get("n_rounds"),
        field="formal_sampling_schedule.n_rounds",
    ) != expected_rounds:
        raise ValueError("The formal sampling schedule has the wrong round count.")
    rows = _require_sequence(
        manifest.get("entries"), field="formal_sampling_schedule.entries"
    )
    entries: set[tuple[int, str, str, tuple[int, ...], int]] = set()
    audit_unit_ids: set[str] = set()
    per_prototype = {prototype_id: 0 for prototype_id in support_members}
    prototype_order = tuple(
        candidate.candidate_id for candidate in preregistration.support_spec.candidates
    )
    if set(prototype_order) != set(support_members):
        raise ValueError("The formal sampling schedule changed prototype order.")
    for raw_row in rows:
        row = _require_mapping(raw_row, field="formal_sampling_schedule.entry")
        if set(row) != {
            "round_index",
            "audit_unit_id",
            "prototype_id",
            "episode_seeds_by_attempt_index",
            "ego_position",
        }:
            raise ValueError("A formal sampling-schedule row has the wrong fields.")
        round_index = _positive_int(
            row.get("round_index"), field="formal_sampling_schedule.round_index"
        )
        audit_unit_id = str(row.get("audit_unit_id", ""))
        prototype_id = str(row.get("prototype_id", ""))
        episode_seeds = tuple(
            _nonnegative_int(
                value,
                field="formal_sampling_schedule.episode_seeds_by_attempt_index",
            )
            for value in _require_sequence(
                row.get("episode_seeds_by_attempt_index"),
                field="formal_sampling_schedule.episode_seeds_by_attempt_index",
            )
        )
        ego_position = _nonnegative_int(
            row.get("ego_position"), field="formal_sampling_schedule.ego_position"
        )
        if (
            round_index > expected_rounds
            or not _is_sha256(audit_unit_id)
            or prototype_id not in support_members
            or ego_position != 1
            or len(episode_seeds)
            != preregistration.statistics.mechanical_replacement_attempts_per_coordinate
        ):
            raise ValueError("A formal sampling-schedule row is invalid.")
        prototype_index = prototype_order.index(prototype_id)
        expected_audit_unit_id = _canonical_value_sha256(
            [
                "r015_formal_audit_unit_v4",
                "r015_formal_sampling_schedule_v4",
                formal_root_seed,
                round_index,
                prototype_index,
                prototype_id,
            ]
        )
        if audit_unit_id != expected_audit_unit_id:
            raise ValueError(
                "A formal sampling-schedule audit unit was not derived from its coordinate."
            )
        expected_episode_seeds = tuple(
            int(
                _canonical_value_sha256(
                    [
                        "r015_formal_prefrozen_replacement_seed_v4",
                        formal_root_seed,
                        round_index,
                        prototype_index,
                        prototype_id,
                        attempt_index,
                    ]
                )[-8:],
                16,
            )
            for attempt_index in range(R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT)
        )
        if episode_seeds != expected_episode_seeds:
            raise ValueError(
                "A formal sampling-schedule seed was not derived from its coordinate."
            )
        entry = (
            round_index,
            audit_unit_id,
            prototype_id,
            episode_seeds,
            ego_position,
        )
        if entry in entries or audit_unit_id in audit_unit_ids:
            raise ValueError("The formal sampling schedule repeats a row or audit unit.")
        entries.add(entry)
        audit_unit_ids.add(audit_unit_id)
        per_prototype[prototype_id] += 1
    if per_prototype != {
        prototype_id: expected_rounds for prototype_id in support_members
    } or len(entries) != len(support_members) * expected_rounds:
        raise ValueError(
            "The formal sampling schedule is incomplete or unequal across prototypes."
        )
    round_members = {
        round_index: {
            prototype_id
            for observed_round, _, prototype_id, _, _ in entries
            if observed_round == round_index
        }
        for round_index in range(1, expected_rounds + 1)
    }
    if any(members != set(support_members) for members in round_members.values()):
        raise ValueError("Every formal round must contain one block per prototype.")
    return frozenset(entries)


def _validate_frozen_scientific_artifacts(
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
) -> None:
    """Read the scientific manifests that cannot be reduced to a file hash."""

    _load_formal_sampling_schedule(preregistration, support_members)

    registry_path = preregistration.artifacts["probe_registry"].path
    projection_path = preregistration.artifacts["response_projection"].path
    response_vocabulary_path = preregistration.artifacts[
        "response_vocabulary"
    ].path
    planner_path = preregistration.artifacts["continuation_planner"].path
    continuation_path = preregistration.artifacts["continuation_controller"].path
    decision_verifier_path = preregistration.artifacts[
        "decision_evidence_verifier"
    ].path
    filter_path = preregistration.artifacts["official_history_filter"].path
    detector_path = preregistration.artifacts["wrong_delivery_detector"].path
    return_bound_path = preregistration.artifacts["return_bound_derivation"].path
    stability_path = preregistration.artifacts["planning_stability_report"].path
    pilot_path = preregistration.artifacts["pilot_wiring_report"].path
    key_derivation_path = preregistration.artifacts["random_key_derivation"].path
    safety_verifier_path = preregistration.artifacts[
        "safety_branch_evidence_verifier"
    ].path
    replay_verifier_path = preregistration.artifacts["trace_replay_verifier"].path
    if any(
        path is None
        for path in (
            registry_path,
            projection_path,
            response_vocabulary_path,
            planner_path,
            continuation_path,
            decision_verifier_path,
            filter_path,
            detector_path,
            return_bound_path,
            stability_path,
            pilot_path,
            key_derivation_path,
            safety_verifier_path,
            replay_verifier_path,
        )
    ):
        raise ValueError("A required R015 scientific manifest is pending.")

    filter_manifest = _load_json_or_yaml(filter_path)
    planner_manifest = _load_json_or_yaml(planner_path)
    continuation_manifest = _load_json_or_yaml(continuation_path)
    _validate_official_history_filter_manifest(filter_manifest, preregistration)
    _validate_sequential_controller_manifest(planner_manifest, preregistration)
    _validate_continuation_controller_manifest(
        continuation_manifest,
        preregistration,
        support_members,
    )
    _validate_manifest_implementation(
        filter_manifest,
        manifest_path=filter_path,
        label="official-history filter",
    )
    _validate_manifest_implementation(
        planner_manifest,
        manifest_path=planner_path,
        label="sequential controller",
    )
    _validate_manifest_implementation(
        continuation_manifest,
        manifest_path=continuation_path,
        label="continuation controller",
    )
    _validate_controller_replay_backend_implementation(
        planner_manifest,
        manifest_path=planner_path,
    )
    decision_verifier_manifest = _load_json_or_yaml(decision_verifier_path)
    _validate_decision_evidence_verifier_manifest(
        decision_verifier_manifest, preregistration
    )
    _validate_manifest_implementation(
        decision_verifier_manifest,
        manifest_path=decision_verifier_path,
        label="decision-evidence verifier",
    )
    replay_verifier_manifest = _load_json_or_yaml(replay_verifier_path)
    _validate_trace_replay_verifier_manifest(
        replay_verifier_manifest, preregistration
    )
    _validate_manifest_implementation(
        replay_verifier_manifest,
        manifest_path=replay_verifier_path,
        label="trace-replay verifier",
    )
    safety_verifier_manifest = _load_json_or_yaml(safety_verifier_path)
    _validate_safety_branch_evidence_verifier_manifest(
        safety_verifier_manifest, preregistration
    )
    _validate_manifest_implementation(
        safety_verifier_manifest,
        manifest_path=safety_verifier_path,
        label="safety-branch evidence verifier",
    )

    registry = _load_json_or_yaml(registry_path)
    if registry.get("schema_version") != "path_c_r015_probe_registry_v1":
        raise ValueError("The probe-registry artifact has the wrong schema.")
    registry_scripts = _probe_scripts_from_experiment(
        {"registered_probe_scripts": registry.get("probe_scripts")}
    )
    if registry_scripts != preregistration.probe_scripts or registry.get(
        "semantic_sha256"
    ) != preregistration.probe_registry_semantic_sha256:
        raise ValueError("The probe registry differs from the preregistered scripts.")
    if _positive_int(
        registry.get("maximum_script_length"),
        field="probe_registry.maximum_script_length",
    ) != 2:
        raise ValueError("The probe registry must keep the two-action length limit.")

    response_vocabulary = _load_json_or_yaml(response_vocabulary_path)
    if response_vocabulary.get(
        "schema_version"
    ) != "path_c_r015_response_vocabulary_v1" or not _is_sha256(
        response_vocabulary.get("response_summary_sha256")
    ):
        raise ValueError("The response-vocabulary artifact has the wrong schema.")
    _validate_manifest_implementation_sources(
        response_vocabulary,
        manifest_path=response_vocabulary_path,
        label="response vocabulary",
        expected_filenames=frozenset(
            {"path_c_response_summary.py", "path_c_response_probe.py"}
        ),
    )

    projection = _load_json_or_yaml(projection_path)
    if projection.get("schema_version") != "path_c_r015_response_projection_v1":
        raise ValueError("The response-projection artifact has the wrong schema.")
    if projection.get("name") != preregistration.information.projection_name:
        raise ValueError("The response-projection name changed after preregistration.")
    if set(projection.get("mask_targets", ())) != set(
        preregistration.information.projection_mask_targets
    ) or set(projection.get("preserved_fields", ())) != set(
        preregistration.information.projection_preserved_fields
    ):
        raise ValueError("The response-projection field contract changed.")
    if projection.get("use_only_route") != preregistration.information.use_only_route:
        raise ValueError("The response-projection use route changed.")
    projection_hash_fields = (
        "implementation_sha256",
        "derived_field_dependency_graph_sha256",
        "explicit_alias_clearlist_sha256",
        "recurrent_state_write_rule_sha256",
    )
    if any(not _is_sha256(projection.get(field)) for field in projection_hash_fields):
        raise ValueError("The response projection lacks a frozen dependency binding.")
    projection_semantics = {
        "name": projection["name"],
        "mask_targets": sorted(str(item) for item in projection["mask_targets"]),
        "preserved_fields": sorted(
            str(item) for item in projection["preserved_fields"]
        ),
        "use_only_route": projection["use_only_route"],
        **{field: projection[field] for field in projection_hash_fields},
    }
    if projection.get("semantic_sha256") != _canonical_mapping_sha256(
        projection_semantics
    ):
        raise ValueError("The response-projection semantic digest is invalid.")

    detector = _load_json_or_yaml(detector_path)
    if detector.get("schema_version") != "path_c_r015_wrong_delivery_detector_v1":
        raise ValueError("The wrong-delivery detector has the wrong schema.")
    if detector.get("event_name") != "wrong_delivery" or detector.get(
        "recipe_indicator_cost_is_safety_event"
    ) is not False:
        raise ValueError("R015 safety must mean wrong delivery only.")
    if detector.get("recipe_indicator_cost_in_raw_return") is not True:
        raise ValueError("The recipe-indicator cost must remain in raw return.")
    if detector.get("environment_source_sha256") != preregistration.artifacts[
        "environment_source"
    ].sha256 or not _is_sha256(detector.get("implementation_sha256")):
        raise ValueError("The wrong-delivery detector is not bound to the environment.")
    environment_source_names = frozenset(
        {"settings", "overcooked", "layouts", "common"}
    )
    detector_environment_sources = _validate_manifest_file_binding_mapping(
        detector,
        field="environment_source_closure",
        label="wrong-delivery detector",
        expected_names=environment_source_names,
    )
    if detector_environment_sources["overcooked"]["sha256"] != (
        preregistration.artifacts["environment_source"].sha256
    ):
        raise ValueError("The wrong-delivery detector source closure changed.")
    _validate_manifest_implementation(
        detector,
        manifest_path=detector_path,
        label="wrong-delivery detector",
    )

    return_bound = _load_json_or_yaml(return_bound_path)
    if return_bound.get(
        "schema_version"
    ) != "path_c_r015_return_bound_derivation_v2":
        raise ValueError("The return-bound derivation has the wrong schema.")
    if return_bound.get("proved") is not True or return_bound.get(
        "reward_accounting"
    ) != "undiscounted_raw_team_reward":
        raise ValueError("The registered deterministic return bound is not proved.")
    if not str(return_bound.get("implementation_path", "")) or not _is_sha256(
        return_bound.get("implementation_sha256")
    ):
        raise ValueError("The return-bound proof implementation is not bound.")
    if _positive_int(
        return_bound.get("horizon_environment_steps"),
        field="return_bound.horizon_environment_steps",
    ) != 400:
        raise ValueError("The deterministic return bound must cover all 400 steps.")
    if return_bound.get("environment_config_sha256") != preregistration.artifacts[
        "environment_config"
    ].sha256 or return_bound.get(
        "environment_source_sha256"
    ) != preregistration.artifacts[
        "environment_source"
    ].sha256:
        raise ValueError("The return bound is not bound to the frozen environment.")
    return_bound_environment_sources = _validate_manifest_file_binding_mapping(
        return_bound,
        field="environment_source_closure",
        label="return-bound proof",
        expected_names=environment_source_names,
    )
    if return_bound_environment_sources != detector_environment_sources:
        raise ValueError("The safety detector and return proof bind different sources.")
    bound_values = {
        "delivery_reward": preregistration.statistics.delivery_reward,
        "return_lower_bound": preregistration.statistics.return_lower_bound,
        "return_upper_bound": preregistration.statistics.return_upper_bound,
        "paired_difference_absolute_bound": (
            preregistration.statistics.paired_difference_absolute_bound
        ),
        "paired_difference_width": preregistration.statistics.paired_difference_width,
    }
    if any(value is None for value in bound_values.values()) or any(
        not math.isclose(
            _finite_float(return_bound.get(field), field=f"return_bound.{field}"),
            float(value),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for field, value in bound_values.items()
    ):
        raise ValueError("The return-bound values differ from the preregistration.")
    if return_bound.get("lockstep_invariant_required") is not True or return_bound.get(
        "arbitrary_suffix_carry_in_excluded"
    ) is not False or _positive_int(
        return_bound.get("minimum_probe_step"), field="return_bound.minimum_probe_step"
    ) != 1 or _positive_int(
        return_bound.get("maximum_probe_step"), field="return_bound.maximum_probe_step"
    ) != 100:
        raise ValueError("The structural paired-difference proof contract changed.")
    _validate_manifest_implementation(
        return_bound,
        manifest_path=return_bound_path,
        label="return-bound proof",
    )

    stability = _load_json_or_yaml(stability_path)
    if stability.get("schema_version") != "path_c_r015_planning_stability_report_v1":
        raise ValueError("The planning-stability report has the wrong schema.")
    branches = _positive_int(
        stability.get("branch_count"), field="planning_stability.branch_count"
    )
    doubled = _positive_int(
        stability.get("doubled_branch_count"),
        field="planning_stability.doubled_branch_count",
    )
    agreement = _finite_float(
        stability.get("action_ranking_agreement"),
        field="planning_stability.action_ranking_agreement",
    )
    if doubled != 2 * branches or not 0.95 <= agreement <= 1.0:
        raise ValueError("Planning must pass the R versus 2R ranking check at 95 percent.")
    if stability.get("independent_design_data") is not True or stability.get(
        "formal_data_included"
    ) is not False or stability.get("passed") is not True:
        raise ValueError("Planning stability must use separate design data and pass.")
    if stability.get("continuation_planner_sha256") != preregistration.artifacts[
        "continuation_planner"
    ].sha256 or stability.get("probe_registry_semantic_sha256") != (
        preregistration.probe_registry_semantic_sha256
    ):
        raise ValueError("The planning-stability report binds different machinery.")

    pilot = _load_json_or_yaml(pilot_path)
    if pilot.get("schema_version") != "path_c_r015_pilot_wiring_report_v2":
        raise ValueError("The pilot wiring report has the wrong schema.")
    raw_counts = _require_mapping(
        pilot.get("completed_blocks_per_prototype"),
        field="pilot.completed_blocks_per_prototype",
    )
    counts = {
        str(prototype_id): _nonnegative_int(
            count, field=f"pilot.completed_blocks_per_prototype.{prototype_id}"
        )
        for prototype_id, count in raw_counts.items()
    }
    if counts != {
        prototype_id: preregistration.statistics.pilot_blocks_per_prototype
        for prototype_id in support_members
    }:
        raise ValueError("The pilot must complete 20 blocks for every fixed prototype.")
    wiring_checks = _require_mapping(
        pilot.get("wiring_checks"), field="pilot.wiring_checks"
    )
    required_checks = {
        "information_isolation",
        "filter_normalization_and_repeat_stability",
        "response_use_pairing",
        "random_key_naming",
        "zero_probe_interpretability",
        "artifact_counts_read_back",
        "lockstep_structural_zero",
        "shared_firing_indicator",
        "firing_count_input_isolation",
    }
    if set(wiring_checks) != required_checks or any(
        wiring_checks[name] is not True for name in required_checks
    ):
        raise ValueError("The pilot wiring checks are incomplete.")
    if pilot.get("scientific_effect_used_for_round_budget") is not False or pilot.get(
        "empirical_variance_used_for_round_budget"
    ) is not False or pilot.get("formal_interval_includes_pilot") is not False:
        raise ValueError("Pilot outcomes may not influence or enter the formal interval.")

    key_derivation = _load_json_or_yaml(key_derivation_path)
    if key_derivation.get(
        "schema_version"
    ) != "path_c_r015_random_key_derivation_v1":
        raise ValueError("The random-key derivation artifact has the wrong schema.")
    required_purposes = {
        "actual_episode",
        "value_planning",
        "score_estimation",
        "safety_validation",
    }
    if tuple(key_derivation.get("coordinates", ())) != _R015_RANDOM_KEY_COORDINATES or set(
        key_derivation.get("purpose_names", ())
    ) != required_purposes:
        raise ValueError("The random-key coordinates or purpose names changed.")
    if key_derivation.get("derivation_algorithm") != (
        "sha256_canonical_json_v1"
    ) or key_derivation.get("key_payload_schema_version") != (
        "path_c_r015_random_key_v1"
    ):
        raise ValueError("The random-key derivation algorithm changed.")
    if key_derivation.get("independent_purpose_namespaces") is not True or (
        key_derivation.get("branch_index_changes_key") is not True
    ) or not _is_sha256(key_derivation.get("implementation_sha256")):
        raise ValueError("The random-key derivation does not establish branch separation.")
    _validate_manifest_implementation(
        key_derivation,
        manifest_path=key_derivation_path,
        label="random-key derivation",
    )


def validate_r015_support_report(
    report: Mapping[str, Any],
    support_spec: R015PartnerSupportSpec,
) -> Mapping[str, Mapping[str, Any]]:
    """Recompute four-candidate provenance and ability from episode evidence."""

    return validate_r015_support_report_evidence(support_spec, report)


def _validate_r015_support_report_legacy(
    report: Mapping[str, Any],
    support_spec: R015PartnerSupportSpec,
) -> Mapping[str, Mapping[str, Any]]:
    """Retained only as a readable description; it is not a formal entry point."""

    if report.get("schema_version") != R015_SUPPORT_REPORT_SCHEMA:
        raise ValueError("Unsupported R015 partner-support report.")
    if report.get("registration_status") != "configured":
        raise ValueError("R015 support report does not match the configured registration.")
    if report.get("support_status") != "admitted" or report.get(
        "support_complete"
    ) is not True:
        raise ValueError("R015 formal readout requires a complete admitted support.")
    if report.get("scientific_readout_allowed") is not False:
        raise ValueError("Partner ability admission cannot be marked as a scientific readout.")
    if report.get("artifact_verification_status") != "verified":
        raise ValueError("R015 support artifacts were not verified as a complete set.")
    expected = {candidate.candidate_id: candidate for candidate in support_spec.candidates}
    members = _require_mapping(report.get("members"), field="support report members")
    if set(members) != set(expected) or len(members) != 4:
        raise ValueError("R015 support report must admit the four registered candidates.")
    if set(report.get("qualified_candidate_ids", ())) != set(expected):
        raise ValueError("R015 qualified-candidate list is incomplete or changed.")
    report_families = _require_mapping(
        report.get("families"), field="support report families"
    )
    if set(report_families) != {family.family_id for family in support_spec.families}:
        raise ValueError("R015 support report changed the registered family set.")
    for family in support_spec.families:
        reported_family = _require_mapping(
            report_families[family.family_id],
            field=f"families.{family.family_id}",
        )
        if reported_family.get("family_spec_sha256") != family.family_spec_sha256:
            raise ValueError("R015 support report changed a family definition.")
        if any(
            reported_family.get(key) != value
            for key, value in family.definition().items()
        ):
            raise ValueError("R015 support report changed family generation semantics.")
    family_counts: dict[str, int] = {
        family.family_id: 0 for family in support_spec.families
    }
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    training_config_hashes: set[str] = set()
    training_manifest_hashes: set[str] = set()
    training_run_ids: set[str] = set()
    environment_hashes: set[str] = set()
    implementation_hashes_by_family: dict[str, set[str]] = {
        family.family_id: set() for family in support_spec.families
    }
    training_runs: set[tuple[str, int]] = set()
    normalized: dict[str, Mapping[str, Any]] = {}
    for candidate_id, raw_member in members.items():
        member = _require_mapping(raw_member, field=f"members.{candidate_id}")
        candidate = expected[candidate_id]
        if member.get("artifact_verified") is not True:
            raise ValueError("Every R015 support member must have verified artifacts.")
        if member.get("admitted") is not True:
            raise ValueError("Every R015 support member must pass ability admission.")
        if member.get("family_id") != candidate.family_id or int(
            member.get("training_seed", -1)
        ) != candidate.training_seed:
            raise ValueError("R015 support member changed family or training run.")
        family = support_spec.family(candidate.family_id)
        if member.get("family_spec_sha256") != family.family_spec_sha256:
            raise ValueError("R015 support member family definition changed.")
        checkpoint_sha = member.get("checkpoint_sha256")
        training_sha = member.get("training_config_sha256")
        manifest_sha = member.get("training_manifest_sha256")
        environment_sha = member.get("environment_config_sha256")
        implementation_sha = member.get("training_implementation_sha256")
        training_run_id = member.get("training_run_id")
        if any(
            not _is_sha256(value)
            for value in (
                checkpoint_sha,
                training_sha,
                manifest_sha,
                environment_sha,
                implementation_sha,
                training_run_id,
            )
        ):
            raise ValueError("R015 admitted artifacts require real SHA-256 digests.")
        expected_run_id = partner_training_run_id(
            family_spec_sha256=family.family_spec_sha256,
            training_seed=candidate.training_seed,
            training_config_sha256=str(training_sha),
            environment_config_sha256=str(environment_sha),
            training_implementation_sha256=str(implementation_sha),
        )
        if training_run_id != expected_run_id:
            raise ValueError("R015 training-run identifier does not match its bindings.")
        if int(member.get("snapshot_environment_steps", -1)) != (
            candidate.expected_environment_steps
        ):
            raise ValueError("R015 admitted checkpoint step changed after registration.")
        checkpoint_path = str(member.get("checkpoint_path", ""))
        if not checkpoint_path:
            raise ValueError("R015 admitted checkpoint path is empty.")
        if Path(checkpoint_path).resolve() != candidate.checkpoint_path.resolve():
            raise ValueError("R015 admitted checkpoint path changed after registration.")
        training_config_path = Path(str(member.get("training_config_path", "")))
        if training_config_path.resolve() != candidate.training_config_path.resolve():
            raise ValueError("R015 admitted training configuration path changed.")
        training_manifest_path = Path(str(member.get("training_manifest_path", "")))
        if training_manifest_path.resolve() != candidate.training_manifest_path.resolve():
            raise ValueError("R015 admitted training manifest path changed.")
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file() or _file_sha256(checkpoint_file) != checkpoint_sha:
            raise ValueError("R015 admitted checkpoint is absent or changed.")
        if not training_config_path.is_file() or _file_sha256(
            training_config_path
        ) != training_sha:
            raise ValueError("R015 admitted training configuration is absent or changed.")
        if not training_manifest_path.is_file() or _file_sha256(
            training_manifest_path
        ) != manifest_sha:
            raise ValueError("R015 admitted training manifest is absent or changed.")
        checkpoint_paths.add(checkpoint_path)
        checkpoint_hashes.add(str(checkpoint_sha))
        training_config_hashes.add(str(training_sha))
        training_manifest_hashes.add(str(manifest_sha))
        training_run_ids.add(str(training_run_id))
        environment_hashes.add(str(environment_sha))
        implementation_hashes_by_family[candidate.family_id].add(
            str(implementation_sha)
        )
        training_runs.add((candidate.family_id, candidate.training_seed))
        family_counts[candidate.family_id] += 1
        normalized[candidate_id] = dict(member)
    if len(checkpoint_paths) != 4 or len(checkpoint_hashes) != 4:
        raise ValueError("R015 requires four unique checkpoint artifacts.")
    if len(training_config_hashes) != 4 or len(training_manifest_hashes) != 4:
        raise ValueError("R015 requires four independently bound training artifacts.")
    if len(training_run_ids) != 4:
        raise ValueError("R015 requires four unique training-run identifiers.")
    if len(environment_hashes) != 1:
        raise ValueError("R015 support members must bind one environment configuration.")
    if any(len(values) != 1 for values in implementation_hashes_by_family.values()):
        raise ValueError("Each R015 family must bind one training implementation.")
    if len(
        {next(iter(values)) for values in implementation_hashes_by_family.values()}
    ) != 2:
        raise ValueError("The two R015 families must bind different implementations.")
    if len(training_runs) != 4:
        raise ValueError("R015 requires four independent training runs.")
    if set(family_counts.values()) != {2}:
        raise ValueError("R015 requires exactly two admitted members per family.")
    reported_by_family = _require_mapping(
        report.get("members_by_family"), field="support report members_by_family"
    )
    expected_by_family = {
        family_id: sorted(
            candidate_id
            for candidate_id, member in normalized.items()
            if member.get("family_id") == family_id
        )
        for family_id in family_counts
    }
    normalized_by_family = {
        str(key): sorted(str(item) for item in _require_sequence(
            value, field=f"members_by_family.{key}"
        ))
        for key, value in reported_by_family.items()
    }
    if normalized_by_family != expected_by_family:
        raise ValueError("R015 support report family membership changed.")
    return normalized


@dataclass(frozen=True)
class ArmOutcome:
    group: str
    prototype_id: str
    family_id: str
    training_seed: int
    training_run_id: str
    ego_position: int
    initial_state_sha256: str
    episode_seed: int
    firing_indicator: bool
    raw_return: float
    valid: bool
    information_source: str
    forbidden_fields_read: tuple[str, ...]
    episode_random_key: str
    controller_contract_sha256: str
    ego_checkpoint_sha256: str
    environment_config_sha256: str
    environment_source_sha256: str
    official_history_filter_sha256: str

    @classmethod
    def from_mapping(cls, group: str, payload: Mapping[str, Any]) -> "ArmOutcome":
        outcome = cls(
            group=group,
            prototype_id=str(payload.get("prototype_id", "")),
            family_id=str(payload.get("family_id", "")),
            training_seed=_nonnegative_int(
                payload.get("training_seed"), field=f"arms.{group}.training_seed"
            ),
            training_run_id=str(payload.get("training_run_id", "")),
            ego_position=_nonnegative_int(
                payload.get("ego_position"), field=f"arms.{group}.ego_position"
            ),
            initial_state_sha256=str(payload.get("initial_state_sha256", "")),
            episode_seed=_nonnegative_int(
                payload.get("episode_seed"), field=f"arms.{group}.episode_seed"
            ),
            firing_indicator=(payload.get("firing_indicator") is True),
            raw_return=_finite_float(
                payload.get("raw_return"), field=f"arms.{group}.raw_return"
            ),
            valid=payload.get("valid") is True,
            information_source=str(payload.get("information_source")),
            forbidden_fields_read=tuple(
                str(item)
                for item in _require_sequence(
                    payload.get("forbidden_fields_read", ()),
                    field=f"arms.{group}.forbidden_fields_read",
                )
            ),
            episode_random_key=str(payload.get("episode_random_key", "")),
            controller_contract_sha256=str(
                payload.get("controller_contract_sha256", "")
            ),
            ego_checkpoint_sha256=str(payload.get("ego_checkpoint_sha256", "")),
            environment_config_sha256=str(
                payload.get("environment_config_sha256", "")
            ),
            environment_source_sha256=str(
                payload.get("environment_source_sha256", "")
            ),
            official_history_filter_sha256=str(
                payload.get("official_history_filter_sha256", "")
            ),
        )
        if not outcome.valid:
            raise ValueError("An invalid formal group invalidates the complete paired block.")
        if outcome.ego_position != 1:
            raise ValueError("R015 formal runtime fixes the ego cook at agent_1.")
        if not isinstance(payload.get("firing_indicator"), bool):
            raise ValueError("Every formal group must carry the shared firing indicator.")
        if outcome.information_source != "official_local_history_only":
            raise ValueError("Formal group read information outside official local history.")
        if outcome.forbidden_fields_read:
            raise ValueError("Formal group recorded a forbidden evaluator-only field read.")
        if not outcome.episode_random_key:
            raise ValueError("Every formal group requires its shared episode random key.")
        if not _is_sha256(outcome.initial_state_sha256):
            raise ValueError("Every formal group must bind the shared initial state.")
        if not _is_sha256(outcome.training_run_id):
            raise ValueError("Every formal group must bind the partner training run.")
        if not _is_sha256(outcome.controller_contract_sha256):
            raise ValueError("Every formal group must bind the controller input contract.")
        if any(
            not _is_sha256(value)
            for value in (
                outcome.ego_checkpoint_sha256,
                outcome.environment_config_sha256,
                outcome.environment_source_sha256,
                outcome.official_history_filter_sha256,
            )
        ):
            raise ValueError("Every formal group must bind its frozen runtime artifacts.")
        return outcome


@dataclass(frozen=True)
class SafetyComparison:
    support_prototype_id: str
    repetitions: int
    wrong_delivery_count: int
    positive_posterior_support: bool
    posterior_support_evidence: Mapping[str, Any]
    compatible_hidden_state_reconstructed: bool
    random_stream_key: str
    audit_unit_id: str
    episode_seed: int
    purpose: str
    environment_step: int
    branch_start_index: int
    branch_count: int
    branch_keys: tuple[str, ...]
    branch_evidence: tuple[Mapping[str, Any], ...]
    branch_results: tuple[Mapping[str, Any], ...]
    branch_keys_sha256: str
    random_key_derivation_sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SafetyComparison":
        return cls(
            support_prototype_id=str(payload.get("support_prototype_id", "")),
            repetitions=_nonnegative_int(
                payload.get("repetitions"), field="safety.comparisons.repetitions"
            ),
            wrong_delivery_count=_nonnegative_int(
                payload.get("wrong_delivery_count"),
                field="safety.comparisons.wrong_delivery_count",
            ),
            positive_posterior_support=payload.get("positive_posterior_support") is True,
            posterior_support_evidence=_require_mapping(
                payload.get("posterior_support_evidence"),
                field="safety.comparisons.posterior_support_evidence",
            ),
            compatible_hidden_state_reconstructed=(
                payload.get("compatible_hidden_state_reconstructed") is True
            ),
            random_stream_key=str(payload.get("random_stream_key", "")),
            audit_unit_id=str(payload.get("audit_unit_id", "")),
            episode_seed=_nonnegative_int(
                payload.get("episode_seed"),
                field="safety.comparisons.episode_seed",
            ),
            purpose=str(payload.get("purpose", "")),
            environment_step=_nonnegative_int(
                payload.get("environment_step"),
                field="safety.comparisons.environment_step",
            ),
            branch_start_index=_nonnegative_int(
                payload.get("branch_start_index"),
                field="safety.comparisons.branch_start_index",
            ),
            branch_count=_nonnegative_int(
                payload.get("branch_count"),
                field="safety.comparisons.branch_count",
            ),
            branch_keys=tuple(
                str(item)
                for item in _require_sequence(
                    payload.get("branch_keys"),
                    field="safety.comparisons.branch_keys",
                )
            ),
            branch_evidence=tuple(
                _require_mapping(item, field="safety.comparisons.branch_evidence item")
                for item in _require_sequence(
                    payload.get("branch_evidence"),
                    field="safety.comparisons.branch_evidence",
                )
            ),
            branch_results=tuple(
                _require_mapping(item, field="safety.comparisons.branch_results item")
                for item in _require_sequence(
                    payload.get("branch_results"),
                    field="safety.comparisons.branch_results",
                )
            ),
            branch_keys_sha256=str(payload.get("branch_keys_sha256", "")),
            random_key_derivation_sha256=str(
                payload.get("random_key_derivation_sha256", "")
            ),
        )


@dataclass(frozen=True)
class R015PairedBlock:
    round_index: int
    audit_unit_id: str
    prototype_id: str
    family_id: str
    training_seed: int
    training_run_id: str
    ego_position: int
    initial_state_sha256: str
    episode_seed: int
    mechanical_attempt_index: int
    formal_effect_look_number: int
    pilot_data: bool
    arms: Mapping[str, ArmOutcome]
    probe_executed: bool
    probe_step: int | None
    no_probe_reason: str | None
    planning_random_stream_key: str
    score_random_stream_key: str
    safety_comparisons: tuple[SafetyComparison, ...]
    candidate_evaluation_count: int
    safety_rejection_count: int
    non_positive_score_count: int
    window_expired_count: int

    @property
    def delta_response(self) -> float:
        return self.arms["A2-use"].raw_return - self.arms["A2-mask"].raw_return

    @property
    def delta_cost(self) -> float:
        return self.arms["A1"].raw_return - self.arms["A2-mask"].raw_return

    @property
    def delta_net(self) -> float:
        return self.arms["A2-use"].raw_return - self.arms["A1"].raw_return

    def validate_identity(self) -> None:
        if not math.isclose(
            self.delta_net,
            self.delta_response - self.delta_cost,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Delta_net must equal Delta_response minus Delta_cost.")
        differences = (self.delta_net, self.delta_response, self.delta_cost)
        if not self.probe_executed and any(value != 0.0 for value in differences):
            raise ValueError("A no-fire block must recompute every paired difference as zero.")


def _validate_probe_decision(
    payload: Mapping[str, Any],
    *,
    preregistration: R015Preregistration,
    probe_executed: bool,
    no_probe_reason: str | None,
) -> tuple[str | None, int | None, int, str, str]:
    """Verify the first-positive, highest-score, one-probe decision rule."""

    consultations = _require_sequence(
        payload.get("consultations"), field="probe_decision.consultations"
    )
    if not consultations:
        raise ValueError("R015 requires a recorded consultation trace.")
    if payload.get("probe_registry_semantic_sha256") != (
        preregistration.probe_registry_semantic_sha256
    ):
        raise ValueError("Probe scoring used different registered script semantics.")
    derivation_sha = preregistration.artifacts["random_key_derivation"].sha256
    if derivation_sha is None or payload.get(
        "random_key_derivation_sha256"
    ) != derivation_sha:
        raise ValueError("Probe scoring used a different random-key derivation.")
    planning_random_stream_key = str(
        payload.get("planning_random_stream_key", "")
    )
    score_random_stream_key = str(payload.get("score_random_stream_key", ""))
    if (
        not planning_random_stream_key
        or not score_random_stream_key
        or planning_random_stream_key == score_random_stream_key
        or payload.get("planning_random_purpose") != "value_planning"
        or payload.get("score_random_purpose") != "score_estimation"
    ):
        raise ValueError("Planning and score estimation require separate named streams.")
    registered_ids = set(preregistration.probe_scripts)
    previous_step = 0
    selected_events: list[tuple[str, int]] = []
    any_eligible_safe = False
    candidate_evaluation_count = 0
    for consultation_index, raw_consultation in enumerate(consultations):
        consultation = _require_mapping(
            raw_consultation,
            field=f"probe_decision.consultations.{consultation_index}",
        )
        step = _positive_int(
            consultation.get("environment_step"),
            field="probe_decision.environment_step",
        )
        if step > 100 or (step - 1) % 5 != 0 or step <= previous_step:
            raise ValueError(
                "Consultations must be strictly ordered five-step points from step 1 through 100."
            )
        previous_step = step
        v_base = _finite_float(
            consultation.get("v_base"), field="probe_decision.v_base"
        )
        raw_candidates = _require_sequence(
            consultation.get("candidates"),
            field="probe_decision.candidates",
        )
        candidate_rows: dict[str, dict[str, Any]] = {}
        for raw_candidate in raw_candidates:
            candidate = _require_mapping(
                raw_candidate, field="probe_decision candidate"
            )
            probe_id = str(candidate.get("probe_id", ""))
            if probe_id not in registered_ids or probe_id in candidate_rows:
                raise ValueError("A decision candidate is absent from the probe registry.")
            script = preregistration.probe_scripts[probe_id]
            if candidate.get("registered") is not True or int(
                candidate.get("script_length", -1)
            ) != len(script.primitive_actions):
                raise ValueError("A decision candidate changed its registered script.")
            if not isinstance(candidate.get("eligible"), bool) or not isinstance(
                candidate.get("static_safety_pass"), bool
            ):
                raise ValueError("Candidate eligibility and static safety must be boolean.")
            values = {
                name: _finite_float(
                    candidate.get(name), field=f"probe_decision.{probe_id}.{name}"
                )
                for name in (
                    "j_use",
                    "j_mask",
                    "v_mask",
                    "i_response",
                    "c_task",
                    "s_seq",
                )
            }
            if not math.isclose(
                values["i_response"],
                values["j_use"] - values["j_mask"],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                values["c_task"],
                values["v_mask"] - values["j_mask"],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                values["s_seq"],
                values["j_use"] - values["v_mask"],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Probe score decomposition is internally inconsistent.")
            candidate_rows[probe_id] = {
                **values,
                "eligible": candidate["eligible"],
                "static_safety_pass": candidate["static_safety_pass"],
            }
        if set(candidate_rows) != registered_ids:
            raise ValueError("Every consultation must score the full probe registry.")
        candidate_evaluation_count += len(candidate_rows)
        eligible_rows = {
            probe_id: row
            for probe_id, row in candidate_rows.items()
            if row["eligible"] and row["static_safety_pass"]
        }
        any_eligible_safe = any_eligible_safe or bool(eligible_rows)
        expected_v_mask = max(
            [v_base] + [row["j_mask"] for row in eligible_rows.values()]
        )
        if any(
            not math.isclose(
                row["v_mask"], expected_v_mask, rel_tol=1.0e-12, abs_tol=1.0e-12
            )
            for row in candidate_rows.values()
        ):
            raise ValueError("V_mask must include the base and every safe masked candidate.")
        positive_rows = {
            probe_id: row for probe_id, row in eligible_rows.items() if row["s_seq"] > 0.0
        }
        selected_id = consultation.get("selected_for_safety_probe_id")
        if positive_rows:
            if selected_events or not isinstance(selected_id, str):
                raise ValueError("The first positive consultation must select one probe.")
            maximum = max(row["s_seq"] for row in eligible_rows.values())
            if selected_id not in positive_rows or not math.isclose(
                positive_rows[selected_id]["s_seq"],
                maximum,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("The selected probe must have the highest positive S_seq.")
            selected_events.append((selected_id, step))
            if consultation_index != len(consultations) - 1:
                raise ValueError("Consultation must stop after the first positive selection.")
        elif selected_id is not None:
            raise ValueError("A non-positive consultation cannot select a probe.")

    selected_id = selected_events[0][0] if selected_events else None
    selected_step = selected_events[0][1] if selected_events else None
    executed_id = payload.get("executed_probe_id")
    raw_probe_step = payload.get("probe_step")
    probe_step = (
        None
        if raw_probe_step is None
        else _positive_int(raw_probe_step, field="probe_decision.probe_step")
    )
    before = _nonnegative_int(
        payload.get("probe_budget_used_before"),
        field="probe_decision.probe_budget_used_before",
    )
    after = _nonnegative_int(
        payload.get("probe_budget_used_after"),
        field="probe_decision.probe_budget_used_after",
    )
    if before != 0:
        raise ValueError("R015 permits no earlier probe in the episode.")
    if probe_executed:
        if executed_id != selected_id or probe_step != selected_step or after != 1:
            raise ValueError("Executed probe must be the first selected positive probe.")
    else:
        if executed_id is not None or probe_step is not None or after != 0:
            raise ValueError("A no-probe block cannot record an executed script.")
        if no_probe_reason in {"support_incompatible", "safety_rejected"}:
            if selected_id is None:
                raise ValueError("A rejected probe requires a selected positive candidate.")
        elif selected_id is not None:
            raise ValueError("This no-probe reason cannot follow a positive selection.")
        if no_probe_reason == "non_positive_score" and not any_eligible_safe:
            raise ValueError("A non-positive result requires at least one eligible probe.")
        if no_probe_reason == "no_safe_candidate" and any_eligible_safe:
            raise ValueError("A no-safe-candidate result cannot contain a safe candidate.")
        if no_probe_reason == "window_expired" and previous_step < 96:
            raise ValueError("Window expiry requires recording the final consultation point.")
    return (
        selected_id,
        selected_step,
        candidate_evaluation_count,
        planning_random_stream_key,
        score_random_stream_key,
    )


def _validate_trace_manifest(
    trace: Mapping[str, Any],
    *,
    trace_sha256: Any,
    preregistration: R015Preregistration,
    support_member: Mapping[str, Any],
    arms: Mapping[str, ArmOutcome],
    audit_unit_id: str,
    prototype_id: str,
    family_id: str,
    training_seed: int,
    training_run_id: str,
    ego_position: int,
    initial_state_sha256: str,
    episode_seed: int,
    mechanical_attempt_index: int,
    episode_random_key: str,
    probe_executed: bool,
    selected_probe_id: str | None,
    selected_probe_step: int | None,
    expected_masked_values: Mapping[str, Any],
    trace_replay_verifier: TraceReplayVerifier,
) -> Mapping[str, Mapping[str, Any]]:
    """Recompute pairing evidence from the embedded content-addressed trace."""

    if trace.get("schema_version") != "path_c_r015_trace_manifest_v1":
        raise ValueError("The paired-block trace manifest has the wrong schema.")
    if trace_sha256 != _canonical_mapping_sha256(trace):
        raise ValueError("The paired-block trace manifest digest is invalid.")
    expected_identity = {
        "audit_unit_id": audit_unit_id,
        "prototype_id": prototype_id,
        "family_id": family_id,
        "training_seed": training_seed,
        "training_run_id": training_run_id,
        "ego_position": ego_position,
        "initial_state_sha256": initial_state_sha256,
        "episode_seed": episode_seed,
        "mechanical_attempt_index": mechanical_attempt_index,
        "episode_random_key": episode_random_key,
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
    }
    if any(trace.get(field) != value for field, value in expected_identity.items()):
        raise ValueError("The trace manifest changed the paired-block identity.")
    raw_groups = _require_mapping(trace.get("groups"), field="trace.groups")
    if set(raw_groups) != set(R015_FORMAL_GROUPS):
        raise ValueError("The trace manifest must contain all three formal groups.")
    groups = {
        group: _require_mapping(raw_groups[group], field=f"trace.groups.{group}")
        for group in R015_FORMAL_GROUPS
    }
    allowed_fields = set(preregistration.information.allowed_controller_fields)
    forbidden_fields = set(preregistration.information.forbidden_controller_fields)
    primitive_actions = {"up", "down", "right", "left", "stay", "interact"}
    for group, record in groups.items():
        raw_steps = _require_sequence(
            record.get("environment_steps"),
            field=f"trace.groups.{group}.environment_steps",
        )
        if len(raw_steps) != 400:
            raise ValueError("Every formal trace must contain exactly 400 environment steps.")
        steps = [
            _require_mapping(item, field=f"trace.groups.{group}.environment_step")
            for item in raw_steps
        ]
        step_fields = {
            "environment_step",
            "ego_action",
            "partner_action",
            "controller_input",
            "raw_team_reward",
            "done",
            "environment_random_key",
            "official_local_observation_after",
            "controller_recurrent_state_before_sha256",
            "controller_recurrent_state_after_sha256",
            "partner_recurrent_state_before_sha256",
            "partner_recurrent_state_after_sha256",
            "belief_update_mode",
            "belief_before_sha256",
            "belief_after_sha256",
        }
        for index, step in enumerate(steps):
            if set(step) != step_fields or _nonnegative_int(
                step.get("environment_step"), field="trace.environment_step"
            ) != index:
                raise ValueError("A formal trace step has the wrong fields or index.")
            if step.get("ego_action") not in primitive_actions or step.get(
                "partner_action"
            ) not in primitive_actions:
                raise ValueError("A formal trace contains an action outside the environment.")
            if not isinstance(step.get("done"), bool) or step.get("done") is not (
                index == 399
            ):
                raise ValueError("A formal trace has the wrong 400-step boundary.")
            if not str(step.get("environment_random_key", "")):
                raise ValueError("A formal trace step lacks its environment random key.")
            if not _is_sha256(
                step.get("controller_recurrent_state_before_sha256")
            ) or not _is_sha256(
                step.get("controller_recurrent_state_after_sha256")
            ):
                raise ValueError("A formal trace step lacks frozen controller-state writes.")
            if index > 0 and step.get(
                "controller_recurrent_state_before_sha256"
            ) != steps[index - 1].get("controller_recurrent_state_after_sha256"):
                raise ValueError("The controller recurrent-state writes are not contiguous.")
            if not _is_sha256(
                step.get("partner_recurrent_state_before_sha256")
            ) or not _is_sha256(
                step.get("partner_recurrent_state_after_sha256")
            ):
                raise ValueError("A formal trace step lacks partner-state writes.")
            if index > 0 and step.get(
                "partner_recurrent_state_before_sha256"
            ) != steps[index - 1].get("partner_recurrent_state_after_sha256"):
                raise ValueError("The partner recurrent-state writes are not contiguous.")
            if step.get("belief_update_mode") != "online_each_environment_step":
                raise ValueError(
                    "A formal executed group must update its belief at every environment step."
                )
            if not _is_sha256(step.get("belief_before_sha256")) or not _is_sha256(
                step.get("belief_after_sha256")
            ):
                raise ValueError("A formal trace step lacks its belief-state binding.")
            if index > 0 and step.get("belief_before_sha256") != steps[index - 1].get(
                "belief_after_sha256"
            ):
                response_update_index = (
                    selected_probe_step
                    + len(
                        preregistration.probe_scripts[
                            str(selected_probe_id)
                        ].primitive_actions
                    )
                    if probe_executed
                    and selected_probe_step is not None
                    and selected_probe_id is not None
                    else None
                )
                if group != "A2-use" or index != response_update_index:
                    raise ValueError("The online belief updates are not contiguous.")
            _finite_float(
                step.get("raw_team_reward"), field="trace.raw_team_reward"
            )
            controller_input = _require_mapping(
                step.get("controller_input"), field="trace.controller_input"
            )
            if index > 0 and controller_input.get(
                "official_local_observation"
            ) != steps[index - 1].get("official_local_observation_after"):
                raise ValueError("The official-observation trace is not contiguous.")
        official_events = _require_sequence(
            record.get("official_history_events"),
            field=f"trace.groups.{group}.official_history_events",
        )
        computed_official_events = [step["controller_input"] for step in steps]
        if list(official_events) != computed_official_events:
            raise ValueError("Official-history events do not match the 400-step trace.")
        for event in official_events:
            event_mapping = _require_mapping(
                event, field=f"trace.groups.{group}.official_history_event"
            )
            if not event_mapping or not set(event_mapping).issubset(allowed_fields):
                raise ValueError("A trace event contains a non-official controller field.")
            if set(event_mapping) & forbidden_fields:
                raise ValueError("A trace event contains evaluator-only information.")
        forbidden_reads = _require_sequence(
            record.get("forbidden_read_events"),
            field=f"trace.groups.{group}.forbidden_read_events",
        )
        if forbidden_reads:
            raise ValueError("The trace records a forbidden evaluator-field read.")
        reward_components = _require_sequence(
            record.get("raw_reward_components"),
            field=f"trace.groups.{group}.raw_reward_components",
        )
        computed_rewards = [step["raw_team_reward"] for step in steps]
        if list(reward_components) != computed_rewards:
            raise ValueError("Raw reward components do not match the 400-step trace.")
        replay = _require_mapping(
            record.get("replay_verification"),
            field=f"trace.groups.{group}.replay_verification",
        )
        expected_replay = {
            "trace_replay_verifier_sha256": preregistration.artifacts[
                "trace_replay_verifier"
            ].sha256,
            "environment_config_sha256": preregistration.artifacts[
                "environment_config"
            ].sha256,
            "environment_source_sha256": preregistration.artifacts[
                "environment_source"
            ].sha256,
            "ego_checkpoint_sha256": preregistration.artifacts[
                "ego_checkpoint"
            ].sha256,
            "continuation_planner_sha256": preregistration.artifacts[
                "continuation_planner"
            ].sha256,
            "official_history_filter_sha256": preregistration.artifacts[
                "official_history_filter"
            ].sha256,
            "response_projection_sha256": preregistration.artifacts[
                "response_projection"
            ].sha256,
            "step_count": 400,
            "actions_legal": True,
            "controller_actions_recomputed": True,
            "recurrent_state_writes_recomputed": True,
            "observation_chain_matches": True,
            "raw_rewards_match": True,
            "done_boundary_matches": True,
            "passed": True,
        }
        if replay != expected_replay:
            raise ValueError("The trace lacks a successful frozen environment replay.")
        traced_return = math.fsum(
            _finite_float(value, field=f"trace.groups.{group}.raw_reward_component")
            for value in reward_components
        )
        if not math.isclose(
            traced_return,
            arms[group].raw_return,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("A raw return does not equal its trace components.")

    masked_reference = _require_mapping(
        groups["A1"].get("masked_value_reference"),
        field="trace.groups.A1.masked_value_reference",
    )
    candidate_values = _require_mapping(
        masked_reference.get("candidate_j_mask"),
        field="trace.groups.A1.masked_value_reference.candidate_j_mask",
    )
    if set(candidate_values) != set(preregistration.probe_scripts) or any(
        not math.isclose(
            _finite_float(value, field=f"masked_value_reference.{probe_id}"),
            _finite_float(
                expected_masked_values["candidate_j_mask"][probe_id],
                field=f"expected_masked_values.{probe_id}",
            ),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for probe_id, value in candidate_values.items()
    ):
        raise ValueError("A1 masked candidate values differ from the score trace.")
    v_base = _finite_float(masked_reference.get("v_base"), field="A1.v_base")
    v_mask = _finite_float(masked_reference.get("v_mask"), field="A1.v_mask")
    selected_value = _finite_float(
        masked_reference.get("selected_value"), field="A1.selected_value"
    )
    selected_action_id = str(masked_reference.get("selected_action_id", ""))
    selectable_candidate_ids = {
        str(item)
        for item in _require_sequence(
            masked_reference.get("selectable_candidate_ids"),
            field="A1.selectable_candidate_ids",
        )
    }
    if selectable_candidate_ids != set(expected_masked_values["safe_candidate_ids"]):
        raise ValueError("A1 safe candidate set differs from the score trace.")
    allowed_action_ids = {"base"} | selectable_candidate_ids
    if selected_action_id not in allowed_action_ids:
        raise ValueError("A1 selected an action outside the frozen candidate set.")
    expected_v_mask = max(
        [v_base]
        + [float(candidate_values[probe_id]) for probe_id in selectable_candidate_ids]
    )
    selected_expected_value = (
        v_base
        if selected_action_id == "base"
        else _finite_float(
            candidate_values.get(selected_action_id), field="A1.selected_action_id"
        )
    )
    if (
        not math.isclose(
            v_base,
            _finite_float(expected_masked_values["v_base"], field="expected.v_base"),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(v_mask, expected_v_mask, rel_tol=1.0e-12, abs_tol=1.0e-12)
        or not math.isclose(
            v_mask,
            _finite_float(expected_masked_values["v_mask"], field="expected.v_mask"),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            selected_value,
            selected_expected_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            selected_value, v_mask, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    ):
        raise ValueError("A1 did not select a highest-valued masked action.")
    if any(
        groups[group].get("masked_value_reference") != masked_reference
        for group in R015_FORMAL_GROUPS
    ):
        raise ValueError("The formal groups use different masked value references.")

    if probe_executed:
        if selected_probe_id not in preregistration.probe_scripts or (
            selected_probe_step is None
        ):
            raise ValueError("An executed trace lacks its registered probe selection.")
        if not 1 <= selected_probe_step <= 100:
            raise ValueError("The firing step must lie from 1 through 100.")
        if trace.get("probe_step") != selected_probe_step:
            raise ValueError("The trace and controller record different probe steps.")
        pre_traces = [
            _require_sequence(
                groups[group].get("pre_probe_trace"),
                field=f"trace.groups.{group}.pre_probe_trace",
            )
            for group in R015_FORMAL_GROUPS
        ]
        pre_actions = [
            tuple(
                str(item)
                for item in _require_sequence(
                    groups[group].get("pre_probe_actions"),
                    field=f"trace.groups.{group}.pre_probe_actions",
                )
            )
            for group in R015_FORMAL_GROUPS
        ]
        pre_random_keys = [
            tuple(
                str(item)
                for item in _require_sequence(
                    groups[group].get("pre_probe_random_keys"),
                    field=f"trace.groups.{group}.pre_probe_random_keys",
                )
            )
            for group in R015_FORMAL_GROUPS
        ]
        if not pre_traces[0] or any(value != pre_traces[0] for value in pre_traces[1:]):
            raise ValueError("A1 and both A2 groups have different pre-probe histories.")
        if any(value != pre_actions[0] for value in pre_actions[1:]) or any(
            value != pre_random_keys[0] for value in pre_random_keys[1:]
        ):
            raise ValueError("A1 and both A2 groups differ before the probe.")
        # t_p counts completed steps, while embedded trace records are zero-indexed.
        # Records 0 through t_p-1 are therefore the shared pre-branch prefix.
        trajectory_prefixes = [
            list(groups[group]["environment_steps"][:selected_probe_step])
            for group in R015_FORMAL_GROUPS
        ]
        if len({_canonical_value_sha256(prefix) for prefix in trajectory_prefixes}) != 1:
            raise ValueError("The formal trajectories are not byte-identical through t_p-1.")
        for group in R015_FORMAL_GROUPS:
            steps = groups[group]["environment_steps"]
            expected_pre_trace = [
                step["controller_input"] for step in steps[: selected_probe_step + 1]
            ]
            expected_pre_actions = [
                step["ego_action"] for step in steps[:selected_probe_step]
            ]
            expected_pre_keys = [
                step["environment_random_key"]
                for step in steps[:selected_probe_step]
            ]
            if list(groups[group]["pre_probe_trace"]) != expected_pre_trace or list(
                groups[group]["pre_probe_actions"]
            ) != expected_pre_actions or list(
                groups[group]["pre_probe_random_keys"]
            ) != expected_pre_keys:
                raise ValueError("Pre-probe evidence does not match the 400-step trace.")
        expected_probe_actions = preregistration.probe_scripts[
            selected_probe_id
        ].primitive_actions
        for group in ("A2-mask", "A2-use"):
            record = groups[group]
            if record.get("selected_probe_id") != selected_probe_id:
                raise ValueError("The trace executed a different registered probe.")
            probe_actions = tuple(
                str(item)
                for item in _require_sequence(
                    record.get("probe_actions"),
                    field=f"trace.groups.{group}.probe_actions",
                )
            )
            if probe_actions != expected_probe_actions:
                raise ValueError("The trace probe actions differ from the registry.")
            steps = record["environment_steps"]
            probe_end = selected_probe_step + len(expected_probe_actions)
            if list(record["probe_actions"]) != [
                step["ego_action"]
                for step in steps[selected_probe_step:probe_end]
            ] or list(record["probe_random_keys"]) != [
                step["environment_random_key"]
                for step in steps[selected_probe_step:probe_end]
            ] or list(record["future_random_branch_keys"]) != [
                step["environment_random_key"] for step in steps[probe_end:]
            ]:
                raise ValueError("Probe and future branches do not match the full trace.")
            for key_field in ("probe_random_keys", "future_random_branch_keys"):
                keys = tuple(
                    str(item)
                    for item in _require_sequence(
                        record.get(key_field),
                        field=f"trace.groups.{group}.{key_field}",
                    )
                )
                if not keys or len(set(keys)) != len(keys):
                    raise ValueError("The paired trace has invalid random branch keys.")
        paired_fields = (
            "masked_history",
            "current_probe_response",
            "masked_value_reference",
            "belief_common_input",
            "continuation_common_input",
            "recurrent_common_input",
            "probe_random_keys",
            "future_random_branch_keys",
        )
        if any(
            groups["A2-mask"].get(field) != groups["A2-use"].get(field)
            for field in paired_fields
        ):
            raise ValueError("The two A2 traces are not paired on state and randomness.")
        if groups["A1"].get("masked_value_reference") != groups["A2-mask"].get(
            "masked_value_reference"
        ):
            raise ValueError("A1 and A2 use different masked value references.")
        current_response = groups["A2-use"].get("current_probe_response")
        if current_response is None:
            raise ValueError("An executed trace lacks the registered response content.")
        if groups["A2-mask"].get("belief_current_response_extra") is not None or (
            groups["A2-use"].get("belief_current_response_extra") != current_response
        ):
            raise ValueError("The trace violates the use-versus-mask belief interface.")
        for group in ("A2-mask", "A2-use"):
            record = groups[group]
            if record.get("continuation_direct_response") is not None or record.get(
                "recurrent_direct_response"
            ) is not None:
                raise ValueError("The trace routes the response around belief update.")
            aliases = _require_sequence(
                record.get("non_belief_response_aliases"),
                field=f"trace.groups.{group}.non_belief_response_aliases",
            )
            if aliases:
                raise ValueError("The trace retains a non-belief response alias.")
    else:
        if trace.get("probe_step") is not None:
            raise ValueError("A no-probe trace cannot contain a probe step.")
        complete_traces = [
            _require_sequence(
                groups[group].get("complete_trace"),
                field=f"trace.groups.{group}.complete_trace",
            )
            for group in R015_FORMAL_GROUPS
        ]
        actions = [
            _require_sequence(
                groups[group].get("actions"),
                field=f"trace.groups.{group}.actions",
            )
            for group in R015_FORMAL_GROUPS
        ]
        random_branches = [
            _require_sequence(
                groups[group].get("future_random_branch_keys"),
                field=f"trace.groups.{group}.future_random_branch_keys",
            )
            for group in R015_FORMAL_GROUPS
        ]
        if not complete_traces[0] or any(
            value != complete_traces[0] for value in complete_traces[1:]
        ):
            raise ValueError("A no-probe trace differs across formal groups.")
        if any(value != actions[0] for value in actions[1:]) or any(
            value != random_branches[0] for value in random_branches[1:]
        ):
            raise ValueError("A no-probe trace changed actions or random branches.")
        complete_trajectories = [
            list(groups[group]["environment_steps"]) for group in R015_FORMAL_GROUPS
        ]
        if len(
            {_canonical_value_sha256(trajectory) for trajectory in complete_trajectories}
        ) != 1:
            raise ValueError("A no-fire block must have byte-identical full trajectories.")
        for group in R015_FORMAL_GROUPS:
            steps = groups[group]["environment_steps"]
            if list(groups[group]["complete_trace"]) != [
                step["controller_input"] for step in steps
            ] or list(groups[group]["actions"]) != [
                step["ego_action"] for step in steps
            ] or list(groups[group]["future_random_branch_keys"]) != [
                step["environment_random_key"] for step in steps
            ]:
                raise ValueError("No-probe evidence does not match the 400-step trace.")
    partner_candidate = next(
        candidate
        for candidate in preregistration.support_spec.candidates
        if candidate.candidate_id == prototype_id
    )
    support_report_sha256 = (
        None
        if preregistration.support_report_path is None
        else _file_sha256(preregistration.support_report_path)
    )
    replay_context = {
        "environment_config_path": str(
            preregistration.artifacts["environment_config"].path
        ),
        "environment_config_sha256": preregistration.artifacts[
            "environment_config"
        ].sha256,
        "environment_source_path": str(
            preregistration.artifacts["environment_source"].path
        ),
        "environment_source_sha256": preregistration.artifacts[
            "environment_source"
        ].sha256,
        "ego_checkpoint_path": str(preregistration.artifacts["ego_checkpoint"].path),
        "ego_checkpoint_sha256": preregistration.artifacts[
            "ego_checkpoint"
        ].sha256,
        "continuation_planner_path": str(
            preregistration.artifacts["continuation_planner"].path
        ),
        "continuation_planner_sha256": preregistration.artifacts[
            "continuation_planner"
        ].sha256,
        "official_history_filter_path": str(
            preregistration.artifacts["official_history_filter"].path
        ),
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "response_projection_path": str(
            preregistration.artifacts["response_projection"].path
        ),
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "ego_evidence_contract_path": str(
            preregistration.artifacts["ego_evidence_contract"].path
        ),
        "ego_evidence_contract_sha256": preregistration.artifacts[
            "ego_evidence_contract"
        ].sha256,
        "response_vocabulary_path": str(
            preregistration.artifacts["response_vocabulary"].path
        ),
        "response_vocabulary_sha256": preregistration.artifacts[
            "response_vocabulary"
        ].sha256,
        "probe_registry_path": str(preregistration.artifacts["probe_registry"].path),
        "probe_registry_sha256": preregistration.artifacts[
            "probe_registry"
        ].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_path": str(
            preregistration.artifacts["random_key_derivation"].path
        ),
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "partner_checkpoint_path": str(partner_candidate.checkpoint_path),
        "partner_checkpoint_sha256": support_member.get("checkpoint_sha256"),
        "partner_model_weights_sha256": support_member.get(
            "model_weights_sha256"
        ),
        "partner_training_config_path": support_member.get("training_config_path"),
        "partner_training_config_sha256": support_member.get(
            "training_config_sha256"
        ),
        "partner_training_manifest_path": support_member.get(
            "training_manifest_path"
        ),
        "partner_training_manifest_sha256": support_member.get(
            "training_manifest_sha256"
        ),
        "partner_family_spec_sha256": support_member.get("family_spec_sha256"),
        "partner_architecture": support_member.get("architecture"),
        "partner_architecture_sha256": support_member.get("architecture_sha256"),
        "partner_action_rule": support_member.get("action_rule"),
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": support_report_sha256,
        "audit_unit_id": audit_unit_id,
    }
    recomputed_replay = _require_mapping(
        trace_replay_verifier(trace, replay_context),
        field="trace replay verifier result",
    )
    group_ego_actions = {
        group: [step["ego_action"] for step in groups[group]["environment_steps"]]
        for group in R015_FORMAL_GROUPS
    }
    group_partner_actions = {
        group: [step["partner_action"] for step in groups[group]["environment_steps"]]
        for group in R015_FORMAL_GROUPS
    }
    group_state_writes = {
        group: [
            {
                "environment_step": step["environment_step"],
                "before_sha256": step[
                    "controller_recurrent_state_before_sha256"
                ],
                "after_sha256": step[
                    "controller_recurrent_state_after_sha256"
                ],
            }
            for step in groups[group]["environment_steps"]
        ]
        for group in R015_FORMAL_GROUPS
    }
    group_partner_state_writes = {
        group: [
            {
                "environment_step": step["environment_step"],
                "before_sha256": step["partner_recurrent_state_before_sha256"],
                "after_sha256": step["partner_recurrent_state_after_sha256"],
            }
            for step in groups[group]["environment_steps"]
        ]
        for group in R015_FORMAL_GROUPS
    }
    group_action_state_summaries_sha256 = {
        group: _canonical_value_sha256(
            [
                {
                    "environment_step": step["environment_step"],
                    "ego_action": step["ego_action"],
                    "before_sha256": step[
                        "controller_recurrent_state_before_sha256"
                    ],
                    "after_sha256": step[
                        "controller_recurrent_state_after_sha256"
                    ],
                }
                for step in groups[group]["environment_steps"]
            ]
        )
        for group in R015_FORMAL_GROUPS
    }
    group_partner_action_state_summaries_sha256 = {
        group: _canonical_value_sha256(
            [
                {
                    "environment_step": step["environment_step"],
                    "partner_action": step["partner_action"],
                    "before_sha256": step[
                        "partner_recurrent_state_before_sha256"
                    ],
                    "after_sha256": step[
                        "partner_recurrent_state_after_sha256"
                    ],
                }
                for step in groups[group]["environment_steps"]
            ]
        )
        for group in R015_FORMAL_GROUPS
    }
    a1_selected_action_id = str(
        _require_mapping(
            groups["A1"].get("masked_value_reference"),
            field="trace.groups.A1.masked_value_reference",
        ).get("selected_action_id", "")
    )
    a1_selected_script = (
        list(preregistration.probe_scripts[a1_selected_action_id].primitive_actions)
        if a1_selected_action_id in preregistration.probe_scripts
        else []
    )
    selected_probe_actions = (
        list(preregistration.probe_scripts[selected_probe_id].primitive_actions)
        if probe_executed and selected_probe_id is not None
        else []
    )
    group_controller_route_summaries = {
        "A1": {
            "belief_route": "B_mask(q,x)",
            "continuation_route": "C(q,x)",
            "selected_action_id": a1_selected_action_id,
            "selected_registered_script_actions": a1_selected_script,
            "selected_action_matches_replay": True,
        },
        "A2-mask": {
            "belief_route": "B_mask(q,x)",
            "continuation_route": "C(q,x)",
            "selected_probe_actions": selected_probe_actions,
            "current_response_used": False,
            "pre_probe_actions_match": True,
            "post_probe_actions_match_replay": True,
        },
        "A2-use": {
            "belief_route": "B_use(q,x,y)" if probe_executed else "B_mask(q,x)",
            "continuation_route": "C(q,x)",
            "selected_probe_actions": selected_probe_actions,
            "current_response_used": probe_executed,
            "pre_probe_actions_match": True,
            "post_probe_actions_match_replay": True,
        },
    }
    expected_replay_result = {
        "schema_version": "path_c_r015_trace_replay_result_v2",
        "verified": True,
        "groups_sha256": _canonical_mapping_sha256(raw_groups),
        "group_step_counts": {group: 400 for group in R015_FORMAL_GROUPS},
        "group_environment_steps_sha256": {
            group: _canonical_value_sha256(groups[group]["environment_steps"])
            for group in R015_FORMAL_GROUPS
        },
        "environment_config_sha256": preregistration.artifacts[
            "environment_config"
        ].sha256,
        "environment_source_sha256": preregistration.artifacts[
            "environment_source"
        ].sha256,
        "ego_checkpoint_sha256": preregistration.artifacts[
            "ego_checkpoint"
        ].sha256,
        "continuation_planner_sha256": preregistration.artifacts[
            "continuation_planner"
        ].sha256,
        "official_history_filter_sha256": preregistration.artifacts[
            "official_history_filter"
        ].sha256,
        "response_projection_sha256": preregistration.artifacts[
            "response_projection"
        ].sha256,
        "ego_evidence_contract_sha256": preregistration.artifacts[
            "ego_evidence_contract"
        ].sha256,
        "response_vocabulary_sha256": preregistration.artifacts[
            "response_vocabulary"
        ].sha256,
        "probe_registry_sha256": preregistration.artifacts[
            "probe_registry"
        ].sha256,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": preregistration.artifacts[
            "random_key_derivation"
        ].sha256,
        "partner_checkpoint_sha256": support_member.get("checkpoint_sha256"),
        "partner_model_weights_sha256": support_member.get(
            "model_weights_sha256"
        ),
        "partner_training_config_sha256": support_member.get(
            "training_config_sha256"
        ),
        "partner_training_manifest_sha256": support_member.get(
            "training_manifest_sha256"
        ),
        "partner_family_spec_sha256": support_member.get("family_spec_sha256"),
        "partner_architecture_sha256": support_member.get("architecture_sha256"),
        "partner_action_rule": support_member.get("action_rule"),
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": support_report_sha256,
        "group_ego_actions": group_ego_actions,
        "group_partner_actions": group_partner_actions,
        "group_recurrent_state_writes_sha256": {
            group: _canonical_value_sha256(group_state_writes[group])
            for group in R015_FORMAL_GROUPS
        },
        "group_action_state_summaries_sha256": (
            group_action_state_summaries_sha256
        ),
        "group_partner_recurrent_state_writes_sha256": {
            group: _canonical_value_sha256(group_partner_state_writes[group])
            for group in R015_FORMAL_GROUPS
        },
        "group_partner_action_state_summaries_sha256": (
            group_partner_action_state_summaries_sha256
        ),
        "group_controller_route_summaries": group_controller_route_summaries,
        "all_controller_actions_recomputed": True,
        "all_recurrent_state_writes_recomputed": True,
        "all_partner_actions_recomputed": True,
        "all_partner_recurrent_state_writes_recomputed": True,
    }
    reported_replay = _require_mapping(
        trace.get("replay_verification"), field="trace.replay_verification"
    )
    if dict(recomputed_replay) != expected_replay_result or dict(
        reported_replay
    ) != expected_replay_result:
        raise ValueError(
            "The frozen controller action or state replay does not match the trace."
        )
    return groups


def _validate_decision_evidence(
    probe_decision: Mapping[str, Any],
    *,
    trace_groups: Mapping[str, Mapping[str, Any]],
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
    audit_unit_id: str,
    prototype_id: str,
    episode_seed: int,
    decision_evidence_verifier: DecisionEvidenceVerifier,
) -> None:
    """Recompute every consultation from frozen planner evidence and history."""

    raw_consultations = _require_sequence(
        probe_decision.get("consultations"), field="probe_decision.consultations"
    )
    a1_history = _require_sequence(
        trace_groups["A1"].get("official_history_events"),
        field="trace.groups.A1.official_history_events",
    )
    verifier_consultations: list[Mapping[str, Any]] = []
    recorded_consultations: list[Mapping[str, Any]] = []
    for consultation_index, raw_consultation in enumerate(raw_consultations):
        consultation = _require_mapping(
            raw_consultation,
            field=f"probe_decision.consultations.{consultation_index}",
        )
        step = _nonnegative_int(
            consultation.get("environment_step"),
            field="probe_decision.environment_step",
        )
        evidence = _require_mapping(
            consultation.get("planner_evidence"),
            field="probe_decision.planner_evidence",
        )
        if set(evidence) != {
            "schema_version",
            "official_history",
            "planning_random_key",
            "score_random_key",
            "planning_branches",
            "branch_sampling_rule_id",
            "branch_belief_rule_id",
            "cost_accounting",
        } or evidence.get(
            "schema_version"
        ) != "path_c_r015_consultation_planner_evidence_v2":
            raise ValueError("A consultation planner-evidence record has the wrong schema.")
        if evidence.get("branch_sampling_rule_id") != R015_BRANCH_SAMPLING_ID or (
            evidence.get("branch_belief_rule_id") != R015_BRANCH_BELIEF_ID
        ):
            raise ValueError("A consultation changed its registered planning rules.")
        official_history = _require_sequence(
            evidence.get("official_history"),
            field="probe_decision.planner_evidence.official_history",
        )
        if list(official_history) != list(a1_history[: step + 1]):
            raise ValueError("Consultation evidence differs from the replayed official history.")
        expected_planning_key = _derive_r015_random_key(
            audit_unit_id=audit_unit_id,
            partner_prototype_id=prototype_id,
            episode_seed=episode_seed,
            purpose="value_planning",
            environment_step=step,
            branch_index=0,
        )
        expected_score_key = _derive_r015_random_key(
            audit_unit_id=audit_unit_id,
            partner_prototype_id=prototype_id,
            episode_seed=episode_seed,
            purpose="score_estimation",
            environment_step=step,
            branch_index=0,
        )
        if evidence.get("planning_random_key") != expected_planning_key or evidence.get(
            "score_random_key"
        ) != expected_score_key:
            raise ValueError("Consultation evidence uses an unregistered random key.")
        planning_branches = _require_sequence(
            evidence.get("planning_branches"),
            field="probe_decision.planner_evidence.planning_branches",
        )
        if not planning_branches or any(
            not isinstance(branch, Mapping) for branch in planning_branches
        ):
            raise ValueError("A consultation lacks replayable planning branches.")
        if len(planning_branches) != (
            preregistration.controller.planning_branches_per_candidate
        ):
            raise ValueError("A consultation has the wrong sampled branch count.")
        seen_canonical_slots: set[int] = set()
        remaining_steps = 400 - step
        for branch_index, branch in enumerate(planning_branches):
            if branch.get("source_belief_filter_algorithm_id") != (
                R015_FILTER_ALGORITHM_ID_V2
            ) or branch.get("source_belief_resampling_algorithm") != (
                R015_RESAMPLING_ALGORITHM_ID_V2
            ):
                raise ValueError(
                    "A formal consultation branch uses a non-v2 filter belief."
                )
            canonical_slot = _nonnegative_int(
                branch.get("canonical_slot_16"),
                field=(
                    "probe_decision.planner_evidence.planning_branches."
                    f"{branch_index}.canonical_slot_16"
                ),
            )
            if canonical_slot >= 16 or canonical_slot in seen_canonical_slots:
                raise ValueError("A consultation has an invalid canonical branch slot.")
            seen_canonical_slots.add(canonical_slot)
            expected_root = _derive_r015_controller_key(
                expected_planning_key,
                R015_BRANCH_SAMPLING_ID,
                "canonical_slot_16",
                canonical_slot,
            )
            expected_random_summary = _future_random_sequence_summary(
                root_key=expected_root,
                sequence_length=remaining_steps,
            )
            recorded_random_summary = {
                field: branch.get(field)
                for field in expected_random_summary
            }
            if recorded_random_summary != expected_random_summary or branch.get(
                "common_random_key"
            ) != expected_root:
                raise ValueError(
                    "A consultation branch changed its compact future-random evidence."
                )
            if "future_random_keys" in json.dumps(
                branch, sort_keys=True, ensure_ascii=True
            ):
                raise ValueError(
                    "Formal planning evidence may not persist stepwise future random keys."
                )
        planning_cost = _require_mapping(
            evidence.get("cost_accounting"),
            field="probe_decision.planner_evidence.cost_accounting",
        )
        if planning_cost.get("schema_version") != "path_c_r015_planning_cost_v2" or (
            planning_cost.get("sample_count")
            != preregistration.controller.planning_branches_per_candidate
        ):
            raise ValueError("A consultation has the wrong planning-cost contract.")
        verifier_consultations.append(
            {
                "environment_step": step,
                "official_history": list(official_history),
                "planning_random_key": expected_planning_key,
                "score_random_key": expected_score_key,
                "planning_branches": list(planning_branches),
                "branch_sampling_rule_id": R015_BRANCH_SAMPLING_ID,
                "branch_belief_rule_id": R015_BRANCH_BELIEF_ID,
                "cost_accounting": dict(planning_cost),
            }
        )
        recorded_consultations.append(
            {
                "environment_step": step,
                "v_base": consultation.get("v_base"),
                "candidates": list(
                    _require_sequence(
                        consultation.get("candidates"),
                        field="probe_decision.candidates",
                    )
                ),
                "selected_for_safety_probe_id": consultation.get(
                    "selected_for_safety_probe_id"
                ),
                "planning_random_key": expected_planning_key,
                "score_random_key": expected_score_key,
            }
        )
    verifier_result = _require_mapping(
        decision_evidence_verifier(
            {"consultations": verifier_consultations},
            {
                "audit_unit_id": audit_unit_id,
                "prototype_id": prototype_id,
                "episode_seed": episode_seed,
                "support_members": support_members,
                "probe_scripts": {
                    probe_id: list(spec.primitive_actions)
                    for probe_id, spec in preregistration.probe_scripts.items()
                },
                "continuation_planner_path": str(
                    preregistration.artifacts["continuation_planner"].path
                ),
                "continuation_planner_sha256": preregistration.artifacts[
                    "continuation_planner"
                ].sha256,
                "official_history_filter_sha256": preregistration.artifacts[
                    "official_history_filter"
                ].sha256,
                "response_projection_sha256": preregistration.artifacts[
                    "response_projection"
                ].sha256,
                "random_key_derivation_sha256": preregistration.artifacts[
                    "random_key_derivation"
                ].sha256,
            },
        ),
        field="decision-evidence verifier result",
    )
    expected_result = {
        "schema_version": "path_c_r015_decision_evidence_verification_v1",
        "verified": True,
        "consultations": recorded_consultations,
        "full_probe_registry_ids": sorted(preregistration.probe_scripts),
        "first_positive_stop_valid": True,
    }
    if verifier_result != expected_result:
        raise ValueError(
            "Frozen decision verification differs from the recorded candidates or stop."
        )


def validate_r015_paired_block(
    payload: Mapping[str, Any],
    *,
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
    trace_replay_verifier: TraceReplayVerifier,
    safety_branch_evidence_verifier: SafetyBranchEvidenceVerifier,
    decision_evidence_verifier: DecisionEvidenceVerifier,
    expected_runtime_binding_sha256: str | None = None,
) -> R015PairedBlock:
    """Validate strict A1/A2-mask/A2-use pairing and current-response routing."""

    if payload.get("schema_version") != R015_PAIRED_BLOCK_SCHEMA:
        raise ValueError("Unsupported R015 paired-block schema.")
    if payload.get("phase") != "formal":
        raise ValueError("R015 paired-block adjudication accepts formal data only.")
    if payload.get("pilot_data") is not False:
        raise ValueError("A formal paired block must explicitly exclude pilot data.")
    recorded_runtime_binding = payload.get("runtime_binding_sha256")
    if recorded_runtime_binding is not None and not _is_sha256(
        recorded_runtime_binding
    ):
        raise ValueError("A formal paired block has an invalid runtime binding.")
    if expected_runtime_binding_sha256 is not None and (
        not _is_sha256(expected_runtime_binding_sha256)
        or recorded_runtime_binding != expected_runtime_binding_sha256
    ):
        raise ValueError("A formal paired block changed its frozen runtime binding.")
    round_index = _positive_int(payload.get("round_index"), field="round_index")
    if round_index > preregistration.statistics.n_rounds_max:
        raise ValueError("A paired block exceeds N_rounds_max.")
    audit_unit_id = str(payload.get("audit_unit_id", ""))
    if not audit_unit_id:
        raise ValueError("Paired block audit-unit identifier is empty.")
    prototype_id = str(payload.get("prototype_id", ""))
    if prototype_id not in support_members:
        raise ValueError("Paired block uses a prototype outside the frozen support.")
    member = support_members[prototype_id]
    family_id = str(payload.get("family_id", ""))
    training_seed = _nonnegative_int(
        payload.get("training_seed"), field="training_seed"
    )
    training_run_id = str(payload.get("training_run_id", ""))
    ego_position = _nonnegative_int(payload.get("ego_position"), field="ego_position")
    if ego_position != 1:
        raise ValueError("R015 formal runtime fixes the ego cook at agent_1.")
    initial_state_sha256 = str(payload.get("initial_state_sha256", ""))
    episode_seed = _nonnegative_int(payload.get("episode_seed"), field="episode_seed")
    mechanical_attempt_index = _nonnegative_int(
        payload.get("mechanical_attempt_index"),
        field="mechanical_attempt_index",
    )
    if mechanical_attempt_index >= R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT:
        raise ValueError("Paired block replacement attempt exceeds the frozen schedule.")
    if (
        family_id != member.get("family_id")
        or training_seed != int(member.get("training_seed", -2))
        or training_run_id != member.get("training_run_id")
    ):
        raise ValueError("Paired block changed its prototype family or training run.")
    raw_arms = _require_mapping(payload.get("arms"), field="arms")
    if set(raw_arms) != set(R015_FORMAL_GROUPS):
        raise ValueError("A formal block must contain exactly A1, A2-mask, and A2-use.")
    arms = {
        group: ArmOutcome.from_mapping(
            group, _require_mapping(raw_arms[group], field=f"arms.{group}")
        )
        for group in R015_FORMAL_GROUPS
    }
    shared_identity = {
        (
            outcome.prototype_id,
            outcome.family_id,
            outcome.training_seed,
            outcome.training_run_id,
            outcome.ego_position,
            outcome.initial_state_sha256,
            outcome.episode_seed,
        )
        for outcome in arms.values()
    }
    if shared_identity != {
        (
            prototype_id,
            family_id,
            training_seed,
            training_run_id,
            ego_position,
            initial_state_sha256,
            episode_seed,
        )
    }:
        raise ValueError(
            "All three formal groups must share prototype, position, initial state, and seed."
        )
    random_keys = {outcome.episode_random_key for outcome in arms.values()}
    if len(random_keys) != 1:
        raise ValueError("All three formal groups must share the episode random key.")
    expected_episode_random_key = _derive_r015_random_key(
        audit_unit_id=audit_unit_id,
        partner_prototype_id=prototype_id,
        episode_seed=episode_seed,
        purpose="actual_episode",
        environment_step=0,
        branch_index=0,
    )
    if random_keys != {expected_episode_random_key}:
        raise ValueError(
            "The actual-episode random key does not match the frozen derivation."
        )
    raw_firing_indicator = payload.get("firing_indicator")
    if not isinstance(raw_firing_indicator, bool) or {
        outcome.firing_indicator for outcome in arms.values()
    } != {raw_firing_indicator}:
        raise ValueError("The firing indicator must be one shared block-level property.")
    evidence_binding = preregistration.artifacts["ego_evidence_contract"]
    if evidence_binding.sha256 is None:
        raise ValueError("The official-history controller contract is still pending.")
    if {
        outcome.controller_contract_sha256 for outcome in arms.values()
    } != {evidence_binding.sha256}:
        raise ValueError("A formal group used a different controller input contract.")
    runtime_arm_bindings = {
        "ego_checkpoint_sha256": "ego_checkpoint",
        "environment_config_sha256": "environment_config",
        "environment_source_sha256": "environment_source",
        "official_history_filter_sha256": "official_history_filter",
    }
    for field, artifact_name in runtime_arm_bindings.items():
        frozen_sha = preregistration.artifacts[artifact_name].sha256
        if frozen_sha is None or {
            getattr(outcome, field) for outcome in arms.values()
        } != {frozen_sha}:
            raise ValueError(f"A formal group used a different {artifact_name}.")
    lower = preregistration.statistics.return_lower_bound
    upper = preregistration.statistics.return_upper_bound
    if lower is None or upper is None:
        raise ValueError("Deterministic return bounds are still pending.")
    if any(not lower <= outcome.raw_return <= upper for outcome in arms.values()):
        raise ValueError("A formal return lies outside the deterministic bounds.")

    safety = _require_mapping(payload.get("safety"), field="safety")
    if not isinstance(safety.get("probe_executed"), bool):
        raise ValueError("safety.probe_executed must be boolean.")
    detector_sha = preregistration.artifacts["wrong_delivery_detector"].sha256
    if detector_sha is None or safety.get("wrong_delivery_detector_sha256") != (
        detector_sha
    ):
        raise ValueError("The paired block used a different wrong-delivery detector.")
    if safety.get("recipe_indicator_cost_counted_as_safety_event") is not False:
        raise ValueError("Recipe-indicator cost is not an R015 safety event.")
    probe_executed = safety.get("probe_executed") is True
    if raw_firing_indicator is not probe_executed:
        raise ValueError("The shared firing indicator differs from the probe decision.")
    telemetry: dict[str, int] = {}
    for field in (
        "candidate_evaluation_count",
        "safety_rejection_count",
        "non_positive_score_count",
        "window_expired_count",
    ):
        raw_count = safety.get(field)
        telemetry[field] = _nonnegative_int(raw_count, field=f"safety.{field}")
    if telemetry["safety_rejection_count"] > 1:
        raise ValueError("Each formal episode permits at most one safety decision.")
    no_probe_reason = safety.get("no_probe_reason")
    probe_decision_payload = _require_mapping(
        payload.get("probe_decision"), field="probe_decision"
    )
    probe_decision_sha256 = _canonical_mapping_sha256(probe_decision_payload)
    (
        selected_for_safety_id,
        selected_probe_step,
        scored_candidate_count,
        planning_random_stream_key,
        score_random_stream_key,
    ) = _validate_probe_decision(
        probe_decision_payload,
        preregistration=preregistration,
        probe_executed=probe_executed,
        no_probe_reason=None if no_probe_reason is None else str(no_probe_reason),
    )
    consultation_rows = _require_sequence(
        probe_decision_payload.get("consultations"),
        field="probe_decision.consultations",
    )
    random_key_step = _nonnegative_int(
        _require_mapping(
            consultation_rows[-1], field="probe_decision.final_consultation"
        ).get("environment_step"),
        field="probe_decision.final_consultation.environment_step",
    )
    expected_planning_key = _derive_r015_random_key(
        audit_unit_id=audit_unit_id,
        partner_prototype_id=prototype_id,
        episode_seed=episode_seed,
        purpose="value_planning",
        environment_step=random_key_step,
        branch_index=0,
    )
    expected_score_key = _derive_r015_random_key(
        audit_unit_id=audit_unit_id,
        partner_prototype_id=prototype_id,
        episode_seed=episode_seed,
        purpose="score_estimation",
        environment_step=random_key_step,
        branch_index=0,
    )
    if planning_random_stream_key != expected_planning_key or (
        score_random_stream_key != expected_score_key
    ):
        raise ValueError(
            "Planning or score random key does not match the frozen derivation."
        )
    if telemetry["candidate_evaluation_count"] != scored_candidate_count:
        raise ValueError("Candidate evaluation telemetry does not match the score trace.")
    episode_random_key = next(iter(arms.values())).episode_random_key
    if len(
        {
            episode_random_key,
            planning_random_stream_key,
            score_random_stream_key,
        }
    ) != 3:
        raise ValueError("Actual, planning, and score streams must be independent.")
    trace_manifest = _require_mapping(
        payload.get("trace_manifest"), field="trace_manifest"
    )
    decision_consultations = _require_sequence(
        probe_decision_payload.get("consultations"),
        field="probe_decision.consultations",
    )
    reference_consultation = _require_mapping(
        decision_consultations[-1], field="probe_decision.reference_consultation"
    )
    reference_candidates = _require_sequence(
        reference_consultation.get("candidates"),
        field="probe_decision.reference_candidates",
    )
    expected_masked_values = {
        "v_base": reference_consultation.get("v_base"),
        "v_mask": _require_mapping(
            reference_candidates[0], field="probe_decision.reference_candidate"
        ).get("v_mask"),
        "candidate_j_mask": {
            str(
                _require_mapping(candidate, field="probe_decision.reference_candidate").get(
                    "probe_id"
                )
            ): _require_mapping(
                candidate, field="probe_decision.reference_candidate"
            ).get("j_mask")
            for candidate in reference_candidates
        },
        "safe_candidate_ids": sorted(
            str(
                _require_mapping(candidate, field="probe_decision.reference_candidate").get(
                    "probe_id"
                )
            )
            for candidate in reference_candidates
            if _require_mapping(
                candidate, field="probe_decision.reference_candidate"
            ).get("eligible")
            is True
            and _require_mapping(
                candidate, field="probe_decision.reference_candidate"
            ).get("static_safety_pass")
            is True
        ),
    }
    trace_groups = _validate_trace_manifest(
        trace_manifest,
        trace_sha256=payload.get("trace_manifest_sha256"),
        preregistration=preregistration,
        support_member=member,
        arms=arms,
        audit_unit_id=audit_unit_id,
        prototype_id=prototype_id,
        family_id=family_id,
        training_seed=training_seed,
        training_run_id=training_run_id,
        ego_position=ego_position,
        initial_state_sha256=initial_state_sha256,
        episode_seed=episode_seed,
        mechanical_attempt_index=mechanical_attempt_index,
        episode_random_key=episode_random_key,
        probe_executed=probe_executed,
        selected_probe_id=selected_for_safety_id,
        selected_probe_step=selected_probe_step,
        expected_masked_values=expected_masked_values,
        trace_replay_verifier=trace_replay_verifier,
    )
    _validate_decision_evidence(
        probe_decision_payload,
        trace_groups=trace_groups,
        preregistration=preregistration,
        support_members=support_members,
        audit_unit_id=audit_unit_id,
        prototype_id=prototype_id,
        episode_seed=episode_seed,
        decision_evidence_verifier=decision_evidence_verifier,
    )
    pairing = _require_mapping(payload.get("pairing"), field="pairing")
    if not _is_sha256(pairing.get("shared_continuation_controller_sha256")):
        raise ValueError("A2 groups must bind their shared continuation controller.")
    frozen_pairing_bindings = {
        "shared_continuation_controller_sha256": "continuation_controller",
        "probe_registry_sha256": "probe_registry",
        "response_vocabulary_sha256": "response_vocabulary",
    }
    for field, artifact_name in frozen_pairing_bindings.items():
        frozen_sha = preregistration.artifacts[artifact_name].sha256
        if frozen_sha is None or pairing.get(field) != frozen_sha:
            raise ValueError(f"The paired block used a different {artifact_name}.")
    if pairing.get("probe_registry_semantic_sha256") != (
        preregistration.probe_registry_semantic_sha256
    ):
        raise ValueError("The paired controllers used different probe scripts.")
    if pairing.get("a1_score_trace_sha256") != probe_decision_sha256 or pairing.get(
        "a2_score_trace_sha256"
    ) != probe_decision_sha256:
        raise ValueError("A1 and A2 must bind the same pre-probe score trace.")
    projection_binding = preregistration.artifacts["response_projection"]
    if projection_binding.sha256 is None or pairing.get(
        "response_projection_sha256"
    ) != projection_binding.sha256:
        raise ValueError("The paired block used a different response projection.")
    if pairing.get("raw_official_local_observation_preserved") is not True:
        raise ValueError("The response projection may not rewrite official observation.")
    if pairing.get("future_passive_response_route") != "shared":
        raise ValueError("Later passive response tokens must be shared by both A2 groups.")
    if pairing.get("a1_registered_response_route") != "masked":
        raise ValueError("A1 must remain the response-masked reference controller.")
    if pairing.get("a1_full_candidate_set") is not True:
        raise ValueError("A1 must retain the complete frozen candidate set.")
    if pairing.get("a1_uses_masked_value_reference") is not True:
        raise ValueError("A1 must choose by the frozen response-masked value reference.")
    a2_mask_pair = _require_mapping(pairing.get("A2-mask"), field="pairing.A2-mask")
    a2_use_pair = _require_mapping(pairing.get("A2-use"), field="pairing.A2-use")
    if probe_executed:
        probe_hashes = {
            "pre_probe_trace_sha256",
            "masked_history_sha256",
            "current_probe_response_sha256",
        }
        if any(
            not _is_sha256(record.get(field))
            for record in (a2_mask_pair, a2_use_pair)
            for field in probe_hashes
        ):
            raise ValueError("An executed probe requires both A2 pairing bindings.")
        trace_digest_fields = {
            "pre_probe_trace_sha256": "pre_probe_trace",
            "masked_history_sha256": "masked_history",
            "current_probe_response_sha256": "current_probe_response",
            "masked_value_reference_sha256": "masked_value_reference",
            "belief_common_input_sha256": "belief_common_input",
            "continuation_common_input_sha256": "continuation_common_input",
            "recurrent_common_input_sha256": "recurrent_common_input",
        }
        for group, pairing_record in (
            ("A2-mask", a2_mask_pair),
            ("A2-use", a2_use_pair),
        ):
            trace_record = trace_groups[group]
            if any(
                pairing_record.get(digest_field)
                != _canonical_value_sha256(trace_record.get(trace_field))
                for digest_field, trace_field in trace_digest_fields.items()
            ):
                raise ValueError("A pairing digest does not match the embedded trace.")
            trace_belief_extra = trace_record.get("belief_current_response_extra")
            expected_belief_extra_sha = (
                None
                if trace_belief_extra is None
                else _canonical_value_sha256(trace_belief_extra)
            )
            if pairing_record.get(
                "belief_current_response_extra_sha256"
            ) != expected_belief_extra_sha:
                raise ValueError("The belief response binding differs from the trace.")
            if list(
                _require_sequence(
                    pairing_record.get("non_belief_response_aliases"),
                    field="pairing.non_belief_response_aliases",
                )
            ) != list(trace_record.get("non_belief_response_aliases", ())):
                raise ValueError("The response-alias record differs from the trace.")
        if any(
            a2_mask_pair.get(field) != a2_use_pair.get(field)
            for field in probe_hashes | {"selected_probe_id"}
        ):
            raise ValueError("A2-mask and A2-use pairing evidence must match.")
        if pairing.get("a1_pre_probe_trace_sha256") != _canonical_value_sha256(
            trace_groups["A1"].get("pre_probe_trace")
        ) or pairing.get("a1_pre_probe_trace_sha256") != a2_mask_pair.get(
            "pre_probe_trace_sha256"
        ):
            raise ValueError("A1 and both A2 groups must share the pre-probe trace.")
        masked_value_hashes = {
            pairing.get("a1_masked_value_reference_sha256"),
            a2_mask_pair.get("masked_value_reference_sha256"),
            a2_use_pair.get("masked_value_reference_sha256"),
        }
        if len(masked_value_hashes) != 1 or not _is_sha256(
            next(iter(masked_value_hashes))
        ) or pairing.get("a1_masked_value_reference_sha256") != (
            _canonical_value_sha256(
                trace_groups["A1"].get("masked_value_reference")
            )
        ):
            raise ValueError("A1 and both A2 groups must share the masked value reference.")
        if not str(a2_mask_pair.get("selected_probe_id", "")):
            raise ValueError("A2-mask and A2-use must bind the same selected probe.")
        if a2_mask_pair.get("selected_probe_id") != selected_for_safety_id:
            raise ValueError("The executed A2 script differs from the scored selection.")
        if a2_mask_pair.get("current_response_route") != "masked":
            raise ValueError("A2-mask must mask the current probe response.")
        if a2_use_pair.get("current_response_route") != "belief_update_only":
            raise ValueError("A2-use may route the current response to belief update only.")
        common_projection_fields = {
            "belief_common_input_sha256",
            "continuation_common_input_sha256",
            "recurrent_common_input_sha256",
            "projection_dependency_audit_sha256",
        }
        if any(
            not _is_sha256(record.get(field))
            for record in (a2_mask_pair, a2_use_pair)
            for field in common_projection_fields
        ) or any(
            a2_mask_pair.get(field) != a2_use_pair.get(field)
            for field in common_projection_fields
        ):
            raise ValueError("Both A2 groups must bind the same projected common inputs.")
        if {
            a2_mask_pair.get("projection_dependency_audit_sha256"),
            a2_use_pair.get("projection_dependency_audit_sha256"),
        } != {projection_binding.sha256}:
            raise ValueError("The projected-input audit differs from the frozen projection.")
        if a2_mask_pair.get("belief_current_response_extra_sha256") is not None:
            raise ValueError("A2-mask cannot pass the current response to belief update.")
        if a2_use_pair.get("belief_current_response_extra_sha256") != (
            a2_use_pair.get("current_probe_response_sha256")
        ):
            raise ValueError("A2-use must add the current response only to belief update.")
        for record in (a2_mask_pair, a2_use_pair):
            if record.get("continuation_direct_response_sha256") is not None or record.get(
                "recurrent_direct_response_sha256"
            ) is not None:
                raise ValueError("The current response cannot bypass belief update.")
            aliases = _require_sequence(
                record.get("non_belief_response_aliases"),
                field="pairing.non_belief_response_aliases",
            )
            if aliases:
                raise ValueError("The projected controller input retains a response alias.")
        if pairing.get("a2_groups_share_pre_probe_actions") is not True:
            raise ValueError("A2-mask and A2-use must share all pre-probe actions.")
        if pairing.get("a2_groups_share_probe_script") is not True:
            raise ValueError("A2-mask and A2-use must execute the same probe script.")
    else:
        if any(
            record.get("selected_probe_id") is not None
            for record in (a2_mask_pair, a2_use_pair)
        ):
            raise ValueError("A block without a probe cannot name a selected probe.")
        if any(
            record.get("current_probe_response_sha256") is not None
            for record in (a2_mask_pair, a2_use_pair)
        ):
            raise ValueError("A block without a probe cannot contain a probe response.")
        if a2_mask_pair.get("current_response_route") != "not_applicable" or (
            a2_use_pair.get("current_response_route") != "not_applicable"
        ):
            raise ValueError("Current-response routing is not applicable without a probe.")
        if not _is_sha256(a2_mask_pair.get("complete_trace_sha256")) or (
            a2_mask_pair.get("complete_trace_sha256")
            != a2_use_pair.get("complete_trace_sha256")
        ):
            raise ValueError("Without a probe, both A2 groups must bind one trace.")
        expected_complete_trace_sha256 = _canonical_value_sha256(
            trace_groups["A2-mask"].get("complete_trace")
        )
        if a2_mask_pair.get("complete_trace_sha256") != (
            expected_complete_trace_sha256
        ):
            raise ValueError("The no-probe trace digest does not match its content.")
        if pairing.get("a1_complete_trace_sha256") != a2_mask_pair.get(
            "complete_trace_sha256"
        ) or pairing.get("a1_complete_trace_sha256") != _canonical_value_sha256(
            trace_groups["A1"].get("complete_trace")
        ):
            raise ValueError("Without a probe, A1 and both A2 groups must share one trace.")
        if pairing.get("a2_groups_share_complete_trace") is not True:
            raise ValueError("Without a probe, the two A2 groups must share their trace.")
        if pairing.get("all_groups_share_complete_trace") is not True:
            raise ValueError("Without a probe, all formal groups must share their trace.")

    raw_comparisons = _require_sequence(
        safety.get("comparisons", ()), field="safety.comparisons"
    )
    comparisons = tuple(
        SafetyComparison.from_mapping(
            _require_mapping(item, field="safety.comparisons item")
        )
        for item in raw_comparisons
    )
    if any(
        not item.support_prototype_id
        or item.repetitions < 0
        or item.wrong_delivery_count < 0
        or item.wrong_delivery_count > item.repetitions
        or not item.random_stream_key
        for item in comparisons
    ):
        raise ValueError("A safety comparison contains an invalid count or identifier.")
    if any(item.support_prototype_id not in support_members for item in comparisons):
        raise ValueError("A safety comparison used a prototype outside the support.")
    if len({item.support_prototype_id for item in comparisons}) != len(comparisons):
        raise ValueError("A safety comparison repeated a support prototype.")
    if len({item.random_stream_key for item in comparisons}) != len(comparisons):
        raise ValueError("Safety comparisons require independent random streams.")
    if len({item.branch_keys_sha256 for item in comparisons}) != len(comparisons):
        raise ValueError("Safety comparisons require distinct branch-key sets.")
    required_repeats = preregistration.statistics.safety_repeats_per_comparison
    if probe_executed and (
        required_repeats is None
        or any(item.repetitions != required_repeats for item in comparisons)
    ):
        raise ValueError("A safety comparison used the wrong repeat count.")
    random_key_derivation_sha256 = preregistration.artifacts[
        "random_key_derivation"
    ].sha256
    if any(
        item.audit_unit_id != audit_unit_id
        or item.episode_seed != episode_seed
        or item.purpose != "safety_validation"
        or item.environment_step != selected_probe_step
        or item.branch_start_index != 0
        or item.branch_count != item.repetitions
        or not _is_sha256(item.branch_keys_sha256)
        or item.random_key_derivation_sha256 != random_key_derivation_sha256
        for item in comparisons
    ):
        raise ValueError("A safety comparison lacks its full independent-branch binding.")
    all_safety_branch_keys: set[str] = set()
    for item in comparisons:
        expected_branch_keys = tuple(
            _derive_r015_random_key(
                audit_unit_id=audit_unit_id,
                partner_prototype_id=item.support_prototype_id,
                episode_seed=episode_seed,
                purpose="safety_validation",
                environment_step=item.environment_step,
                branch_index=branch_index,
            )
            for branch_index in range(
                item.branch_start_index,
                item.branch_start_index + item.branch_count,
            )
        )
        if item.branch_keys != expected_branch_keys:
            raise ValueError(
                "Safety branch keys do not match the frozen coordinate derivation."
            )
        if item.branch_keys_sha256 != _canonical_value_sha256(
            list(expected_branch_keys)
        ):
            raise ValueError("The safety branch-key digest does not match its key list.")
        if len(item.branch_evidence) != item.branch_count:
            raise ValueError("Safety evidence must contain every registered branch.")
        for offset, evidence in enumerate(item.branch_evidence):
            expected_index = item.branch_start_index + offset
            required_evidence_fields = {
                "branch_index",
                "random_key",
                "compatible_hidden_state_sha256",
                "environment_steps",
            }
            if set(evidence) != required_evidence_fields or _nonnegative_int(
                evidence.get("branch_index"), field="safety.branch_evidence.branch_index"
            ) != expected_index:
                raise ValueError("A safety branch evidence record has the wrong fields.")
            if evidence.get("random_key") != expected_branch_keys[offset] or not _is_sha256(
                evidence.get("compatible_hidden_state_sha256")
            ):
                raise ValueError("A safety branch evidence record has the wrong binding.")
            environment_steps = _require_sequence(
                evidence.get("environment_steps"),
                field="safety.branch_evidence.environment_steps",
            )
            if not environment_steps or any(
                not isinstance(step, Mapping) for step in environment_steps
            ):
                raise ValueError("A safety branch requires replayable environment steps.")
        if len(item.branch_results) != item.branch_count:
            raise ValueError("Safety results must contain every registered branch.")
        for offset, result in enumerate(item.branch_results):
            expected_index = item.branch_start_index + offset
            if set(result) != {
                "branch_index",
                "random_key",
                "wrong_delivery_detected",
            } or _nonnegative_int(
                result.get("branch_index"), field="safety.branch_results.branch_index"
            ) != expected_index:
                raise ValueError("A safety branch result has the wrong fields or index.")
            if result.get("random_key") != expected_branch_keys[offset] or not isinstance(
                result.get("wrong_delivery_detected"), bool
            ):
                raise ValueError("A safety branch result has the wrong binding.")
        if sum(
            result["wrong_delivery_detected"] is True for result in item.branch_results
        ) != item.wrong_delivery_count:
            raise ValueError("Safety branch results do not match the wrong-delivery count.")
        if selected_for_safety_id not in preregistration.probe_scripts:
            raise ValueError("Safety replay requires one registered selected probe.")
        selected_probe_actions_for_safety = list(
            preregistration.probe_scripts[
                selected_for_safety_id
            ].primitive_actions
        )
        official_history_prefix = list(
            _require_sequence(
                trace_groups["A1"].get("official_history_events"),
                field="trace.groups.A1.official_history_events",
            )[: item.environment_step + 1]
        )
        comparison_support_member = support_members[item.support_prototype_id]
        verification = safety_branch_evidence_verifier(
            {
                "support_prototype_id": item.support_prototype_id,
                "posterior_support_evidence": item.posterior_support_evidence,
                "branch_evidence": list(item.branch_evidence),
                "official_history": official_history_prefix,
                "selected_probe_id": selected_for_safety_id,
                "selected_probe_actions": selected_probe_actions_for_safety,
            },
            {
                "audit_unit_id": audit_unit_id,
                "episode_seed": episode_seed,
                "environment_step": item.environment_step,
                "branch_keys": list(expected_branch_keys),
                "environment_config_path": str(
                    preregistration.artifacts["environment_config"].path
                ),
                "environment_config_sha256": preregistration.artifacts[
                    "environment_config"
                ].sha256,
                "environment_source_path": str(
                    preregistration.artifacts["environment_source"].path
                ),
                "environment_source_sha256": preregistration.artifacts[
                    "environment_source"
                ].sha256,
                "wrong_delivery_detector_path": str(
                    preregistration.artifacts["wrong_delivery_detector"].path
                ),
                "wrong_delivery_detector_sha256": preregistration.artifacts[
                    "wrong_delivery_detector"
                ].sha256,
                "random_key_derivation_path": str(
                    preregistration.artifacts["random_key_derivation"].path
                ),
                "random_key_derivation_sha256": random_key_derivation_sha256,
                "support_checkpoint_path": comparison_support_member.get(
                    "checkpoint_path"
                ),
                "support_checkpoint_sha256": comparison_support_member.get(
                    "checkpoint_sha256"
                ),
                "support_model_weights_sha256": comparison_support_member.get(
                    "model_weights_sha256"
                ),
                "support_training_config_path": comparison_support_member.get(
                    "training_config_path"
                ),
                "support_training_config_sha256": comparison_support_member.get(
                    "training_config_sha256"
                ),
                "support_training_manifest_path": comparison_support_member.get(
                    "training_manifest_path"
                ),
                "support_training_manifest_sha256": comparison_support_member.get(
                    "training_manifest_sha256"
                ),
                "support_architecture": comparison_support_member.get("architecture"),
                "support_architecture_sha256": comparison_support_member.get(
                    "architecture_sha256"
                ),
                "support_action_rule": comparison_support_member.get("action_rule"),
                "continuation_planner_path": str(
                    preregistration.artifacts["continuation_planner"].path
                ),
                "continuation_planner_sha256": preregistration.artifacts[
                    "continuation_planner"
                ].sha256,
                "support_family_spec_sha256": comparison_support_member.get(
                    "family_spec_sha256"
                ),
                "support_registration_sha256": (
                    preregistration.support_registration_sha256
                ),
                "support_report_sha256": (
                    None
                    if preregistration.support_report_path is None
                    else _file_sha256(preregistration.support_report_path)
                ),
                "official_history_filter_path": str(
                    preregistration.artifacts["official_history_filter"].path
                ),
                "official_history_filter_sha256": preregistration.artifacts[
                    "official_history_filter"
                ].sha256,
                "response_projection_path": str(
                    preregistration.artifacts["response_projection"].path
                ),
                "response_projection_sha256": preregistration.artifacts[
                    "response_projection"
                ].sha256,
                "ego_evidence_contract_path": str(
                    preregistration.artifacts["ego_evidence_contract"].path
                ),
                "ego_evidence_contract_sha256": preregistration.artifacts[
                    "ego_evidence_contract"
                ].sha256,
                "response_vocabulary_path": str(
                    preregistration.artifacts["response_vocabulary"].path
                ),
                "response_vocabulary_sha256": preregistration.artifacts[
                    "response_vocabulary"
                ].sha256,
                "probe_registry_path": str(
                    preregistration.artifacts["probe_registry"].path
                ),
                "probe_registry_sha256": preregistration.artifacts[
                    "probe_registry"
                ].sha256,
                "probe_registry_semantic_sha256": (
                    preregistration.probe_registry_semantic_sha256
                ),
            },
        )
        expected_verification = {
            "schema_version": "path_c_r015_safety_branch_verification_v2",
            "verified": True,
            "support_prototype_id": item.support_prototype_id,
            "branch_count": item.branch_count,
            "branch_keys_sha256": item.branch_keys_sha256,
            "wrong_delivery_count": item.wrong_delivery_count,
            "positive_posterior_support": item.positive_posterior_support,
            "compatible_hidden_state_reconstructed": (
                item.compatible_hidden_state_reconstructed
            ),
            "all_branches_replayed": True,
            "support_checkpoint_sha256": comparison_support_member.get(
                "checkpoint_sha256"
            ),
            "support_model_weights_sha256": comparison_support_member.get(
                "model_weights_sha256"
            ),
            "support_architecture_sha256": comparison_support_member.get(
                "architecture_sha256"
            ),
            "support_training_config_sha256": comparison_support_member.get(
                "training_config_sha256"
            ),
            "support_training_manifest_sha256": comparison_support_member.get(
                "training_manifest_sha256"
            ),
            "support_family_spec_sha256": comparison_support_member.get(
                "family_spec_sha256"
            ),
            "support_action_rule": comparison_support_member.get("action_rule"),
            "support_registration_sha256": (
                preregistration.support_registration_sha256
            ),
            "support_report_sha256": (
                None
                if preregistration.support_report_path is None
                else _file_sha256(preregistration.support_report_path)
            ),
            "selected_probe_id": selected_for_safety_id,
            "selected_probe_actions": selected_probe_actions_for_safety,
            "probe_registry_sha256": preregistration.artifacts[
                "probe_registry"
            ].sha256,
            "probe_registry_semantic_sha256": (
                preregistration.probe_registry_semantic_sha256
            ),
            "official_history_filter_sha256": preregistration.artifacts[
                "official_history_filter"
            ].sha256,
            "response_projection_sha256": preregistration.artifacts[
                "response_projection"
            ].sha256,
            "ego_evidence_contract_sha256": preregistration.artifacts[
                "ego_evidence_contract"
            ].sha256,
            "response_vocabulary_sha256": preregistration.artifacts[
                "response_vocabulary"
            ].sha256,
            "official_history_sha256": _canonical_value_sha256(
                official_history_prefix
            ),
            "branch_evidence_sha256": _canonical_value_sha256(
                list(item.branch_evidence)
            ),
            "branch_results_sha256": _canonical_value_sha256(
                list(item.branch_results)
            ),
        }
        if verification != expected_verification:
            raise ValueError(
                "Frozen safety verification does not match the wrong delivery or support summary."
            )
        if all_safety_branch_keys.intersection(expected_branch_keys):
            raise ValueError("Safety comparisons reused an individual branch key.")
        all_safety_branch_keys.update(expected_branch_keys)
    controller_random_keys = {
        episode_random_key,
        planning_random_stream_key,
        score_random_stream_key,
    }
    if any(item.random_stream_key in controller_random_keys for item in comparisons):
        raise ValueError("Safety and controller random streams must be independent.")
    if all_safety_branch_keys & controller_random_keys:
        raise ValueError("A safety branch reused an actual, planning, or score key.")
    if any(
        not item.positive_posterior_support
        or not item.compatible_hidden_state_reconstructed
        for item in comparisons
    ):
        raise ValueError("Safety comparison lacks supported compatible hidden state.")
    if probe_executed:
        if telemetry["candidate_evaluation_count"] <= 0:
            raise ValueError("An executed probe requires at least one candidate evaluation.")
        if telemetry["safety_rejection_count"] != 0:
            raise ValueError("An executed probe cannot also be counted as safety rejected.")
        if no_probe_reason is not None:
            raise ValueError("An executed probe cannot carry a no-probe reason.")
        if len(comparisons) != preregistration.statistics.partner_prototype_count:
            raise ValueError("Every executed probe requires four safety comparisons.")
        if {item.support_prototype_id for item in comparisons} != set(support_members):
            raise ValueError("Safety comparisons must cover the four support prototypes.")
        if required_repeats is None:
            raise ValueError("The safety repeat count is still pending.")
        if any(item.repetitions != required_repeats for item in comparisons):
            raise ValueError("A safety comparison used the wrong repeat count.")
        if any(item.wrong_delivery_count != 0 for item in comparisons):
            raise ValueError("A probe with any simulated wrong delivery must be rejected.")
    else:
        if no_probe_reason not in _NO_PROBE_REASONS:
            raise ValueError("A non-executed probe requires a registered reason.")
        if (
            no_probe_reason == "non_positive_score"
            and telemetry["non_positive_score_count"] <= 0
        ):
            raise ValueError("A non-positive decision must record its candidate count.")
        if (
            no_probe_reason == "window_expired"
            and telemetry["window_expired_count"] <= 0
        ):
            raise ValueError("A window-expired decision must record its count.")
        if no_probe_reason in {
            "support_incompatible",
            "safety_rejected",
            "no_safe_candidate",
        } and telemetry["safety_rejection_count"] <= 0:
            raise ValueError("A safety rejection must be counted.")
        if no_probe_reason in {"non_positive_score", "window_expired"} and telemetry[
            "safety_rejection_count"
        ] != 0:
            raise ValueError("A non-safety outcome cannot be counted as safety rejected.")
        required_repeats = preregistration.statistics.safety_repeats_per_comparison
        if no_probe_reason == "safety_rejected":
            if not comparisons or not any(
                item.wrong_delivery_count > 0 for item in comparisons
            ):
                raise ValueError(
                    "A safety-rejected probe must record an observed wrong delivery."
                )
            if required_repeats is None or any(
                item.repetitions <= 0 or item.repetitions > required_repeats
                for item in comparisons
            ):
                raise ValueError("Rejected safety comparisons have invalid repeat counts.")
        elif comparisons:
            raise ValueError("Only a safety rejection may retain failed comparisons.")
        if not math.isclose(
            arms["A2-use"].raw_return,
            arms["A2-mask"].raw_return,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Without a probe, A2-use and A2-mask must have equal return.")
        if not math.isclose(
            arms["A1"].raw_return,
            arms["A2-mask"].raw_return,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Without a probe, all formal returns must be equal.")

    block = R015PairedBlock(
        round_index=round_index,
        audit_unit_id=audit_unit_id,
        prototype_id=prototype_id,
        family_id=family_id,
        training_seed=training_seed,
        training_run_id=training_run_id,
        ego_position=ego_position,
        initial_state_sha256=initial_state_sha256,
        episode_seed=episode_seed,
        mechanical_attempt_index=mechanical_attempt_index,
        formal_effect_look_number=_positive_int(
            payload.get("formal_effect_look_number"),
            field="formal_effect_look_number",
        ),
        pilot_data=False,
        arms=arms,
        probe_executed=probe_executed,
        probe_step=selected_probe_step,
        no_probe_reason=None if no_probe_reason is None else str(no_probe_reason),
        planning_random_stream_key=planning_random_stream_key,
        score_random_stream_key=score_random_stream_key,
        safety_comparisons=comparisons,
        candidate_evaluation_count=telemetry["candidate_evaluation_count"],
        safety_rejection_count=telemetry["safety_rejection_count"],
        non_positive_score_count=telemetry["non_positive_score_count"],
        window_expired_count=telemetry["window_expired_count"],
    )
    if block.ego_position != 1:
        raise ValueError("Paired block ego must remain the agent_1 cook.")
    if not _is_sha256(block.initial_state_sha256):
        raise ValueError("Paired block must bind its initial state.")
    if block.episode_seed < 0:
        raise ValueError("Paired block episode seed is invalid.")
    if block.formal_effect_look_number != 1 or block.pilot_data:
        raise ValueError("Formal data permit one effect look and exclude pilot blocks.")
    block.validate_identity()
    if any(
        abs(value) > preregistration.statistics.paired_difference_absolute_bound
        for value in (block.delta_net, block.delta_response, block.delta_cost)
    ):
        raise ValueError("A paired difference exceeds the checked structural range.")
    return block


@dataclass(frozen=True)
class EffectInterval:
    estimate: float
    lower: float
    upper: float
    radius: float
    sample_variance: float | None = None
    formula_id: str | None = None


def maurer_pontil_empirical_bernstein_interval(
    values: Sequence[float],
    *,
    alpha: float,
    lower_bound: float,
    upper_bound: float,
) -> EffectInterval:
    """Return the frozen bounded empirical-Bernstein interval."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("Empirical-Bernstein alpha must lie strictly between zero and one.")
    if not lower_bound < upper_bound:
        raise ValueError("Empirical-Bernstein bounds are not ordered.")
    samples = tuple(_finite_float(value, field="round_average") for value in values)
    if len(samples) < 2:
        raise ValueError("Empirical-Bernstein intervals require at least two rounds.")
    if any(not lower_bound <= value <= upper_bound for value in samples):
        raise ValueError("A round average exceeds the frozen paired-difference range.")
    sample_count = len(samples)
    estimate = math.fsum(samples) / sample_count
    sample_variance = math.fsum(
        (value - estimate) ** 2 for value in samples
    ) / (sample_count - 1)
    log_term = math.log(2.0 / alpha)
    radius = math.sqrt(
        2.0 * sample_variance * log_term / sample_count
    ) + (
        7.0
        * (upper_bound - lower_bound)
        * log_term
        / (3.0 * (sample_count - 1))
    )
    return EffectInterval(
        estimate=estimate,
        lower=max(lower_bound, estimate - radius),
        upper=min(upper_bound, estimate + radius),
        radius=radius,
        sample_variance=sample_variance,
        formula_id="maurer_pontil_empirical_bernstein_bounded_v1",
    )


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(success + 1)
        - math.lgamma(trials - success + 1)
        + success * log_probability
        + (trials - success) * log_complement
        for success in range(successes + 1)
    ]
    largest = max(log_terms)
    return math.exp(largest) * math.fsum(
        math.exp(value - largest) for value in log_terms
    )


def clopper_pearson_upper_bound(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    """One-sided exact binomial upper bound, inverted by deterministic bisection."""

    successes = _nonnegative_int(successes, field="successes")
    trials = _positive_int(trials, field="trials")
    if successes > trials:
        raise ValueError("Binomial successes cannot exceed trials.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Clopper-Pearson alpha must lie strictly between zero and one.")
    if successes == trials:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


@dataclass(frozen=True)
class R015KillDecision:
    verdict: str
    registered_negative_reason: str | None
    checkpoint_round_count: int
    firing_counts_by_prototype: Mapping[str, int]
    total_fires: int
    rho_bar_upper: float
    effect_absolute_upper: float
    alpha_checkpoint: float
    method: str
    full_state_upper_bound_status: str = "not_provided"
    substrate_negative_claim_allowed: bool = False


def evaluate_r015_firing_count_checkpoint(
    payload: Mapping[str, Any],
    *,
    statistics: R015Statistics,
    expected_prototype_ids: Sequence[str],
) -> R015KillDecision:
    """Evaluate one frozen checkpoint without accepting any effect value."""

    required_fields = {
        "schema_version",
        "experiment_id",
        "checkpoint_round_count",
        "firing_counts_by_prototype",
    }
    if set(payload) != required_fields:
        raise ValueError("The firing-count evaluator accepts count-only input fields.")
    if payload.get("schema_version") != R015_FIRING_COUNT_SCHEMA or payload.get(
        "experiment_id"
    ) != R015_EXPERIMENT_ID:
        raise ValueError("The firing-count checkpoint has the wrong schema or experiment.")
    checkpoint = _positive_int(
        payload.get("checkpoint_round_count"), field="checkpoint_round_count"
    )
    if checkpoint not in statistics.kill_checkpoint_rounds:
        raise ValueError("The firing-count checkpoint is outside the frozen schedule.")
    raw_counts = _require_mapping(
        payload.get("firing_counts_by_prototype"),
        field="firing_counts_by_prototype",
    )
    prototype_ids = tuple(str(item) for item in expected_prototype_ids)
    if len(set(prototype_ids)) != statistics.partner_prototype_count or set(
        raw_counts
    ) != set(prototype_ids):
        raise ValueError("Firing counts must cover the four frozen prototypes exactly.")
    counts = {
        prototype_id: _nonnegative_int(
            raw_counts[prototype_id],
            field=f"firing_counts_by_prototype.{prototype_id}",
        )
        for prototype_id in prototype_ids
    }
    if any(count > checkpoint for count in counts.values()):
        raise ValueError("A firing count exceeds the completed round count.")
    total_fires = sum(counts.values())
    if total_fires == 0:
        rho_bar_upper = -math.log(statistics.alpha_rho_per_checkpoint) / (
            checkpoint * statistics.partner_prototype_count
        )
        method = "zero_count_exponential_bound"
    else:
        rho_bar_upper = math.fsum(
            clopper_pearson_upper_bound(
                count,
                checkpoint,
                alpha=statistics.alpha_rho_per_prototype_cp,
            )
            for count in counts.values()
        ) / statistics.partner_prototype_count
        method = "per_prototype_one_sided_clopper_pearson_average"
    effect_absolute_upper = (
        rho_bar_upper * statistics.paired_difference_absolute_bound
    )
    verdict = (
        "REGISTERED_NEGATIVE"
        if effect_absolute_upper < statistics.practical_margin
        else "CONTINUE"
    )
    return R015KillDecision(
        verdict=verdict,
        registered_negative_reason=(
            "firing_count_upper_below_margin"
            if verdict == "REGISTERED_NEGATIVE"
            else None
        ),
        checkpoint_round_count=checkpoint,
        firing_counts_by_prototype=counts,
        total_fires=total_fires,
        rho_bar_upper=rho_bar_upper,
        effect_absolute_upper=effect_absolute_upper,
        alpha_checkpoint=statistics.alpha_rho_per_checkpoint,
        method=method,
    )


@dataclass(frozen=True)
class R015Decision:
    verdict: str
    delta_response: EffectInterval
    delta_cost_estimate: float
    delta_net: EffectInterval
    practical_margin: float
    paired_block_count: int
    formal_round_count: int
    firing_counts_by_prototype: Mapping[str, int]
    registered_negative_reason: str | None
    full_state_upper_bound_status: str
    substrate_negative_claim_allowed: bool


def decide_r015(
    *,
    delta_net: EffectInterval,
    delta_response: EffectInterval,
    practical_margin: float,
) -> str:
    """Apply the frozen three-state rule without treating overlap as failure."""

    if delta_net.lower > practical_margin and delta_response.lower > 0.0:
        return "GO"
    if delta_net.upper < practical_margin:
        return "REGISTERED_NEGATIVE"
    return "UNDECIDABLE"


def _aggregate_validated_r015_blocks(
    blocks: Sequence[R015PairedBlock],
    *,
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
    expected_sampling_schedule: FormalSamplingSchedule,
    kill_checkpoint_decisions: Sequence[R015KillDecision],
) -> R015Decision:
    """Internal arithmetic over blocks already accepted by the complete validator."""

    preregistration.statistics.require_formal_ready()
    if len(support_members) != preregistration.statistics.partner_prototype_count:
        raise ValueError("Formal aggregation requires exactly four support prototypes.")
    family_counts: dict[str, int] = {}
    training_runs: set[tuple[str, int]] = set()
    training_run_ids: set[str] = set()
    for member in support_members.values():
        family_id = str(member.get("family_id", ""))
        training_seed = int(member.get("training_seed", -1))
        training_run_id = member.get("training_run_id")
        if not family_id or training_seed < 0 or not _is_sha256(training_run_id):
            raise ValueError("A formal support member lacks family or training-run data.")
        family_counts[family_id] = family_counts.get(family_id, 0) + 1
        training_runs.add((family_id, training_seed))
        training_run_ids.add(str(training_run_id))
    if (
        sorted(family_counts.values()) != [2, 2]
        or len(training_runs) != 4
        or len(training_run_ids) != 4
    ):
        raise ValueError("Formal aggregation requires two independent runs per family.")
    if any(
        decision.registered_negative_reason == "firing_count_upper_below_margin"
        for decision in kill_checkpoint_decisions
    ):
        raise ValueError("Effect values cannot be read after a firing-count kill.")
    expected_checkpoints = preregistration.statistics.kill_checkpoint_rounds
    if tuple(
        decision.checkpoint_round_count for decision in kill_checkpoint_decisions
    ) != expected_checkpoints:
        raise ValueError("Effect adjudication lacks the complete checkpoint history.")
    expected_rounds = preregistration.statistics.n_rounds_max
    if len(blocks) != len(support_members) * expected_rounds:
        raise ValueError("Formal dataset has the wrong total number of paired blocks.")
    expected_schedule_by_coordinate = {
        (round_index, audit_unit_id, prototype_id, ego_position): episode_seeds
        for (
            round_index,
            audit_unit_id,
            prototype_id,
            episode_seeds,
            ego_position,
        ) in expected_sampling_schedule
    }
    observed_sampling_coordinates: set[tuple[int, str, str, int]] = set()
    for block in blocks:
        coordinate = (
            block.round_index,
            block.audit_unit_id,
            block.prototype_id,
            block.ego_position,
        )
        attempt_seeds = expected_schedule_by_coordinate.get(coordinate)
        if attempt_seeds is None or not (
            0 <= block.mechanical_attempt_index < len(attempt_seeds)
        ) or attempt_seeds[block.mechanical_attempt_index] != block.episode_seed:
            raise ValueError("Formal blocks differ from the frozen sampling schedule.")
        observed_sampling_coordinates.add(coordinate)
    if observed_sampling_coordinates != set(expected_schedule_by_coordinate):
        raise ValueError("Formal blocks do not cover every frozen sampling coordinate.")
    by_prototype: dict[str, list[R015PairedBlock]] = {
        prototype_id: [] for prototype_id in support_members
    }
    by_round: dict[int, dict[str, R015PairedBlock]] = {
        round_index: {} for round_index in range(1, expected_rounds + 1)
    }
    audit_unit_ids: set[str] = set()
    independence_keys: set[str] = set()
    safety_random_keys: set[str] = set()
    safety_branch_keys: set[str] = set()
    for block in blocks:
        block.validate_identity()
        if block.round_index not in by_round:
            raise ValueError("A formal block has an out-of-range round index.")
        if block.prototype_id not in by_prototype:
            raise ValueError("Formal dataset contains an unregistered prototype.")
        member = support_members[block.prototype_id]
        if (
            block.family_id != member.get("family_id")
            or block.training_seed != int(member.get("training_seed", -1))
            or block.training_run_id != member.get("training_run_id")
        ):
            raise ValueError("Formal block changed prototype family or training run.")
        if block.formal_effect_look_number != 1 or block.pilot_data:
            raise ValueError("Formal aggregation permits one effect look and excludes pilot data.")
        if set(block.arms) != set(R015_FORMAL_GROUPS):
            raise ValueError("Formal block has the wrong experiment groups.")
        arm_identities = {
            (
                arm.prototype_id,
                arm.family_id,
                arm.training_seed,
                arm.training_run_id,
                arm.ego_position,
                arm.initial_state_sha256,
                arm.episode_seed,
            )
            for arm in block.arms.values()
        }
        if arm_identities != {
            (
                block.prototype_id,
                block.family_id,
                block.training_seed,
                block.training_run_id,
                block.ego_position,
                block.initial_state_sha256,
                block.episode_seed,
            )
        }:
            raise ValueError("Formal block groups do not share their paired identity.")
        if any(
            not arm.valid
            or arm.information_source != "official_local_history_only"
            or arm.forbidden_fields_read
            for arm in block.arms.values()
        ):
            raise ValueError("Formal block contains invalid or privileged controller input.")
        expected_arm_bindings = {
            "controller_contract_sha256": preregistration.artifacts[
                "ego_evidence_contract"
            ].sha256,
            "ego_checkpoint_sha256": preregistration.artifacts[
                "ego_checkpoint"
            ].sha256,
            "environment_config_sha256": preregistration.artifacts[
                "environment_config"
            ].sha256,
            "environment_source_sha256": preregistration.artifacts[
                "environment_source"
            ].sha256,
            "official_history_filter_sha256": preregistration.artifacts[
                "official_history_filter"
            ].sha256,
        }
        if any(
            frozen_sha is None
            or {getattr(arm, field) for arm in block.arms.values()} != {frozen_sha}
            for field, frozen_sha in expected_arm_bindings.items()
        ):
            raise ValueError("Formal block changed a frozen runtime artifact.")
        lower = preregistration.statistics.return_lower_bound
        upper = preregistration.statistics.return_upper_bound
        if any(not lower <= arm.raw_return <= upper for arm in block.arms.values()):
            raise ValueError("Formal block return exceeds the deterministic bound.")
        if not block.probe_executed and len(
            {arm.raw_return for arm in block.arms.values()}
        ) != 1:
            raise ValueError("A no-probe block must have identical formal returns.")
        # v4 合同允许 32 位 episode seed 偶然碰撞。正式块的唯一身份由冻结日程坐标和
        # audit_unit_id 给出，不能再用 seed 与确定性初态作全局去重。
        if block.audit_unit_id in audit_unit_ids:
            raise ValueError("Formal paired-block audit-unit identifiers must be unique.")
        audit_unit_ids.add(block.audit_unit_id)
        random_key = next(iter(block.arms.values())).episode_random_key
        expected_random_key = _derive_r015_random_key(
            audit_unit_id=block.audit_unit_id,
            partner_prototype_id=block.prototype_id,
            episode_seed=block.episode_seed,
            purpose="actual_episode",
            environment_step=0,
            branch_index=0,
        )
        if random_key != expected_random_key:
            raise ValueError(
                "A formal block actual-episode key does not match the frozen derivation."
            )
        block_random_keys = {
            random_key,
            block.planning_random_stream_key,
            block.score_random_stream_key,
        }
        if len(block_random_keys) != 3 or block_random_keys & independence_keys:
            raise ValueError("Formal paired blocks reused a controller random stream.")
        independence_keys.update(block_random_keys)
        for comparison in block.safety_comparisons:
            if comparison.random_stream_key in safety_random_keys:
                raise ValueError("Formal safety comparisons reused a random stream.")
            safety_random_keys.add(comparison.random_stream_key)
            if safety_branch_keys.intersection(comparison.branch_keys):
                raise ValueError("Formal safety comparisons reused a branch random key.")
            safety_branch_keys.update(comparison.branch_keys)
        by_prototype[block.prototype_id].append(block)
        if block.prototype_id in by_round[block.round_index]:
            raise ValueError("A formal round repeats one prototype.")
        by_round[block.round_index][block.prototype_id] = block
    if independence_keys & safety_random_keys:
        raise ValueError("Formal episodes and safety checks share a random stream.")
    if independence_keys & safety_branch_keys:
        raise ValueError("Formal episodes and safety checks share a branch random key.")
    if len(safety_random_keys) > (
        preregistration.statistics.maximum_safety_comparisons
    ):
        raise ValueError("Formal data exceed the frozen number of safety comparisons.")
    if any(len(rows) != expected_rounds for rows in by_prototype.values()):
        raise ValueError("Every fixed prototype must contribute the same block count.")
    if any(set(rows) != set(support_members) for rows in by_round.values()):
        raise ValueError("Every formal round must contain one block per prototype.")

    total = len(blocks)
    cost_estimate = math.fsum(block.delta_cost for block in blocks) / total
    response_round_averages = tuple(
        math.fsum(block.delta_response for block in by_round[round_index].values())
        / preregistration.statistics.partner_prototype_count
        for round_index in range(1, expected_rounds + 1)
    )
    net_round_averages = tuple(
        math.fsum(block.delta_net for block in by_round[round_index].values())
        / preregistration.statistics.partner_prototype_count
        for round_index in range(1, expected_rounds + 1)
    )
    response_estimate = math.fsum(response_round_averages) / expected_rounds
    net_estimate = math.fsum(net_round_averages) / expected_rounds
    if not math.isclose(
        net_estimate,
        response_estimate - cost_estimate,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Aggregate Delta identity failed.")
    difference_limit = preregistration.statistics.paired_difference_absolute_bound
    response_interval = maurer_pontil_empirical_bernstein_interval(
        response_round_averages,
        alpha=preregistration.statistics.alpha_eb_response,
        lower_bound=-difference_limit,
        upper_bound=difference_limit,
    )
    net_interval = maurer_pontil_empirical_bernstein_interval(
        net_round_averages,
        alpha=preregistration.statistics.alpha_eb_net,
        lower_bound=-difference_limit,
        upper_bound=difference_limit,
    )
    firing_counts = {
        prototype_id: sum(block.probe_executed for block in rows)
        for prototype_id, rows in by_prototype.items()
    }
    for checkpoint_decision in kill_checkpoint_decisions:
        recomputed_checkpoint_counts = {
            prototype_id: sum(
                block.probe_executed
                for block in rows
                if block.round_index <= checkpoint_decision.checkpoint_round_count
            )
            for prototype_id, rows in by_prototype.items()
        }
        if dict(checkpoint_decision.firing_counts_by_prototype) != (
            recomputed_checkpoint_counts
        ):
            raise ValueError("Firing counts do not match the validated block records.")
    verdict = decide_r015(
        delta_net=net_interval,
        delta_response=response_interval,
        practical_margin=preregistration.statistics.practical_margin,
    )
    return R015Decision(
        verdict=verdict,
        delta_response=response_interval,
        delta_cost_estimate=cost_estimate,
        delta_net=net_interval,
        practical_margin=preregistration.statistics.practical_margin,
        paired_block_count=total,
        formal_round_count=expected_rounds,
        firing_counts_by_prototype=firing_counts,
        registered_negative_reason=(
            "effect_interval_upper_below_margin"
            if verdict == "REGISTERED_NEGATIVE"
            else None
        ),
        full_state_upper_bound_status=preregistration.full_state_upper_bound_status,
        substrate_negative_claim_allowed=False,
    )


def _load_firing_count_checkpoint_decisions(
    preregistration: R015Preregistration,
    support_members: Mapping[str, Mapping[str, Any]],
) -> tuple[R015KillDecision, ...]:
    paths = preregistration.firing_count_checkpoint_paths
    if paths is None or len(paths) != len(
        preregistration.statistics.kill_checkpoint_rounds
    ):
        raise ValueError("The firing-count checkpoint path list is incomplete.")
    decisions: list[R015KillDecision] = []
    previous_counts = {prototype_id: 0 for prototype_id in support_members}
    for checkpoint_index, (expected_checkpoint, path) in enumerate(
        zip(preregistration.statistics.kill_checkpoint_rounds, paths, strict=True)
    ):
        if not path.is_file():
            raise ValueError("A frozen firing-count checkpoint record is missing.")
        decision = evaluate_r015_firing_count_checkpoint(
            _load_json_or_yaml(path),
            statistics=preregistration.statistics,
            expected_prototype_ids=tuple(support_members),
        )
        if decision.checkpoint_round_count != expected_checkpoint:
            raise ValueError("A firing-count file is bound to the wrong checkpoint.")
        if any(
            decision.firing_counts_by_prototype[prototype_id]
            < previous_counts[prototype_id]
            for prototype_id in support_members
        ):
            raise ValueError("Cumulative firing counts decreased between checkpoints.")
        previous_counts = dict(decision.firing_counts_by_prototype)
        decisions.append(decision)
        if decision.registered_negative_reason == "firing_count_upper_below_margin":
            if any(later_path.is_file() for later_path in paths[checkpoint_index + 1 :]):
                raise ValueError(
                    "A later checkpoint exists after the audit should have stopped."
                )
            return tuple(decisions)
    return tuple(decisions)


def load_and_evaluate_r015_firing_checkpoint(
    preregistration_path: str | Path,
    *,
    checkpoint_round: int,
) -> R015KillDecision:
    """Read only cumulative firing counts at one frozen checkpoint."""

    preregistration = load_r015_preregistration(preregistration_path)
    support_members, completed_paths = preregistration.require_firing_checkpoint_ready(
        checkpoint_round
    )
    previous_counts = {prototype_id: 0 for prototype_id in support_members}
    current_decision: R015KillDecision | None = None
    for expected_checkpoint, path in zip(
        preregistration.statistics.kill_checkpoint_rounds[: len(completed_paths)],
        completed_paths,
        strict=True,
    ):
        decision = evaluate_r015_firing_count_checkpoint(
            _load_json_or_yaml(path),
            statistics=preregistration.statistics,
            expected_prototype_ids=tuple(support_members),
        )
        if decision.checkpoint_round_count != expected_checkpoint:
            raise ValueError("A firing-count file is bound to the wrong checkpoint.")
        if any(
            decision.firing_counts_by_prototype[prototype_id]
            < previous_counts[prototype_id]
            for prototype_id in support_members
        ):
            raise ValueError("Cumulative firing counts decreased between checkpoints.")
        if current_decision is not None and current_decision.registered_negative_reason == (
            "firing_count_upper_below_margin"
        ):
            raise ValueError("A later checkpoint exists after the audit should have stopped.")
        previous_counts = dict(decision.firing_counts_by_prototype)
        current_decision = decision
    if current_decision is None or current_decision.checkpoint_round_count != checkpoint_round:
        raise ValueError("The requested firing-count checkpoint was not read.")
    return current_decision


def _read_dataset_bound_formal_ledger(
    dataset: Mapping[str, Any],
    *,
    field: str,
) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    """Read one dataset-bound ledger from its exact path and byte digest."""

    binding = _require_mapping(dataset.get(field), field=field)
    if set(binding) != {"path", "sha256"} or not _is_sha256(
        binding.get("sha256")
    ):
        raise ValueError(f"Formal dataset has an invalid {field} binding.")
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Formal dataset {field} path is missing or not absolute.")
    payload = path.read_bytes()
    if _bytes_sha256(payload) != binding["sha256"]:
        raise ValueError(f"Formal dataset {field} digest changed.")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"Formal dataset {field} has an empty ledger line.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Formal dataset {field} has an incomplete line {line_number}."
            ) from error
        rows.append(_require_mapping(row, field=f"{field} line {line_number}"))
    return path.resolve(), tuple(rows)


def _validate_final_formal_execution_ledgers(
    dataset: Mapping[str, Any],
    *,
    raw_blocks: Sequence[Any],
    expected_sampling_schedule: FormalSamplingSchedule,
    prototype_order: Sequence[str],
    expected_runtime_binding_sha256: str,
) -> str:
    """Independently bind final blocks to the complete mechanical-attempt history."""

    runtime_binding_sha256 = dataset.get("runtime_binding_sha256")
    if not _is_sha256(expected_runtime_binding_sha256) or not _is_sha256(
        runtime_binding_sha256
    ) or runtime_binding_sha256 != expected_runtime_binding_sha256:
        raise ValueError("Formal dataset lacks its frozen runtime binding.")
    ledger_path, ledger_rows = _read_dataset_bound_formal_ledger(
        dataset,
        field="formal_block_ledger",
    )
    invalid_path, invalid_rows = _read_dataset_bound_formal_ledger(
        dataset,
        field="mechanically_invalid_attempts",
    )
    if ledger_path == invalid_path or ledger_path.parent != invalid_path.parent:
        raise ValueError("Formal execution ledgers must be distinct sibling files.")
    if ledger_path.name != "formal_block_ledger.jsonl" or invalid_path.name != (
        "mechanically_invalid_attempts.jsonl"
    ):
        raise ValueError("Formal execution ledger filenames changed.")
    if len(ledger_rows) != len(expected_sampling_schedule) or len(raw_blocks) != len(
        ledger_rows
    ):
        raise ValueError("Formal execution ledgers do not cover every frozen coordinate.")

    schedule_by_coordinate = {
        (round_index, audit_unit_id, prototype_id, ego_position): episode_seeds
        for (
            round_index,
            audit_unit_id,
            prototype_id,
            episode_seeds,
            ego_position,
        ) in expected_sampling_schedule
    }
    prototype_rank = {
        str(prototype_id): rank
        for rank, prototype_id in enumerate(prototype_order)
    }
    if len(prototype_rank) != len(prototype_order) or set(prototype_rank) != {
        prototype_id
        for _, _, prototype_id, _, _ in expected_sampling_schedule
    }:
        raise ValueError("Formal ledger prototype order differs from frozen support.")
    expected_coordinate_order = tuple(
        sorted(
            schedule_by_coordinate,
            key=lambda item: (item[0], prototype_rank[item[2]]),
        )
    )
    invalid_by_sequence: dict[int, list[Mapping[str, Any]]] = {}
    previous_sequence = -1
    invalid_schema_fields = {
        "schema_version",
        "sequence_index",
        "round_index",
        "audit_unit_id",
        "prototype_id",
        "episode_seed",
        "ego_position",
        "mechanical_attempt_index",
        "reason",
        "schedule_bound_seed",
        "outcome_field_read",
        "runtime_binding_sha256",
    }
    for row in invalid_rows:
        if set(row) != invalid_schema_fields or row.get("schema_version") != (
            "path_c_r015_mechanical_replacement_attempt_v2"
        ):
            raise ValueError("Formal invalid-attempt ledger has the wrong schema.")
        sequence_index = _nonnegative_int(
            row.get("sequence_index"), field="invalid sequence_index"
        )
        if sequence_index >= len(ledger_rows) or sequence_index < previous_sequence:
            raise ValueError("Formal invalid attempts are outside execution order.")
        previous_sequence = sequence_index
        invalid_by_sequence.setdefault(sequence_index, []).append(row)

    observed_coordinates: set[tuple[int, str, str, int]] = set()
    ledger_schema_fields = {
        "schema_version",
        "sequence_index",
        "round_index",
        "audit_unit_id",
        "prototype_id",
        "episode_seed",
        "ego_position",
        "mechanical_attempt_index",
        "firing_indicator",
        "block_path",
        "block_sha256",
        "runtime_binding_sha256",
    }
    block_directory = ledger_path.parent / "paired_blocks"
    for sequence_index, (row, raw_block_value) in enumerate(
        zip(ledger_rows, raw_blocks, strict=True)
    ):
        if set(row) != ledger_schema_fields or row.get("schema_version") != (
            "path_c_r015_formal_block_ledger_v2"
        ):
            raise ValueError("Formal block ledger has the wrong schema.")
        if _nonnegative_int(
            row.get("sequence_index"), field="formal sequence_index"
        ) != sequence_index:
            raise ValueError("Formal block ledger sequence is not contiguous.")
        coordinate = (
            _positive_int(row.get("round_index"), field="formal round_index"),
            str(row.get("audit_unit_id", "")),
            str(row.get("prototype_id", "")),
            _nonnegative_int(row.get("ego_position"), field="formal ego_position"),
        )
        if coordinate != expected_coordinate_order[sequence_index]:
            raise ValueError("Formal block ledger changed the frozen execution order.")
        attempt_seeds = schedule_by_coordinate.get(coordinate)
        if attempt_seeds is None or coordinate in observed_coordinates:
            raise ValueError("Formal block ledger changed a frozen coordinate.")
        observed_coordinates.add(coordinate)
        selected_attempt_index = _nonnegative_int(
            row.get("mechanical_attempt_index"),
            field="formal mechanical_attempt_index",
        )
        invalid_prefix = invalid_by_sequence.get(sequence_index, [])
        if selected_attempt_index != len(invalid_prefix) or not (
            selected_attempt_index < len(attempt_seeds)
        ):
            raise ValueError(
                "Formal success does not immediately follow its complete invalid prefix."
            )
        for attempt_index, invalid_row in enumerate(invalid_prefix):
            expected_invalid_identity = (
                coordinate[0],
                coordinate[1],
                coordinate[2],
                attempt_seeds[attempt_index],
                coordinate[3],
                attempt_index,
            )
            observed_invalid_identity = (
                _positive_int(
                    invalid_row.get("round_index"), field="invalid round_index"
                ),
                str(invalid_row.get("audit_unit_id", "")),
                str(invalid_row.get("prototype_id", "")),
                _nonnegative_int(
                    invalid_row.get("episode_seed"), field="invalid episode_seed"
                ),
                _nonnegative_int(
                    invalid_row.get("ego_position"), field="invalid ego_position"
                ),
                _nonnegative_int(
                    invalid_row.get("mechanical_attempt_index"),
                    field="invalid mechanical_attempt_index",
                ),
            )
            if observed_invalid_identity != expected_invalid_identity or (
                invalid_row.get("sequence_index") != sequence_index
            ) or invalid_row.get("schedule_bound_seed") is not True or (
                invalid_row.get("outcome_field_read") is not False
            ) or invalid_row.get("runtime_binding_sha256") != (
                runtime_binding_sha256
            ) or not str(invalid_row.get("reason", "")):
                raise ValueError(
                    "Formal invalid-attempt prefix changed its seed or runtime identity."
                )

        expected_seed = attempt_seeds[selected_attempt_index]
        if _nonnegative_int(
            row.get("episode_seed"), field="formal episode_seed"
        ) != expected_seed or row.get(
            "runtime_binding_sha256"
        ) != runtime_binding_sha256 or not isinstance(
            row.get("firing_indicator"), bool
        ) or not _is_sha256(row.get("block_sha256")):
            raise ValueError("Formal block ledger changed its selected attempt binding.")
        block_path = Path(str(row.get("block_path", ""))).resolve()
        try:
            block_path.relative_to(block_directory.resolve())
        except ValueError as error:
            raise ValueError("Formal block ledger points outside its block directory.") from error
        if not block_path.is_file():
            raise ValueError("Formal block ledger block digest is missing or changed.")
        block_bytes = block_path.read_bytes()
        if _bytes_sha256(block_bytes) != row["block_sha256"]:
            raise ValueError("Formal block ledger block digest is missing or changed.")
        persisted_block = _parse_json_or_yaml_bytes(block_path, block_bytes)
        raw_block = _require_mapping(
            raw_block_value, field=f"paired_blocks item {sequence_index}"
        )
        expected_block_identity = (
            coordinate[0],
            coordinate[1],
            coordinate[2],
            expected_seed,
            coordinate[3],
            selected_attempt_index,
        )
        observed_block_identity = (
            _positive_int(
                persisted_block.get("round_index"), field="block round_index"
            ),
            str(persisted_block.get("audit_unit_id", "")),
            str(persisted_block.get("prototype_id", "")),
            _nonnegative_int(
                persisted_block.get("episode_seed"), field="block episode_seed"
            ),
            _nonnegative_int(
                persisted_block.get("ego_position"), field="block ego_position"
            ),
            _nonnegative_int(
                persisted_block.get("mechanical_attempt_index"),
                field="block mechanical_attempt_index",
            ),
        )
        if observed_block_identity != expected_block_identity or persisted_block.get(
            "runtime_binding_sha256"
        ) != runtime_binding_sha256 or persisted_block.get(
            "firing_indicator"
        ) is not row["firing_indicator"] or persisted_block != raw_block:
            raise ValueError(
                "Formal dataset block differs from its persisted ledger-bound block."
            )
    if observed_coordinates != set(schedule_by_coordinate):
        raise ValueError("Formal execution ledgers do not match the frozen schedule.")
    return str(runtime_binding_sha256)


def load_and_adjudicate_r015(
    preregistration_path: str | Path,
    *,
    expected_runtime_binding_sha256: str | None = None,
) -> R015Decision | R015KillDecision:
    """Return an early count verdict or perform the single final effect readout."""

    preregistration = load_r015_preregistration(preregistration_path)
    support_members = preregistration._require_frozen_support()
    if preregistration.formal_view_record_path is None:
        raise ValueError("The formal-view consumption record path is pending.")
    if preregistration.formal_view_record_path.exists():
        raise ValueError("The single R015 formal view has already been consumed.")
    kill_checkpoint_decisions = _load_firing_count_checkpoint_decisions(
        preregistration, support_members
    )
    for kill_decision in kill_checkpoint_decisions:
        if kill_decision.registered_negative_reason == (
            "firing_count_upper_below_margin"
        ):
            return kill_decision
    if not _is_sha256(expected_runtime_binding_sha256):
        raise ValueError(
            "The final R015 effect view requires the current frozen runtime binding."
        )
    support_members = preregistration.require_formal_ready()
    expected_sampling_schedule = _load_formal_sampling_schedule(
        preregistration, support_members
    )
    trace_replay_verifier = _load_trace_replay_verifier(preregistration)
    safety_branch_evidence_verifier = _load_safety_branch_evidence_verifier(
        preregistration
    )
    decision_evidence_verifier = _load_decision_evidence_verifier(preregistration)
    if preregistration.formal_dataset_path is None or not (
        preregistration.formal_dataset_path.is_file()
    ):
        raise ValueError("The single final effect dataset is pending.")
    assert preregistration.formal_view_record_path is not None
    assert preregistration.support_report_path is not None
    support_report_sha256 = _file_sha256(preregistration.support_report_path)
    # Exclusive creation occurs before the first dataset byte is read. A failed
    # adjudication therefore still consumes the only formal view, which is fail-closed.
    with preregistration.formal_view_record_path.open(
        "x", encoding="utf-8"
    ) as record_handle:
        dataset_bytes = preregistration.formal_dataset_path.read_bytes()
        consumption_record = {
            "schema_version": "path_c_r015_formal_effect_view_receipt_v3",
            "experiment_id": R015_EXPERIMENT_ID,
            "formal_effect_look_number": 1,
            "formal_effect_look_round": preregistration.statistics.n_rounds_max,
            "preregistration_sha256": preregistration.source_sha256,
            "formal_dataset_path": str(preregistration.formal_dataset_path),
            "formal_dataset_sha256": _bytes_sha256(dataset_bytes),
            "support_report_sha256": support_report_sha256,
            "status": "consumed_before_adjudication",
        }
        json.dump(
            consumption_record,
            record_handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        record_handle.write("\n")
        record_handle.flush()
    dataset = _parse_json_or_yaml_bytes(
        preregistration.formal_dataset_path, dataset_bytes
    )
    if dataset.get("schema_version") != R015_FORMAL_DATASET_SCHEMA:
        raise ValueError("Unsupported R015 formal dataset.")
    if dataset.get("experiment_id") != R015_EXPERIMENT_ID:
        raise ValueError("Formal dataset experiment identifier changed.")
    if dataset.get("analysis_kind") != "effect_final":
        raise ValueError("The final dataset must request the single effect look.")
    if _positive_int(
        dataset.get("formal_effect_look_number"), field="formal_effect_look_number"
    ) != 1 or _nonnegative_int(
        dataset.get("prior_formal_effect_look_count"),
        field="prior_formal_effect_look_count",
    ) != 0:
        raise ValueError("R015 permits exactly one formal effect look.")
    if _positive_int(
        dataset.get("formal_effect_look_round"), field="formal_effect_look_round"
    ) != preregistration.statistics.n_rounds_max:
        raise ValueError("The formal effect look is not at N_rounds_max.")
    if dataset.get("pilot_data_included") is not False:
        raise ValueError("R015 pilot data may not enter the formal interval.")
    if dataset.get("preregistration_sha256") != preregistration.source_sha256:
        raise ValueError("Formal dataset binds a different preregistration.")
    if dataset.get("support_report_sha256") != support_report_sha256:
        raise ValueError("Formal dataset binds a different partner-support report.")
    checkpoint_hashes = _require_sequence(
        dataset.get("firing_count_checkpoint_sha256"),
        field="firing_count_checkpoint_sha256",
    )
    expected_checkpoint_hashes = [
        _file_sha256(path)
        for path in (preregistration.firing_count_checkpoint_paths or ())
    ]
    if list(checkpoint_hashes) != expected_checkpoint_hashes:
        raise ValueError("The effect dataset binds different firing-count checkpoints.")
    raw_blocks = _require_sequence(dataset.get("paired_blocks"), field="paired_blocks")
    runtime_binding_sha256 = _validate_final_formal_execution_ledgers(
        dataset,
        raw_blocks=raw_blocks,
        expected_sampling_schedule=expected_sampling_schedule,
        prototype_order=tuple(
            candidate.candidate_id
            for candidate in preregistration.support_spec.candidates
        ),
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
    )
    blocks = tuple(
        validate_r015_paired_block(
            _require_mapping(item, field="paired_blocks item"),
            preregistration=preregistration,
            support_members=support_members,
            trace_replay_verifier=trace_replay_verifier,
            safety_branch_evidence_verifier=safety_branch_evidence_verifier,
            decision_evidence_verifier=decision_evidence_verifier,
            expected_runtime_binding_sha256=runtime_binding_sha256,
        )
        for item in raw_blocks
    )
    return _aggregate_validated_r015_blocks(
        blocks,
        preregistration=preregistration,
        support_members=support_members,
        expected_sampling_schedule=expected_sampling_schedule,
        kill_checkpoint_decisions=kill_checkpoint_decisions,
    )
