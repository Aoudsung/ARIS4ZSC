"""Remote stage implementations that bind the Path C core to OvercookedV2."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.path_c.contracts.records import (
    CheckpointMetadata,
    DecisionRecord,
    EpisodeRecord,
    MetricRow,
)
from src.path_c.contracts.outer_units import load_outer_units_manifest
from src.path_c.model.adaptation_model import build_model, initialize_from_official
from src.path_c.model.checkpoint import (
    MANIFEST_FILE,
    WEIGHTS_FILE,
    load_checkpoint,
    save_checkpoint,
    tree_sha256,
)
from src.path_c.pipeline.run import canonical_sha256
from src.path_c.pipeline.stages import run_pool_check, write_stage_summary
from src.path_c.probe.calibration import calibrate, write_calibration_artifacts
from src.path_c.training.adaptation import (
    adaptation_loss,
    adaptation_update,
    environment_minibatch,
    environment_minibatches,
    make_adaptation_optimizer,
)
from src.path_c.training.metrics import (
    JsonlLedger,
    MetricWriter,
    effective_counts_from_rows,
)
from src.path_c.training.prefit import make_prefit_optimizer, prefit_loss, prefit_update
from src.path_c.training.rollout import (
    RolloutCallbacks,
    initialize_rollout,
    make_compiled_collector,
)

from .env_dock import (
    OvercookedV2VectorEnvironment,
    event_values,
    verify_environment_startup_contracts,
    visible_goal_safe_action_mask,
)
from .official_dock import (
    EXPLICIT_OFFICIAL_LEAF_MAPPING,
    OfficialBackboneDock,
    load_frozen_partner_pool,
    make_batched_partner_step,
    validate_checkpoint_reference,
    verify_initialized_actor_critic_parity,
)
from .response_dock import registered_response_vocabulary, response_tokens


def _host_scalar(value: Any) -> float:
    return float(np.asarray(value))


def _host_int(value: Any) -> int:
    return int(np.asarray(value))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _startup_guard_report(
    environment_guards: Mapping[str, Any],
    official_parity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require every registered startup guard to report an explicit pass."""

    if not isinstance(environment_guards, Mapping) or not isinstance(
        official_parity, Mapping
    ):
        raise RuntimeError("Startup guard results must be mappings.")
    sections: dict[str, Mapping[str, Any]] = {}
    for name in ("observation_layout", "delivery_counters"):
        value = environment_guards.get(name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Startup guard result is missing: {name}.")
        sections[name] = dict(value)
    sections["official_warm_start_parity"] = dict(official_parity)
    all_passed = all(section.get("passed") is True for section in sections.values())
    if not all_passed:
        failed = sorted(
            name for name, section in sections.items() if section.get("passed") is not True
        )
        raise RuntimeError("Startup guard did not pass: " + ", ".join(failed) + ".")
    return {
        "schema_version": "path_c_startup_guards_v1",
        "all_passed": all_passed,
        **sections,
    }


def _completed_return(batch: Any) -> float | None:
    values = np.asarray(batch.completed_episode_returns, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(finite.mean())


def _metric_row(state: Any, batch: Any, metrics: Mapping[str, Any]) -> MetricRow:
    loss_values = {
        str(name): _host_scalar(value)
        for name, value in metrics.items()
        if name not in {"gradient_norm", "parameters_finite"}
    }
    probe_count = int(np.asarray(batch.probed).sum())
    decision_count = int(np.asarray(batch.probed).size)
    return MetricRow(
        environment_steps=_host_int(state.effective_environment_steps),
        completed_episodes=_host_int(state.completed_episodes),
        losses=loss_values,
        raw_step_return=float(np.asarray(batch.rewards, dtype=np.float64).mean()),
        completed_episode_return=_completed_return(batch),
        probe_count=probe_count,
        probe_rate=float(probe_count / max(decision_count, 1)),
        gradient_norms={"all": _host_scalar(metrics["gradient_norm"])},
        parameters_finite=bool(np.asarray(metrics["parameters_finite"])),
    )


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _read_metric_counts(path: Path) -> Mapping[str, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return effective_counts_from_rows(rows)


def _family_episode_coverage(path: Path, *, expected_episodes: int) -> Mapping[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_episodes:
        raise RuntimeError("Family schedule episode ledger changed its row count.")
    family_counts = {
        str(index): sum(int(row.get("prototype_index", -1)) == index for row in rows)
        for index in range(4)
    }
    member_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("partner_member_id", row.get("partner_id")))
        member_counts[key] = member_counts.get(key, 0) + 1
    if set(family_counts.values()) != {expected_episodes // 4}:
        raise RuntimeError("The four partner families were not used equally.")
    grouped: dict[int, list[int]] = {}
    for prototype_index in range(4):
        counts = [
            count
            for member, count in member_counts.items()
            if any(
                int(row.get("prototype_index", -1)) == prototype_index
                and str(row.get("partner_member_id", row.get("partner_id"))) == member
                for row in rows
            )
        ]
        grouped[prototype_index] = counts
        if not counts or max(counts) - min(counts) > 1:
            raise RuntimeError("Members within one partner family are not balanced.")
    return {
        "family_episode_counts": family_counts,
        "member_episode_counts": dict(sorted(member_counts.items())),
    }


def _completed_safe_opportunity_counts(
    *,
    episode_ids: np.ndarray,
    episode_steps: np.ndarray,
    has_safe_candidate: np.ndarray,
    episode_completed: np.ndarray,
) -> Mapping[tuple[int, int], int]:
    """Count safe opportunities after a complete device segment is collected."""

    arrays = (
        np.asarray(episode_ids),
        np.asarray(episode_steps),
        np.asarray(has_safe_candidate, dtype=np.bool_),
        np.asarray(episode_completed, dtype=np.bool_),
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Episode identifiers, steps, opportunities, and completions must align.")
    if arrays[0].ndim != 2:
        raise ValueError("Completed rollout records must have [time, environment] shape.")

    completed: dict[tuple[int, int], int] = {}
    unused_time_count, environment_count = arrays[0].shape
    for environment_index in range(environment_count):
        episode_ids = arrays[0][:, environment_index]
        unique_ids, inverse = np.unique(episode_ids, return_inverse=True)
        opportunity_counts = np.bincount(
            inverse,
            weights=arrays[2][:, environment_index].astype(np.int64),
            minlength=len(unique_ids),
        ).astype(np.int64)
        for local_index in range(len(unique_ids)):
            locations = np.flatnonzero(inverse == local_index)
            observed_steps = arrays[1][locations, environment_index]
            if not np.array_equal(
                observed_steps,
                np.arange(len(locations), dtype=observed_steps.dtype),
            ):
                if int(observed_steps[0]) != 0:
                    raise RuntimeError(
                        "Episode opportunity records must begin at episode step zero."
                    )
                raise RuntimeError(
                    "Episode opportunity records must have consecutive steps."
                )
        for time_index in np.flatnonzero(arrays[3][:, environment_index]):
            completed[(int(time_index), environment_index)] = int(
                opportunity_counts[inverse[time_index]]
            )
    return completed


def _write_rollout_records(
    *,
    decision_ledger: JsonlLedger,
    episode_ledger: JsonlLedger,
    batch: Any,
    controller: str,
    partner_pool: Sequence[Any],
    maximum_probe_budget: int,
    record_all_decisions: bool = True,
) -> None:
    """Move one completed device segment into host decision and episode ledgers."""

    arrays = {
        name: np.asarray(getattr(batch, name))
        for name in (
            "episode_ids",
            "episode_steps",
            "base_actions",
            "actions",
            "probed",
            "actor_owned_action",
            "candidate_actions",
            "decision_scores",
            "information_scores",
            "has_safe_candidate",
            "budget_remaining",
            "chosen_j_use",
            "chosen_j_mask",
            "value_base",
            "value_mask",
            "chosen_s_seq",
            "chosen_response_information",
            "partner_indices",
            "ego_seats",
            "episode_completed",
            "completed_episode_returns",
            "completed_episode_probe_counts",
            "correct_deliveries",
            "wrong_deliveries",
            "indicator_costs",
        )
    }
    arrays["partner_member_indices"] = np.asarray(
        getattr(batch, "partner_member_indices", batch.partner_indices)
    )
    completed_safe_opportunities = _completed_safe_opportunity_counts(
        episode_ids=arrays["episode_ids"],
        episode_steps=arrays["episode_steps"],
        has_safe_candidate=arrays["has_safe_candidate"],
        episode_completed=arrays["episode_completed"],
    )
    decisions = []
    episodes = []
    decision_mask = (
        np.ones_like(arrays["probed"], dtype=np.bool_)
        if record_all_decisions
        else arrays["probed"].astype(np.bool_)
    )
    for time_index, environment_index in np.argwhere(decision_mask):
        index = (int(time_index), int(environment_index))
        probed = bool(arrays["probed"][index])
        if controller == "registered_response_sequential_branch_v1":
            score_components = {
                "j_use": _finite_or_none(arrays["chosen_j_use"][index]),
                "j_mask": _finite_or_none(arrays["chosen_j_mask"][index]),
                "v_base": _finite_or_none(arrays["value_base"][index]),
                "v_mask": _finite_or_none(arrays["value_mask"][index]),
                "s_seq": _finite_or_none(arrays["chosen_s_seq"][index]),
            }
        elif controller == "generic_response_information":
            score_components = {
                "response_information": _finite_or_none(
                    arrays["chosen_response_information"][index]
                )
            }
        else:
            score_components = {}
        decisions.append(
            DecisionRecord(
                episode_id=int(arrays["episode_ids"][index]),
                environment_index=int(environment_index),
                episode_step=int(arrays["episode_steps"][index]),
                base_action=int(arrays["base_actions"][index]),
                chosen_action=int(arrays["actions"][index]),
                probed=probed,
                actor_owned_action=bool(arrays["actor_owned_action"][index]),
                candidate=(
                    int(arrays["candidate_actions"][index])
                    if controller != "off"
                    and bool(arrays["has_safe_candidate"][index])
                    else None
                ),
                score_components=score_components,
                budget_remaining=int(arrays["budget_remaining"][index]),
                controller=controller,
            ).to_mapping()
        )
    for time_index, environment_index in np.argwhere(
        arrays["episode_completed"].astype(np.bool_)
    ):
        index = (int(time_index), int(environment_index))
        partner_index = int(arrays["partner_member_indices"][index])
        partner = partner_pool[partner_index]
        probe_count = int(arrays["completed_episode_probe_counts"][index])
        safe_opportunities = completed_safe_opportunities[index]
        episode = EpisodeRecord(
                episode_id=int(arrays["episode_ids"][index]),
                partner_id=str(partner.training_run_id),
                ego_seat=int(arrays["ego_seats"][index]),
                raw_return=float(arrays["completed_episode_returns"][index]),
                environment_steps=int(arrays["episode_steps"][index]) + 1,
                probe_count=probe_count,
                safe_candidate_opportunity_count=safe_opportunities,
                probe_trigger_rate=(
                    probe_count / safe_opportunities if safe_opportunities else 0.0
                ),
                maximum_probe_budget=maximum_probe_budget,
                probe_budget_usage_rate=probe_count / maximum_probe_budget,
                correct_delivery_count=int(arrays["correct_deliveries"][index]),
                wrong_delivery_count=int(arrays["wrong_deliveries"][index]),
                indicator_cost=float(arrays["indicator_costs"][index]),
            ).to_mapping()
        if hasattr(partner, "family_id"):
            episode["partner_family_id"] = str(partner.family_id)
        if hasattr(partner, "member_id"):
            episode["partner_member_id"] = str(partner.member_id)
        episode["prototype_index"] = int(arrays["partner_indices"][index])
        episodes.append(episode)
    decision_ledger.append(decisions)
    episode_ledger.append(episodes)


class PathCStageRuntime:
    """Own one lazily built remote model, environment, and frozen partner pool."""

    def __init__(
        self,
        config: Any,
        *,
        backbone_launch_config: str | Path,
        partner_launch_configs: Sequence[str | Path],
    ) -> None:
        self.config = config
        self.backbone_launch_config = Path(backbone_launch_config).resolve()
        self.partner_launch_configs = tuple(Path(path).resolve() for path in partner_launch_configs)
        self._built: Mapping[str, Any] | None = None

    def stage_runners(self) -> Mapping[str, Any]:
        """Expose training stages only; familiar-partner evaluation is historical."""

        if self.config.is_family_pool:
            return {
                "pool_check": self.pool_check,
                "prefit": self.prefit,
                "training_calibration": self.training_calibration,
                "adaptation": self.adaptation,
                "deployment_calibration": self.deployment_calibration,
            }
        return {
            "pool_check": self.pool_check,
            "prefit": self.prefit,
            "calibration": self.calibration,
            "adaptation": self.adaptation,
        }

    def _decision_ledger_path(self, stage_directory: Path) -> Path:
        """Keep full development traces and compact formal probe evidence."""

        name = (
            "probe_decisions.jsonl.gz"
            if self.config.run_kind == "formal"
            else "decisions.jsonl"
        )
        return stage_directory / name

    def pool_check(self, context: Any) -> Mapping[str, Path]:
        member_launches = (
            tuple(member.launch_config_path for member in self.config.partner_pool)
            if self.config.is_family_pool
            else self.partner_launch_configs
        )
        launch_by_checkpoint = {
            str(self.config.backbone_init.checkpoint_path): self.backbone_launch_config,
            **{
                str(member.checkpoint_path): launch
                for member, launch in zip(
                    self.config.partner_pool,
                    member_launches,
                    strict=True,
                )
            },
        }
        expected_sources: dict[str, tuple[Any, str]] = {}
        binding = self.config.formal_outer_unit
        if binding is not None:
            manifest = load_outer_units_manifest(binding.outer_units_manifest_path)
            if manifest.sha256 != binding.outer_units_manifest_sha256:
                raise ValueError("The resolved formal outer-unit manifest hash changed.")
            unit = manifest.unit(binding.outer_unit_id)
            expected_launches = (
                unit.backbone.launch_config_path,
                *(member.launch_config_path for member in unit.partners),
            )
            if expected_launches != (
                self.backbone_launch_config,
                *self.partner_launch_configs,
            ):
                raise ValueError("The formal runtime changed an outer-unit launch config.")
            for source in unit.sources:
                expected_sources[str(source.checkpoint_path)] = (
                    source,
                    source.flax_weights_sha256,
                )
                for snapshot in source.checkpoint_history:
                    expected_sources[str(snapshot.checkpoint_path)] = (
                        source,
                        snapshot.flax_weights_sha256,
                    )

        def validate(reference: Any) -> Mapping[str, Any]:
            launch = launch_by_checkpoint.get(str(reference.checkpoint_path))
            if launch is None:
                raise ValueError("The pool check has no official launch config for a checkpoint.")
            report = validate_checkpoint_reference(
                reference, launch_config_path=launch
            )
            if binding is not None:
                expected_record = expected_sources.get(str(reference.checkpoint_path))
                if expected_record is None:
                    raise ValueError(
                        "An official checkpoint does not match its formal outer-unit source."
                    )
                expected, expected_weights = expected_record
                if (
                    report.get("training_seed") != expected.training_seed
                    or report.get("layout") != manifest.layout
                    or report.get("family_id") != expected.family_id
                    or report.get("training_run_id") != expected.training_run_id
                    or report.get("flax_weights_sha256")
                    != expected_weights
                ):
                    raise ValueError(
                        "An official checkpoint does not match its formal outer-unit source."
                    )
            return report

        artifacts = dict(run_pool_check(context, validate_checkpoint=validate))
        startup_path = context.stage_directory / "startup_guards.json"
        _atomic_json(startup_path, self._build()["startup_guards"])
        artifacts["startup_guards"] = startup_path
        return artifacts

    def _build(self) -> Mapping[str, Any]:
        if self._built is not None:
            return self._built
        import jax

        environment = OvercookedV2VectorEnvironment.create(
            num_envs=self.config.environment.num_envs,
            layout=self.config.environment.layout,
        )
        dock = OfficialBackboneDock.from_launch_config(self.backbone_launch_config)
        environment_guards = verify_environment_startup_contracts(
            environment,
            action_order=dock.action_order(),
        )
        vocabulary = registered_response_vocabulary()
        model = build_model(
            num_prototypes=self.config.num_prototypes,
            action_count=len(dock.action_order()),
            response_count=vocabulary.size,
            official_dimensions=dock.network_dimensions(),
            model_config=self.config.model,
        )
        initialization_key = jax.random.PRNGKey(self.config.seeds.model_seed)
        observation_key, parameter_key = jax.random.split(initialization_key)
        unused_state, observations = environment.reset(observation_key)
        del unused_state
        example = observations[:, 0, ...][None, ...]
        starts = np.ones((1, environment.num_envs), dtype=np.bool_)
        official_params = dock.official_parameter_tree(self.config.backbone_init)
        params = initialize_from_official(
            model,
            random_key=parameter_key,
            official_params=official_params,
            explicit_leaf_mapping=EXPLICIT_OFFICIAL_LEAF_MAPPING,
            example_observations=example,
            example_episode_start=starts,
        )
        official_parity = verify_initialized_actor_critic_parity(
            dock=dock,
            model=model,
            initialized_params=params,
            official_params=official_params,
            observation_shape=environment.observation_shape,
        )
        partners = load_frozen_partner_pool(
            self.config.partner_pool,
            launch_config_paths=(
                tuple(member.launch_config_path for member in self.config.partner_pool)
                if self.config.is_family_pool
                else self.partner_launch_configs
            ),
        )
        unit = getattr(self.config, "formal_outer_unit", None)
        backbone_policy_id = (
            f"official_backbone_outer_unit_{unit.outer_unit_id:02d}"
            if unit is not None
            else "official_backbone_seed_100"
        )
        backbone_policy = dock.frozen_policy(
            official_params, policy_id=backbone_policy_id
        )
        startup_guards = _startup_guard_report(
            environment_guards,
            official_parity,
        )

        def model_apply(
            current_params: Mapping[str, Any], carry: Any, observations_at_step: Any, episode_start: Any
        ) -> tuple[Any, Mapping[str, Any]]:
            return model.apply(
                {"params": current_params}, carry, observations_at_step, episode_start
            )

        callbacks = RolloutCallbacks(
            model_apply=model_apply,
            partner_step=make_batched_partner_step(partners),
            response_tokens=response_tokens,
            safe_action_mask=functools.partial(
                visible_goal_safe_action_mask, action_order=dock.action_order()
            ),
            event_values=event_values,
        )
        self._built = {
            "environment": environment,
            "dock": dock,
            "vocabulary": vocabulary,
            "partners": partners,
            "partner_audit_members": (
                (
                    SimpleNamespace(
                        training_run_id=(
                            f"current_policy_outer_unit_{unit.outer_unit_id:02d}"
                            if unit is not None
                            else "current_policy"
                        ),
                        member_id="current_policy_rollout_snapshot",
                        family_id="adapted_ego_family",
                    ),
                    *self.config.partner_pool,
                )
                if self.config.is_family_pool
                else self.config.partner_pool
            ),
            "backbone_policy": backbone_policy,
            "model": model,
            "model_apply": model_apply,
            "callbacks": callbacks,
            "initial_params": params,
            "startup_guards": startup_guards,
        }
        return self._built

    def _new_rollout_state(self, *, stage_number: int) -> Any:
        import jax

        built = self._build()
        environment = built["environment"]
        key = jax.random.fold_in(
            jax.random.PRNGKey(self.config.seeds.environment_seed), stage_number
        )
        return initialize_rollout(
            environment=environment,
            model_initial_carry=built["model"].initial_carry,
            partner_initial_carry=built["partners"][0].initial_state,
            random_key=key,
            num_prototypes=self.config.num_prototypes,
            budget_per_episode=self.config.probe.budget_per_episode,
            family_member_counts=(
                (1, 3, 6, 6) if self.config.is_family_pool else None
            ),
        )

    def _collector(
        self,
        *,
        controller: str,
        decision_threshold: float,
        information_threshold: float,
        random_trigger_probability: float,
    ) -> Any:
        built = self._build()
        return make_compiled_collector(
            length=self.config.adaptation.unroll_length,
            environment=built["environment"],
            callbacks=built["callbacks"],
            controller=controller,
            candidate_window=self.config.probe.candidate_window,
            budget_per_episode=self.config.probe.budget_per_episode,
            gamma=self.config.adaptation.gamma,
            gae_lambda=self.config.adaptation.gae_lambda,
            probability_floor=self.config.probe.belief_probability_floor,
            decision_threshold=float(decision_threshold),
            information_threshold=float(information_threshold),
            random_trigger_probability=float(random_trigger_probability),
            num_prototypes=self.config.num_prototypes,
            partner_prototype_lookup=(
                (0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3)
                if self.config.is_family_pool
                else None
            ),
            family_member_counts=(
                (1, 3, 6, 6) if self.config.is_family_pool else None
            ),
            current_policy_partner=self.config.is_family_pool,
        )

    def _checkpoint_metadata(
        self,
        *,
        stage: str,
        context: Any,
        params: Mapping[str, Any],
        effective_environment_steps: int,
        completed_episodes: int,
    ) -> CheckpointMetadata:
        vocabulary = self._build()["vocabulary"]
        shared = stage == "prefit"
        config_hash = (
            canonical_sha256(self.config.shared_stage_mapping())
            if shared
            else self.config.config_sha256
        )
        extra = {
            "stage_input_sha256": context.stage_input_sha256,
            "response_vocabulary": list(vocabulary.tokens),
            "action_order": list(self._build()["dock"].action_order()),
            "resolved_config": (
                self.config.shared_stage_mapping()
                if shared
                else self.config.to_mapping()
            ),
            "model_weights_sha256": tree_sha256(params),
        }
        if not shared:
            calibration_key = (
                "training_calibration.summary"
                if self.config.is_family_pool
                else "calibration.summary"
            )
            calibration = context.preceding_artifacts.get(calibration_key)
            if not isinstance(calibration, Mapping) or not calibration.get("sha256"):
                raise ValueError(
                    "An adaptation checkpoint must bind its training calibration."
                )
            extra[
                (
                    "training_calibration_summary_sha256"
                    if self.config.is_family_pool
                    else "calibration_summary_sha256"
                )
            ] = str(calibration["sha256"])
        return CheckpointMetadata(
            schema_version="path_c_model_checkpoint_metadata_v1",
            stage=stage,
            condition_id=None if shared else self.config.condition_id,
            controller=None if shared else self.config.controller,
            shared_across_conditions=shared,
            run_kind=self.config.run_kind,
            scientific_readout_allowed=self.config.scientific_readout_allowed,
            environment_steps=int(effective_environment_steps),
            completed_episodes=int(completed_episodes),
            response_vocabulary_sha256=vocabulary.sha256,
            backbone_origin_sha256=self.config.backbone_init.flax_weights_sha256,
            config_sha256=config_hash,
            extra=extra,
        )

    def prefit(self, context: Any) -> Mapping[str, Path]:
        import jax
        import jax.numpy as jnp

        built = self._build()
        params = built["initial_params"]
        optimizer, optimizer_state = make_prefit_optimizer(
            params, learning_rate=self.config.prefit.learning_rate
        )
        collector = self._collector(
            controller="off",
            decision_threshold=0.0,
            information_threshold=0.0,
            random_trigger_probability=0.0,
        )
        state = self._new_rollout_state(stage_number=0)

        def update(current_params: Any, current_optimizer_state: Any, batch: Any) -> Any:
            initial_carry = jax.tree_util.tree_map(lambda value: value[0], batch.model_carry)

            def loss_function(candidate_params: Any) -> Any:
                unused_carry, outputs = built["model"].apply(
                    {"params": candidate_params},
                    initial_carry,
                    batch.observations,
                    batch.episode_start,
                )
                del unused_carry
                return prefit_loss(
                    outputs,
                    batch,
                    gamma=self.config.adaptation.gamma,
                    loss_weights=self.config.prefit.loss_weights,
                )

            return prefit_update(
                params=current_params,
                optimizer_state=current_optimizer_state,
                optimizer=optimizer,
                loss_function=loss_function,
            )

        compiled_update = jax.jit(update)
        rollout_size = self.config.environment.num_envs * self.config.adaptation.unroll_length
        rollout_count = self.config.prefit.env_steps // rollout_size
        prefit_minibatch_count = rollout_size // self.config.prefit.batch_size
        writer = MetricWriter(context.stage_directory / "metrics.jsonl", truncate=True)
        decision_path = self._decision_ledger_path(context.stage_directory)
        decision_ledger = JsonlLedger(decision_path, truncate=True)
        episode_ledger = JsonlLedger(
            context.stage_directory / "episodes.jsonl", truncate=True
        )
        last_metrics: Mapping[str, Any] | None = None
        last_metric_row: MetricRow | None = None
        for rollout_index in range(rollout_count):
            state, batch = collector(state=state, params=params)
            _write_rollout_records(
                decision_ledger=decision_ledger,
                episode_ledger=episode_ledger,
                batch=batch,
                controller="off",
                partner_pool=built["partner_audit_members"],
                maximum_probe_budget=self.config.probe.budget_per_episode,
                record_all_decisions=self.config.run_kind != "formal",
            )
            if prefit_minibatch_count == 1:
                minibatches = (batch,)
            else:
                minibatches = environment_minibatches(
                    batch,
                    jnp.arange(self.config.environment.num_envs),
                    prefit_minibatch_count,
                )
            for minibatch in minibatches:
                params, optimizer_state, last_metrics = compiled_update(
                    params, optimizer_state, minibatch
                )
            jax.block_until_ready((params, optimizer_state, last_metrics))
            steps = _host_int(state.effective_environment_steps)
            should_record = (
                steps % self.config.adaptation.metrics_interval_env_steps == 0
            )
            if should_record or rollout_index == rollout_count - 1:
                last_metric_row = _metric_row(state, batch, last_metrics)
            if should_record:
                writer.append(last_metric_row)
            del minibatch, minibatches, batch
        if last_metrics is None or last_metric_row is None:
            raise RuntimeError("Prefit completed no rollout.")
        if not writer.rows or int(writer.rows[-1]["environment_steps"]) != _host_int(
            state.effective_environment_steps
        ):
            writer.append(last_metric_row)
        metrics_path = context.stage_directory / "metrics.jsonl"
        effective = _read_metric_counts(metrics_path)
        if effective != {
            "effective_environment_steps": _host_int(state.effective_environment_steps),
            "completed_episodes": _host_int(state.completed_episodes),
        }:
            raise RuntimeError("Prefit metric artifacts do not match the completed rollout state.")
        episode_path = context.stage_directory / "episodes.jsonl"
        family_coverage = (
            _family_episode_coverage(
                episode_path, expected_episodes=effective["completed_episodes"]
            )
            if self.config.is_family_pool
            else {}
        )
        checkpoint_dir = context.stage_directory / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            params=params,
            metadata=self._checkpoint_metadata(
                stage="prefit",
                context=context,
                params=params,
                effective_environment_steps=effective["effective_environment_steps"],
                completed_episodes=effective["completed_episodes"],
            ),
        )
        summary = write_stage_summary(
            context,
            stage="prefit",
            effective_environment_steps=effective["effective_environment_steps"],
            completed_episodes=effective["completed_episodes"],
            extra={
                "checkpoint_path": str(checkpoint_dir),
                "decision_recording": (
                    "probed_steps_only_gzip"
                    if self.config.run_kind == "formal"
                    else "all_steps_jsonl"
                ),
                **family_coverage,
            },
        )
        return {
            "manifest": checkpoint_dir / MANIFEST_FILE,
            "weights": checkpoint_dir / WEIGHTS_FILE,
            "metrics": metrics_path,
            "decisions": decision_path,
            "episodes": episode_path,
            "summary": summary,
        }

    @staticmethod
    def _preceding_path(context: Any, key: str) -> Path:
        reference = context.preceding_artifacts.get(key)
        if not isinstance(reference, Mapping):
            raise ValueError(f"A preceding pipeline artifact is missing: {key}")
        return Path(str(reference["path"])).resolve()

    def _run_calibration(
        self,
        context: Any,
        *,
        parameter_artifact: str,
        episodes: int,
        stage_number: int,
    ) -> Mapping[str, Mapping[str, str]]:
        manifest_path = self._preceding_path(context, parameter_artifact)
        params, unused_metadata, unused_manifest = load_checkpoint(manifest_path.parent)
        del unused_metadata, unused_manifest
        collector = self._collector(
            controller="off",
            decision_threshold=0.0,
            information_threshold=0.0,
            random_trigger_probability=0.0,
        )
        state = self._new_rollout_state(stage_number=stage_number)
        episodes_per_segment = self.config.environment.num_envs
        if episodes % episodes_per_segment:
            raise ValueError("Calibration episodes must contain complete vector segments.")
        segment_count = episodes // episodes_per_segment
        rows: list[dict[str, Any]] = []
        for unused_index in range(segment_count):
            del unused_index
            state, batch = collector(state=state, params=params)
            probed = np.asarray(batch.probed, dtype=np.bool_)
            if probed.any():
                raise RuntimeError("Calibration must not execute a probe.")
            safe = np.asarray(batch.has_safe_candidate, dtype=np.bool_).reshape(-1)
            decision = np.asarray(batch.decision_scores, dtype=np.float64).reshape(-1)
            information = np.asarray(batch.information_scores, dtype=np.float64).reshape(-1)
            completed = np.asarray(batch.episode_completed, dtype=np.bool_).reshape(-1)
            prototype_indices = np.asarray(
                batch.partner_indices, dtype=np.int64
            ).reshape(-1)
            member_indices = np.asarray(
                batch.partner_member_indices, dtype=np.int64
            ).reshape(-1)
            for index, has_safe in enumerate(safe):
                rows.append(
                    {
                        "decision_index": len(rows),
                        "has_safe_candidate": bool(has_safe),
                        "max_decision_score": float(decision[index]) if has_safe else None,
                        "max_information_score": float(information[index]) if has_safe else None,
                        "probed": False,
                        "episode_completed": bool(completed[index]),
                        "prototype_index": int(prototype_indices[index]),
                        "partner_member_index": int(member_indices[index]),
                    }
                )
        effective_environment_steps = len(rows)
        completed_episodes = sum(bool(row["episode_completed"]) for row in rows)
        if (
            effective_environment_steps != _host_int(state.effective_environment_steps)
            or completed_episodes != _host_int(state.completed_episodes)
            or completed_episodes != episodes
        ):
            raise RuntimeError("Calibration completed-episode count differs from its budget.")
        result = calibrate(
            rows,
            decision_null_quantile=self.config.probe.decision_null_quantile,
            information_quantile=self.config.probe.information_quantile,
            decision_threshold_override=self.config.probe.manual_threshold,
        )
        completed_rows = [row for row in rows if row["episode_completed"]]
        family_coverage = {
            str(index): sum(
                row["prototype_index"] == index for row in completed_rows
            )
            for index in range(self.config.num_prototypes)
        }
        member_coverage = {
            str(index): sum(
                row["partner_member_index"] == index for row in completed_rows
            )
            for index in sorted(
                {row["partner_member_index"] for row in completed_rows}
            )
        }
        if self.config.is_family_pool:
            member_groups = ((0,), (1, 2, 3), tuple(range(4, 10)), tuple(range(10, 16)))
            unbalanced_members = any(
                max(member_coverage.get(str(index), 0) for index in group)
                - min(member_coverage.get(str(index), 0) for index in group)
                > 1
                for group in member_groups
            )
            if (
                set(family_coverage.values()) != {episodes // 4}
                or unbalanced_members
            ):
                raise RuntimeError(
                    "Calibration did not preserve the balanced family schedule."
                )
        artifacts = write_calibration_artifacts(
            context.stage_directory,
            rows,
            result,
            metadata={
                "run_kind": self.config.run_kind,
                "scientific_readout_allowed": self.config.scientific_readout_allowed,
                "effective_environment_steps": effective_environment_steps,
                "completed_episodes": completed_episodes,
                "response_vocabulary_sha256": self._build()["vocabulary"].sha256,
                "parameter_artifact": parameter_artifact,
                "parameter_manifest_sha256": context.preceding_artifacts[
                    parameter_artifact
                ]["sha256"],
                "stage_input_sha256": context.stage_input_sha256,
                "decision_threshold_source": self.config.probe.threshold_source,
                "family_episode_coverage": family_coverage,
                "member_episode_coverage": member_coverage,
            },
        )
        persisted_text = Path(artifacts["rows"]["path"]).read_text(encoding="utf-8")
        persisted_rows = [
            json.loads(line)
            for line in persisted_text.splitlines()
            if line.strip()
        ]
        if len(persisted_rows) != effective_environment_steps or sum(
            bool(row["episode_completed"]) for row in persisted_rows
        ) != completed_episodes:
            raise RuntimeError("Calibration row artifacts changed their effective data counts.")
        return artifacts

    def calibration(self, context: Any) -> Mapping[str, Mapping[str, str]]:
        return self._run_calibration(
            context,
            parameter_artifact="prefit.manifest",
            episodes=self.config.calibration.episodes,
            stage_number=1,
        )

    def training_calibration(
        self, context: Any
    ) -> Mapping[str, Mapping[str, str]]:
        return self._run_calibration(
            context,
            parameter_artifact="prefit.manifest",
            episodes=self.config.calibration.episodes,
            stage_number=1,
        )

    def deployment_calibration(
        self, context: Any
    ) -> Mapping[str, Mapping[str, str]]:
        if self.config.deployment_calibration is None:
            raise ValueError("This configuration has no deployment calibration.")
        return self._run_calibration(
            context,
            parameter_artifact="adaptation.manifest",
            episodes=self.config.deployment_calibration.episodes,
            stage_number=3,
        )

    def backbone_evaluation(self, context: Any) -> Mapping[str, Path]:
        from .evaluation_runtime import run_backbone_evaluation

        return run_backbone_evaluation(
            context=context,
            built=self._build(),
            config=self.config,
        )

    def adaptation(self, context: Any) -> Mapping[str, Path]:
        import jax

        built = self._build()
        prefit_manifest = self._preceding_path(context, "prefit.manifest")
        params, unused_metadata, unused_manifest = load_checkpoint(prefit_manifest.parent)
        del unused_metadata, unused_manifest
        calibration_key = (
            "training_calibration.summary"
            if self.config.is_family_pool
            else "calibration.summary"
        )
        calibration_summary = json.loads(
            self._preceding_path(context, calibration_key).read_text(encoding="utf-8")
        )
        collector = self._collector(
            controller=self.config.controller,
            decision_threshold=float(calibration_summary["decision_threshold"]),
            information_threshold=float(calibration_summary["information_threshold"]),
            random_trigger_probability=float(calibration_summary["random_trigger_probability"]),
        )
        rollout_size = self.config.environment.num_envs * self.config.adaptation.unroll_length
        rollout_count = self.config.adaptation.env_steps // rollout_size
        optimizer_updates = (
            rollout_count
            * self.config.adaptation.update_epochs
            * self.config.adaptation.num_minibatches
        )
        optimizer, optimizer_state = make_adaptation_optimizer(
            params,
            critic_learning_rate=self.config.adaptation.critic_learning_rate,
            actor_learning_rate=self.config.adaptation.actor_learning_rate,
            auxiliary_learning_rate=self.config.adaptation.auxiliary_learning_rate,
            gradient_clip_norm=self.config.adaptation.gradient_clip_norm,
            total_updates=optimizer_updates,
            warmup_fraction=self.config.adaptation.learning_rate_warmup_fraction,
        )
        state = self._new_rollout_state(stage_number=2)

        def update(current_params: Any, current_optimizer_state: Any, batch: Any) -> Any:
            initial_carry = jax.tree_util.tree_map(lambda value: value[0], batch.model_carry)

            def loss_function(candidate_params: Any) -> Any:
                unused_carry, outputs = built["model"].apply(
                    {"params": candidate_params},
                    initial_carry,
                    batch.observations,
                    batch.episode_start,
                )
                del unused_carry
                return adaptation_loss(
                    outputs,
                    batch,
                    gamma=self.config.adaptation.gamma,
                    clip_epsilon=self.config.adaptation.ppo_clip,
                    entropy_coefficient=self.config.adaptation.entropy_coefficient,
                    value_coefficient=self.config.adaptation.value_coefficient,
                    auxiliary_loss_weights=self.config.prefit.loss_weights,
                )

            return adaptation_update(
                params=current_params,
                optimizer_state=current_optimizer_state,
                optimizer=optimizer,
                loss_function=loss_function,
            )

        def update_epoch(
            current_params: Any,
            current_optimizer_state: Any,
            full_batch: Any,
            environment_groups: Any,
        ) -> Any:
            def update_group(carry: Any, indices: Any) -> Any:
                group_params, group_optimizer_state = carry
                minibatch = environment_minibatch(full_batch, indices)
                next_params, next_optimizer_state, metrics = update(
                    group_params, group_optimizer_state, minibatch
                )
                return (next_params, next_optimizer_state), metrics

            (next_params, next_optimizer_state), metric_history = jax.lax.scan(
                update_group,
                (current_params, current_optimizer_state),
                environment_groups,
            )
            final_metrics = jax.tree_util.tree_map(
                lambda values: values[-1], metric_history
            )
            return next_params, next_optimizer_state, final_metrics

        compiled_epoch_update = jax.jit(update_epoch)
        shuffle_key = jax.random.fold_in(
            jax.random.PRNGKey(self.config.seeds.model_seed), 2
        )
        writer = MetricWriter(context.stage_directory / "metrics.jsonl", truncate=True)
        decision_path = self._decision_ledger_path(context.stage_directory)
        decision_ledger = JsonlLedger(decision_path, truncate=True)
        episode_ledger = JsonlLedger(
            context.stage_directory / "episodes.jsonl", truncate=True
        )
        last_metrics: Mapping[str, Any] | None = None
        last_metric_row: MetricRow | None = None
        for rollout_index in range(rollout_count):
            state, batch = collector(state=state, params=params)
            _write_rollout_records(
                decision_ledger=decision_ledger,
                episode_ledger=episode_ledger,
                batch=batch,
                controller=self.config.controller,
                partner_pool=built["partner_audit_members"],
                maximum_probe_budget=self.config.probe.budget_per_episode,
                record_all_decisions=self.config.run_kind != "formal",
            )
            for epoch in range(self.config.adaptation.update_epochs):
                shuffle_key, epoch_key = jax.random.split(shuffle_key)
                permutation = jax.random.permutation(
                    epoch_key, self.config.environment.num_envs
                )
                environment_groups = permutation.reshape(
                    (
                        self.config.adaptation.num_minibatches,
                        self.config.environment.num_envs
                        // self.config.adaptation.num_minibatches,
                    )
                )
                params, optimizer_state, last_metrics = compiled_epoch_update(
                    params, optimizer_state, batch, environment_groups
                )
            jax.block_until_ready((params, optimizer_state, last_metrics))
            steps = _host_int(state.effective_environment_steps)
            should_record = (
                steps % self.config.adaptation.metrics_interval_env_steps == 0
            )
            if should_record or rollout_index == rollout_count - 1:
                last_metric_row = _metric_row(state, batch, last_metrics)
            if should_record:
                writer.append(last_metric_row)
            del environment_groups, permutation, batch
        if last_metrics is None or last_metric_row is None:
            raise RuntimeError("Adaptation completed no rollout.")
        if not writer.rows or int(writer.rows[-1]["environment_steps"]) != _host_int(
            state.effective_environment_steps
        ):
            writer.append(last_metric_row)
        metrics_path = context.stage_directory / "metrics.jsonl"
        effective = _read_metric_counts(metrics_path)
        if effective != {
            "effective_environment_steps": _host_int(state.effective_environment_steps),
            "completed_episodes": _host_int(state.completed_episodes),
        }:
            raise RuntimeError("Adaptation metric artifacts do not match the completed rollout state.")
        episode_path = context.stage_directory / "episodes.jsonl"
        family_coverage = (
            _family_episode_coverage(
                episode_path, expected_episodes=effective["completed_episodes"]
            )
            if self.config.is_family_pool
            else {}
        )
        checkpoint_dir = context.stage_directory / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            params=params,
            metadata=self._checkpoint_metadata(
                stage="adaptation",
                context=context,
                params=params,
                effective_environment_steps=effective["effective_environment_steps"],
                completed_episodes=effective["completed_episodes"],
            ),
        )
        summary = write_stage_summary(
            context,
            stage="adaptation",
            effective_environment_steps=effective["effective_environment_steps"],
            completed_episodes=effective["completed_episodes"],
            extra={
                "checkpoint_path": str(checkpoint_dir),
                "update_epochs": self.config.adaptation.update_epochs,
                "optimizer_updates": optimizer_updates,
                "raw_team_reward_only": True,
                "probe_cost_applied_twice": False,
                "decision_recording": (
                    "probed_steps_only_gzip"
                    if self.config.run_kind == "formal"
                    else "all_steps_jsonl"
                ),
                **family_coverage,
            },
        )
        return {
            "manifest": checkpoint_dir / MANIFEST_FILE,
            "weights": checkpoint_dir / WEIGHTS_FILE,
            "metrics": metrics_path,
            "decisions": decision_path,
            "episodes": episode_path,
            "summary": summary,
        }

    def evaluation(self, context: Any) -> Mapping[str, Path]:
        from .evaluation_runtime import run_development_evaluation

        if self.config.evaluation.schedule == "formal_manifest":
            raise ValueError(
                "Formal evaluation requires a caller-supplied ten-policy manifest; "
                "this release intentionally does not generate that manifest."
            )
        checkpoint_manifest = self._preceding_path(context, "adaptation.manifest")
        params, unused_metadata, unused_manifest = load_checkpoint(checkpoint_manifest.parent)
        del unused_metadata, unused_manifest
        calibration_summary = json.loads(
            self._preceding_path(context, "calibration.summary").read_text(encoding="utf-8")
        )
        return run_development_evaluation(
            context=context,
            built=self._build(),
            params=params,
            config=self.config,
            calibration_summary=calibration_summary,
        )
