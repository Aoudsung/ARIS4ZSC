"""Standard four-mode OvercookedV2 matrix evaluator for Path C model version four."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.path_c.vqbc.checkpoint import (
    assert_pairing_reference_ownership,
    load_vqbc_deployment,
)
from src.path_c.vqbc.evaluation import (
    VQBCEpisodeRow,
    VQBCPairing,
    VQBCPopulation,
    standard_episode_seed,
    standard_pairings,
    summarize_standard_rows,
    validate_standard_rows,
    write_rows,
)
from src.path_c.vqbc.config import (
    VQBC_DEPLOYMENT_MODES,
    VQBCConfig,
)
from src.path_c.vqbc.model import (
    build_vqbc_model,
    encode_response_codes,
)
from src.path_c.vqbc.policy import (
    deployment_belief_after_response,
    uniform_slot_log_belief,
    update_log_temperature,
)
from src.path_c.vqbc.quotient import slot_bayes_update
from src.path_c.vqbc.rollout import (
    VQBCRolloutCallbacks,
    initialize_policy_state,
    policy_action,
)

from .env_dock import OvercookedV2VectorEnvironment
from .official_dock import OfficialBackboneDock


class DeploymentStep:
    __slots__ = ("state", "action", "output", "decision")

    def __init__(self, state: Any, action: Any, output: Any, decision: Any) -> None:
        self.state = state
        self.action = action
        self.output = output
        self.decision = decision


@dataclass(frozen=True)
class _DevelopmentEntry:
    config: VQBCConfig
    checkpoint_path: Path


@dataclass(frozen=True)
class VQBCDeploymentPolicy:
    config: Any
    model: Any
    params: Mapping[str, Any]
    codebook_embeddings: Any
    reference_dock: OfficialBackboneDock
    reference_params: Mapping[str, Any]
    log_temperature: Any
    generic_log_temperature: Any

    @classmethod
    def load(cls, entry: Any) -> "VQBCDeploymentPolicy":
        dock = OfficialBackboneDock.from_launch_config(
            entry.config.backbone_init.launch_config_path
        )
        reference_params = dock.official_parameter_tree(
            entry.config.backbone_init
        )
        model = build_vqbc_model(
            model_config=entry.config.model,
            official_dimensions=dock.network_dimensions(),
        )
        deployment, unused_metadata, unused_manifest = load_vqbc_deployment(
            entry.checkpoint_path,
            expected_config=entry.config,
        )
        del unused_metadata, unused_manifest
        return cls(
            config=entry.config,
            model=model,
            params=deployment["online_params"],
            codebook_embeddings=deployment["codebook"]["embeddings"],
            reference_dock=dock,
            reference_params=reference_params,
            log_temperature=deployment["kl_state"]["log_temperature"],
            generic_log_temperature=deployment["kl_state"][
                "generic_log_temperature"
            ],
        )

    def callbacks(self) -> VQBCRolloutCallbacks:
        def model_apply(
            params: Mapping[str, Any],
            carry: Any,
            observations: Any,
            previous_actions: Any,
            previous_rewards: Any,
            starts: Any,
        ) -> Any:
            return self.model.apply(
                {"params": params},
                carry,
                observations,
                previous_actions,
                previous_rewards,
                starts,
            )

        def reference_apply(
            carry: Any, observations: Any, starts: Any
        ) -> tuple[Any, Any]:
            next_carry, logits, unused_value = (
                self.reference_dock.apply_reference_actor_critic(
                    self.reference_params,
                    carry,
                    observations[None, ...],
                    starts[None, ...],
                )
            )
            del unused_value
            return next_carry, logits[0]

        return VQBCRolloutCallbacks(
            model_apply=model_apply,
            reference_apply=reference_apply,
            partner_step=lambda *unused: None,
            partner_observe=lambda *unused: None,
        )

    def initial_state(self, batch_size: int) -> Any:
        import jax.numpy as jnp

        state = initialize_policy_state(
            model_initial_carry=self.model.initial_carry,
            reference_initial_carry=self.reference_dock.initial_recurrent_state,
            batch_size=batch_size,
            slot_count=self.config.model.slot_count,
            action_count=self.config.model.action_count,
            initial_temperature=1.0,
        )
        return state._replace(
            log_temperature=jnp.full(
                (batch_size,), self.log_temperature, dtype=jnp.float32
            ),
            generic_log_temperature=jnp.full(
                (batch_size,),
                self.generic_log_temperature,
                dtype=jnp.float32,
            ),
        )

    def act(
        self,
        state: Any,
        observations: Any,
        key: Any,
        deployment_mode: str,
    ) -> DeploymentStep:
        (
            next_state,
            action,
            output,
            decision,
            unused_generic,
        ) = policy_action(
            callbacks=self.callbacks(),
            params=self.params,
            policy_state=state,
            observations=observations,
            key=key,
            deployment_mode=deployment_mode,
            gamma=self.config.training.gamma,
        )
        del unused_generic
        return DeploymentStep(next_state, action, output, decision)

    def observe(
        self,
        step: DeploymentStep,
        *,
        observations: Any,
        next_observations: Any,
        rewards: Any,
        dones: Any,
        deployment_mode: str,
        mask_response: bool = False,
    ) -> tuple[Any, Any]:
        import jax.numpy as jnp

        codes, unused_logits, unused_signatures = encode_response_codes(
            model=self.model,
            params=self.params,
            observations=observations[None, ...],
            actions=step.action[None, ...],
            next_observations=next_observations[None, ...],
            dones=dones[None, ...],
            codebook_embeddings=self.codebook_embeddings,
        )
        del unused_logits, unused_signatures
        codes = codes[0]
        step.decision = step.decision._replace(response_code=codes)
        output = step.output
        updated = slot_bayes_update(
            slot_log_belief=step.state.slot_log_belief,
            slot_response_probabilities=output.response_probabilities,
            action=step.action,
            response_code=codes,
        )
        normal_update = deployment_belief_after_response(
            mode=deployment_mode,
            current_log_belief=step.state.slot_log_belief,
            updated_log_belief=updated,
        )
        if isinstance(mask_response, bool):
            updated = (
                step.state.slot_log_belief
                if mask_response
                else normal_update
            )
        else:
            updated = jnp.where(
                jnp.asarray(mask_response, dtype=jnp.bool_)[:, None],
                step.state.slot_log_belief,
                normal_update,
            )
        uniform = uniform_slot_log_belief(
            updated.shape[:-1], self.config.model.slot_count
        )
        next_state = step.state._replace(
            slot_log_belief=jnp.where(dones[:, None], uniform, updated),
            previous_action=jnp.where(
                dones,
                jnp.full(dones.shape, self.config.model.action_count, jnp.int32),
                step.action,
            ),
            previous_team_reward=jnp.where(dones, 0.0, rewards),
            episode_start=dones,
        )
        return next_state, codes

    def advance_temperature(
        self, *, deployment_mode: str, mean_kl: Any
    ) -> "VQBCDeploymentPolicy":
        config = self.config.kl_control
        if deployment_mode == "generic_response_information":
            return replace(
                self,
                generic_log_temperature=update_log_temperature(
                    log_temperature=self.generic_log_temperature,
                    mean_kl=mean_kl,
                    target_kl=config.target_per_step,
                    learning_rate=config.dual_learning_rate,
                    minimum_temperature=config.minimum_temperature,
                    maximum_temperature=config.maximum_temperature,
                ),
            )
        return replace(
            self,
            log_temperature=update_log_temperature(
                log_temperature=self.log_temperature,
                mean_kl=mean_kl,
                target_kl=config.target_per_step,
                learning_rate=config.dual_learning_rate,
                minimum_temperature=config.minimum_temperature,
                maximum_temperature=config.maximum_temperature,
            ),
        )


def _evaluate_batch(
    *,
    left: VQBCDeploymentPolicy,
    right: VQBCDeploymentPolicy,
    pairing: VQBCPairing,
    episode_indexes: tuple[int, ...],
    episode_seeds: tuple[int, ...],
    population_id: str,
    runner: Any | None = None,
) -> tuple[list[VQBCEpisodeRow], Any, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    count = len(episode_seeds)
    seed_array = jnp.asarray(episode_seeds, dtype=jnp.uint32)
    reset_keys = jax.vmap(jax.random.PRNGKey)(seed_array)
    if runner is None:
        runner = _make_evaluation_batch_runner(
            left=left,
            right=right,
            deployment_mode=pairing.deployment_mode,
            count=count,
        )
    (
        returns,
        correct,
        wrong,
        cumulative_kl,
        left_cumulative_kl,
        right_cumulative_kl,
        deviations,
        quotient_sum,
        entropy_sum,
        response_counts,
    ) = runner(
        reset_keys,
        left.log_temperature,
        left.generic_log_temperature,
        right.log_temperature,
        right.generic_log_temperature,
    )
    rows = []
    for lane, (episode_index, episode_seed) in enumerate(
        zip(episode_indexes, episode_seeds, strict=True)
    ):
        rows.append(
            VQBCEpisodeRow(
                schema_version="path_c_vqbc_evaluation_rows_v1",
                population_id=population_id,
                layout=left.config.environment.layout,
                deployment_mode=pairing.deployment_mode,
                split=pairing.split,
                pairing_id=pairing.pairing_id,
                left_outer_unit_id=pairing.left_outer_unit_id,
                right_outer_unit_id=pairing.right_outer_unit_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
                environment_steps=400,
                raw_return=float(np.asarray(returns[lane])),
                correct_delivery_count=int(np.asarray(correct[lane])),
                wrong_delivery_count=int(np.asarray(wrong[lane])),
                cumulative_kl=float(np.asarray(cumulative_kl[lane])),
                reference_action_deviation_count=int(
                    np.asarray(deviations[lane])
                ),
                mean_quotient_count=float(
                    np.asarray(quotient_sum[lane] / 800.0)
                ),
                mean_belief_entropy=float(
                    np.asarray(entropy_sum[lane] / 800.0)
                ),
                response_code_counts=tuple(
                    int(value)
                    for value in np.asarray(response_counts[lane]).tolist()
                ),
            )
        )
    return rows, jnp.mean(left_cumulative_kl / 400.0), jnp.mean(
        right_cumulative_kl / 400.0
    )


def _make_evaluation_batch_runner(
    *,
    left: VQBCDeploymentPolicy,
    right: VQBCDeploymentPolicy,
    deployment_mode: str,
    count: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    environment = OvercookedV2VectorEnvironment.create(
        num_envs=count, layout=left.config.environment.layout
    )

    def run_episode(
        reset_keys: Any,
        left_log_temperature: Any,
        left_generic_log_temperature: Any,
        right_log_temperature: Any,
        right_generic_log_temperature: Any,
    ) -> tuple[Any, ...]:
        current_left = replace(
            left,
            log_temperature=left_log_temperature,
            generic_log_temperature=left_generic_log_temperature,
        )
        current_right = replace(
            right,
            log_temperature=right_log_temperature,
            generic_log_temperature=right_generic_log_temperature,
        )
        environment_state, observations = environment.reset_with_keys(
            reset_keys
        )
        initial = (
            environment_state,
            observations,
            current_left.initial_state(count),
            current_right.initial_state(count),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count,), dtype=jnp.float32),
            jnp.zeros((count, 16), dtype=jnp.int32),
        )

        def one_step(current: tuple[Any, ...], step_index: Any) -> tuple[Any, None]:
            (
                current_environment_state,
                current_observations,
                left_state,
                right_state,
                returns,
                correct,
                wrong,
                cumulative_kl,
                left_cumulative_kl,
                right_cumulative_kl,
                deviations,
                quotient_sum,
                entropy_sum,
                response_counts,
            ) = current
            step_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, step_index)
            )(reset_keys)
            left_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 1)
            )(step_keys)
            right_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 2)
            )(step_keys)
            environment_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 3)
            )(step_keys)
            left_step = current_left.act(
                left_state,
                current_observations[:, 0],
                left_keys,
                deployment_mode,
            )
            right_step = current_right.act(
                right_state,
                current_observations[:, 1],
                right_keys,
                deployment_mode,
            )
            joint_actions = jnp.stack(
                (left_step.action, right_step.action), axis=-1
            )
            (
                next_environment_state,
                next_observations,
                rewards,
                dones,
                info,
            ) = environment.step_with_keys(
                current_environment_state,
                joint_actions,
                environment_keys,
            )
            terminal = info["terminal_observations"]
            mask = dones.reshape(
                dones.shape
                + (1,) * (next_observations[:, 0].ndim - 1)
            )
            left_next = jnp.where(
                mask, terminal[:, 0], next_observations[:, 0]
            )
            right_next = jnp.where(
                mask, terminal[:, 1], next_observations[:, 1]
            )
            left_state, left_codes = current_left.observe(
                left_step,
                observations=current_observations[:, 0],
                next_observations=left_next,
                rewards=rewards,
                dones=dones,
                deployment_mode=deployment_mode,
            )
            right_state, right_codes = current_right.observe(
                right_step,
                observations=current_observations[:, 1],
                next_observations=right_next,
                rewards=rewards,
                dones=dones,
                deployment_mode=deployment_mode,
            )
            left_kl = left_step.decision.kl_divergence
            right_kl = right_step.decision.kl_divergence
            return (
                next_environment_state,
                next_observations,
                left_state,
                right_state,
                returns + rewards,
                correct + info["correct_delivery"],
                wrong + info["wrong_delivery"],
                cumulative_kl + left_kl + right_kl,
                left_cumulative_kl + left_kl,
                right_cumulative_kl + right_kl,
                deviations
                + (
                    left_step.action
                    != left_step.decision.reference_greedy_action
                ).astype(jnp.int32)
                + (
                    right_step.action
                    != right_step.decision.reference_greedy_action
                ).astype(jnp.int32),
                quotient_sum
                + left_step.decision.quotient_count
                + right_step.decision.quotient_count,
                entropy_sum
                + left_step.decision.belief_entropy
                + right_step.decision.belief_entropy,
                response_counts
                + jax.nn.one_hot(left_codes, 16, dtype=jnp.int32)
                + jax.nn.one_hot(right_codes, 16, dtype=jnp.int32),
            ), None

        final, unused = jax.lax.scan(
            one_step, initial, jnp.arange(400, dtype=jnp.int32)
        )
        del unused
        return final[4:]

    return jax.jit(run_episode)


def evaluate_pairing(
    population: VQBCPopulation, pairing: VQBCPairing
) -> list[VQBCEpisodeRow]:
    left_entry = population.entry(pairing.left_outer_unit_id)
    right_entry = population.entry(pairing.right_outer_unit_id)
    assert_pairing_reference_ownership(
        left_metadata=left_entry.checkpoint_metadata,
        left_reference=left_entry.config.backbone_init,
        left_outer_unit_id=pairing.left_outer_unit_id,
        right_metadata=right_entry.checkpoint_metadata,
        right_reference=right_entry.config.backbone_init,
        right_outer_unit_id=pairing.right_outer_unit_id,
    )
    left = VQBCDeploymentPolicy.load(
        left_entry
    )
    right = VQBCDeploymentPolicy.load(
        right_entry
    )
    rows = []
    batch_size = left.config.environment.num_envs
    runners = {}
    for start in range(0, 500, batch_size):
        indexes = tuple(range(start, min(start + batch_size, 500)))
        count = len(indexes)
        if count not in runners:
            runners[count] = _make_evaluation_batch_runner(
                left=left,
                right=right,
                deployment_mode=pairing.deployment_mode,
                count=count,
            )
        seeds = tuple(
            standard_episode_seed(
                population_id=population.population_id,
                layout=population.layout,
                left_outer_unit_id=pairing.left_outer_unit_id,
                right_outer_unit_id=pairing.right_outer_unit_id,
                episode_index=index,
            )
            for index in indexes
        )
        batch_rows, left_kl, right_kl = _evaluate_batch(
            left=left,
            right=right,
            pairing=pairing,
            episode_indexes=indexes,
            episode_seeds=seeds,
            population_id=population.population_id,
            runner=runners[count],
        )
        rows.extend(batch_rows)
        left = left.advance_temperature(
            deployment_mode=pairing.deployment_mode, mean_kl=left_kl
        )
        right = right.advance_temperature(
            deployment_mode=pairing.deployment_mode, mean_kl=right_kl
        )
    return rows


def run_standard_evaluation(population: VQBCPopulation) -> Mapping[str, Path]:
    rows = []
    for pairing in standard_pairings():
        rows.extend(evaluate_pairing(population, pairing))
    validate_standard_rows(rows)
    rows_path = write_rows(population.output_root / "rows.jsonl", rows)
    summary_path = population.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(summarize_standard_rows(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return {"rows": rows_path, "summary": summary_path}


def _response_code_summary(rows: list[VQBCEpisodeRow]) -> Mapping[str, Any]:
    counts = [
        sum(row.response_code_counts[index] for row in rows)
        for index in range(16)
    ]
    total = sum(counts)
    probabilities = [
        count / total if total else 0.0 for count in counts
    ]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return {
        "counts": counts,
        "active_code_count": sum(count > 0 for count in counts),
        "perplexity": math.exp(entropy) if total else 0.0,
    }


def _summarize_development_rows(
    rows: list[VQBCEpisodeRow],
) -> Mapping[str, Any]:
    by_mode = {
        mode: [row for row in rows if row.deployment_mode == mode]
        for mode in VQBC_DEPLOYMENT_MODES
    }
    if any(len(mode_rows) != 500 for mode_rows in by_mode.values()):
        raise ValueError(
            "Development evaluation requires 500 matched episodes per mode."
        )
    expected_seeds = {
        row.episode_index: row.episode_seed
        for row in by_mode[VQBC_DEPLOYMENT_MODES[0]]
    }
    if any(
        {
            row.episode_index: row.episode_seed
            for row in mode_rows
        }
        != expected_seeds
        for mode_rows in by_mode.values()
    ):
        raise ValueError("Development deployment modes changed episode seeds.")

    mode_summary = {}
    returns_by_mode = {}
    for mode, mode_rows in by_mode.items():
        returns = {
            row.episode_index: float(row.raw_return) for row in mode_rows
        }
        returns_by_mode[mode] = returns
        mode_summary[mode] = {
            "episode_count": len(mode_rows),
            "environment_steps": sum(
                row.environment_steps for row in mode_rows
            ),
            "mean_raw_return": sum(returns.values()) / len(mode_rows),
            "mean_correct_delivery_count": sum(
                row.correct_delivery_count for row in mode_rows
            )
            / len(mode_rows),
            "mean_wrong_delivery_count": sum(
                row.wrong_delivery_count for row in mode_rows
            )
            / len(mode_rows),
            "mean_kl_per_policy_step": sum(
                row.cumulative_kl for row in mode_rows
            )
            / (len(mode_rows) * 800.0),
            "sampled_action_reference_greedy_deviation_rate": sum(
                row.reference_action_deviation_count for row in mode_rows
            )
            / (len(mode_rows) * 800.0),
            "mean_quotient_count": sum(
                row.mean_quotient_count for row in mode_rows
            )
            / len(mode_rows),
            "mean_belief_entropy": sum(
                row.mean_belief_entropy for row in mode_rows
            )
            / len(mode_rows),
            "response_codes": _response_code_summary(mode_rows),
        }

    posterior = returns_by_mode["posterior_use"]
    matched_differences = {}
    for comparison in (
        "prior_only",
        "reference_only",
        "generic_response_information",
    ):
        alternative = returns_by_mode[comparison]
        matched_differences[
            f"posterior_use_minus_{comparison}"
        ] = sum(
            posterior[index] - alternative[index]
            for index in sorted(expected_seeds)
        ) / len(expected_seeds)

    return {
        "schema_version": "path_c_vqbc_development_evaluation_summary_v1",
        "run_kind": "development",
        "scientific_readout_allowed": False,
        "matched_episode_seeds": True,
        "deployment_modes": mode_summary,
        "matched_mean_raw_return_differences": matched_differences,
    }


def run_development_evaluation(
    *,
    resolved_config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path | None = None,
) -> Mapping[str, Path]:
    """Evaluate one development checkpoint in four matched self-play modes."""

    config_path = Path(resolved_config_path).resolve()
    config = VQBCConfig.from_mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        base_dir=config_path.parent,
    )
    if config.run_kind != "development" or config.outer_unit is not None:
        raise ValueError(
            "The one-checkpoint evaluator accepts only a development config."
        )
    checkpoint = Path(checkpoint_path).resolve()
    entry = _DevelopmentEntry(config=config, checkpoint_path=checkpoint)
    population_id = (
        f"path_c_vqbc_v4_development_{config.environment.layout}_"
        f"{config.seeds.model_seed}"
    )
    rows = []
    for mode in VQBC_DEPLOYMENT_MODES:
        base_policy = VQBCDeploymentPolicy.load(entry)
        left = base_policy
        right = base_policy
        runners = {}
        pairing = VQBCPairing(
            deployment_mode=mode,
            split="sp",
            left_outer_unit_id=0,
            right_outer_unit_id=0,
            pairing_id="development_self_play",
        )
        for start in range(0, 500, config.environment.num_envs):
            indexes = tuple(
                range(start, min(start + config.environment.num_envs, 500))
            )
            count = len(indexes)
            if count not in runners:
                runners[count] = _make_evaluation_batch_runner(
                    left=left,
                    right=right,
                    deployment_mode=mode,
                    count=count,
                )
            seeds = tuple(
                standard_episode_seed(
                    population_id=population_id,
                    layout=config.environment.layout,
                    left_outer_unit_id=0,
                    right_outer_unit_id=0,
                    episode_index=index,
                )
                for index in indexes
            )
            batch_rows, left_kl, right_kl = _evaluate_batch(
                left=left,
                right=right,
                pairing=pairing,
                episode_indexes=indexes,
                episode_seeds=seeds,
                population_id=population_id,
                runner=runners[count],
            )
            rows.extend(batch_rows)
            left = left.advance_temperature(
                deployment_mode=mode, mean_kl=left_kl
            )
            right = right.advance_temperature(
                deployment_mode=mode, mean_kl=right_kl
            )
    destination = (
        config.output_root / "development_evaluation"
        if output_root is None
        else Path(output_root).resolve()
    )
    rows_path = write_rows(destination / "rows.jsonl", rows)
    summary_path = destination / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _summarize_development_rows(rows),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return {"rows": rows_path, "summary": summary_path}


__all__ = [
    "VQBCDeploymentPolicy",
    "evaluate_pairing",
    "run_development_evaluation",
    "run_standard_evaluation",
]
