"""Load one trained Path C policy and advance its deployment state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.official_adapter import OfficialNetwork, restore_official_checkpoint
from src.path_c.experiment import METHOD_VERSION, PopulationEntry
from src.path_c.method import (
    CodebookState,
    deployment_belief_after_response,
    slot_bayes_update,
    uniform_slot_log_belief,
)
from src.path_c.model import build_model, encode_response_codes
from src.path_c.runner import RunnerFunctions, initialize_policy_state
from src.path_c.storage import (
    orbax_manager,
    read_run_identity,
    restore_latest_checkpoint,
)

@dataclass(frozen=True, slots=True)
class Deployment:
    outer_unit_id: int
    network: OfficialNetwork
    reference_params: Any
    online_params: Any
    heads: Any
    head_params: Any
    codebook: Any
    log_temperature: Any
    generic_log_temperature: Any

    def functions(self) -> RunnerFunctions:
        def online_step(
            params: Mapping[str, Any],
            carry: Any,
            observations: Any,
            episode_start: Any,
        ) -> tuple[Any, Any, Any, Any]:
            return self.network.step(
                params, carry, observations, episode_start
            )

        def reference_step(
            carry: Any, observations: Any, episode_start: Any
        ) -> tuple[Any, Any, Any]:
            next_carry, unused_feature, logits, value = self.network.step(
                self.reference_params,
                carry,
                observations,
                episode_start,
            )
            del unused_feature
            return next_carry, logits, value

        def heads_apply(
            params: Mapping[str, Any],
            control_carry: Any,
            features: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            episode_start: Any,
            belief: Any,
        ) -> tuple[Any, Mapping[str, Any]]:
            return self.heads.apply(
                {"params": params},
                control_carry,
                features,
                previous_actions,
                previous_team_rewards,
                episode_start,
                belief,
                method=self.heads.step,
            )

        def response_apply(
            params: Mapping[str, Any],
            observations: Any,
            actions: Any,
            next_observations: Any,
            dones: Any,
            embeddings: Any,
            terminal_response: int,
        ) -> tuple[Any, Any, Any]:
            return encode_response_codes(
                model=self.heads,
                params=params,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                dones=dones,
                codebook_embeddings=embeddings,
                terminal_response=terminal_response,
            )

        return RunnerFunctions(
            online_step=online_step,
            reference_step=reference_step,
            heads_apply=heads_apply,
            encode_response=response_apply,
            partner_step=lambda *unused: None,
            partner_observe=lambda *unused: None,
        )


def load_deployment(
    entry: PopulationEntry, config: Any
) -> Deployment:
    identity = read_run_identity(entry.run_directory)
    if identity.get("stage") != "train" or identity.get("method") != METHOD_VERSION:
        raise ValueError("Population entry does not point to this Path C training method.")
    if int(identity.get("outer_unit_id", -1)) != entry.outer_unit_id:
        raise ValueError("Population outer-unit identity differs from its training run.")
    if identity.get("config") != config.to_mapping():
        raise ValueError("Population training config differs from evaluation config.")
    reference_path = Path(str(identity["reference_checkpoint"])).resolve()
    reference_config, reference_params = restore_official_checkpoint(reference_path)
    network = OfficialNetwork(reference_config)
    if network.layout != config.environment.layout:
        raise ValueError("Training reference and evaluation layout differ.")
    manager = orbax_manager(entry.run_directory / "checkpoints", create=False)
    restored = restore_latest_checkpoint(manager)
    if restored is None:
        raise FileNotFoundError(
            f"No Orbax step in {entry.run_directory / 'checkpoints'}."
        )
    unused_step, checkpoint = restored
    del unused_step
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Orbax deployment checkpoint must restore as a mapping.")
    required = {"online_params", "codebook", "runner_state"}
    if not required.issubset(checkpoint):
        raise ValueError("Orbax deployment checkpoint is missing training fields.")
    online_params = checkpoint["online_params"]
    codebook_values = checkpoint["codebook"]
    runner_values = checkpoint["runner_state"]
    if not all(
        isinstance(value, Mapping)
        for value in (online_params, codebook_values, runner_values)
    ):
        raise TypeError("Orbax deployment fields must be mappings.")
    policy_values = runner_values.get("ego_policy")
    if not isinstance(policy_values, Mapping):
        raise TypeError("Orbax checkpoint lacks the ego policy state.")
    codebook = CodebookState(**codebook_values)
    heads = build_model(
        hidden_dim=config.model.hidden_dim,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        response_count=config.model.response_count,
        prior_scale=config.model.prior_scale,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=(
            config.model.log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=(
            config.model.log_standard_deviation_maximum
        ),
    )
    return Deployment(
        outer_unit_id=entry.outer_unit_id,
        network=network,
        reference_params=reference_params,
        online_params=online_params["official"],
        heads=heads,
        head_params=online_params["heads"],
        codebook=codebook,
        log_temperature=policy_values["log_temperature"],
        generic_log_temperature=policy_values["generic_log_temperature"],
    )

def reset_deployment_state(
    deployment: Deployment,
    *,
    batch_size: int,
    config: Any,
) -> Any:
    import jax.numpy as jnp

    state = initialize_policy_state(
        official_initial_carry=deployment.network.initial_carry,
        batch_size=batch_size,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        hidden_dim=config.model.hidden_dim,
        initial_temperature=config.kl.initial_temperature,
    )
    return state._replace(
        log_temperature=jnp.full(
            (batch_size,),
            jnp.ravel(deployment.log_temperature)[0],
            dtype=jnp.float32,
        ),
        generic_log_temperature=jnp.full(
            (batch_size,),
            jnp.ravel(deployment.generic_log_temperature)[0],
            dtype=jnp.float32,
        ),
    )


def update_deployment_after_transition(
    *,
    deployment: Deployment,
    functions: RunnerFunctions,
    state: Any,
    output: Any,
    observations: Any,
    actions: Any,
    next_observations: Any,
    rewards: Any,
    dones: Any,
    deployment_mode: str,
    terminal_response: int,
    mask_response: Any = False,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    response_codes, unused_logits, unused_signatures = (
        functions.encode_response(
            deployment.head_params,
            observations[None, ...],
            actions[None, ...],
            next_observations[None, ...],
            dones[None, ...],
            deployment.codebook.embeddings,
            terminal_response,
        )
    )
    del unused_logits, unused_signatures
    response_codes = response_codes[0]
    posterior = slot_bayes_update(
        slot_log_belief=state.slot_log_belief,
        slot_response_probabilities=output.response_probabilities,
        action=actions,
        response_code=response_codes,
    )
    mask = jnp.asarray(mask_response, dtype=jnp.bool_)
    if mask.ndim == 0:
        mask = jnp.broadcast_to(mask, posterior.shape[:-1])
    posterior = jnp.where(
        mask[..., None], state.slot_log_belief, posterior
    )
    posterior = deployment_belief_after_response(
        mode=deployment_mode,
        current_log_belief=state.slot_log_belief,
        updated_log_belief=posterior,
    )
    uniform = uniform_slot_log_belief(
        posterior.shape[:-1], int(posterior.shape[-1])
    )
    return state._replace(
        slot_log_belief=jnp.where(dones[:, None], uniform, posterior),
        previous_action=jnp.where(
            dones,
            jnp.full(dones.shape, output.q_values.shape[-1], dtype=jnp.int32),
            actions,
        ),
        previous_team_reward=jnp.where(dones, 0.0, rewards),
        episode_start=dones,
    ), response_codes



__all__ = [
    "Deployment",
    "load_deployment",
    "reset_deployment_state",
    "update_deployment_after_transition",
]
