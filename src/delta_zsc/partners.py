"""Static, lineage-aware partner distribution for unified DELTA-ZSC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from .manifest import normalized_mechanism
from .runner import PartnerFunctions


@dataclass(frozen=True, slots=True)
class PartnerPoolMember:
    run_id: str
    parent_training_run_id: str
    mechanism: str
    hyperparameter_family: str
    checkpoint_stage: float
    checkpoint: Any
    probability: float


class StaticPartnerState(NamedTuple):
    carry: Any
    member: Any


class StaticPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


def build_training_partner_pool(config: Any, manifest: Any) -> tuple[PartnerPoolMember, ...]:
    rows = tuple(manifest.by_role("development_support"))
    if not rows:
        raise ValueError("DELTA training support is empty.")
    candidates = []
    for row in rows:
        mechanism = normalized_mechanism(row.generation_mechanism)
        if mechanism not in {"sp", "op"}:
            raise ValueError("Unified DELTA training support contains only SP/OP.")
        if float(row.checkpoint_stage) not in tuple(config.partner_pool.checkpoint_stages):
            raise ValueError(f"Unregistered support checkpoint stage: {row.run_id}")
        if row.owner_seed_index is not None:
            raise ValueError("Training support partners must be owner-free.")
        candidates.append((row, mechanism))
    mechanisms = sorted({mechanism for _, mechanism in candidates})
    if set(mechanisms) != {"op", "sp"}:
        raise ValueError("Training support must contain both SP and OP mechanisms.")

    # In formal runs every parent contributes the registered three progress
    # checkpoints and each mechanism contains at least ten independent parents.
    if config.run_kind == "formal":
        for mechanism in mechanisms:
            parents = {
                row.parent_training_run_id
                for row, current in candidates
                if current == mechanism
            }
            if len(parents) < 10:
                raise ValueError("Formal support needs ten independent parents per mechanism.")
            for parent in parents:
                stages = {
                    float(row.checkpoint_stage)
                    for row, current in candidates
                    if current == mechanism and row.parent_training_run_id == parent
                }
                if stages != set(config.partner_pool.checkpoint_stages):
                    raise ValueError(
                        f"Formal support parent {parent} lacks stages 0/0.5/1."
                    )

    members: list[PartnerPoolMember] = []
    for mechanism in mechanisms:
        families = sorted(
            {
                row.hyperparameter_family
                for row, current in candidates
                if current == mechanism
            }
        )
        for family in families:
            family_rows = [
                row
                for row, current in candidates
                if current == mechanism and row.hyperparameter_family == family
            ]
            stages = sorted({float(row.checkpoint_stage) for row in family_rows})
            for stage in stages:
                stage_rows = [
                    row for row in family_rows if float(row.checkpoint_stage) == stage
                ]
                probability = (
                    1.0
                    / len(mechanisms)
                    / len(families)
                    / len(stages)
                    / len(stage_rows)
                )
                for row in stage_rows:
                    members.append(
                        PartnerPoolMember(
                            run_id=row.run_id,
                            parent_training_run_id=row.parent_training_run_id,
                            mechanism=mechanism,
                            hyperparameter_family=row.hyperparameter_family,
                            checkpoint_stage=float(row.checkpoint_stage),
                            checkpoint=row.checkpoint,
                            probability=float(probability),
                        )
                    )
    if abs(sum(row.probability for row in members) - 1.0) > 1.0e-8:
        raise AssertionError("Partner-pool probabilities must sum to one.")
    return tuple(members)


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def one(left: Any, right: Any):
        value = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = value.reshape(value.shape + (1,) * (jnp.ndim(left) - value.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(one, selected, alternative)


CURRICULUM_SHARPNESS = 4.0
"""Initial log-odds penalty applied per unit of missing partner capability.

At progress zero a stage-0 partner is exp(-4) ~ 1.8% as likely as a fully
trained one; the penalty decays linearly to zero, so the registered uniform
mixture is reached at progress one and holds thereafter.

The permanently uniform mixture it replaces gave every episode a one-in-three
chance of a randomly initialised partner.  A partner that cannot complete a
dish emits no protocol to infer and completes no joint task, so those episodes
carry no coordination signal at all -- they only add variance to an advantage
that is already averaged over conflicting partners.  Late in training the same
episodes are useful as a robustness perturbation, which is where they now land.
"""


def partner_curriculum_weights(
    base_probabilities: Any,
    stages: Any,
    group_indexes: Any,
    group_count: int,
    progress: Any,
) -> Any:
    """Re-weight the pool toward capable partners early in training.

    Renormalisation is per (mechanism, family) group so the registered
    ``mechanism_uniform_sampling`` invariant survives the reweighting: the
    curriculum moves mass between stages, never between mechanisms.
    """

    import jax.numpy as jnp

    base = jnp.asarray(base_probabilities, dtype=jnp.float32)
    stage = jnp.asarray(stages, dtype=jnp.float32)
    groups = jnp.asarray(group_indexes, dtype=jnp.int32)
    fraction = jnp.clip(jnp.asarray(progress, dtype=jnp.float32), 0.0, 1.0)
    sharpness = jnp.asarray(CURRICULUM_SHARPNESS, dtype=jnp.float32) * (1.0 - fraction)
    unnormalised = base * jnp.exp(sharpness * (stage - 1.0))
    group_mass = jnp.zeros((int(group_count),), dtype=jnp.float32).at[groups].add(
        unnormalised
    )
    group_base = jnp.zeros((int(group_count),), dtype=jnp.float32).at[groups].add(base)
    scale = group_base[groups] / jnp.maximum(group_mass[groups], 1.0e-30)
    return unnormalised * scale


def make_static_partner_functions(
    *,
    pool: Any,
    probabilities: Any,
    run_ids: Any,
    checkpoint_stages: Any,
    group_indexes: Any,
    group_count: int,
) -> PartnerFunctions:
    import jax
    import jax.numpy as jnp

    weights = jnp.asarray(probabilities, dtype=jnp.float32)
    if weights.ndim != 1 or weights.shape[0] != int(pool.member_count):
        raise ValueError("Static partner probabilities do not match pool members.")
    weights = weights / jnp.sum(weights)
    identifiers = jnp.asarray(run_ids, dtype=jnp.int32)
    if identifiers.shape != weights.shape:
        raise ValueError("Static partner run IDs do not match pool members.")
    stages = jnp.asarray(checkpoint_stages, dtype=jnp.float32)
    groups = jnp.asarray(group_indexes, dtype=jnp.int32)
    if stages.shape != weights.shape or groups.shape != weights.shape:
        raise ValueError("Static partner stage/group arrays do not match pool members.")

    def sample(key: Any, size: int, progress: Any) -> Any:
        current = partner_curriculum_weights(
            weights, stages, groups, int(group_count), progress
        )
        return jax.random.categorical(
            key,
            jnp.log(jnp.maximum(current, 1.0e-30)),
            shape=(int(size),),
        ).astype(jnp.int32)

    def initial_state(batch_size: int, key: Any) -> StaticPartnerState:
        # Training starts at progress zero; the curriculum is at its sharpest.
        return StaticPartnerState(
            carry=pool.initial_carry(int(batch_size)),
            member=sample(key, int(batch_size), 0.0),
        )

    def step(
        parameters: Any,
        state: StaticPartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ):
        del parameters
        action, carry = pool.step_with_keys(
            state.member, observations, state.carry, episode_start, keys
        )
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 911))(keys)
        return action, StaticPartnerState(carry, state.member), StaticPartnerContext(
            state.member, reset_keys
        ), jnp.zeros_like(action, dtype=jnp.float32)

    def observe(
        parameters: Any,
        state: StaticPartnerState,
        context: StaticPartnerContext,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ):
        del observations, actions, rewards, next_observations
        # ``parameters`` carries the curriculum progress for this update.  The
        # static pool has no learnable parameters, so the slot was unused.
        progress = 0.0 if parameters is None else parameters
        done = jnp.asarray(dones, dtype=jnp.bool_)
        # Partner resampling and fresh recurrent carries are only observable on
        # real episode boundaries.  Avoid constructing both for every step.
        def reset_members(reset_keys: Any) -> StaticPartnerState:
            fresh_member = jax.vmap(lambda key: sample(key, 1, progress)[0])(
                reset_keys
            )
            return StaticPartnerState(
                carry=pool.initial_carry(int(done.shape[0])), member=fresh_member
            )

        fresh = jax.lax.cond(
            jnp.any(done), reset_members, lambda unused: state, context.reset_keys
        )
        return _tree_select(done, fresh, state)

    def run_id(parameters: Any, state: StaticPartnerState, context: Any):
        del parameters, context
        return identifiers[state.member]

    def diagnostics(parameters: Any, state: StaticPartnerState, context: Any):
        del parameters, context
        return {
            "source": jnp.full(state.member.shape, 1, dtype=jnp.int32),
            "member": state.member,
            "run_id": identifiers[state.member],
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


__all__ = [
    "PartnerPoolMember",
    "StaticPartnerContext",
    "StaticPartnerState",
    "build_training_partner_pool",
    "make_static_partner_functions",
]
