"""The single static partner-pool implementation used by DEPI.

The only R0/B0--B2 training distribution is the manifest-backed,
mechanism/family/stage/run-stratified pool below;
the family-disjoint test split is represented by two checkpoint-free Official
heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .counterfactual_anchor import tree_select
from .runner import PartnerFunctions


class StaticPoolPartnerState(NamedTuple):
    carry: Any
    member: Any
    sampling_group: Any


class StaticPoolPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


@dataclass(frozen=True, slots=True)
class PartnerPoolMember:
    """One explicit stratum in the static partner distribution."""

    run_id: str
    parent_training_run_id: str
    family_id: str
    hyperparameter_family: str
    mechanism: str
    checkpoint_stage: float
    seed: int
    checkpoint: Path | None
    split: str
    probability: float


def _normalized_mechanism(value: str) -> str:
    lowered = str(value).strip().lower()
    aliases = {
        "rnn-sp": "sp",
        "rnn-op": "op",
        "state-augmented": "sa",
        "greedy-courier": "heuristic",
        "stationary-helper": "heuristic",
    }
    return aliases.get(lowered, lowered)


def build_partner_pool(config: Any, manifest: Any, split: str) -> tuple[PartnerPoolMember, ...]:
    """Build the sole family/stage/seed-stratified partner distribution.

    Sampling is uniform over mechanisms, then hyperparameter families within
    a mechanism, stages within a family, and runs within a stage.  Adding an
    OP width family therefore cannot silently increase total OP probability.
    The heuristic family is excluded from training and
    accepted only in family-disjoint test splits.
    """

    split_name = str(split)
    if split_name not in {
        "train",
        "development",
        "comparator_fit",
        "comparator_validation",
        "test",
        "calibration",
    }:
        raise ValueError(f"Unknown partner-pool split: {split_name}")
    stages = tuple(float(value) for value in config.partner_pool.checkpoint_stages)
    if split_name == "test":
        if not bool(config.partner_pool.heuristic_family_test_only):
            raise ValueError("The DEPI test family must remain heuristic-only.")
        from .heuristic_partners import HEURISTIC_RUN_NAMES

        probability = 1.0 / float(len(HEURISTIC_RUN_NAMES))
        return tuple(
            PartnerPoolMember(
                run_id=run_id,
                parent_training_run_id=run_id,
                family_id="heuristic",
                hyperparameter_family="deterministic-official-script",
                mechanism="heuristic",
                checkpoint_stage=1.0,
                seed=index,
                checkpoint=None,
                split="test",
                probability=probability,
            )
            for index, run_id in enumerate(HEURISTIC_RUN_NAMES)
        )
    role = {
        "train": "development_support",
        "development": "development_support",
        "comparator_fit": "comparator_fit",
        "comparator_validation": "comparator_validation",
        "calibration": "calibration",
    }.get(split_name, "confirmatory")
    candidates = []
    for run in manifest.by_role(role):
        mechanism = _normalized_mechanism(run.generation_mechanism)
        heuristic = mechanism == "heuristic"
        if split_name in {"train", "development"} and heuristic:
            continue
        if heuristic:
            # Heuristics are virtual policies, never checkpoint manifest rows.
            continue
        checkpoint_stage = float(run.checkpoint_stage)
        if checkpoint_stage not in stages:
            raise ValueError(
                f"Partner {run.run_id} uses an unregistered checkpoint stage."
            )
        hyperparameter_family = str(run.hyperparameter_family)
        candidates.append(
            (
                run,
                hyperparameter_family,
                mechanism,
                checkpoint_stage,
            )
        )
    if not candidates:
        raise ValueError(f"Partner pool split {split_name!r} is empty.")

    mechanisms = sorted({mechanism for _, _, mechanism, _ in candidates})
    members: list[PartnerPoolMember] = []
    for mechanism in mechanisms:
        mechanism_families = sorted(
            {
                hyperparameter_family
                for _, hyperparameter_family, row_mechanism, _ in candidates
                if row_mechanism == mechanism
            }
        )
        for hyperparameter_family in mechanism_families:
            family_rows = [
                row
                for row in candidates
                if row[2] == mechanism and row[1] == hyperparameter_family
            ]
            family_stages = sorted({row[3] for row in family_rows})
            family_id = f"{mechanism}:{hyperparameter_family}"
            for stage in family_stages:
                stage_rows = [row for row in family_rows if row[3] == stage]
                probability = (
                    1.0
                    / len(mechanisms)
                    / len(mechanism_families)
                    / len(family_stages)
                    / len(stage_rows)
                )
                for run, _, _, _ in stage_rows:
                    members.append(
                        PartnerPoolMember(
                            run_id=str(run.run_id),
                            parent_training_run_id=str(run.parent_training_run_id),
                            family_id=family_id,
                            hyperparameter_family=hyperparameter_family,
                            mechanism=mechanism,
                            checkpoint_stage=stage,
                            seed=int(run.seed),
                            checkpoint=Path(run.checkpoint),
                            split=split_name,
                            probability=probability,
                        )
                    )
    total = sum(member.probability for member in members)
    if abs(total - 1.0) > 1.0e-9:
        raise AssertionError("Static partner-pool probabilities must sum to one.")
    return tuple(members)


def make_static_pool_partner_functions(
    *,
    external_pool: Any,
    member_probabilities: Any | None = None,
    member_family_ids: Any | None = None,
    member_checkpoint_stages: Any | None = None,
    member_sampling_groups: Any | None = None,
) -> PartnerFunctions:
    """Default static wide partner pool (METHOD_SPEC §7.3).

    Every episode reset resamples a pool member from the registered hierarchy;
    between resets the member is fixed. Pool composition (SP/OP x10 seeds x3
    checkpoint stages and OP width variants) is decided where the pool itself
    is built; the heuristic family is a separate test-only panel.
    """

    import jax
    import jax.numpy as jnp

    probabilities = None
    if member_probabilities is not None:
        probabilities = jnp.asarray(member_probabilities, dtype=jnp.float32)
        probabilities = probabilities / jnp.sum(probabilities)
    family_ids = (
        None
        if member_family_ids is None
        else jnp.asarray(member_family_ids, dtype=jnp.int32)
    )
    checkpoint_stages = (
        None
        if member_checkpoint_stages is None
        else jnp.asarray(member_checkpoint_stages, dtype=jnp.float32)
    )
    if (family_ids is None) != (checkpoint_stages is None):
        raise ValueError("Static partner metadata must provide both family and stage.")
    sampling_groups = (
        None
        if member_sampling_groups is None
        else jnp.asarray(member_sampling_groups, dtype=jnp.int32)
    )
    if sampling_groups is not None:
        if probabilities is None or sampling_groups.shape != probabilities.shape:
            raise ValueError("Role-partitioned sampling needs one group per pool member.")
        if set(map(int, jnp.unique(sampling_groups).tolist())) != {0, 1, 2}:
            raise ValueError("Static pool sampling groups must be support/fit/validation.")

    def sample_members(key: Any, count: int, groups: Any | None = None) -> Any:
        if probabilities is None:
            return external_pool.sample_members(key, int(count))
        if sampling_groups is None or groups is None:
            return jax.random.categorical(
                key,
                jnp.log(jnp.maximum(probabilities, 1.0e-12)),
                shape=(int(count),),
            ).astype(jnp.int32)
        group_values = jnp.asarray(groups, dtype=jnp.int32)
        keys = jax.random.split(key, int(count))

        def one(sample_key: Any, group: Any) -> Any:
            weight = jnp.where(sampling_groups == group, probabilities, 0.0)
            weight = weight / jnp.maximum(jnp.sum(weight), 1.0e-12)
            return jax.random.categorical(
                sample_key,
                jnp.where(weight > 0.0, jnp.log(weight), -jnp.inf),
            ).astype(jnp.int32)

        return jax.vmap(one)(keys, group_values)

    def initial_state(batch_size: int, key: Any) -> StaticPoolPartnerState:
        if sampling_groups is None:
            groups = jnp.zeros((int(batch_size),), dtype=jnp.int32)
        else:
            lane = jnp.arange(int(batch_size), dtype=jnp.int32)
            # Before the comparator is frozen, reserve one eighth of lanes for
            # fit and one eighth for validation. The remaining six eighths are
            # the policy-support distribution.
            groups = jnp.where(lane % 8 == 6, 1, jnp.where(lane % 8 == 7, 2, 0))
        if sampling_groups is None:
            member = sample_members(key, int(batch_size), groups)
        else:
            # Systematic stratified draws guarantee that every registered
            # comparator run with mass >= 1 / reserved-lane-count appears in
            # the first anchor rollout; independent categorical draws would
            # leave a nonzero probability of an invalid one-run split.
            member = jnp.zeros((int(batch_size),), dtype=jnp.int32)
            host_groups = [int(value) for value in groups.tolist()]
            for group in (0, 1, 2):
                indexes = [
                    index for index, value in enumerate(host_groups) if value == group
                ]
                if not indexes:
                    continue
                weight = jnp.where(sampling_groups == group, probabilities, 0.0)
                weight = weight / jnp.sum(weight)
                offset = jax.random.uniform(jax.random.fold_in(key, group))
                quantiles = (
                    jnp.arange(len(indexes), dtype=jnp.float32) + offset
                ) / float(len(indexes))
                selected = jnp.searchsorted(jnp.cumsum(weight), quantiles, side="right")
                member = member.at[jnp.asarray(indexes, dtype=jnp.int32)].set(
                    selected.astype(jnp.int32)
                )
        return StaticPoolPartnerState(
            carry=external_pool.initial_carry(int(batch_size)),
            member=member,
            sampling_group=groups,
        )

    def step(parameters: Any, state: StaticPoolPartnerState, observations: Any, episode_start: Any, keys: Any):
        del parameters
        action_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(keys)
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 911))(keys)
        action, next_carry = external_pool.step_with_keys(
            state.member, observations, state.carry, episode_start, action_keys
        )
        next_state = state._replace(carry=next_carry)
        context = StaticPoolPartnerContext(state.member, reset_keys)
        return action, next_state, context, jnp.zeros_like(action, dtype=jnp.float32)

    def observe(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext, observations: Any, actions: Any, rewards: Any, dones: Any, next_observations: Any):
        del observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        count = int(done.shape[0])
        comparator_frozen = jnp.asarray(
            False if parameters is None else parameters, dtype=jnp.bool_
        )
        fresh_groups = jnp.where(
            comparator_frozen,
            jnp.zeros_like(state.sampling_group),
            state.sampling_group,
        )
        fresh_members = jax.vmap(
            lambda key, group: sample_members(key, 1, group[None])[0]
        )(context.reset_keys, fresh_groups)
        fresh = StaticPoolPartnerState(
            carry=external_pool.initial_carry(count),
            member=fresh_members,
            sampling_group=fresh_groups,
        )
        return tree_select(done, fresh, state)

    def run_id(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext):
        del parameters, context
        return (10_000 + state.member).astype(jnp.int32)

    def diagnostics(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext):
        del parameters
        count = context.member.shape[0]
        return {
            "source": jnp.full((count,), 2, dtype=jnp.int32),
            "member": context.member,
            "family_id": (
                jnp.zeros((count,), dtype=jnp.int32)
                if family_ids is None
                else family_ids[context.member]
            ),
            "checkpoint_stage": (
                jnp.ones((count,), dtype=jnp.float32)
                if checkpoint_stages is None
                else checkpoint_stages[context.member]
            ),
            "partition": state.sampling_group,
            "ppo_mask": (state.sampling_group == 0).astype(jnp.float32),
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


__all__ = [
    "PartnerPoolMember",
    "StaticPoolPartnerContext",
    "StaticPoolPartnerState",
    "build_partner_pool",
    "make_static_pool_partner_functions",
]
