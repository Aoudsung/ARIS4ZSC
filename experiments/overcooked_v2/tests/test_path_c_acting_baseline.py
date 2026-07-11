from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from experiments.overcooked_v2.baselines.belief_filter import (
    BayesianHMMBeliefFilter,
    BeliefFilterActingPolicy,
    HMMFilterConfig,
    OnlineModeBeliefFilter,
    ParticleBeliefFilter,
    ParticleFilterConfig,
)
from experiments.overcooked_v2.path_c_protocol import (
    ActingEnvironmentStep,
    ActingEpisodeSpec,
    ActingTransition,
    CallableActingPolicy,
    DirectInformationActingPolicy,
    FORMAL_ACTING_POLICY_NAMES,
    FORMAL_LOCKED_ACTING_POLICY_NAMES,
    FormalActingBenchmarkContract,
    FormalActingFactoryUnavailableError,
    FormalActingPolicyFactoryEntry,
    FormalActingPolicyFactoryRegistry,
    NoProbeActingPolicy,
    PathCActingEnvironment,
    PathCActingPolicy,
    PathCActingRunner,
    PolicyAction,
    ProbeBudgetState,
    RandomProbeActingPolicy,
    ReturnProbeBudgetPoint,
)


class _StaticActingPolicy:
    policy_name = "static_exploitation"
    oracle_baseline = False

    def __init__(self) -> None:
        self.seed = -1
        self.observed: list[ActingTransition] = []
        self.reset_budget: ProbeBudgetState | None = None

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self.seed = int(seed)
        self.observed = []
        self.reset_budget = probe_budget_state

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        del observation, probe_budget_state
        return PolicyAction(
            action_id=0,
            is_probe=False,
            propensity=1.0,
            candidate_action_ids=(0,),
            candidate_scores=(1.0,),
            estimated_probe_cost=0.0,
            policy_kind="static_exploitation",
        )

    def observe(self, transition: ActingTransition) -> None:
        self.observed.append(transition)

    def representation(self) -> np.ndarray:
        return np.asarray([float(self.seed), float(len(self.observed))], dtype=np.float32)

    def metrics(self) -> Mapping[str, Any]:
        return {"seed": self.seed, "observed": len(self.observed)}


def _transition(action: PolicyAction, *, token: str = "claim_signal") -> ActingTransition:
    return ActingTransition(
        observation={"belief_observation_token": "before"},
        action=action,
        reward=1.0,
        next_observation={"belief_observation_token": token},
        terminated=False,
        truncated=False,
        realized_probe_cost=0.25 if action.is_probe else 0.0,
        environment_steps=2,
    )


class _TwoDecisionEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.step_calls = 0

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        self.events.append(f"environment.reset:{int(seed)}")
        self.step_calls = 0
        return self._observation("start")

    def step(self, action_id: int) -> ActingEnvironmentStep:
        self.events.append(f"environment.step:{int(action_id)}")
        self.step_calls += 1
        if self.step_calls == 1:
            if int(action_id) != 2:
                raise AssertionError("The first decision must use the probe action.")
            return ActingEnvironmentStep(
                observation=self._observation("claim_signal"),
                reward=3.0,
                terminated=False,
                truncated=False,
                environment_steps=2,
            )
        if int(action_id) != 0:
            raise AssertionError("The second decision must use the non-probe action.")
        return ActingEnvironmentStep(
            observation=self._observation("claim_signal"),
            reward=2.0,
            terminated=True,
            truncated=False,
            environment_steps=1,
        )

    @staticmethod
    def _observation(token: str) -> Mapping[str, Any]:
        return {
            "belief_observation_token": str(token),
            "valid_action_ids": (0, 2),
            "probe_candidate_action_ids": (2,),
            "probe_only_action_ids": (2,),
        }


class _CallableController:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.transitions: list[ActingTransition] = []

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self.events.append(f"policy.reset:{int(seed)}")
        self.transitions = []
        assert probe_budget_state.probes_used == 0

    def __call__(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        del observation
        self.events.append(f"policy.act:{probe_budget_state.remaining}")
        is_probe = probe_budget_state.remaining > 0
        action_id = 2 if is_probe else 0
        return PolicyAction(
            action_id=action_id,
            is_probe=is_probe,
            propensity=1.0,
            candidate_action_ids=(0, 2),
            candidate_scores=(0.0, 1.0) if is_probe else (1.0, 0.0),
            estimated_probe_cost=(
                float(probe_budget_state.cost_per_probe) if is_probe else 0.0
            ),
            policy_kind="callable_recurrent_controller",
        )

    def observe(self, transition: ActingTransition) -> None:
        self.events.append(f"policy.observe:{int(transition.action.action_id)}")
        self.transitions.append(transition)

    def representation(self) -> np.ndarray:
        return np.asarray([len(self.transitions)], dtype=np.float32)

    def metrics(self) -> Mapping[str, Any]:
        return {
            "probe_count": sum(int(item.action.is_probe) for item in self.transitions),
            "realized_probe_cost": sum(
                float(item.realized_probe_cost) for item in self.transitions
            ),
            "environment_steps": sum(
                int(item.environment_steps) for item in self.transitions
            ),
        }


def test_structural_acting_policy_contract_exposes_full_lifecycle():
    policy = _StaticActingPolicy()
    assert isinstance(policy, PathCActingPolicy)
    budget = ProbeBudgetState(budget=1, cost_per_probe=0.25)
    policy.reset(seed=17, probe_budget_state=budget)
    action = policy.act({}, budget)
    policy.observe(_transition(action))
    assert action.is_probe is False
    assert np.array_equal(policy.representation(), np.asarray([17.0, 1.0]))
    assert policy.metrics() == {"seed": 17, "observed": 1}


def test_random_probe_uses_only_registered_support_and_reports_propensity():
    exploitation = _StaticActingPolicy()
    policy = RandomProbeActingPolicy(exploitation_policy=exploitation)
    budget = ProbeBudgetState(budget=1, cost_per_probe=0.25)
    policy.reset(seed=3, probe_budget_state=budget)
    action = policy.act(
        {"probe_candidate_action_ids": (2, 4)},
        budget,
    )
    assert action.action_id in {2, 4}
    assert action.is_probe is True
    assert action.propensity == pytest.approx(0.5)
    assert action.candidate_action_ids == (2, 4)
    assert action.estimated_probe_cost == pytest.approx(0.25)
    policy.observe(_transition(action))
    assert policy.metrics()["random_probe_count"] == 1
    assert policy.metrics()["probe_support_policy"] == (
        "matched_valid_action_support"
    )


def test_no_probe_wrapper_passes_zero_budget_and_rejects_probe_emission():
    exploitation = _StaticActingPolicy()
    policy = NoProbeActingPolicy(exploitation_policy=exploitation)
    assigned_budget = ProbeBudgetState(budget=4, cost_per_probe=0.25)
    policy.reset(seed=5, probe_budget_state=assigned_budget)
    assert exploitation.reset_budget is not None
    assert exploitation.reset_budget.budget == 0
    action = policy.act({"probe_candidate_action_ids": (2,)}, assigned_budget)
    assert action.is_probe is False
    assert policy.metrics()["probe_count"] == 0

    class _AlwaysProbe(_StaticActingPolicy):
        def act(self, observation, probe_budget_state):
            del observation, probe_budget_state
            return PolicyAction(
                action_id=2,
                is_probe=True,
                propensity=1.0,
                candidate_action_ids=(2,),
                candidate_scores=(1.0,),
                estimated_probe_cost=0.25,
                policy_kind="invalid_no_probe_emission",
            )

    invalid = NoProbeActingPolicy(exploitation_policy=_AlwaysProbe())
    invalid.reset(seed=0, probe_budget_state=assigned_budget)
    with pytest.raises(ValueError, match="attempted to emit a probe"):
        invalid.act({}, assigned_budget)


def test_belief_filter_baseline_acts_updates_and_reports_under_one_protocol():
    belief_filter = BayesianHMMBeliefFilter(
        HMMFilterConfig(
            mode_names=("yield", "claim"),
            transition_stay_prob=0.9,
        ),
        {
            "claim_signal": {"yield": 0.05, "claim": 0.95},
            "yield_signal": {"yield": 0.95, "claim": 0.05},
        },
    )
    policy = BeliefFilterActingPolicy(
        policy_name="exact_belief_filter",
        belief_filter=belief_filter,
        action_values_by_mode=np.asarray(
            [
                [1.0, 0.0, 4.0],
                [0.0, 5.0, 2.0],
            ]
        ),
    )
    assert isinstance(policy, PathCActingPolicy)
    budget = ProbeBudgetState(budget=1, cost_per_probe=0.25)
    policy.reset(seed=11, probe_budget_state=budget)
    observation = {
        "valid_action_ids": (0, 1, 2),
        "probe_candidate_action_ids": (2,),
        "probe_only_action_ids": (2,),
    }

    first = policy.act(observation, budget)
    assert first.action_id == 2
    assert first.is_probe is True
    policy.observe(_transition(first, token="claim_signal"))
    posterior = policy.representation()
    assert posterior.shape == (2,)
    assert posterior[1] > posterior[0]

    exhausted = budget.consume(is_probe=True)
    second = policy.act(observation, exhausted)
    assert second.action_id == 1
    assert second.is_probe is False
    policy.observe(_transition(second, token="claim_signal"))

    metrics = policy.metrics()
    assert metrics["acting_protocol_version"] == "path_c_acting_policy_v1"
    assert metrics["action_count"] == 2
    assert metrics["probe_count"] == 1
    assert metrics["environment_steps"] == 4
    assert metrics["realized_probe_cost"] == pytest.approx(0.25)
    assert metrics["oracle_baseline"] is False

    policy.reset(seed=11, probe_budget_state=budget)
    reset_metrics = policy.metrics()
    assert reset_metrics["action_count"] == 0
    assert reset_metrics["probe_count"] == 0


def test_belief_acting_policy_value_head_changes_with_public_state():
    belief_filter = BayesianHMMBeliefFilter(
        HMMFilterConfig(mode_names=("yield", "claim")),
        {"signal": {"yield": 0.5, "claim": 0.5}},
    )
    # rows are [one public feature, two belief coordinates, intercept]
    policy = BeliefFilterActingPolicy(
        policy_name="state_conditioned_belief",
        belief_filter=belief_filter,
        public_context_weights=np.asarray([
            [2.0, -2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
    )
    budget = ProbeBudgetState(budget=0, cost_per_probe=0.0)
    policy.reset(seed=0, probe_budget_state=budget)
    common = {
        "valid_action_ids": (0, 1),
        "probe_candidate_action_ids": (),
        "probe_only_action_ids": (),
    }
    positive = policy.act(
        {**common, "belief_public_context": np.asarray([1.0])},
        budget,
    )
    negative = policy.act(
        {**common, "belief_public_context": np.asarray([-1.0])},
        budget,
    )
    assert positive.action_id == 0
    assert negative.action_id == 1
    assert policy.metrics()["state_conditioned_action_values"] is True


def test_common_runner_executes_lifecycle_and_emits_one_return_budget_point():
    events: list[str] = []
    environment = _TwoDecisionEnvironment(events)
    assert isinstance(environment, PathCActingEnvironment)
    controller = _CallableController(events)
    policy = CallableActingPolicy(
        policy_name="full_history_rnn",
        action_fn=controller,
        observe_fn=controller.observe,
        representation_fn=controller.representation,
        metrics_fn=controller.metrics,
    )
    runner = PathCActingRunner(
        cost_per_probe=0.25,
        maximum_environment_steps=4,
    )
    episode = runner.run_episode(
        environment=environment,
        policy=policy,
        spec=ActingEpisodeSpec(
            split_role="locked_audit",
            episode_uid="locked:full_history_rnn:held_out_identity_3:7:1",
            split_group_id="locked-group-3",
            mechanism="mechanism-a",
            identity_group="held_out_identity_3",
            style_group="style-3",
            layout_group="cramped_room",
            seed_group="seed_7",
            seed=7,
            probe_budget=1,
            training_environment_steps=1000,
            gradient_updates=12,
            evaluation_environment_step_limit=4,
            evaluation_schedule_id="locked_schedule_v1",
        ),
    )

    assert events == [
        "environment.reset:7",
        "policy.reset:7",
        "policy.act:1",
        "environment.step:2",
        "policy.observe:2",
        "policy.act:0",
        "environment.step:0",
        "policy.observe:0",
    ]
    assert episode.raw_return == pytest.approx(5.0)
    assert episode.realized_probe_cost == pytest.approx(0.25)
    assert episode.net_return == pytest.approx(4.75)
    assert episode.environment_steps == 3
    assert episode.probes_used == 1
    assert episode.steps[0].budget_before.probes_used == 0
    assert episode.steps[0].budget_after.probes_used == 1
    assert episode.steps[0].net_reward == pytest.approx(2.75)
    assert episode.steps[1].realized_probe_cost == 0.0

    point = episode.return_probe_budget_point()
    assert isinstance(point, ReturnProbeBudgetPoint)
    assert point.raw_return == pytest.approx(5.0)
    assert point.realized_probe_cost == pytest.approx(0.25)
    assert point.net_return == pytest.approx(4.75)
    assert point.episode_environment_steps == 3
    assert point.training_environment_steps == 1000
    assert point.evaluation_environment_step_limit == 4
    assert point.gradient_updates == 12


def test_runner_rejects_wrong_probe_price_before_environment_step():
    class _WrongCostPolicy(_StaticActingPolicy):
        def act(self, observation, probe_budget_state):
            del observation, probe_budget_state
            return PolicyAction(
                action_id=2,
                is_probe=True,
                propensity=1.0,
                candidate_action_ids=(2,),
                candidate_scores=(1.0,),
                estimated_probe_cost=0.1,
                policy_kind="wrong_probe_price",
            )

    environment = _TwoDecisionEnvironment([])
    runner = PathCActingRunner(
        cost_per_probe=0.25,
        maximum_environment_steps=4,
    )
    with pytest.raises(ValueError, match="outside the assigned contract"):
        runner.run_episode(
            environment=environment,
            policy=_WrongCostPolicy(),
            spec=ActingEpisodeSpec(
                split_role="design",
                episode_uid="design:wrong-cost:identity:0:1",
                split_group_id="design-group-0",
                mechanism="mechanism-a",
                identity_group="identity",
                style_group="style-0",
                layout_group="layout",
                seed_group="seed",
                seed=0,
                probe_budget=1,
                training_environment_steps=1000,
                gradient_updates=0,
                evaluation_environment_step_limit=4,
                evaluation_schedule_id="schedule",
            ),
        )
    assert environment.step_calls == 0

    with pytest.raises(ValueError, match="non-probe action"):
        PolicyAction(
            action_id=0,
            is_probe=False,
            propensity=1.0,
            candidate_action_ids=(0,),
            candidate_scores=(1.0,),
            estimated_probe_cost=0.01,
            policy_kind="wrong_non_probe_price",
        )


def test_runner_requires_policy_metrics_to_reconcile_with_its_cost_ledger():
    class _MissingLedgerMetrics(_CallableController):
        def metrics(self) -> Mapping[str, Any]:
            return {"probe_count": 1}

    events: list[str] = []
    controller = _MissingLedgerMetrics(events)
    policy = CallableActingPolicy(
        policy_name="incomplete_metrics_rnn",
        action_fn=controller,
        observe_fn=controller.observe,
        representation_fn=controller.representation,
        metrics_fn=controller.metrics,
    )
    with pytest.raises(ValueError, match="authoritative realized_probe_cost"):
        PathCActingRunner(
            cost_per_probe=0.25,
            maximum_environment_steps=4,
        ).run_episode(
            environment=_TwoDecisionEnvironment(events),
            policy=policy,
            spec=ActingEpisodeSpec(
                split_role="design",
                episode_uid="design:incomplete-metrics:identity:0:1",
                split_group_id="design-group-0",
                mechanism="mechanism-a",
                identity_group="identity",
                style_group="style-0",
                layout_group="layout",
                seed_group="seed",
                seed=0,
                probe_budget=1,
                training_environment_steps=1000,
                gradient_updates=0,
                evaluation_environment_step_limit=4,
                evaluation_schedule_id="schedule",
            ),
        )


def test_scripted_learned_and_particle_filters_share_online_filter_protocol():
    config = HMMFilterConfig(
        mode_names=("yield", "claim"),
        transition_stay_prob=0.9,
    )
    likelihood = {
        "claim_signal": {"yield": 0.01, "claim": 0.99},
        "yield_signal": {"yield": 0.99, "claim": 0.01},
    }
    scripted = BayesianHMMBeliefFilter(config, likelihood)
    learned = BayesianHMMBeliefFilter.fit_from_training_split(
        config,
        observations=(
            "yield_signal",
            "yield_signal",
            "claim_signal",
            "claim_signal",
        ),
        mode_labels=("yield", "yield", "claim", "claim"),
    )
    particle = ParticleBeliefFilter(
        ParticleFilterConfig(
            mode_names=("yield", "claim"),
            particle_count=2048,
            transition_stay_prob=0.9,
        ),
        likelihood,
    )

    for belief_filter in (scripted, learned, particle):
        assert isinstance(belief_filter, OnlineModeBeliefFilter)
        belief_filter.reset(seed=19)
        posterior = belief_filter.update("claim_signal")
        assert posterior.shape == (2,)
        assert posterior[1] > posterior[0]
    assert scripted.inference_kind == "scripted_hmm_filter"
    assert learned.inference_kind == "learned_hmm_filter"
    assert particle.inference_kind == "particle_belief_filter"

    particle.reset(seed=23)
    first_trace = particle.update("claim_signal")
    particle.reset(seed=23)
    second_trace = particle.update("claim_signal")
    assert np.array_equal(first_trace, second_trace)


def test_particle_filter_is_deployable_through_belief_acting_policy():
    particle = ParticleBeliefFilter(
        ParticleFilterConfig(
            mode_names=("yield", "claim"),
            particle_count=2048,
            transition_stay_prob=0.9,
        ),
        {
            "claim_signal": {"yield": 0.01, "claim": 0.99},
            "yield_signal": {"yield": 0.99, "claim": 0.01},
        },
    )
    policy = BeliefFilterActingPolicy(
        policy_name="particle_belief_filter",
        belief_filter=particle,
        action_values_by_mode=np.asarray(
            [
                [1.0, 0.0, 4.0],
                [0.0, 5.0, 2.0],
            ]
        ),
    )
    budget = ProbeBudgetState(budget=1, cost_per_probe=0.25)
    policy.reset(seed=31, probe_budget_state=budget)
    action = policy.act(
        {
            "valid_action_ids": (0, 1, 2),
            "probe_candidate_action_ids": (2,),
            "probe_only_action_ids": (2,),
        },
        budget,
    )
    assert action.action_id == 2
    assert action.is_probe is True
    policy.observe(_transition(action, token="claim_signal"))
    assert policy.representation()[1] > policy.representation()[0]
    metrics = policy.metrics()
    assert metrics["inference_kind"] == "particle_belief_filter"
    assert metrics["particle_count"] == 2048
    assert 0.0 < metrics["effective_sample_size"] <= 2048.0


def test_belief_acting_policy_optimizes_probe_cost_adjusted_action_value():
    belief_filter = BayesianHMMBeliefFilter(
        HMMFilterConfig(mode_names=("single_mode",)),
        {"signal": {"single_mode": 1.0}},
    )
    policy = BeliefFilterActingPolicy(
        policy_name="exact_belief_filter",
        belief_filter=belief_filter,
        action_values_by_mode=np.asarray([[1.0, 1.1]]),
    )
    budget = ProbeBudgetState(budget=1, cost_per_probe=0.25)
    policy.reset(seed=0, probe_budget_state=budget)
    action = policy.act(
        {
            "valid_action_ids": (0, 1),
            "probe_candidate_action_ids": (1,),
            "probe_only_action_ids": (1,),
        },
        budget,
    )
    assert action.action_id == 0
    assert action.is_probe is False
    assert action.estimated_probe_cost == 0.0


def _formal_benchmark_contract() -> FormalActingBenchmarkContract:
    return FormalActingBenchmarkContract(
        split_manifest_sha256="a" * 64,
        training_environment_steps=40000,
        gradient_updates=40000,
        evaluation_environment_step_limit=1000,
        evaluation_schedule_id="schedule-v1",
        probe_budget_grid=(0, 1, 2),
        probe_cost_per_use=0.25,
        action_support_sha256="b" * 64,
        effective_episode_floor=2000,
        effective_transition_floor=40000,
    )


def _formal_factory_entry(
    policy_name: str,
    *,
    available: bool = True,
) -> FormalActingPolicyFactoryEntry:
    def factory(spec, environment):
        del spec, environment
        return CallableActingPolicy(
            policy_name=policy_name,
            action_fn=lambda observation, budget: PolicyAction(
                action_id=0,
                is_probe=False,
                propensity=1.0,
                candidate_action_ids=(0,),
                candidate_scores=(0.0,),
                estimated_probe_cost=0.0,
                policy_kind="test_factory",
            ),
            observe_fn=lambda transition: None,
            representation_fn=lambda: np.zeros(1, dtype=np.float32),
            metrics_fn=lambda: {
                "probe_count": 0,
                "environment_steps": 0,
                "realized_probe_cost": 0.0,
            },
        )

    return FormalActingPolicyFactoryEntry(
        policy_name=policy_name,
        factory=factory if available else None,
        factory_id=f"test:{policy_name}",
        allowed_split_roles=(
            ("design",)
            if policy_name == "direct_information"
            else ("design", "locked_audit")
        ),
        unavailable_reason=None if available else "artifact is absent",
    )


def test_formal_factory_registry_has_nine_design_and_eight_locked_policies():
    registry = FormalActingPolicyFactoryRegistry(
        entries=tuple(
            _formal_factory_entry(name) for name in FORMAL_ACTING_POLICY_NAMES
        ),
        benchmark_contract=_formal_benchmark_contract(),
    )
    assert tuple(registry.factories_for_role("design")) == FORMAL_ACTING_POLICY_NAMES
    assert tuple(registry.factories_for_role("locked_audit")) == (
        FORMAL_LOCKED_ACTING_POLICY_NAMES
    )
    assert "direct_information" not in registry.factories_for_role("locked_audit")


def test_formal_factory_registry_reports_missing_policy_before_rollout():
    registry = FormalActingPolicyFactoryRegistry(
        entries=tuple(
            _formal_factory_entry(
                name,
                available=name != "rnn_residualized",
            )
            for name in FORMAL_ACTING_POLICY_NAMES
        ),
        benchmark_contract=_formal_benchmark_contract(),
    )
    with pytest.raises(FormalActingFactoryUnavailableError) as captured:
        registry.factories_for_role("design")
    assert captured.value.missing_by_policy == {
        "rnn_residualized": "artifact is absent"
    }


def test_direct_information_policy_rejects_nonfinite_response_table():
    with pytest.raises(ValueError, match="finite"):
        DirectInformationActingPolicy(
            response_probabilities=np.asarray([[[np.nan, 1.0]]]),
            exploitation_policy=_StaticActingPolicy(),
        )


def test_belief_training_artifact_rejects_non_train_collection_role(tmp_path):
    from experiments.overcooked_v2.evaluate_aris import (
        _validated_belief_training_artifact,
    )

    payload = {
        "schema_version": "path_c_belief_training_artifact_v2",
        "collection_role": "design",
        "split_manifest_sha256": "a" * 64,
        "effective_episodes": 2000,
        "effective_transitions": 40000,
        "training_environment_steps": 40000,
        "gradient_updates": 40000,
        "trainable_parameters": 1,
        "training_flops": 1.0,
        "wall_clock_seconds": 1.0,
        "inference_latency_ms": 0.1,
        "mode_names": ["mode"],
        "observations": ["response"],
        "mode_labels": ["mode"],
        "public_context_key": "belief_public_context",
        "public_context_dim": 1,
        "public_context_weights": [[1.0], [0.0], [0.0]],
        "action_value_provenance": {
            "schema_version": "path_c_action_value_provenance_v1",
            "collection_role": "train",
            "split_manifest_sha256": "a" * 64,
            "source_artifact": {"path": "unused", "sha256": "c" * 64},
            "estimator_id": "train_only_value_fit_v1",
        },
    }
    with pytest.raises(ValueError, match="collection_role=train"):
        _validated_belief_training_artifact(
            payload,
            mode_names=("mode",),
            root=tmp_path,
            benchmark_contract=_formal_benchmark_contract(),
            require_observations=True,
        )
