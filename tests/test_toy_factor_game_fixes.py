import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from toy_factor_game.ce_estimation import _all_conventions, estimate_ce_matrix, induce_graph  # noqa: E402
from toy_factor_game.env import DELIVER_RIGHT, Action, ToyFactorGameEnv  # noqa: E402
from toy_factor_game.evaluate import (  # noqa: E402
    EVAL_SCHEMA,
    EXP3_ROUTING_CONDITIONS,
    EXPERIMENT_SCHEMA,
    oracle_planner_option_value,
    oracle_planner_select,
    per_factor_swap_diagnostics,
)
from toy_factor_game.evidence import option_relevance_mask  # noqa: E402
from toy_factor_game.graph_config import get_graph_config  # noqa: E402
from toy_factor_game.options import NUM_OPTIONS, OptionID, get_option_action, get_option_cost, get_valid_options  # noqa: E402
from toy_factor_game.policy import ActiveFactorAgent, uniform_marginals  # noqa: E402
from toy_factor_game.train import TRAIN_METHODS  # noqa: E402


def make_agent(graph, method="aris_bellman", hidden_dim=16):
    env = ToyFactorGameEnv()
    return ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph,
        hidden_dim=hidden_dim,
        method=method,
    )


def test_prior_centered_residual_zero_at_uniform():
    graph = get_graph_config("full_support")
    agent = make_agent(graph)
    obs = torch.randn(3, ToyFactorGameEnv().obs_dim)
    q_base = agent.q_base_values(obs)
    q_uniform = agent.q_uniform_values(obs)
    assert torch.allclose(q_uniform, q_base, atol=1e-6)


def test_irrelevant_factor_belief_does_not_affect_non_relevant_option_q():
    graph = get_graph_config("overcomplete")
    mask = option_relevance_mask(graph)
    option = int(OptionID.GOTO_RESOURCE_A)
    factor_idx = next(idx for idx, row in enumerate(mask) if not row[option])
    agent = make_agent(graph)
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)
    q_before = agent.q_values(obs, marginals)
    changed = [m.clone() for m in marginals]
    changed[factor_idx] = torch.zeros_like(changed[factor_idx])
    changed[factor_idx][0, -1] = 1.0
    q_after = agent.q_values(obs, changed)
    assert torch.allclose(q_before[:, option], q_after[:, option], atol=1e-6)


def test_base_only_ignores_belief():
    graph = get_graph_config("full_support")
    agent = make_agent(graph, method="base_only")
    obs = torch.zeros(2, ToyFactorGameEnv().obs_dim)
    uniform = uniform_marginals(graph.factor_modes, 2, obs.device)
    changed = [m.clone() for m in uniform]
    for marginal in changed:
        marginal.zero_()
        marginal[:, 0] = 1.0
    assert torch.allclose(agent.q_values(obs, uniform), agent.q_values(obs, changed))
    assert torch.allclose(agent.q_values(obs, uniform), agent.q_base_values(obs))


def test_true_belief_methods_dispatch_to_matching_q_architecture():
    graph = get_graph_config("full_support")
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)

    factor_agent = make_agent(graph, method="true_belief_factorq")
    flat_agent = make_agent(graph, method="true_belief_flatq")

    assert torch.allclose(factor_agent.q_values(obs, marginals), factor_agent.factor_q(obs, marginals))
    assert torch.allclose(flat_agent.q_values(obs, marginals), flat_agent.flat_q(obs, marginals))


def test_full_support_is_induced_from_all_option_pair_ce_matrix():
    ce_matrix = estimate_ce_matrix(n_conventions=None, seed=42)
    expected_pairs = {(option_i, option_j) for option_i, option_j, _value in induce_graph(ce_matrix)}
    graph = get_graph_config("full_support")
    actual_pairs = {(factor.option_i, factor.option_j) for factor in graph.factors}
    assert actual_pairs == expected_pairs


def test_ce_default_uses_all_conventions_and_metadata_records_it():
    ce_default = estimate_ce_matrix(n_conventions=None, seed=42)
    ce_all = estimate_ce_matrix(n_conventions=len(_all_conventions()), seed=42)
    assert np.allclose(ce_default, ce_all)

    graph = get_graph_config("full_support")
    metadata = graph.ce_metadata()
    assert metadata["mode"] == "all_conventions"
    assert metadata["n_conventions"] == len(_all_conventions())
    assert metadata["n_scenarios"] == 3


def test_shuffled_route_and_relevance_ablate_different_graph_edges():
    full = get_graph_config("full_support")
    shuffled_routes = get_graph_config("shuffled_routes")
    shuffled_relevance = get_graph_config("shuffled_relevance")

    assert [f.name for f in shuffled_routes.factors] == [f.name for f in full.factors]
    assert [f.name for f in shuffled_relevance.factors] == [f.name for f in full.factors]
    assert shuffled_routes.route_permutation is not None
    assert shuffled_routes.relevance_permutation is None
    assert shuffled_relevance.route_permutation is None
    assert shuffled_relevance.relevance_permutation is not None

    if full.n_factors > 1:
        assert shuffled_routes.route_permutation != tuple(range(full.n_factors))
        assert option_relevance_mask(shuffled_relevance) != option_relevance_mask(full)


def test_old_method_and_graph_names_fail_visibly():
    with pytest.raises(ValueError):
        get_graph_config("full_graph")
    with pytest.raises(ValueError):
        get_graph_config("overcomplete_minus_noncritical")
    assert get_graph_config("overcomplete_minus_low_ce").name == "overcomplete_minus_low_ce"
    with pytest.raises(ValueError):
        make_agent(get_graph_config("full_support"), method="oracle_belief")
    with pytest.raises(ValueError):
        make_agent(get_graph_config("full_support"), method="oracle_belief_factorq")
    with pytest.raises(ValueError):
        make_agent(get_graph_config("full_support"), method="oracle_belief_flatq")


def test_oracle_planner_is_evaluation_only():
    graph = get_graph_config("full_support")
    assert "oracle_planner" not in TRAIN_METHODS
    assert "random_policy" not in TRAIN_METHODS
    planner_agent = make_agent(graph, method="oracle_planner")
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)
    with pytest.raises(RuntimeError):
        planner_agent.q_values(obs, marginals)


def test_oracle_planner_horizon_one_matches_one_step_enumeration():
    env = ToyFactorGameEnv(max_steps=5, seed=0)
    gamma = 0.99
    selected = oracle_planner_select(env, horizon=1, gamma=gamma)
    enumerated = []
    for option in get_valid_options(env):
        sim_env = copy.deepcopy(env)
        action = get_option_action(option, sim_env.ego_pos, sim_env.ego_carrying)
        _obs, reward, _done, _info = sim_env.step(action)
        enumerated.append((float(reward - get_option_cost(option)), int(option), option))
    enumerated.sort(key=lambda item: (-item[0], item[1]))
    assert selected == enumerated[0][2]


def test_oracle_planner_higher_horizon_not_worse_when_first_step_completes():
    env = ToyFactorGameEnv(max_steps=5, seed=0)
    env.ego_pos = list(DELIVER_RIGHT)
    env.ego_carrying = True
    env.deliveries_left = 1
    option = OptionID.DROP
    value_h1 = oracle_planner_option_value(env, option, horizon=1, gamma=0.99)
    value_h2 = oracle_planner_option_value(env, option, horizon=2, gamma=0.99)
    assert value_h2 >= value_h1


def test_per_factor_swap_reports_maxq_and_action_flip_fields():
    graph = get_graph_config("full_support")
    agent = make_agent(graph)
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)
    valid_mask = torch.ones(1, NUM_OPTIONS, dtype=torch.bool)
    rows = per_factor_swap_diagnostics(agent, obs, marginals, valid_mask)
    assert rows
    assert {"selected_delta", "maxq_delta", "abs_maxq_delta", "action_flip"}.issubset(rows[0])


def test_exp3_contains_routing_and_relevance_controls():
    assert ("aris_bellman", "shuffled_routes") in EXP3_ROUTING_CONDITIONS
    assert ("aris_bellman", "shuffled_relevance") in EXP3_ROUTING_CONDITIONS
    assert ("aris_bellman", "random_same_size") in EXP3_ROUTING_CONDITIONS


def test_schema_bumped_to_v4_1():
    assert EXPERIMENT_SCHEMA == "aris_bellman_v4.1"
    assert EVAL_SCHEMA == "aris_bellman_eval_v4.1"


def test_task_completion_terminates_episode():
    env = ToyFactorGameEnv(max_steps=50, seed=0)
    env.ego_pos = list(DELIVER_RIGHT)
    env.ego_carrying = True
    env.deliveries_left = 1
    _obs, _reward, done, info = env.step(int(Action.DROP), partner_action_override=int(Action.NOOP))
    assert done
    assert info["completed"]
