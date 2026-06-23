import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from toy_factor_game.ce_estimation import estimate_ce_matrix, induce_graph  # noqa: E402
from toy_factor_game.env import DELIVER_RIGHT, Action, ToyFactorGameEnv  # noqa: E402
from toy_factor_game.evidence import option_relevance_mask  # noqa: E402
from toy_factor_game.graph_config import get_graph_config  # noqa: E402
from toy_factor_game.options import NUM_OPTIONS, OptionID  # noqa: E402
from toy_factor_game.policy import ActiveFactorAgent, uniform_marginals  # noqa: E402


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


def test_oracle_method_split_dispatches_to_matching_q_architecture():
    graph = get_graph_config("full_support")
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)

    factor_agent = make_agent(graph, method="oracle_belief_factorq")
    flat_agent = make_agent(graph, method="oracle_belief_flatq")

    assert torch.allclose(factor_agent.q_values(obs, marginals), factor_agent.factor_q(obs, marginals))
    assert torch.allclose(flat_agent.q_values(obs, marginals), flat_agent.flat_q(obs, marginals))


def test_full_support_is_induced_from_all_option_pair_ce_matrix():
    ce_matrix = estimate_ce_matrix()
    expected_pairs = {(option_i, option_j) for option_i, option_j, _value in induce_graph(ce_matrix)}
    graph = get_graph_config("full_support")
    actual_pairs = {(factor.option_i, factor.option_j) for factor in graph.factors}
    assert actual_pairs == expected_pairs


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
        make_agent(get_graph_config("full_support"), method="oracle_belief")


def test_task_completion_terminates_episode():
    env = ToyFactorGameEnv(max_steps=50, seed=0)
    env.ego_pos = list(DELIVER_RIGHT)
    env.ego_carrying = True
    env.deliveries_left = 1
    _obs, _reward, done, info = env.step(int(Action.DROP), partner_action_override=int(Action.NOOP))
    assert done
    assert info["completed"]
