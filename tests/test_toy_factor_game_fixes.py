import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from toy_factor_game.env import (  # noqa: E402
    DELIVER_RIGHT,
    RESOURCE_A,
    RESOURCE_B,
    Action,
    ConventionAssignment,
    ToyFactorGameEnv,
)
from toy_factor_game.evidence import route_event_to_factors, option_relevance_mask  # noqa: E402
from toy_factor_game.graph_config import get_graph_config  # noqa: E402
from toy_factor_game.options import NUM_OPTIONS, OptionID  # noqa: E402
from toy_factor_game.policy import ActiveFactorAgent, uniform_marginals  # noqa: E402
from toy_factor_game.train import collect_episode, train_step  # noqa: E402
from toy_factor_game import gtvoi, policy  # noqa: E402


def make_event(**overrides):
    event = {
        "ego_pos_before": (0, 0),
        "partner_pos_before": (6, 6),
        "ego_pos_after": (0, 1),
        "partner_pos_after": (6, 5),
        "ego_carrying_before": False,
        "partner_carrying_before": False,
        "ego_carrying_after": False,
        "partner_carrying_after": False,
        "resource_a_available_before": True,
        "resource_b_available_before": True,
        "resource_a_available_after": True,
        "resource_b_available_after": True,
        "deliveries_left_before": 0,
        "deliveries_right_before": 0,
        "deliveries_left_after": 0,
        "deliveries_right_after": 0,
        "ego_action": int(Action.RIGHT),
        "partner_action": int(Action.LEFT),
        "collision": False,
        "completed": False,
        "ego_option": int(OptionID.GOTO_RESOURCE_A),
    }
    event.update(overrides)
    return event


def test_unrelated_event_does_not_change_bottleneck_evidence():
    graph = get_graph_config("full_graph")
    bottleneck_idx = next(i for i, factor in enumerate(graph.factors) if factor.env_factor_id == 0)
    base = route_event_to_factors(make_event(), graph)[bottleneck_idx]
    resource_only = route_event_to_factors(
        make_event(
            ego_pos_after=RESOURCE_A,
            ego_action=int(Action.PICKUP),
            resource_a_available_after=False,
        ),
        graph,
    )[bottleneck_idx]
    assert base == resource_only


def test_deleted_factor_removes_relevance_route():
    full = get_graph_config("full_graph")
    minus = get_graph_config("minus_critical")
    removed_ids = {factor.env_factor_id for factor in full.factors} - {
        factor.env_factor_id for factor in minus.factors
    }
    assert removed_ids
    for factor in minus.factors:
        assert factor.env_factor_id not in removed_ids
    mask = option_relevance_mask(minus)
    assert len(mask) == minus.n_factors


def test_unrelated_factor_belief_does_not_affect_action_q():
    graph = get_graph_config("plus_irrelevant")
    noop_idx = next(i for i, factor in enumerate(graph.factors) if factor.option_i == OptionID.NOOP)
    agent = ActiveFactorAgent(
        obs_dim=ToyFactorGameEnv().obs_dim,
        n_actions=ToyFactorGameEnv().n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph,
        hidden_dim=16,
        method="aris_bellman",
    )
    obs = torch.zeros(1, ToyFactorGameEnv().obs_dim)
    marginals = uniform_marginals(graph.factor_modes, 1, obs.device)
    q_before = agent.q_values(obs, marginals)
    changed = [m.clone() for m in marginals]
    changed[noop_idx] = torch.zeros_like(changed[noop_idx])
    changed[noop_idx][0, -1] = 1.0
    q_after = agent.q_values(obs, changed)
    assert torch.allclose(
        q_before[:, int(OptionID.GOTO_RESOURCE_A)],
        q_after[:, int(OptionID.GOTO_RESOURCE_A)],
    )


def test_deprecated_selector_and_auxiliary_training_classes_are_absent():
    assert not hasattr(policy, "ResponsePredictor")
    assert not hasattr(policy, "BeliefTransitionModel")
    assert not hasattr(gtvoi, "OptionSelector")


def test_train_step_reports_only_bellman_losses():
    device = torch.device("cpu")
    graph = get_graph_config("full_graph")
    env = ToyFactorGameEnv(
        partner_convention=ConventionAssignment({0: 0, 1: 0, 2: 0}),
        max_steps=4,
        seed=0,
    )
    agent = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph,
        hidden_dim=16,
        method="aris_bellman",
    )
    target = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph,
        hidden_dim=16,
        method="aris_bellman",
    )
    target.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    episode = collect_episode(env, agent, device, graph, explore_eps=1.0)
    losses = train_step(agent, target, optimizer, episode, device, graph)
    assert {"total", "td", "q_pred_mean", "q_target_mean"}.issubset(losses)
    assert "response" not in losses
    assert "calibration" not in losses
    assert "sparsity" not in losses
    assert "control" not in losses


def test_task_completion_terminates_episode():
    env = ToyFactorGameEnv(max_steps=50, seed=0)
    env.ego_pos = list(DELIVER_RIGHT)
    env.ego_carrying = True
    env.deliveries_left = 1
    _obs, _reward, done, info = env.step(int(Action.DROP), partner_action_override=int(Action.NOOP))
    assert done
    assert info["completed"]
