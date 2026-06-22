import ast
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common.metrics import pareto_frontier
from toy_factor_game.ce_estimation import DEFAULT_CE_THRESHOLD, estimate_ce_matrix, induce_graph
from toy_factor_game.evaluate import exp1_gtvoi_vs_mi, first_alignment_time, real_factor_belief_aligned
from toy_factor_game.env import Action, ToyFactorGameEnv
from toy_factor_game.graph_config import _candidate_factor_specs, get_graph_config, stable_convention_seed
from toy_factor_game.gtvoi import OptionSelector, belief_to_features, compute_gtvoi
from toy_factor_game.factor_belief import FactorBeliefModel
from toy_factor_game.options import GROUND_TRUTH_FACTORS, NUM_OPTIONS, OptionID
from toy_factor_game.policy import ActiveFactorAgent, brier_calibration_loss
from toy_factor_game.train import MODES, batch_episodes, collect_episode, normalize_advantages, train_step


def make_agent(graph_config):
    env = ToyFactorGameEnv(max_steps=5, seed=0)
    return ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        n_factors=graph_config.n_factors,
        factor_modes=graph_config.factor_modes,
        hidden_dim=16,
        pairwise_pairs=list(graph_config.pairwise_pairs),
    )


def test_brier_calibration_loss_rewards_true_mode_probability():
    labels = torch.tensor([[1]])
    good = [torch.tensor([[0.05, 0.90, 0.05]])]
    bad = [torch.tensor([[0.45, 0.10, 0.45]])]

    assert brier_calibration_loss(good, labels, [3]) < brier_calibration_loss(bad, labels, [3])


def test_brier_calibration_ignores_synthetic_factor_labels():
    labels = torch.tensor([[1, 0]])
    gt = torch.tensor([[0.05, 0.90, 0.05]])
    synthetic_bad = torch.tensor([[0.0, 1.0]])
    synthetic_good = torch.tensor([[1.0, 0.0]])

    loss_bad = brier_calibration_loss([gt, synthetic_bad], labels, [3, 2], gt_mask=[True, False])
    loss_good = brier_calibration_loss([gt, synthetic_good], labels, [3, 2], gt_mask=[True, False])

    assert torch.allclose(loss_bad, loss_good)


def test_belief_connected_value_loss_backpropagates_to_belief_model():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    obs_seq = torch.randn(1, 3, agent.obs_dim)
    ego_seq = torch.zeros(1, 3, agent.n_actions)
    partner_seq = torch.zeros(1, 3, agent.n_actions)
    ego_seq[:, :, 0] = 1.0
    partner_seq[:, :, 0] = 1.0
    partner_next = torch.tensor([0])
    labels = torch.tensor([graph_config.labels_from_convention({0: 0, 1: 0, 2: 0})])

    losses = agent.compute_factor_losses(
        obs_seq,
        ego_seq,
        partner_seq,
        partner_next,
        labels,
        obs_at_eval=obs_seq[:, -1],
        value_target=torch.tensor([1.0]),
        value_coef=1.0,
        gt_mask=graph_config.ground_truth_mask,
    )
    agent.zero_grad()
    losses["belief_value"].backward()

    grad_sum = sum(
        float(param.grad.abs().sum())
        for param in agent.belief_model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0.0


def test_compute_factor_losses_intermediates_match_belief_model_outputs():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    obs_seq = torch.randn(1, 3, agent.obs_dim)
    ego_seq = torch.zeros(1, 3, agent.n_actions)
    partner_seq = torch.zeros(1, 3, agent.n_actions)
    ego_seq[:, :, 0] = 1.0
    partner_seq[:, :, 0] = 1.0
    labels = torch.tensor([graph_config.labels_from_convention({0: 0, 1: 0, 2: 0})])
    losses = agent.compute_factor_losses(
        obs_seq,
        ego_seq,
        partner_seq,
        torch.tensor([0]),
        labels,
        return_intermediates=True,
    )
    h = agent.belief_model.encode_history(obs_seq, ego_seq, partner_seq)
    marginals = agent.belief_model._marginals_from_h(h)

    assert torch.allclose(losses["h"], h)
    for got, expected in zip(losses["marginals"], marginals):
        assert torch.allclose(got, expected)


def test_gtvoi_is_pure_value_delta_without_cost_penalty():
    class SumValue(torch.nn.Module):
        def forward(self, obs, belief_features):
            return belief_features.sum(dim=-1)

    obs = torch.zeros(1, 2)
    current = torch.tensor([[1.0, 2.0, 3.0]])
    simulated = torch.tensor([[2.0, 3.0, 5.0]])

    assert torch.allclose(compute_gtvoi(SumValue(), obs, current, simulated), torch.tensor([4.0]))


def test_transition_model_predicts_normalized_option_dependent_posteriors():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    marginals = [torch.ones(1, n_modes) / n_modes for n_modes in graph_config.factor_modes]
    belief_features = belief_to_features(marginals)
    option0 = torch.nn.functional.one_hot(torch.tensor([0]), NUM_OPTIONS).float()
    option1 = torch.nn.functional.one_hot(torch.tensor([1]), NUM_OPTIONS).float()

    after0 = agent.belief_transition.forward_marginals(belief_features, option0)
    after1 = agent.belief_transition.forward_marginals(belief_features, option1)

    for marginal in after0 + after1:
        assert torch.allclose(marginal.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert sum(float((a - b).abs().sum()) for a, b in zip(after0, after1)) > 1e-6


def test_batched_option_transition_matches_per_option_loop():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    marginals = [torch.ones(1, n_modes) / n_modes for n_modes in graph_config.factor_modes]
    belief_features = belief_to_features(marginals)
    batched_marginals, batched_features = agent.belief_transition.forward_all_options(belief_features)

    for option_id in range(NUM_OPTIONS):
        option_onehot = torch.nn.functional.one_hot(torch.tensor([option_id]), NUM_OPTIONS).float()
        loop_marginals = agent.belief_transition.forward_marginals(belief_features, option_onehot)
        loop_features = belief_to_features(loop_marginals)
        for got, expected in zip(batched_marginals[option_id], loop_marginals):
            assert torch.allclose(got, expected)
        assert torch.allclose(batched_features[option_id], loop_features)


def test_packed_gru_ignores_padded_timesteps():
    model = FactorBeliefModel(obs_dim=4, n_actions=2, n_factors=1, factor_modes=[2], hidden_dim=4)
    obs = torch.randn(2, 4, 4)
    ego = torch.zeros(2, 4, 2)
    partner = torch.zeros(2, 4, 2)
    ego[:, :, 0] = 1.0
    partner[:, :, 0] = 1.0
    lengths = torch.tensor([4, 2])
    obs[1, 2:] = torch.randn(2, 4) * 100.0

    h_packed = model.encode_history(obs, ego, partner, lengths=lengths)
    h_short = model.encode_history(obs[1:2, :2], ego[1:2, :2], partner[1:2, :2])

    assert torch.allclose(h_packed[1], h_short[0], atol=1e-6)


def test_factor_belief_bp_matches_two_node_sum_product():
    model = FactorBeliefModel(
        obs_dim=4,
        n_actions=2,
        n_factors=2,
        factor_modes=[2, 2],
        hidden_dim=4,
        pairwise_pairs=[(0, 1)],
    )
    obs_seq = torch.randn(1, 2, 4)
    act_seq = torch.zeros(1, 2, 2)
    act_seq[:, :, 0] = 1.0

    marginals = model(obs_seq, act_seq, act_seq)
    h = model.encode_history(obs_seq, act_seq, act_seq)
    unary0 = model.unary_potentials[0](h)
    unary1 = model.unary_potentials[1](h)
    pairwise = model.pairwise_potentials[0](h)
    joint_logits = unary0.unsqueeze(-1) + unary1.unsqueeze(-2) + pairwise
    joint = torch.softmax(joint_logits.view(1, -1), dim=-1).view(1, 2, 2)

    assert torch.allclose(marginals[0], joint.sum(dim=-1), atol=1e-5)
    assert torch.allclose(marginals[1], joint.sum(dim=-2), atol=1e-5)


def test_pareto_frontier_scans_from_low_cost_to_high_cost():
    costs = np.array([1.0, 2.0, 3.0])
    rewards = np.array([5.0, 4.0, 6.0])

    assert pareto_frontier(costs, rewards).tolist() == [0, 2]


def test_pareto_frontier_keeps_low_cost_tradeoff_points():
    costs = np.array([2.0, 3.0, 0.5, 0.0])
    rewards = np.array([8.0, 7.0, 6.0, 3.0])

    assert pareto_frontier(costs, rewards).tolist() == [3, 2, 0]


def test_normalize_advantages_keeps_singleton_and_standardizes_batch():
    single = torch.tensor([3.0])
    batch = torch.tensor([1.0, 2.0, 3.0])
    normalized = normalize_advantages(batch)

    assert torch.equal(normalize_advantages(single), single)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_passive_scores_include_cost_and_random_samples_valid_options():
    class FixedQ(torch.nn.Module):
        def forward(self, obs, belief_features):
            return torch.zeros(1, NUM_OPTIONS)

    class ZeroValue(torch.nn.Module):
        def forward(self, obs, belief_features):
            return torch.zeros(1)

    obs = torch.zeros(1, 2)
    belief = torch.zeros(1, 2)
    costs = torch.arange(NUM_OPTIONS, dtype=torch.float32).view(1, -1)
    passive = OptionSelector(FixedQ(), ZeroValue(), beta=0.5, mode="passive")
    passive_scores = passive.scores(obs, belief, option_costs=costs)
    assert torch.allclose(passive_scores, -0.5 * costs)

    random = OptionSelector(FixedQ(), ZeroValue(), mode="random")
    valid_mask = torch.zeros(1, NUM_OPTIONS, dtype=torch.bool)
    valid_mask[0, [0, 1]] = True
    sampled = {
        int(random.select(obs, belief, valid_mask=valid_mask, deterministic=True)[0].item())
        for _ in range(100)
    }
    assert sampled == {0, 1}


def test_actor_critic_ignores_exploration_for_policy_loss():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    env = ToyFactorGameEnv(max_steps=5, seed=0)

    episode = collect_episode(
        env,
        agent,
        torch.device("cpu"),
        mode="gtvoi",
        graph_config=graph_config,
        explore_eps=1.0,
        deterministic=False,
    )
    losses = train_step(agent, optimizer, episode, torch.device("cpu"), graph_config, loss_variant="full")

    assert losses["control"] == 0.0
    assert not any(episode["policy_mask"])


def test_batch_episodes_preserves_single_episode_shapes_and_train_keys():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    env = ToyFactorGameEnv(max_steps=5, seed=0)
    episode = collect_episode(
        env,
        agent,
        torch.device("cpu"),
        mode="gtvoi",
        graph_config=graph_config,
        explore_eps=0.0,
        deterministic=True,
    )
    batch = batch_episodes([episode], torch.device("cpu"), graph_config, gamma=0.99)
    losses = train_step(agent, optimizer, [episode], torch.device("cpu"), graph_config, loss_variant="full")

    assert batch["obs_seq"].shape[0] == 1
    assert batch["valid_mask"].sum().item() == len(episode["obs_history"])
    assert losses["batch_size"] == 1
    assert {"total", "value", "critic_value", "transition", "control"}.issubset(losses)


def test_collect_episode_modes_select_valid_primitive_actions():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    valid_actions = {int(action) for action in Action}

    for mode in MODES:
        env = ToyFactorGameEnv(max_steps=5, seed=0)
        episode = collect_episode(
            env,
            agent,
            torch.device("cpu"),
            mode=mode,
            graph_config=graph_config,
            explore_eps=0.0,
            deterministic=True,
        )
        assert episode["option_history"]
        assert set(episode["primitive_action_history"]).issubset(valid_actions)


def test_observation_static_landmark_channel_is_nonzero_and_shape_stable():
    env = ToyFactorGameEnv(seed=0)
    obs = env.reset()
    landmark_channel = obs[3 * 49:4 * 49]

    assert obs.shape == (4 * 49 + 6,)
    assert landmark_channel.sum() > 0


def test_stable_convention_seed_is_non_negative_and_stable():
    convention = {0: 2, 1: 1, 2: 0}
    first = stable_convention_seed(42, convention, trial=3)
    second = stable_convention_seed(42, convention, trial=3)

    assert first == second
    assert first >= 0


def test_first_alignment_time_uses_belief_not_collision_flag():
    graph_config = get_graph_config("full_graph")
    conv = ToyFactorGameEnv(seed=0).partner_convention
    labels = graph_config.labels_from_convention(conv.modes)
    wrong_labels = [(label + 1) % n_modes for label, n_modes in zip(labels, graph_config.factor_modes)]

    def marginals_from(labels_):
        out = []
        for label, n_modes in zip(labels_, graph_config.factor_modes):
            probs = torch.zeros(1, n_modes)
            probs[0, label] = 1.0
            out.append(probs)
        return out

    class FakeBeliefModel:
        def __init__(self):
            self.calls = 0

        def step_history(self, obs_t, ego_act_t, partner_act_t, hidden=None):
            self.calls += 1
            return torch.tensor([[[float(self.calls)]]])

        def _marginals_from_h(self, h):
            return marginals_from(wrong_labels if int(h.item()) == 1 else labels)

    class FakeAgent:
        def __init__(self):
            self.belief_model = FakeBeliefModel()

    episode_data = {
        "obs_history": [torch.zeros(4), torch.zeros(4)],
        "ego_act_history": [torch.zeros(2), torch.zeros(2)],
        "partner_act_history": [torch.zeros(2), torch.zeros(2)],
        "infos": [{"collision": False}, {"collision": False}],
    }

    assert not real_factor_belief_aligned(marginals_from(wrong_labels), conv, graph_config)
    assert real_factor_belief_aligned(marginals_from(labels), conv, graph_config)
    assert first_alignment_time(
        FakeAgent(), torch.device("cpu"), episode_data, conv, graph_config, max_steps=5
    ) == 2


def test_incremental_alignment_matches_prefix_recompute():
    graph_config = get_graph_config("full_graph")
    agent = make_agent(graph_config)
    conv = ToyFactorGameEnv(seed=0).partner_convention
    env = ToyFactorGameEnv(partner_convention=conv, max_steps=5, seed=0)
    episode = collect_episode(
        env,
        agent,
        torch.device("cpu"),
        mode="gtvoi",
        graph_config=graph_config,
        explore_eps=0.0,
        deterministic=True,
    )

    prefix_time = 6
    for idx in range(len(episode["obs_history"])):
        obs_seq = torch.stack(episode["obs_history"][:idx + 1]).unsqueeze(0)
        ego_seq = torch.stack(episode["ego_act_history"][:idx + 1]).unsqueeze(0)
        partner_seq = torch.stack(episode["partner_act_history"][:idx + 1]).unsqueeze(0)
        marginals = agent.get_belief(obs_seq, ego_seq, partner_seq)
        if real_factor_belief_aligned(marginals, conv, graph_config):
            prefix_time = idx + 1
            break

    assert first_alignment_time(
        agent, torch.device("cpu"), episode, conv, graph_config, max_steps=5
    ) == prefix_time


def test_ce_induction_drives_default_support_and_threshold_changes_it():
    estimate_ce_matrix.cache_clear()
    _candidate_factor_specs.cache_clear()
    ce_matrix = estimate_ce_matrix()
    assert estimate_ce_matrix() is ce_matrix
    assert not ce_matrix.flags.writeable
    try:
        ce_matrix[0, 0] = 0.0
    except ValueError:
        pass
    else:
        raise AssertionError("cached CE matrix should be read-only")

    assert _candidate_factor_specs() is _candidate_factor_specs()
    induced_pairs = {(a, b) for a, b, _ in induce_graph(ce_matrix, DEFAULT_CE_THRESHOLD)}
    required = {(factor.option_i, factor.option_j) for factor in GROUND_TRUTH_FACTORS}

    assert required.issubset(induced_pairs)
    assert len(induce_graph(ce_matrix, ce_matrix.max() + 1.0)) < len(induced_pairs)


def test_exp1_outputs_graph_variant_missing_status(tmp_path):
    result = exp1_gtvoi_vs_mi(
        tmp_path,
        torch.device("cpu"),
        seed=0,
        loss_variant="full",
        hidden_dim=16,
        modes=["gtvoi", "mi", "passive", "random", "oracle"],
        n_per_conv=1,
        max_steps=3,
        graph_variants=["full_graph", "plus_irrelevant"],
    )

    assert set(result) == {"full_graph", "plus_irrelevant"}
    assert result["plus_irrelevant"]["status"] == "missing"


def test_train_default_n_episodes_accounts_for_batching():
    train_path = ROOT / "experiments" / "toy_factor_game" / "train.py"
    tree = ast.parse(train_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--n_episodes":
            continue
        defaults = [kw.value.value for kw in node.keywords if kw.arg == "default"]
        assert defaults == [8000]
        return
    raise AssertionError("--n_episodes parser argument not found")


def test_parallel_launcher_dry_run_lists_exp3_and_exp4_jobs():
    script = ROOT / "experiments" / "toy_factor_game" / "launch_parallel.sh"
    exp3 = subprocess.run(
        ["bash", str(script), "exp3", "--dry_run", "--gpus", "0,1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exp4 = subprocess.run(
        ["bash", str(script), "exp4", "--dry_run", "--gpus", "0,1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    override = subprocess.run(
        ["bash", str(script), "exp3", "--dry_run", "--n_episodes", "2000"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert exp3.count("train.py") == 3
    assert "--loss_variant response_value" in exp3
    assert "--n_episodes 8000" in exp3
    assert exp4.count("train.py") == 6
    assert "--graph_variant complete_graph" in exp4
    assert "--n_episodes 8000" in exp4
    assert "--n_episodes 2000" in override
