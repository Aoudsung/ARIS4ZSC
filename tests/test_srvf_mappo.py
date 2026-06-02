from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from raob.benchmarks.overcooked_classic import (
    CLASSIC_OVERCOOKED_LAYOUTS,
    canonical_classic_layout_name,
)
from raob.srvf_mappo import (
    IRFTable,
    MAPPOActorCritic,
    NeuralSRVFHeads,
    SRVFBelief,
    UnifiedLoss,
    build_arg_parser,
    collect_classic_rollout_batch,
    evaluate_offline_target_regret,
    gradient_audit,
    initialize_source_beta,
    iter_source_batches,
    source_table_to_batch,
    _concat_rollout_batches,
    _discounted_cumsum,
    _env_step_progress,
    _estimated_updates_remaining,
)


def test_classic_layout_aliases_include_gamma_multi_strategy_counter() -> None:
    expected = "counter_circuit_6x5_2pots_3orders"

    assert canonical_classic_layout_name("Multi-Strategy Counter") == expected
    assert canonical_classic_layout_name("multi-strategy-counter") == expected
    assert canonical_classic_layout_name("diverse_counter_circuit_6x5") == expected
    assert canonical_classic_layout_name("counter_circuit") == "counter_circuit"
    assert expected in CLASSIC_OVERCOOKED_LAYOUTS


def test_srvf_score_alpha_extremes() -> None:
    heads = NeuralSRVFHeads(g_dim=3, num_actions=2, factor_dim=2)
    belief = SRVFBelief(factor_dim=2, dz_dim=3).reset(2)
    g = torch.randn(2, 3)
    beta = torch.randn(2, 2)
    precision = torch.eye(2).unsqueeze(0).expand(2, -1, -1).clone()
    eta = torch.einsum("bkl,bl->bk", precision, beta)

    belief.set_state(precision, eta, torch.zeros(2))
    fallback = belief.score(g, heads)
    head_out = heads(g)
    assert torch.allclose(fallback, head_out.a0, atol=1e-5)

    belief.set_state(precision, eta, torch.ones(2))
    raw = belief.score(g, heads)
    expected = head_out.a0 + torch.einsum("bak,bk->ba", head_out.u_c, beta)
    assert torch.allclose(raw, expected, atol=1e-4)


def test_source_table_to_batch_shapes() -> None:
    state_g = torch.randn(4, 3)
    delta_z = torch.randn(4, 2, 3, 3)
    q_raw = torch.randn(4, 2, 3)
    valid = torch.ones(4, 2, 3, dtype=torch.bool)
    table = IRFTable.from_tensors(
        state_g=state_g,
        delta_z=delta_z,
        a_raw=q_raw - q_raw.mean(dim=1, keepdim=True),
        valid_mask=valid,
        partner_ids=("p0", "p1", "p2"),
    )
    init = initialize_source_beta(table, factor_dim=2)
    batch = source_table_to_batch(table, init.beta_source, state_indices=[0, 2], partner_indices=[1, 0])

    assert batch.g.shape == (4, 3)
    assert torch.equal(batch.g, torch.stack([state_g[0], state_g[0], state_g[2], state_g[2]], dim=0))
    assert batch.beta.shape == (4, 2)
    assert torch.equal(batch.beta, torch.stack([init.beta_source[1], init.beta_source[0], init.beta_source[1], init.beta_source[0]], dim=0))
    assert batch.delta_z_target.shape == (4, 2, 3)
    assert torch.equal(batch.delta_z_target[0], table.delta_z[0, :, 1, :])
    assert torch.equal(batch.delta_z_target[1], table.delta_z[0, :, 0, :])
    assert batch.a_bar_target.shape == (4, 2)
    assert batch.action_mask is not None
    assert batch.action_mask.shape == (4, 2)


def test_iter_source_batches_preserves_flat_order_without_tolist() -> None:
    state_g = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    delta_z = torch.arange(4 * 2 * 3 * 3, dtype=torch.float32).reshape(4, 2, 3, 3)
    a_raw = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    valid = torch.ones(4, 2, 3, dtype=torch.bool)
    beta = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    table = IRFTable.from_tensors(
        state_g=state_g,
        delta_z=delta_z,
        a_raw=a_raw,
        valid_mask=valid,
    )

    batches = iter_source_batches(table, beta, batch_size=5, shuffle=False)
    merged_g = torch.cat([batch.g for batch in batches], dim=0)
    merged_beta = torch.cat([batch.beta for batch in batches], dim=0)
    flat = torch.arange(table.num_states * table.num_partners)
    states = torch.div(flat, table.num_partners, rounding_mode="floor")
    partners = flat.remainder(table.num_partners)

    assert torch.equal(merged_g, state_g.index_select(0, states))
    assert torch.equal(merged_beta, beta.index_select(0, partners))


def test_discounted_cumsum_matches_reference_recurrence() -> None:
    values = torch.tensor([1.0, -2.0, 3.5, 0.25], dtype=torch.float32)
    expected = torch.zeros_like(values)
    running = 0.0
    for idx in range(values.numel() - 1, -1, -1):
        running = float(values[idx]) + 0.93 * running
        expected[idx] = running

    assert torch.allclose(_discounted_cumsum(values, 0.93), expected, atol=1e-6, rtol=1e-6)


@dataclass
class _FakeStep:
    observation: Mapping[str, Any]
    reward: float
    done: bool
    state_g: torch.Tensor
    affordance: Any = None


class _FakeAdapter:
    num_actions = 2

    def __init__(self) -> None:
        self.timestep = 0
        self.g = torch.zeros(3)

    def reset(self, seed: int) -> Mapping[str, Any]:
        self.timestep = 0
        self.g = torch.tensor([float(seed % 2), 0.0, 0.0])
        return {"timestep": self.timestep}

    def public_chart_tensor(self) -> torch.Tensor:
        return self.g.clone()

    def ego_observation_tensor(self, *, agent_index: int = 0) -> torch.Tensor:
        assert agent_index == 0
        return self.g.clone()

    def global_state_tensor(self) -> torch.Tensor:
        return torch.cat([self.g, -self.g], dim=0)

    def legal_action_mask(self) -> torch.Tensor:
        return torch.ones(self.num_actions, dtype=torch.bool)

    def step(self, ego_action: int, partner_action: int) -> _FakeStep:
        del partner_action
        self.timestep += 1
        inc = torch.tensor([0.0, float(ego_action), 1.0])
        self.g = self.g + inc
        done = self.timestep >= 3
        return _FakeStep(
            observation={"timestep": self.timestep},
            reward=float(ego_action),
            done=done,
            state_g=self.g.clone(),
        )


class _FakePartner:
    def reset(self, seed: int | None = None) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        state: Any = None,
        rng: Any = None,
    ) -> tuple[int, Any]:
        del observation, rng
        return 0, state


def test_collect_classic_rollout_batch_and_loss() -> None:
    heads = NeuralSRVFHeads(g_dim=3, num_actions=2, factor_dim=2)
    actor_critic = MAPPOActorCritic(
        obs_dim=3,
        global_state_dim=6,
        g_dim=3,
        num_actions=2,
        factor_dim=2,
    )
    belief = SRVFBelief(factor_dim=2, dz_dim=3)
    rollout = collect_classic_rollout_batch(
        _FakeAdapter(),
        [("p0", _FakePartner())],
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief=belief,
        episodes_per_partner=1,
        horizon=3,
        seed=1,
    )
    assert rollout.obs_ego.shape == (3, 3)
    assert rollout.global_state.shape == (3, 6)
    assert rollout.g.shape == (3, 3)
    assert rollout.phase.shape == (3, 0)
    assert rollout.actions.shape == (3,)
    assert rollout.monitor["episode_count"] == 1
    assert rollout.monitor["step_count"] == 3
    assert rollout.monitor["target_data_used"] is False
    assert "p0" in rollout.monitor["by_partner"]
    histogram = rollout.monitor["action_histogram"]
    assert len(histogram) == 2
    assert sum(histogram) == 3
    assert rollout.monitor["episode_return_mean"] == float(histogram[1])
    assert abs(rollout.monitor["step_reward_nonzero_fraction"] - float(histogram[1] / 3)) < 1e-6

    source = source_table_to_batch(
        IRFTable.from_tensors(
            state_g=torch.randn(2, 3),
            delta_z=torch.randn(2, 2, 2, 3),
            a_raw=torch.randn(2, 2, 2),
            valid_mask=torch.ones(2, 2, 2, dtype=torch.bool),
        ),
        torch.randn(2, 2),
    )
    loss, logs = UnifiedLoss(actor_critic, heads).compute(rollout, source)
    assert loss.ndim == 0
    assert "L_policy" in logs
    assert "L_delta" in logs


def test_concat_rollout_batches_merges_monitoring() -> None:
    heads = NeuralSRVFHeads(g_dim=3, num_actions=2, factor_dim=2)
    actor_critic = MAPPOActorCritic(
        obs_dim=3,
        global_state_dim=6,
        g_dim=3,
        num_actions=2,
        factor_dim=2,
    )
    partners = [("p0", _FakePartner())]
    first = collect_classic_rollout_batch(
        _FakeAdapter(),
        partners,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief=SRVFBelief(factor_dim=2, dz_dim=3),
        episodes_per_partner=1,
        horizon=3,
        seed=1,
    )
    second = collect_classic_rollout_batch(
        _FakeAdapter(),
        partners,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief=SRVFBelief(factor_dim=2, dz_dim=3),
        episodes_per_partner=1,
        horizon=3,
        seed=2,
    )
    merged = _concat_rollout_batches([first, second], device="cpu")

    assert merged.actions.shape == (6,)
    assert merged.monitor["worker_count"] == 2
    assert merged.monitor["episode_count"] == 2
    assert merged.monitor["step_count"] == 6
    assert sum(merged.monitor["action_histogram"]) == 6
    assert "p0" in merged.monitor["by_partner"]


def test_evaluate_offline_target_regret_vectorized_contract() -> None:
    torch.manual_seed(0)
    state_g = torch.randn(3, 3)
    delta_z = torch.randn(3, 2, 2, 3)
    a_raw = torch.randn(3, 2, 2)
    valid = torch.tensor(
        [
            [[True, True], [False, True]],
            [[True, False], [True, True]],
            [[False, True], [True, False]],
        ],
        dtype=torch.bool,
    )
    table = IRFTable.from_tensors(
        state_g=state_g,
        delta_z=delta_z,
        a_raw=a_raw,
        valid_mask=valid,
        partner_ids=("p0", "p1"),
    )
    init = initialize_source_beta(table, factor_dim=2)
    heads = NeuralSRVFHeads(g_dim=3, num_actions=2, factor_dim=2)
    actor_critic = MAPPOActorCritic(
        obs_dim=3,
        global_state_dim=6,
        g_dim=3,
        num_actions=2,
        factor_dim=2,
    )
    belief = SRVFBelief(
        factor_dim=2,
        dz_dim=3,
        prior_mean=init.prior_mean,
        prior_covariance=init.prior_covariance,
        source_beta=init.beta_source,
    )

    result = evaluate_offline_target_regret(
        table,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief_template=belief,
    )

    expected_rows = int(valid.any(dim=1).sum().item())
    assert result["aggregate"]["evaluated_rows"] == expected_rows
    assert result["leakage_guard"]["target_responses_used_for_posterior"] is True
    for partner_idx, partner_id in enumerate(table.partner_ids):
        assert result["by_partner"][partner_id]["posterior_observations"] == int(valid[:, :, partner_idx].sum().item())


def test_gradient_audit_contract() -> None:
    audit = gradient_audit()
    for key in (
        "policy_reaches_actor",
        "policy_blocked_from_srvf",
        "value_reaches_critic",
        "delta_reaches_srvf",
        "delta_blocked_from_actor",
        "A_reaches_srvf",
        "A_blocked_from_actor",
    ):
        assert audit[key]


def test_formal_classic_cli_args_are_parseable() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "formal-classic",
            "--smoke",
            "--seeds",
            "7,8",
            "--monitor-every",
            "2",
            "--workers",
            "3",
            "--learner-epochs",
            "2",
            "--torch-num-threads",
            "1",
        ]
    )

    assert args.command == "formal-classic"
    assert args.smoke
    assert args.seeds == "7,8"
    assert args.layout == "cramped_room"
    assert args.device == "cuda"
    assert args.monitor_every == 2
    assert args.workers == 3
    assert args.learner_epochs == 2
    assert args.torch_num_threads == 1
    assert args.updates == 0
    assert args.target_env_steps == 100_000_000


def test_formal_classic_resource_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["formal-classic"])

    assert args.workers == 18
    assert args.learner_epochs == 4
    assert args.worker_policy_device == "cpu"
    assert args.updates == 0
    assert args.target_env_steps == 100_000_000


def test_formal_classic_fixed_update_override_is_parseable() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["formal-classic", "--updates", "300"])

    assert args.updates == 300
    assert args.target_env_steps == 100_000_000


def test_estimated_updates_remaining_uses_actual_rollout_rows() -> None:
    target = 100_000_000

    assert _estimated_updates_remaining(7_200, target, 1) == 13_888
    assert _estimated_updates_remaining(target, target, 13_889) == 0
    assert _estimated_updates_remaining(0, target, 0) == 0
    assert _env_step_progress(50, 100) == 0.5
    assert _env_step_progress(120, 100) == 1.0
    assert _env_step_progress(120, 0) == 0.0
