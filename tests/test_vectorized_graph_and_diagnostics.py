from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tcaso_critic.graph_builder import CertifiedGraphBuilder
from tcaso_critic.public_edge_checker import CertifiedEdgeChecker
from tcaso_critic.state_codec import EnvConfigSnapshot, PublicStateCodec
from tcaso_critic.tau_family import TauFamilyGenerator
from tcaso_critic.diagnostics import diagnose_exact_matching
from tcaso_critic.models import Override, ProbeRecord, ProbeClass, CertificateStatus, KappaSignature


@dataclass
class Pos:
    x: np.ndarray
    y: np.ndarray


@dataclass
class Agents:
    pos: Pos
    dir: np.ndarray
    inventory: np.ndarray


@dataclass
class FakeState:
    agents: Agents
    grid: np.ndarray
    time: np.ndarray
    terminal: np.ndarray
    recipe: np.ndarray
    new_correct_delivery: np.ndarray
    ingredient_permutations: np.ndarray


@dataclass
class FakeOutcome:
    next_state: FakeState
    rewards: dict
    dones: dict
    info: dict


class FakeVectorBackend:
    num_actions = 2
    num_agents = 2

    def __init__(self):
        self.batch_calls = 0

    def joint_actions(self):
        return [(0, 0), (1, 0), (0, 1), (1, 1)]

    def reset_state(self, seed: int):
        return make_state(1, 1, 2, 1, time=0)

    def batch_step(self, states, joint_actions, base_seed: int):
        self.batch_calls += 1
        outs = []
        for s, ja in zip(states, joint_actions):
            x = np.array(s.agents.pos.x, copy=True)
            y = np.array(s.agents.pos.y, copy=True)
            if ja[0] == 0:
                x[0] += 1
            if ja[1] == 0:
                x[1] += 1
            outs.append(FakeOutcome(make_state(int(x[0]), int(y[0]), int(x[1]), int(y[1]), time=int(s.time) + 1), {}, {}, {}))
        return outs

    def step(self, state, joint_action, seed):  # not used in vectorized mode
        raise AssertionError("sequential step should not be called")


def make_state(x0, y0, x1, y1, time=0):
    grid = np.zeros((4, 5, 3), dtype=np.int32)
    grid[:, 0, 0] = 1
    grid[:, -1, 0] = 1
    grid[0, :, 0] = 1
    grid[-1, :, 0] = 1
    grid[1, 3, 0] = 5  # pot
    return FakeState(
        agents=Agents(pos=Pos(np.array([x0, x1]), np.array([y0, y1])), dir=np.array([2, 3]), inventory=np.array([0, 0])),
        grid=grid,
        time=np.array(time),
        terminal=np.array(False),
        recipe=np.array(0),
        new_correct_delivery=np.array(False),
        ingredient_permutations=np.array([]),
    )


def test_vectorized_builder_uses_batch_step():
    backend = FakeVectorBackend()
    cfg = EnvConfigSnapshot("fake", 5, False, False, False, False, False, False, False)
    codec = PublicStateCodec("fake", cfg)
    checker = CertifiedEdgeChecker(codec, backend)
    graph = CertifiedGraphBuilder(backend=backend, codec=codec, checker=checker, max_depth=2, max_nodes=100, seed=0, step_mode="vectorized", batch_size=8).build_from_reset()
    assert backend.batch_calls > 0
    assert len(graph.nodes) > 1
    assert graph.edges


def test_tau_family_generates_multiple_task_types():
    cfg = EnvConfigSnapshot("fake", 5, False, False, False, False, False, False, False)
    codec = PublicStateCodec("fake", cfg)
    v0 = codec.alpha_tau(make_state(1, 1, 2, 1, time=0))
    taus = TauFamilyGenerator(audited_agent_i=0).from_initial_state(v0)
    types = {t.task_type for t in taus}
    assert "agent_at_any_of" in types
    assert "agent_inventory_nonempty" in types
    assert "interface_changed_from_initial" in types
    assert "time_at_least" in types


def test_diagnostics_identifies_kappa_strictness():
    override = Override(agent_i=0, action=1)
    p = ProbeRecord("p", "L", "tau", "s1", override, 0.7, 1, 1, "pi1", "kp", ProbeClass.STRUCTURAL_POSITIVE_CANDIDATE, CertificateStatus.DRAFT, failure_label="NO_EXACT_MATCH")
    c = ProbeRecord("c", "L", "tau", "s2", override, 0.0, 0, 1, "pi2", "kc", ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL, CertificateStatus.DRAFT, failure_label="NO_EXACT_MATCH")
    sigp = KappaSignature("sp", {"layout_id": "L", "tau_id": "tau", "public_geometry_hash": "A", "interface_profile_hash": "I1"}, "kp")
    sigc = KappaSignature("sc", {"layout_id": "L", "tau_id": "tau", "public_geometry_hash": "B", "interface_profile_hash": "I2"}, "kc")
    d = diagnose_exact_matching(probes=[p, c], matched_controls=[], kappa_signatures={"p": sigp, "c": sigc}, num_tau_states=1, max_depth=2)
    assert d["diagnostic_primary_label"] == "KAPPA_FIELD_STRICTNESS_DOMINATES_NO_MATCH"
