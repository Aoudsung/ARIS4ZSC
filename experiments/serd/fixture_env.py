"""Deterministic fixture data for SERD sanity checks.

The fixture mimics the branch records expected from Overcooked/JaxMARL adapters:
same pre-shock probe id, a semantic branch, and multiple non-semantic controls.
It is intentionally simple and should not be used as scientific evidence.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from .serd_core import BranchRecord


CONTROL_FAMILIES = (
    "random_lag",
    "state_block",
    "reward_shaping",
    "naive_replanning",
)


def _phi(probe_index: int, shock: float, jitter: float = 0.0) -> dict[str, float]:
    return {
        "pre_agent_x": float(probe_index % 3),
        "pre_partner_x": float((probe_index + 1) % 3),
        "pre_subtask_progress": 0.2 + 0.01 * (probe_index % 5),
        "pre_role_state": float(probe_index % 2),
        "h_reward_drop": shock + jitter,
        "h_partner_suboptimality": 0.5 * shock + jitter,
        "h_blocking_count": float(probe_index % 2),
        "h_distance_delta": 0.25 * shock + jitter,
    }


def make_fixture_records(
    probes: int,
    seed: int,
    domain: str = "fixture_counter_circuit",
) -> Tuple[List[BranchRecord], List[BranchRecord]]:
    rng = random.Random(seed)
    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []

    policies = {
        "fcp_fixture": {
            "semantic_loss": 2.0,
            "control_losses": {
                "random_lag": 1.9,
                "state_block": 2.1,
                "reward_shaping": 1.8,
                "naive_replanning": 2.0,
            },
        },
        "pecan_fixture": {
            "semantic_loss": 1.0,
            "control_losses": {
                "random_lag": 1.8,
                "state_block": 1.7,
                "reward_shaping": 1.6,
                "naive_replanning": 1.5,
            },
        },
    }

    for policy, profile in policies.items():
        for i in range(probes):
            probe_id = f"{policy}_p{i:03d}"
            no_shock_return = 10.0 + rng.uniform(-0.1, 0.1)
            semantic_shock = 1.0 + rng.uniform(-0.02, 0.02)
            semantic_loss = profile["semantic_loss"] + rng.uniform(-0.08, 0.08)
            semantic_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=policy,
                    domain=domain,
                    disruption="missed_handoff",
                    family="semantic",
                    no_shock_return=no_shock_return,
                    branch_return=no_shock_return - semantic_loss,
                    shock_magnitude=semantic_shock,
                    phi_pre_h=_phi(i, semantic_shock),
                )
            )
            for family in CONTROL_FAMILIES:
                control_shock = semantic_shock + rng.uniform(-0.01, 0.01)
                control_loss = profile["control_losses"][family] + rng.uniform(-0.08, 0.08)
                control_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=policy,
                        domain=domain,
                        disruption="missed_handoff",
                        family=family,
                        no_shock_return=no_shock_return,
                        branch_return=no_shock_return - control_loss,
                        shock_magnitude=control_shock,
                        phi_pre_h=_phi(i, control_shock, jitter=rng.uniform(-0.005, 0.005)),
                    )
                )

    return semantic_records, control_records
