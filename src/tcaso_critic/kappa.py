from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from .canonical import canonical_hash
from .models import GraphBuildResult, KappaSignature, Override, PublicState, TaskSpec
from .validators import validate_kappa_signature


class KappaComputer:
    """Exact kappa_tau signature construction.

    The signature deliberately excludes U_tau, D_tau, policy labels, returns, and
    rollout outcomes. All structured fields are represented by canonical hashes
    so the record invariant validator can enforce scalar value domains.
    """

    def __init__(self, *, graph: GraphBuildResult, task_spec: TaskSpec) -> None:
        self.graph = graph
        self.task_spec = task_spec
        self.out_edges = defaultdict(list)
        for e in graph.edges:
            self.out_edges[e.src_hash].append(e)

    def compute(self, state_hash: str, override: Override, distance_to_tau: int | None) -> KappaSignature:
        v = self.graph.nodes[state_hash]
        field_values = {
            "layout_id": v.layout_id,
            "tau_id": self.task_spec.tau_id,
            "agent_i": int(override.agent_i),
            "channel_c": override.channel_c,
            "override_action": int(override.action),
            "distance_to_tau_bucket": self._distance_bucket(distance_to_tau),
            "terminal_bucket": "terminal" if v.T.get("terminal") else "nonterminal",
            "active_recipe_hash": canonical_hash(v.R_recipe) if v.R_recipe is not None else None,
            "public_geometry_hash": canonical_hash({"L": v.L.get("component_cells"), "R": v.R.get("component_cells"), "C": v.C, "W_dyn": v.W_dyn}),
            "inventory_profile_hash": canonical_hash([a["inventory"] for a in v.I.get("agents", [])]),
            "interface_profile_hash": canonical_hash({"dynamic_grid": v.I.get("dynamic_grid"), "extra_grid": v.I.get("extra_grid"), "new_correct_delivery": v.I.get("new_correct_delivery")}),
            "partner_action_feasibility_hash": canonical_hash(self._partner_feasible_actions(state_hash, override)),
            "override_precondition": self._override_precondition(v, override),
            "time_bucket": self._time_bucket(v),
            "reset_domain_tag": str(v.ResetDomainTag),
        }
        khash = canonical_hash(field_values)
        sig = KappaSignature(signature_id=canonical_hash({"state": state_hash, "override": asdict(override), "kappa_hash": khash}, prefix="kappa_sig"), field_values=field_values, kappa_hash=khash)
        validate_kappa_signature(sig)
        return sig

    @staticmethod
    def _distance_bucket(distance_to_tau: int | None) -> str:
        if distance_to_tau is None:
            return "unreachable"
        if distance_to_tau <= 0:
            return "d0"
        if distance_to_tau <= 2:
            return f"d{distance_to_tau}"
        if distance_to_tau <= 5:
            return "d3_5"
        return "d6_plus"

    def _partner_feasible_actions(self, state_hash: str, override: Override) -> list[int]:
        partner_actions = sorted({int(e.joint_action[1 - override.agent_i]) for e in self.out_edges.get(state_hash, []) if int(e.joint_action[override.agent_i]) == int(override.action) and e.source_certified})
        return partner_actions

    def _override_precondition(self, v: PublicState, override: Override) -> str:
        agent = next((a for a in v.I.get("agents", []) if int(a["agent_i"]) == int(override.agent_i)), None)
        if agent is None:
            return "agent_missing"
        action = int(override.action)
        if action == 4:
            return "stay"
        if action == 5:
            return self._interact_precondition(v, agent)
        return self._move_precondition(v, agent, action)

    @staticmethod
    def _move_precondition(v: PublicState, agent: dict[str, Any], action: int) -> str:
        delta = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}.get(action)
        if delta is None:
            return "unknown_move_action"
        x = int(agent["x"]) + delta[0]
        y = int(agent["y"]) + delta[1]
        occupied = set(tuple(p) for p in v.W_dyn.get("occupied_positions", []))
        walkable = set(tuple(p) for p in v.W_dyn.get("walkable_empty_cells", []))
        if (x, y) in occupied:
            return "move_into_agent"
        if (x, y) in walkable:
            return "move_into_empty"
        return "move_blocked_static"

    @staticmethod
    def _interact_precondition(v: PublicState, agent: dict[str, Any]) -> str:
        direction = int(agent["dir"])
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}.get(direction, (0, 0))
        x = int(agent["x"]) + dx
        y = int(agent["y"]) + dy
        static_hash = v.C.get("static_grid_hash", "unknown")
        # The full static object is already inside public_geometry_hash; this
        # scalar label keeps kappa exact but avoids embedding arrays.
        return f"interact_facing_{x}_{y}_staticgrid_{static_hash[:8]}"

    @staticmethod
    def _time_bucket(v: PublicState) -> str:
        time = int(v.T.get("time", 0))
        max_steps = int(v.T.get("max_steps", 0))
        if max_steps <= 0:
            return f"t{time}"
        if time == 0:
            return "t0"
        if time >= max_steps:
            return "terminal_time"
        if time <= 2:
            return f"t{time}"
        return "t3_plus"
