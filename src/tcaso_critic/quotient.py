from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from .canonical import canonical_hash
from .models import EdgeRecord, GraphBuildResult, Override, PublicState, QuotientStateInfo, TaskSpec
from .state_codec import PublicStateCodec


class TaskPredicate:
    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    def __call__(self, v: PublicState) -> bool:
        t = self.spec.task_type
        p = dict(self.spec.params)
        if t == "agent_at":
            agent_i = int(p["agent_i"])
            target = tuple(p["position"])
            return PublicStateCodec.agent_position(v, agent_i) == target
        if t == "agent_at_any_of":
            agent_i = int(p["agent_i"])
            targets = {tuple(x) for x in p["positions"]}
            return PublicStateCodec.agent_position(v, agent_i) in targets
        if t == "terminal":
            return bool(v.T.get("terminal"))
        if t == "recipe_equals":
            return int(v.R_recipe.get("recipe_encoding")) == int(p["recipe_encoding"])
        if t == "agent_inventory_nonempty":
            agent_i = int(p["agent_i"])
            row = PublicStateCodec.agent_row(v, agent_i)
            return row is not None and int(row.get("inventory", 0)) != 0
        if t == "agent_facing_static_type":
            agent_i = int(p["agent_i"])
            static_type = str(p["static_type"])
            row = PublicStateCodec.agent_row(v, agent_i)
            if row is None:
                return False
            return _static_value_to_type(row.get("facing_static")) == static_type
        if t == "interface_changed_from_initial":
            initial_hash = str(p["initial_interface_hash"])
            return str(v.I.get("interface_hash")) != initial_hash
        if t == "time_at_least":
            return int(v.T.get("time", 0)) >= int(p["time_min"])
        raise ValueError(f"Unknown task_type={t!r}")


def _static_value_to_type(value: object) -> str:
    if value is None:
        return "out_of_bounds"
    iv = int(value)
    if iv == 0:
        return "empty"
    if iv == 1:
        return "wall_or_counter"
    if iv == 4:
        return "delivery"
    if iv == 5:
        return "pot"
    if iv == 6:
        return "recipe_indicator"
    if iv == 7:
        return "button_recipe_indicator"
    if iv == 9:
        return "plate_pile"
    if iv >= 10:
        return "ingredient_pile"
    return f"static_{iv}"


@dataclass(frozen=True)
class QuotientResult:
    state_info: dict[str, QuotientStateInfo]
    pi_classes: dict[str, list[str]]
    target_states: tuple[str, ...]


class ContinuationQuotientComputer:
    """Compute J_tau, Pi_tau, U_tau, and D_tau on a bounded certified graph.

    J_tau is represented by shortest certified continuation distances to tau.
    Pi_tau groups states by canonical continuation signature. U_tau is the log
    cardinality of optimal first-action classes for the audited agent. D_tau is
    computed separately by override-disruption queries.
    """

    def __init__(self, *, graph: GraphBuildResult, task_spec: TaskSpec, audited_agent_i: int) -> None:
        self.graph = graph
        self.task_spec = task_spec
        self.predicate = TaskPredicate(task_spec)
        self.audited_agent_i = int(audited_agent_i)
        self.out_edges: dict[str, list[EdgeRecord]] = defaultdict(list)
        self.in_edges: dict[str, list[EdgeRecord]] = defaultdict(list)
        for e in graph.edges:
            self.out_edges[e.src_hash].append(e)
            self.in_edges[e.dst_hash].append(e)

    def compute(self) -> QuotientResult:
        targets = tuple(sorted([h for h, v in self.graph.nodes.items() if self.predicate(v)]))
        dist: dict[str, int] = {h: 0 for h in targets}
        q: deque[str] = deque(targets)
        while q:
            h = q.popleft()
            for e in self.in_edges.get(h, []):
                if e.src_hash not in dist:
                    dist[e.src_hash] = dist[h] + 1
                    q.append(e.src_hash)
        state_info: dict[str, QuotientStateInfo] = {}
        pi_classes: dict[str, list[str]] = defaultdict(list)
        for h in self.graph.nodes:
            d = dist.get(h)
            optimal_classes: tuple[int, ...]
            if d is None or d == 0:
                optimal_classes = tuple()
            else:
                opts = sorted({int(e.joint_action[self.audited_agent_i]) for e in self.out_edges.get(h, []) if e.dst_hash in dist and dist[e.dst_hash] == d - 1})
                optimal_classes = tuple(opts)
            U = math.log(len(optimal_classes)) if len(optimal_classes) > 1 else 0.0
            signature_obj = {
                "tau_id": self.task_spec.tau_id,
                "distance_to_tau": d,
                "optimal_agent_action_classes": optimal_classes,
                "terminal": bool(self.graph.nodes[h].T.get("terminal")),
                "recipe": self.graph.nodes[h].R_recipe.get("recipe_encoding"),
            }
            pi = canonical_hash(signature_obj, prefix="pi")
            info = QuotientStateInfo(state_hash=h, distance_to_tau=d, optimal_action_classes=optimal_classes, U_tau=U, pi_signature=pi)
            state_info[h] = info
            pi_classes[pi].append(h)
        return QuotientResult(state_info=state_info, pi_classes={k: sorted(v) for k, v in pi_classes.items()}, target_states=targets)

    def D_tau(self, quotient: QuotientResult, state_hash: str, override: Override) -> int:
        info = quotient.state_info[state_hash]
        if info.distance_to_tau is None or info.distance_to_tau == 0:
            return 0
        unrestricted = set(info.optimal_action_classes)
        if not unrestricted:
            return 0
        if int(override.action) not in unrestricted:
            return 1
        # If the action class remains optimal, compare next Pi sets. The override
        # disrupts if it removes every optimal successor class except a strict subset.
        all_next_pi = {quotient.state_info[e.dst_hash].pi_signature for e in self.out_edges.get(state_hash, []) if e.dst_hash in quotient.state_info and quotient.state_info[e.dst_hash].distance_to_tau == info.distance_to_tau - 1}
        forced_next_pi = {quotient.state_info[e.dst_hash].pi_signature for e in self.out_edges.get(state_hash, []) if int(e.joint_action[override.agent_i]) == int(override.action) and e.dst_hash in quotient.state_info and quotient.state_info[e.dst_hash].distance_to_tau == info.distance_to_tau - 1}
        return int(forced_next_pi != all_next_pi)
