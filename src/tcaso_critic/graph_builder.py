from __future__ import annotations

from collections import deque
from typing import Any

from .models import GraphBuildResult
from .public_edge_checker import CertifiedEdgeChecker
from .state_codec import PublicStateCodec


class GraphBuildError(RuntimeError):
    pass


class CertifiedGraphBuilder:
    """Bounded certified graph construction from source-backed edges only.

    Two modes are supported:
      * ``vectorized``: frontier-action batches are evaluated with JAX ``vmap``.
      * ``sequential``: explicit debug mode, never used as hidden fallback.

    The graph includes only edges returned by ``CertifiedEdgeChecker`` with
    ``source_certified=True``. Candidate public edges are never admitted.
    """

    def __init__(
        self,
        *,
        backend: Any,
        codec: PublicStateCodec,
        checker: CertifiedEdgeChecker,
        max_depth: int,
        max_nodes: int,
        seed: int,
        step_mode: str = "vectorized",
        batch_size: int = 512,
    ) -> None:
        self.backend = backend
        self.codec = codec
        self.checker = checker
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.seed = int(seed)
        self.step_mode = str(step_mode)
        self.batch_size = int(batch_size)
        if self.step_mode not in {"vectorized", "sequential"}:
            raise ValueError("step_mode must be 'vectorized' or 'sequential'")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def build_from_reset(self) -> GraphBuildResult:
        raw0 = self.backend.reset_state(self.seed)
        v0 = self.codec.alpha_tau(raw0)
        h0 = self.codec.public_state_hash(v0)
        result = GraphBuildResult(nodes={h0: v0}, raw_representatives={h0: raw0}, depths={h0: 0})
        frontier = [h0]
        for depth in range(self.max_depth):
            if not frontier:
                break
            next_frontier: list[str] = []
            requests: list[tuple[str, Any, tuple[int, ...], int]] = []
            for src_hash in frontier:
                raw = result.raw_representatives[src_hash]
                for idx, joint_action in enumerate(self.backend.joint_actions()):
                    edge_seed = self.seed + 1000003 * (depth + 1) + 9176 * idx + 104729 * len(requests)
                    requests.append((src_hash, raw, joint_action, edge_seed))
            for chunk in _chunks(requests, self.batch_size):
                if self.step_mode == "vectorized":
                    states = [raw for _, raw, _, _ in chunk]
                    actions = [ja for _, _, ja, _ in chunk]
                    base_seed = min(seed for *_, seed in chunk)
                    outcomes = self.backend.batch_step(states, actions, base_seed)
                else:
                    outcomes = [self.backend.step(raw, ja, seed) for _, raw, ja, seed in chunk]
                if len(outcomes) != len(chunk):
                    raise GraphBuildError("backend returned an outcome count different from request count")
                for (src_hash, raw, joint_action, _seed), outcome in zip(chunk, outcomes):
                    src_public = result.nodes[src_hash]
                    checked = self.checker.check_precomputed_outcome(src_public, src_hash, raw, joint_action, outcome.next_state)
                    if not checked.edge.source_certified:
                        result.rejected_edges.append(checked.edge)
                        continue
                    result.edges.append(checked.edge)
                    if checked.dst_hash not in result.nodes:
                        if len(result.nodes) >= self.max_nodes:
                            raise GraphBuildError(f"max_nodes={self.max_nodes} exceeded before depth={depth + 1}")
                        result.nodes[checked.dst_hash] = checked.dst_public
                        result.raw_representatives[checked.dst_hash] = checked.next_state
                        result.depths[checked.dst_hash] = depth + 1
                        next_frontier.append(checked.dst_hash)
            frontier = next_frontier
        return result


def _chunks(items: list[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]
