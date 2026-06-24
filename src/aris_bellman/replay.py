from __future__ import annotations

from dataclasses import fields
from typing import Any, Iterable

import numpy as np

from .specs import OptionTransition


class EvidenceBuffer:
    def __init__(
        self,
        num_factors: int,
        window: int,
        evidence_dim: int,
        dtype=np.float32,
    ):
        if num_factors < 0 or window <= 0 or evidence_dim <= 0:
            raise ValueError("Expected num_factors >= 0, window > 0, evidence_dim > 0.")
        self.num_factors = int(num_factors)
        self.window = int(window)
        self.evidence_dim = int(evidence_dim)
        self.dtype = dtype
        self._buffer = np.zeros(
            (self.num_factors, self.window, self.evidence_dim),
            dtype=self.dtype,
        )
        self._count = 0
        self._start = 0

    def append(self, x_f: np.ndarray) -> None:
        evidence = np.asarray(x_f, dtype=self.dtype)
        expected = (self.num_factors, self.evidence_dim)
        if evidence.shape != expected:
            raise ValueError(f"x_f must have shape {expected}; got {evidence.shape}.")

        if self._count < self.window:
            write_idx = self._count
            self._count += 1
        else:
            write_idx = self._start
            self._start = (self._start + 1) % self.window
        self._buffer[:, write_idx, :] = evidence

    def snapshot(self) -> np.ndarray:
        out = np.zeros_like(self._buffer)
        if self._count == 0:
            return out.copy()
        if self._count < self.window:
            out[:, : self._count, :] = self._buffer[:, : self._count, :]
            return out.copy()

        indices = [(self._start + idx) % self.window for idx in range(self.window)]
        out[:, :, :] = self._buffer[:, indices, :]
        return out.copy()

    def reset(self) -> None:
        self._buffer.fill(0)
        self._count = 0
        self._start = 0

    @property
    def count(self) -> int:
        return self._count


class OptionReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._storage: list[OptionTransition] = []
        self._next_idx = 0
        self._rng = np.random.default_rng(seed)

    def add(self, transition: OptionTransition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.capacity

    def extend(self, transitions: Iterable[OptionTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._storage:
            raise ValueError("Cannot sample from an empty OptionReplayBuffer.")
        replace = batch_size > len(self._storage)
        indices = self._rng.choice(len(self._storage), size=batch_size, replace=replace)
        rows = [self._storage[int(idx)] for idx in indices]
        return _stack_transitions(rows)

    def __len__(self) -> int:
        return len(self._storage)


def _stack_transitions(rows: list[OptionTransition]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for field in fields(OptionTransition):
        name = field.name
        values = [getattr(row, name) for row in rows]
        first = values[0]
        if isinstance(first, np.ndarray):
            batch[name] = np.stack(values, axis=0)
        elif isinstance(first, (bool, int, float, np.bool_, np.integer, np.floating)):
            batch[name] = np.asarray(values)
        else:
            batch[name] = values
    return batch
