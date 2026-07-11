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
        self._belief_hidden: np.ndarray | None = None
        self._belief_window_bases: np.ndarray | None = None
        self._last_write_idx: int | None = None
        # Optional full-episode Path C sequence.  The generic replay module keeps
        # this value opaque so the core ARIS package does not depend on an
        # experiment-specific evidence schema.
        self._sequence_episode: Any | None = None

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
        self._last_write_idx = int(write_idx)

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

    def snapshot_mask(self) -> np.ndarray:
        """Boolean [F,T] mask identifying real evidence rows.

        P4/S2: zero padding in early windows is not evidence. The mask is
        stored with each transition and consumed by the belief encoder.
        """
        mask = np.zeros((self.num_factors, self.window), dtype=bool)
        if self._count <= 0:
            return mask.copy()
        active = min(self._count, self.window)
        mask[:, :active] = True
        return mask.copy()

    def snapshot_with_mask(self) -> tuple[np.ndarray, np.ndarray]:
        return self.snapshot(), self.snapshot_mask()

    def length(self) -> int:
        """Number of real rows represented in ``snapshot()``.

        ``snapshot()`` is zero-padded until the evidence window is full. This
        length is the mask source used by the belief model so early padding is
        not treated as observed evidence (P4/S2).
        """
        return int(self._count)

    def mask_snapshot(self) -> np.ndarray:
        mask = np.zeros((self.window,), dtype=bool)
        mask[: min(self._count, self.window)] = True
        return mask.copy()

    def set_belief_hidden(self, hidden: np.ndarray | None) -> None:
        """Persist factor-belief hidden state across option decisions (P4)."""
        if hidden is None:
            self._belief_hidden = None
            return
        arr = np.asarray(hidden, dtype=self.dtype)
        if arr.ndim != 2 or arr.shape[0] != self.num_factors:
            raise ValueError(
                "belief hidden must have shape [num_factors, hidden_dim]; "
                f"got {arr.shape}."
            )
        self._belief_hidden = arr.copy()

    def belief_hidden_snapshot(self) -> np.ndarray | None:
        if self._belief_hidden is None:
            return None
        return self._belief_hidden.copy()

    def record_belief_window_base(self, hidden: np.ndarray) -> None:
        """Record the hidden state that preceded the most recently appended row."""
        if self._last_write_idx is None:
            raise RuntimeError("record_belief_window_base() requires a prior append().")
        arr = np.asarray(hidden, dtype=self.dtype)
        if arr.ndim != 2 or arr.shape[0] != self.num_factors:
            raise ValueError(
                "belief window base must have shape [num_factors, hidden_dim]; "
                f"got {arr.shape}."
            )
        hidden_dim = int(arr.shape[1])
        if (
            self._belief_window_bases is None
            or self._belief_window_bases.shape != (self.num_factors, self.window, hidden_dim)
        ):
            self._belief_window_bases = np.zeros(
                (self.num_factors, self.window, hidden_dim),
                dtype=self.dtype,
            )
        self._belief_window_bases[:, int(self._last_write_idx), :] = arr

    def belief_window_base_snapshot(self) -> np.ndarray | None:
        """Hidden state immediately before the first row in ``snapshot()``.

        TD replay re-encodes the visible evidence window from this detached base,
        so the recurrent filter remains trainable on the current window while still
        carrying history that preceded the window.
        """
        if self._belief_hidden is None:
            return None
        if self._count <= 0 or self._belief_window_bases is None:
            return self._belief_hidden.copy()
        base_idx = 0 if self._count < self.window else int(self._start)
        return self._belief_window_bases[:, base_idx, :].copy()

    def reset(self) -> None:
        self._buffer.fill(0)
        self._count = 0
        self._start = 0
        self._belief_hidden = None
        self._belief_window_bases = None
        self._last_write_idx = None
        self._sequence_episode = None

    def set_sequence_episode(self, episode: Any | None) -> None:
        self._sequence_episode = episode

    def sequence_episode(self) -> Any | None:
        return self._sequence_episode

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


class EpisodeSequenceReplayBuffer:
    """Replay complete ordered episodes while bounding stored transitions.

    Sampling returns episode records, never independently sampled transitions.
    This keeps recurrent hidden-state reconstruction and the episode-level
    bootstrap assignment intact.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._storage: list[Any] = []
        self._transition_count = 0
        self._rng = np.random.default_rng(seed)

    def add(self, episode: Any) -> None:
        length = int(getattr(episode, "transition_length", 0))
        if length <= 0:
            raise ValueError("Sequence replay accepts only non-empty episode records.")
        if length > self.capacity:
            raise ValueError(
                "One episode exceeds the sequence replay transition capacity; "
                "refusing to truncate the recurrent trace."
            )
        self._storage.append(episode)
        self._transition_count += length
        while self._transition_count > self.capacity:
            removed = self._storage.pop(0)
            self._transition_count -= int(removed.transition_length)

    def sample(self, batch_size: int) -> list[Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._storage:
            raise ValueError("Cannot sample from an empty EpisodeSequenceReplayBuffer.")
        replace = batch_size > len(self._storage)
        indices = self._rng.choice(len(self._storage), size=batch_size, replace=replace)
        return [self._storage[int(index)] for index in indices]

    @property
    def episode_count(self) -> int:
        return len(self._storage)

    def __len__(self) -> int:
        return int(self._transition_count)


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
