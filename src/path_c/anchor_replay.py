"""Epoch-isolated counterfactual training replay.

Audit anchors deliberately use a separate manifest type and have no sampling
API, preventing confirmatory replicas from leaking into a loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from .policy_epoch import TargetPolicyEpoch, validate_label_policy


@dataclass(frozen=True, slots=True)
class AnchorReplayItem:
    target_policy_epoch_id: int
    target_policy_fingerprint: str
    partner_source: str
    partner_run_id: str
    uniform_or_opportunity: str
    fit_return_mean: tuple[float, ...]
    fit_return_standard_error: tuple[float, ...]
    collection_step: int
    payload: Any = None

    def __post_init__(self) -> None:
        if self.uniform_or_opportunity not in ("uniform", "opportunity"):
            raise ValueError("Anchor stratum must be uniform or opportunity.")
        if not self.partner_source or not self.partner_run_id:
            raise ValueError("Every anchor must preserve partner lineage.")
        if len(self.fit_return_mean) < 2:
            raise ValueError("All-action anchor means are required.")
        if len(self.fit_return_mean) != len(self.fit_return_standard_error):
            raise ValueError("Anchor means and standard errors differ in length.")
        if self.collection_step < 0:
            raise ValueError("Anchor collection step cannot be negative.")


@dataclass(frozen=True, slots=True)
class AnchorTrainingReplay:
    epoch_id: int
    capacity: int = 512
    items: tuple[AnchorReplayItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        if len(self.items) > self.capacity:
            raise ValueError("Replay exceeds its per-epoch capacity.")
        if any(int(item.target_policy_epoch_id) != int(self.epoch_id) for item in self.items):
            raise ValueError("Replay cannot mix target-policy epochs.")

    def add(self, additions: Iterable[AnchorReplayItem]) -> "AnchorTrainingReplay":
        rows = self.items + tuple(additions)
        for row in rows:
            if int(row.target_policy_epoch_id) != int(self.epoch_id):
                raise ValueError("Old target-policy epoch item rejected from replay.")
        return AnchorTrainingReplay(
            epoch_id=self.epoch_id,
            capacity=self.capacity,
            items=rows[-self.capacity :],
        )

    def sample_indexes(self, *, batch_size: int, key: int) -> np.ndarray:
        if batch_size <= 0 or not self.items:
            raise ValueError("Cannot sample an empty replay or non-positive batch.")
        rng = np.random.default_rng(int(key))
        return rng.integers(0, len(self.items), size=int(batch_size), dtype=np.int64)

    def sample(
        self,
        *,
        batch_size: int,
        key: int,
        current_epoch: TargetPolicyEpoch,
    ) -> tuple[AnchorReplayItem, ...]:
        indexes = self.sample_indexes(batch_size=batch_size, key=key)
        rows = tuple(self.items[int(index)] for index in indexes)
        for row in rows:
            validate_label_policy(
                label_epoch_id=row.target_policy_epoch_id,
                label_policy_fingerprint=row.target_policy_fingerprint,
                current_epoch=current_epoch,
            )
        return rows

    @property
    def fingerprint(self) -> str:
        payload = [
            {key: value for key, value in asdict(item).items() if key != "payload"}
            for item in self.items
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class AnchorReplayState(NamedTuple):
    """Orbax-safe current-epoch replay payload used by the CUDA auxiliary scan."""

    target_policy_epoch_id: Any
    target_policy_fingerprint: str
    batch: Any
    item_count: Any
    capacity: int


def append_replay_batch(
    state: AnchorReplayState | None,
    *,
    batch: Any,
    epoch: TargetPolicyEpoch,
    capacity: int,
) -> AnchorReplayState:
    import jax
    import jax.numpy as jnp

    fingerprint = ":".join(
        (epoch.actor_fingerprint, epoch.belief_fingerprint, epoch.raw_q_fingerprint)
    )
    if state is None:
        combined = batch
    else:
        validate_label_policy(
            label_epoch_id=int(state.target_policy_epoch_id),
            label_policy_fingerprint=state.target_policy_fingerprint,
            current_epoch=epoch,
        )
        combined = jax.tree_util.tree_map(
            lambda first, second: jnp.concatenate((first, second), axis=0),
            state.batch,
            batch,
        )
    count = int(jnp.asarray(combined.anchor_ids).shape[0])
    if count > int(capacity):
        combined = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[-int(capacity) :], combined
        )
        count = int(capacity)
    return AnchorReplayState(
        target_policy_epoch_id=jnp.asarray(epoch.epoch_id, dtype=jnp.int32),
        target_policy_fingerprint=fingerprint,
        batch=combined,
        item_count=jnp.asarray(count, dtype=jnp.int32),
        capacity=int(capacity),
    )


def current_epoch_replay_batch(
    state: AnchorReplayState | None,
    *,
    epoch: TargetPolicyEpoch,
) -> Any | None:
    if state is None:
        return None
    validate_label_policy(
        label_epoch_id=int(state.target_policy_epoch_id),
        label_policy_fingerprint=state.target_policy_fingerprint,
        current_epoch=epoch,
    )
    return state.batch


@dataclass(frozen=True, slots=True)
class AnchorAuditRecord:
    milestone: str
    artifact_path: str
    artifact_fingerprint: str
    random_domain: str
    fit_replicas: int
    evaluation_replicas: int
    continuation_horizon: int


@dataclass(frozen=True, slots=True)
class AnchorAuditManifest:
    records: tuple[AnchorAuditRecord, ...] = field(default_factory=tuple)
    expected_fit_replicas: int = 32
    expected_evaluation_replicas: int = 64
    expected_continuation_horizon: int = 400

    def __post_init__(self) -> None:
        if self.expected_fit_replicas <= 0 or self.expected_evaluation_replicas <= 0:
            raise ValueError("Audit replica expectations must be positive.")
        if self.expected_continuation_horizon <= 0:
            raise ValueError("Audit continuation expectation must be positive.")

    def add(self, record: AnchorAuditRecord) -> "AnchorAuditManifest":
        if record.milestone not in ("C0", "15M", "22.5M", "final"):
            raise ValueError("Audit anchor milestone is not preregistered.")
        if (
            record.fit_replicas != self.expected_fit_replicas
            or record.evaluation_replicas != self.expected_evaluation_replicas
        ):
            raise ValueError(
                "Audit replica split differs from the run's registered split "
                f"{self.expected_fit_replicas}/{self.expected_evaluation_replicas}."
            )
        if record.continuation_horizon != self.expected_continuation_horizon:
            raise ValueError(
                "Audit continuation horizon differs from the run's registered "
                f"horizon {self.expected_continuation_horizon}."
            )
        return AnchorAuditManifest(
            records=self.records + (record,),
            expected_fit_replicas=self.expected_fit_replicas,
            expected_evaluation_replicas=self.expected_evaluation_replicas,
            expected_continuation_horizon=self.expected_continuation_horizon,
        )


def audit_manifest_to_mapping(
    manifest: AnchorAuditManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Convert validated host metadata to a fixed-structure checkpoint tree.

    Orbax restoration requires the template and stored PyTrees to have the same
    structure.  A run begins with no audit records but can finish with four, so
    a variable-length tuple is not resumable.  The preregistered milestone set
    provides a strict upper bound; padding preserves one stable tree definition
    without inventing scientific records.
    """

    value = audit_manifest_from_mapping(manifest)
    capacity = 4
    if len(value.records) > capacity:
        raise ValueError("Audit manifest exceeds the preregistered milestones.")
    padding = AnchorAuditRecord(
        milestone="",
        artifact_path="",
        artifact_fingerprint="",
        random_domain="",
        fit_replicas=0,
        evaluation_replicas=0,
        continuation_horizon=0,
    )
    records = value.records + (padding,) * (capacity - len(value.records))
    return {
        "record_count": int(len(value.records)),
        "records": tuple(asdict(record) for record in records),
        "expected_fit_replicas": int(value.expected_fit_replicas),
        "expected_evaluation_replicas": int(value.expected_evaluation_replicas),
        "expected_continuation_horizon": int(
            value.expected_continuation_horizon
        ),
    }


def audit_manifest_from_mapping(
    value: AnchorAuditManifest | Mapping[str, Any],
) -> AnchorAuditManifest:
    """Restore and revalidate host audit metadata after checkpoint loading."""

    if isinstance(value, AnchorAuditManifest):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Anchor audit manifest must be a mapping or manifest.")
    raw_records = tuple(
        item
        if isinstance(item, AnchorAuditRecord)
        else AnchorAuditRecord(**dict(item))
        for item in value.get("records", ())
    )
    record_count = int(value.get("record_count", len(raw_records)))
    if not 0 <= record_count <= 4 or record_count > len(raw_records):
        raise ValueError("Invalid audit checkpoint record count.")
    records = raw_records[:record_count]
    manifest = AnchorAuditManifest(
        expected_fit_replicas=int(value["expected_fit_replicas"]),
        expected_evaluation_replicas=int(value["expected_evaluation_replicas"]),
        expected_continuation_horizon=int(
            value["expected_continuation_horizon"]
        ),
    )
    for record in records:
        manifest = manifest.add(record)
    return manifest


__all__ = [
    "AnchorAuditManifest",
    "AnchorAuditRecord",
    "AnchorReplayItem",
    "AnchorReplayState",
    "AnchorTrainingReplay",
    "append_replay_batch",
    "audit_manifest_from_mapping",
    "audit_manifest_to_mapping",
    "current_epoch_replay_batch",
]
