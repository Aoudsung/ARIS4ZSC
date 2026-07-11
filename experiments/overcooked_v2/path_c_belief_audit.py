from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import math
from numbers import Integral
from pathlib import Path
import struct
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from experiments.overcooked_v2.path_c_audit_battery import (
    AuditRole,
    FrozenAuditBatteryV1,
    ProbeScriptV1,
)


SNAPSHOT_SCHEMA_VERSION = "path_c_snapshot_v1"
RNG_KEY_SCHEDULE_VERSION = "path_c_rng_key_schedule_v1"
KERNEL_RECORD_SCHEMA_VERSION = "path_c_outer_replica_kernel_v1"
EXACT_HISTORY_SCHEMA_VERSION = "path_c_exact_history_v1"
RNG_STREAM_NAMES = ("jax", "numpy", "python", "torch")
INSTRUMENT_EVIDENCE_SCHEMA_VERSION = "path_c_instrument_evidence_v2"
INSTRUMENT_CELL_REGISTRY_SCHEMA_VERSION = "path_c_instrument_cell_registry_v2"
INSTRUMENT_SAMPLING_DESIGN = "iid_full_posterior_with_replacement_v1"
INSTRUMENT_SAMPLING_SOURCE = "categorical_full_state_posterior_v1"

PosteriorMode = Literal["exact", "approximate"]
RNGMode = Literal["replay", "fork"]


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _categorical_full_state_draw_seed(
    *,
    base_seed: int,
    audit_unit_id: str,
    outer_id: int,
) -> int:
    """Return the frozen per-outer seed used by the categorical full-state sampler."""

    return int(
        canonical_sha256(
            {
                "sampler": "categorical_full_state_v1",
                "base_seed": int(base_seed),
                "audit_unit_id": str(audit_unit_id),
                "outer_id": int(outer_id),
            }
        )[:16],
        16,
    )


@dataclass(frozen=True)
class NamedRNGStreamsV1:
    """Named random-number-generator seeds for all supported runtimes."""

    jax: int
    numpy: int
    python: int
    torch: int
    mode: RNGMode
    schedule_version: str = RNG_KEY_SCHEDULE_VERSION
    coordinate: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"replay", "fork"}:
            raise ValueError("RNG stream mode must be 'replay' or 'fork'.")
        if self.schedule_version != RNG_KEY_SCHEDULE_VERSION:
            raise ValueError(
                f"RNG schedule version must be {RNG_KEY_SCHEDULE_VERSION!r}."
            )
        for name in RNG_STREAM_NAMES:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) < 2**64
            ):
                raise ValueError(f"RNG seed {name} must be an unsigned 64-bit integer.")
        if not isinstance(self.coordinate, tuple) or any(
            not isinstance(value, str) or not value
            for value in self.coordinate
        ):
            raise TypeError("RNG coordinate must be a tuple of non-empty strings.")
        if self.mode == "fork" and len(self.coordinate) != 4:
            raise ValueError(
                "Fork RNG coordinate must contain unit, outer replica, probe, and inner fork."
            )
        if self.mode == "replay" and tuple(self.coordinate) != ("generation",):
            raise ValueError("Replay RNG keys must use the generation coordinate.")

    def seed(self, stream_name: str) -> int:
        if stream_name not in RNG_STREAM_NAMES:
            raise KeyError(f"Unknown RNG stream: {stream_name!r}")
        return int(getattr(self, stream_name))

    def jax_seed_uint32(self) -> int:
        """Return seed material suitable for a JAX PRNGKey without importing JAX."""

        return int(self.jax) & 0xFFFFFFFF

    def jax_key_words_uint32(self) -> tuple[int, int]:
        """Return both 32-bit words of the frozen 64-bit JAX key material."""

        value = int(self.jax)
        return ((value >> 32) & 0xFFFFFFFF, value & 0xFFFFFFFF)

    def to_payload(self) -> dict[str, Any]:
        return {
            "jax": int(self.jax),
            "numpy": int(self.numpy),
            "python": int(self.python),
            "torch": int(self.torch),
            "mode": str(self.mode),
            "schedule_version": str(self.schedule_version),
            "coordinate": list(self.coordinate),
        }


@dataclass(frozen=True)
class RNGKeyScheduleV1:
    """Separates deterministic replay keys from fresh continuation fork keys."""

    manifest_seed: int
    original_replay_keys: NamedRNGStreamsV1
    version: str = RNG_KEY_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.version != RNG_KEY_SCHEDULE_VERSION:
            raise ValueError(f"RNG key schedule version must be {RNG_KEY_SCHEDULE_VERSION!r}.")
        if (
            isinstance(self.manifest_seed, bool)
            or not isinstance(self.manifest_seed, Integral)
            or int(self.manifest_seed) < 0
        ):
            raise ValueError("RNG manifest_seed must be a non-negative integer.")
        if not isinstance(self.original_replay_keys, NamedRNGStreamsV1):
            raise TypeError("original_replay_keys must be NamedRNGStreamsV1.")
        if self.original_replay_keys.mode != "replay":
            raise ValueError("original_replay_keys must be marked as replay keys.")
        if self.original_replay_keys.schedule_version != self.version:
            raise ValueError("Replay keys and key schedule use different versions.")

    @classmethod
    def from_original_seeds(
        cls,
        *,
        manifest_seed: int,
        jax: int,
        numpy: int,
        python: int,
        torch: int,
    ) -> "RNGKeyScheduleV1":
        return cls(
            manifest_seed=manifest_seed,
            original_replay_keys=NamedRNGStreamsV1(
                jax=jax,
                numpy=numpy,
                python=python,
                torch=torch,
                mode="replay",
                coordinate=("generation",),
            ),
        )

    def replay_keys(self) -> NamedRNGStreamsV1:
        """Return the exact generation keys; only replay is allowed to call this."""

        return self.original_replay_keys

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest_seed": int(self.manifest_seed),
            "original_replay_keys": self.original_replay_keys.to_payload(),
        }

    def fork_keys(
        self,
        *,
        unit_id: str,
        outer_id: int,
        probe_id: str,
        inner_id: int,
    ) -> NamedRNGStreamsV1:
        if not str(unit_id).strip() or not str(probe_id).strip():
            raise ValueError("Fork unit_id and probe_id must be non-empty.")
        if (
            isinstance(outer_id, bool)
            or not isinstance(outer_id, Integral)
            or int(outer_id) < 0
        ):
            raise ValueError("Fork outer_id must be non-negative.")
        if (
            isinstance(inner_id, bool)
            or not isinstance(inner_id, Integral)
            or int(inner_id) < 0
        ):
            raise ValueError("Fork inner_id must be non-negative.")
        coordinate = (str(unit_id), str(int(outer_id)), str(probe_id), str(int(inner_id)))
        derived = {
            stream_name: self._fresh_seed(stream_name, coordinate)
            for stream_name in RNG_STREAM_NAMES
        }
        return NamedRNGStreamsV1(
            jax=derived["jax"],
            numpy=derived["numpy"],
            python=derived["python"],
            torch=derived["torch"],
            mode="fork",
            schedule_version=self.version,
            coordinate=coordinate,
        )

    def _fresh_seed(self, stream_name: str, coordinate: tuple[str, ...]) -> int:
        original = self.original_replay_keys.seed(stream_name)
        counter = 0
        while True:
            payload = {
                "version": self.version,
                "manifest_seed": int(self.manifest_seed),
                "mode": "fork",
                "coordinate": list(coordinate),
                "stream_name": stream_name,
                "collision_counter": counter,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).digest()
            candidate = int.from_bytes(digest[:8], "big", signed=False)
            if candidate != original:
                return candidate
            counter += 1


@dataclass(frozen=True)
class SnapshotV1:
    """Fork-complete immutable snapshot represented only by immutable byte strings."""

    unit_id: str
    episode_uid: str
    decision_index: int
    env_state_bytes: bytes
    raw_observation_bytes: bytes
    partner_state_bytes: bytes
    ego_state_bytes: bytes
    rng_key_schedule: RNGKeyScheduleV1
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"Snapshot schema_version must be {SNAPSHOT_SCHEMA_VERSION!r}.")
        if not str(self.unit_id).strip() or not str(self.episode_uid).strip():
            raise ValueError("Snapshot unit_id and episode_uid must be non-empty.")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, Integral)
            or int(self.decision_index) < 0
        ):
            raise ValueError("Snapshot decision_index must be non-negative.")
        if not isinstance(self.rng_key_schedule, RNGKeyScheduleV1):
            raise TypeError("Snapshot rng_key_schedule must be RNGKeyScheduleV1.")
        for name in (
            "env_state_bytes",
            "raw_observation_bytes",
            "partner_state_bytes",
            "ego_state_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"Snapshot {name} must be a non-empty immutable bytes value.")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.schema_version.encode("utf-8"),
            self.unit_id.encode("utf-8"),
            self.episode_uid.encode("utf-8"),
            struct.pack(">Q", int(self.decision_index)),
            self.env_state_bytes,
            self.raw_observation_bytes,
            self.partner_state_bytes,
            self.ego_state_bytes,
            json.dumps(
                {
                    "manifest_seed": int(self.rng_key_schedule.manifest_seed),
                    "original_replay_keys": self.rng_key_schedule.replay_keys().to_payload(),
                    "version": self.rng_key_schedule.version,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        ):
            digest.update(struct.pack(">Q", len(value)))
            digest.update(value)
        return digest.hexdigest()

    def snapshot_copy(self) -> "SnapshotV1":
        """Return a value copy; rollout code never receives a mutable shared object."""

        return replace(self)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "episode_uid": self.episode_uid,
            "decision_index": int(self.decision_index),
            "env_state_sha256": hashlib.sha256(self.env_state_bytes).hexdigest(),
            "raw_observation_sha256": hashlib.sha256(
                self.raw_observation_bytes
            ).hexdigest(),
            "partner_state_sha256": hashlib.sha256(
                self.partner_state_bytes
            ).hexdigest(),
            "ego_state_sha256": hashlib.sha256(self.ego_state_bytes).hexdigest(),
            "rng_key_schedule": self.rng_key_schedule.to_payload(),
            "snapshot_sha256": self.sha256,
        }


@runtime_checkable
class StateCodec(Protocol):
    def encode(self, value: Any) -> bytes:
        ...

    def decode(self, payload: bytes) -> Any:
        ...


def capture_snapshot_v1(
    *,
    unit_id: str,
    episode_uid: str,
    decision_index: int,
    env_state: Any,
    raw_observation: Any,
    partner_state: Any,
    ego_state: Any,
    state_codec: StateCodec,
    observation_codec: StateCodec,
    rng_key_schedule: RNGKeyScheduleV1,
) -> SnapshotV1:
    """Encode every continuation-relevant state component into an immutable snapshot."""

    encoded = {
        "env_state_bytes": state_codec.encode(env_state),
        "raw_observation_bytes": observation_codec.encode(raw_observation),
        "partner_state_bytes": state_codec.encode(partner_state),
        "ego_state_bytes": state_codec.encode(ego_state),
    }
    for name, value in encoded.items():
        if not isinstance(value, bytes):
            raise TypeError(f"Snapshot codec returned a non-bytes value for {name}.")
    return SnapshotV1(
        unit_id=str(unit_id),
        episode_uid=str(episode_uid),
        decision_index=decision_index,
        rng_key_schedule=rng_key_schedule,
        **encoded,
    )


@dataclass(frozen=True)
class FullHiddenStateV1:
    theta_id: str
    execution_state_key: str
    execution_state_bytes: bytes

    def __post_init__(self) -> None:
        if not str(self.theta_id).strip() or not str(self.execution_state_key).strip():
            raise ValueError("Full hidden state theta_id and execution_state_key are required.")
        if not isinstance(self.execution_state_bytes, bytes) or not self.execution_state_bytes:
            raise ValueError("Full hidden execution state must be encoded as non-empty bytes.")

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "theta_id": self.theta_id,
                "execution_state_key": self.execution_state_key,
                "execution_state_bytes_sha256": hashlib.sha256(
                    self.execution_state_bytes
                ).hexdigest(),
            }
        )


@dataclass(frozen=True)
class ForwardHypothesisV1:
    theta_id: str
    state_key: str
    state_bytes: bytes
    mass: float

    def __post_init__(self) -> None:
        if not str(self.theta_id).strip() or not str(self.state_key).strip():
            raise ValueError("Forward hypothesis theta_id and state_key are required.")
        if not isinstance(self.state_bytes, bytes) or not self.state_bytes:
            raise ValueError("Forward hypothesis state_bytes must be non-empty bytes.")
        if not math.isfinite(float(self.mass)) or float(self.mass) <= 0.0:
            raise ValueError("Forward hypothesis mass must be positive and finite.")


@dataclass(frozen=True)
class SparseForwardStateV1:
    theta_id: str
    state_key: str
    state_bytes: bytes


@dataclass(frozen=True)
class ForwardBranchV1:
    state_key: str
    state_bytes: bytes
    probability: float

    def __post_init__(self) -> None:
        if not str(self.state_key).strip():
            raise ValueError("Forward branch state_key is required.")
        if not isinstance(self.state_bytes, bytes) or not self.state_bytes:
            raise ValueError("Forward branch state_bytes must be non-empty bytes.")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Forward branch probability must be in [0, 1].")


ForwardTransition = Callable[
    [SparseForwardStateV1, Any, int],
    Iterable[ForwardBranchV1],
]


@dataclass(frozen=True)
class PosteriorComponentV1:
    hidden_state: FullHiddenStateV1
    posterior_probability: float

    def __post_init__(self) -> None:
        probability = float(self.posterior_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Posterior component probability must be in (0, 1].")


@dataclass(frozen=True)
class SparseForwardStepV1:
    step_index: int
    hypotheses_before: int
    emitted_branches: int
    hypotheses_after_merge: int
    merged_branch_count: int
    pruned_hypotheses: int
    pruned_mass_upper_this_step: float


@dataclass(frozen=True)
class SparseForwardResultV1:
    mode: PosteriorMode
    prune_below: float
    components: tuple[PosteriorComponentV1, ...]
    steps: tuple[SparseForwardStepV1, ...]
    log_retained_evidence: float
    log_evidence_upper: float
    discarded_mass_upper: float
    rho_prune: float

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "approximate"}:
            raise ValueError("Forward result mode must be exact or approximate.")
        if not self.components:
            raise ValueError("Forward result must contain posterior components.")
        total = math.fsum(item.posterior_probability for item in self.components)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("Forward posterior probabilities must sum to one.")
        if self.mode == "exact" and (self.prune_below != 0.0 or self.rho_prune != 0.0):
            raise ValueError("Exact forward recursion cannot prune positive posterior mass.")
        if not 0.0 <= float(self.rho_prune) <= 1.0:
            raise ValueError("rho_prune must be in [0, 1].")


def sparse_state_merging_forward_recursion(
    initial_hypotheses: Sequence[ForwardHypothesisV1],
    observations: Sequence[Any],
    transition: ForwardTransition,
    *,
    mode: PosteriorMode = "exact",
    prune_below: float = 0.0,
) -> SparseForwardResultV1:
    """Run a log-space forward recursion and merge identical sparse states.

    In exact mode any positive pruning threshold is rejected. Approximate mode
    reports ``rho_prune``, a conservative upper bound on posterior mass discarded
    by pruning. Branch probabilities for one source hypothesis must sum to at most
    one, which makes each pruned mass an upper bound on all of its descendants.
    """

    if mode not in {"exact", "approximate"}:
        raise ValueError("Forward recursion mode must be exact or approximate.")
    threshold = float(prune_below)
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("prune_below must be in [0, 1).")
    if mode == "exact" and threshold > 0.0:
        raise ValueError("Exact forward recursion forbids positive-mass pruning.")
    if not initial_hypotheses:
        raise ValueError("Forward recursion requires initial hypotheses.")

    log_masses: dict[tuple[str, str], float] = {}
    payloads: dict[tuple[str, str], bytes] = {}
    for hypothesis in initial_hypotheses:
        key = (hypothesis.theta_id, hypothesis.state_key)
        _merge_log_hypothesis(
            log_masses,
            payloads,
            key,
            hypothesis.state_bytes,
            math.log(float(hypothesis.mass)),
        )
    initial_total = _logsumexp(log_masses.values())
    log_masses = {key: value - initial_total for key, value in log_masses.items()}

    discarded_log_upper = -math.inf
    step_reports: list[SparseForwardStepV1] = []
    for step_index, observation in enumerate(observations):
        next_masses: dict[tuple[str, str], float] = {}
        next_payloads: dict[tuple[str, str], bytes] = {}
        emitted_branches = 0
        for (theta_id, state_key), log_mass in sorted(log_masses.items()):
            state = SparseForwardStateV1(
                theta_id=theta_id,
                state_key=state_key,
                state_bytes=payloads[(theta_id, state_key)],
            )
            branches = tuple(transition(state, observation, step_index))
            branch_total = math.fsum(float(branch.probability) for branch in branches)
            if branch_total > 1.0 + 1.0e-12:
                raise ValueError(
                    "Forward transition branch probabilities exceed one for one hypothesis."
                )
            for branch in branches:
                probability = float(branch.probability)
                if probability == 0.0:
                    continue
                emitted_branches += 1
                key = (theta_id, branch.state_key)
                _merge_log_hypothesis(
                    next_masses,
                    next_payloads,
                    key,
                    branch.state_bytes,
                    log_mass + math.log(probability),
                )
        if not next_masses:
            raise ValueError(f"Forward recursion has zero evidence at step {step_index}.")

        after_merge = len(next_masses)
        pruned_keys: list[tuple[str, str]] = []
        pruned_step_log = -math.inf
        if mode == "approximate" and threshold > 0.0:
            step_total = _logsumexp(next_masses.values())
            for key, log_mass in next_masses.items():
                normalized_mass = math.exp(log_mass - step_total)
                if normalized_mass < threshold:
                    pruned_keys.append(key)
                    pruned_step_log = _logaddexp(pruned_step_log, log_mass)
            if len(pruned_keys) == len(next_masses):
                raise ValueError(
                    f"Approximate forward pruning removed every hypothesis at step {step_index}."
                )
            for key in pruned_keys:
                del next_masses[key]
                del next_payloads[key]
            discarded_log_upper = _logaddexp(discarded_log_upper, pruned_step_log)

        step_reports.append(
            SparseForwardStepV1(
                step_index=step_index,
                hypotheses_before=len(log_masses),
                emitted_branches=emitted_branches,
                hypotheses_after_merge=after_merge,
                merged_branch_count=max(0, emitted_branches - after_merge),
                pruned_hypotheses=len(pruned_keys),
                pruned_mass_upper_this_step=(
                    0.0 if pruned_step_log == -math.inf else math.exp(pruned_step_log)
                ),
            )
        )
        log_masses, payloads = next_masses, next_payloads

    retained_log = _logsumexp(log_masses.values())
    evidence_upper_log = _logaddexp(retained_log, discarded_log_upper)
    rho_prune = (
        0.0
        if discarded_log_upper == -math.inf
        else math.exp(discarded_log_upper - evidence_upper_log)
    )
    components = tuple(
        PosteriorComponentV1(
            hidden_state=FullHiddenStateV1(
                theta_id=theta_id,
                execution_state_key=state_key,
                execution_state_bytes=payloads[(theta_id, state_key)],
            ),
            posterior_probability=math.exp(log_mass - retained_log),
        )
        for (theta_id, state_key), log_mass in sorted(log_masses.items())
    )
    return SparseForwardResultV1(
        mode=mode,
        prune_below=threshold,
        components=components,
        steps=tuple(step_reports),
        log_retained_evidence=retained_log,
        log_evidence_upper=evidence_upper_log,
        discarded_mass_upper=(
            0.0 if discarded_log_upper == -math.inf else math.exp(discarded_log_upper)
        ),
        rho_prune=rho_prune,
    )


@dataclass(frozen=True)
class OuterReplicaDrawV1:
    audit_unit_id: str
    outer_id: int
    hidden_state: FullHiddenStateV1
    posterior_draw_probability: float
    sampler_seed: int
    posterior_mode: PosteriorMode
    rho_prune: float
    source_episode_uid: str | None = None

    def __post_init__(self) -> None:
        if not str(self.audit_unit_id).strip():
            raise ValueError("Outer replica audit_unit_id is required.")
        if isinstance(self.outer_id, bool) or not isinstance(self.outer_id, Integral):
            raise TypeError("Outer replica id must be an integer.")
        if int(self.outer_id) < 0:
            raise ValueError("Outer replica id must be non-negative.")
        if isinstance(self.sampler_seed, bool) or not isinstance(
            self.sampler_seed, Integral
        ):
            raise TypeError("Outer replica sampler_seed must be an integer.")
        if not 0 <= int(self.sampler_seed) < 2**64:
            raise ValueError("Outer replica sampler_seed must be an unsigned 64-bit integer.")
        if not isinstance(self.hidden_state, FullHiddenStateV1):
            raise TypeError("Outer replica hidden_state must be a complete hidden state.")
        if self.source_episode_uid is not None and not str(self.source_episode_uid).strip():
            raise ValueError("Outer replica source_episode_uid cannot be empty.")
        probability = float(self.posterior_draw_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Outer replica draw probability must be in (0, 1].")
        if self.posterior_mode not in {"exact", "approximate"}:
            raise ValueError("Outer replica posterior_mode must be exact or approximate.")
        if not 0.0 <= float(self.rho_prune) <= 1.0:
            raise ValueError("Outer replica rho_prune must be in [0, 1].")
        if self.posterior_mode == "exact" and float(self.rho_prune) != 0.0:
            raise ValueError("An exact outer sampler cannot report positive rho_prune.")


@runtime_checkable
class PosteriorFullStateSampler(Protocol):
    """Draw complete hidden states from a registered posterior over ``U=(theta,s)``."""

    def sample(
        self,
        audit_unit_id: str,
        n: int,
        seed: int,
        *,
        outer_id_start: int = 0,
    ) -> tuple[OuterReplicaDrawV1, ...]:
        ...


@dataclass(frozen=True)
class CategoricalPosteriorFullStateSampler:
    """Finite full-state sampler backed by a sparse forward-recursion posterior."""

    posterior: SparseForwardResultV1

    def sample(
        self,
        audit_unit_id: str,
        n: int,
        seed: int,
        *,
        outer_id_start: int = 0,
    ) -> tuple[OuterReplicaDrawV1, ...]:
        if not str(audit_unit_id).strip():
            raise ValueError("audit_unit_id must be non-empty.")
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("Posterior sampler n must be positive.")
        if (
            isinstance(outer_id_start, bool)
            or not isinstance(outer_id_start, Integral)
            or int(outer_id_start) < 0
        ):
            raise ValueError("outer_id_start must be non-negative.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError("Posterior sampler seed must be a non-negative integer.")
        components = tuple(sorted(
            self.posterior.components,
            key=lambda item: (
                item.hidden_state.theta_id,
                item.hidden_state.execution_state_key,
                item.hidden_state.sha256,
            ),
        ))
        probabilities = np.asarray(
            [item.posterior_probability for item in components],
            dtype=np.float64,
        )
        draws = []
        for offset in range(int(n)):
            outer_id = int(outer_id_start) + offset
            draw_seed = _categorical_full_state_draw_seed(
                base_seed=int(seed),
                audit_unit_id=str(audit_unit_id),
                outer_id=outer_id,
            )
            rng = np.random.default_rng(draw_seed)
            component_index = int(
                rng.choice(
                    len(components),
                    size=None,
                    replace=True,
                    p=probabilities,
                )
            )
            component = components[int(component_index)]
            draws.append(
                OuterReplicaDrawV1(
                    audit_unit_id=str(audit_unit_id),
                    outer_id=outer_id,
                    hidden_state=component.hidden_state,
                    posterior_draw_probability=float(component.posterior_probability),
                    sampler_seed=draw_seed,
                    posterior_mode=self.posterior.mode,
                    rho_prune=float(self.posterior.rho_prune),
                )
            )
        return tuple(draws)


@dataclass(frozen=True)
class CategoricalRolloutResultV1:
    response_token_id: int
    support_violation: bool
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.response_token_id, bool) or int(self.response_token_id) < 0:
            raise ValueError("Categorical response token id must be non-negative.")
        if not _is_sha256(self.trajectory_sha256):
            raise ValueError("Rollout trajectory_sha256 must be a SHA-256 digest.")


@runtime_checkable
class SnapshotForkRunner(Protocol):
    def run(
        self,
        *,
        snapshot: SnapshotV1,
        hidden_state: FullHiddenStateV1 | None,
        probe_script: ProbeScriptV1,
        rng_keys: NamedRNGStreamsV1,
    ) -> CategoricalRolloutResultV1:
        ...


@dataclass(frozen=True)
class SnapshotExecutionV1:
    rng_keys: NamedRNGStreamsV1
    result: CategoricalRolloutResultV1


def replay_from_snapshot(
    snapshot: SnapshotV1,
    probe_script: ProbeScriptV1,
    runner: SnapshotForkRunner,
) -> SnapshotExecutionV1:
    """Replay with original generation keys and without hidden-state replacement."""

    rng_keys = snapshot.rng_key_schedule.replay_keys()
    result = runner.run(
        snapshot=snapshot.snapshot_copy(),
        hidden_state=None,
        probe_script=probe_script,
        rng_keys=rng_keys,
    )
    if not isinstance(result, CategoricalRolloutResultV1):
        raise TypeError("Snapshot replay runner must return CategoricalRolloutResultV1.")
    return SnapshotExecutionV1(rng_keys=rng_keys, result=result)


def fork_from_snapshot(
    snapshot: SnapshotV1,
    outer_draw: OuterReplicaDrawV1,
    probe_script: ProbeScriptV1,
    runner: SnapshotForkRunner,
    *,
    inner_id: int,
) -> SnapshotExecutionV1:
    """Run a posterior-state continuation with fresh, coordinate-derived RNG keys."""

    if snapshot.unit_id != outer_draw.audit_unit_id:
        raise ValueError("Snapshot unit and posterior outer draw refer to different audit units.")
    if isinstance(inner_id, bool) or not isinstance(inner_id, Integral) or int(inner_id) < 0:
        raise ValueError("Fork inner_id must be a non-negative integer.")
    rng_keys = snapshot.rng_key_schedule.fork_keys(
        unit_id=outer_draw.audit_unit_id,
        outer_id=outer_draw.outer_id,
        probe_id=probe_script.script_id,
        inner_id=inner_id,
    )
    result = runner.run(
        snapshot=snapshot.snapshot_copy(),
        hidden_state=outer_draw.hidden_state,
        probe_script=probe_script,
        rng_keys=rng_keys,
    )
    if not isinstance(result, CategoricalRolloutResultV1):
        raise TypeError("Snapshot fork runner must return CategoricalRolloutResultV1.")
    return SnapshotExecutionV1(rng_keys=rng_keys, result=result)


def verify_snapshot_runner_purity(
    snapshot: SnapshotV1,
    outer_draw: OuterReplicaDrawV1,
    probe_script: ProbeScriptV1,
    runner: SnapshotForkRunner,
    *,
    inner_id: int,
) -> bool:
    """Require identical outputs for the same immutable fork coordinate."""

    before = snapshot.sha256
    first = fork_from_snapshot(
        snapshot, outer_draw, probe_script, runner, inner_id=inner_id
    )
    second = fork_from_snapshot(
        snapshot, outer_draw, probe_script, runner, inner_id=inner_id
    )
    if snapshot.sha256 != before:
        raise RuntimeError("Snapshot runner mutated the source snapshot.")
    if first != second:
        raise RuntimeError(
            "Snapshot runner is not pure for an identical state and RNG coordinate."
        )
    return True


@dataclass(frozen=True)
class InnerForkResponseV1:
    inner_id: int
    response_token_id: int
    support_violation: bool
    trajectory_sha256: str
    rng_keys: NamedRNGStreamsV1

    def __post_init__(self) -> None:
        if (
            isinstance(self.inner_id, bool)
            or not isinstance(self.inner_id, Integral)
            or int(self.inner_id) < 0
        ):
            raise ValueError("Inner fork id must be non-negative.")
        if (
            isinstance(self.response_token_id, bool)
            or not isinstance(self.response_token_id, Integral)
            or int(self.response_token_id) < 0
        ):
            raise ValueError("Inner response token id must be non-negative.")
        if type(self.support_violation) is not bool:
            raise TypeError("Inner response support_violation must be boolean.")
        if not _is_sha256(self.trajectory_sha256):
            raise ValueError("Inner response trajectory_sha256 must be a SHA-256 digest.")
        if not isinstance(self.rng_keys, NamedRNGStreamsV1):
            raise TypeError("Inner response rng_keys must be NamedRNGStreamsV1.")
        if self.rng_keys.mode != "fork":
            raise ValueError("Inner fork responses must carry fork RNG keys.")


@dataclass(frozen=True)
class PairedProbeResponsesV1:
    probe_id: str
    sampling_probability: float
    responses: tuple[InnerForkResponseV1, ...]

    def __post_init__(self) -> None:
        if not str(self.probe_id).strip():
            raise ValueError("Paired probe id must be non-empty.")
        probability = float(self.sampling_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Paired probe sampling_probability must be in (0, 1].")
        if not self.responses:
            raise ValueError("Each paired probe must contain at least one inner response.")
        ids = [response.inner_id for response in self.responses]
        if ids != list(range(len(ids))):
            raise ValueError("Inner fork response ids must be contiguous from zero.")


@dataclass(frozen=True)
class OuterReplicaKernelRecordV1:
    """One row for one posterior outer replica, with all probes kept paired."""

    audit_unit_id: str
    outer_id: int
    outer_cluster_id: str
    snapshot_sha256: str
    battery_sha256: str
    theta_id: str
    execution_state_key: str
    hidden_state_sha256: str
    posterior_draw_probability: float
    sampler_seed: int
    source_episode_uid: str | None
    posterior_mode: PosteriorMode
    rho_prune: float
    probe_responses: tuple[PairedProbeResponsesV1, ...]
    schema_version: str = KERNEL_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != KERNEL_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"Kernel record schema_version must be {KERNEL_RECORD_SCHEMA_VERSION!r}."
            )
        if not self.probe_responses:
            raise ValueError("Outer replica record must contain paired probe responses.")
        if not all(
            str(value).strip()
            for value in (
                self.audit_unit_id,
                self.theta_id,
                self.execution_state_key,
            )
        ):
            raise ValueError("Kernel record unit and hidden-state identifiers are required.")
        if (
            isinstance(self.outer_id, bool)
            or not isinstance(self.outer_id, Integral)
            or int(self.outer_id) < 0
        ):
            raise ValueError("Kernel record outer_id must be a non-negative integer.")
        if self.outer_cluster_id != f"{self.audit_unit_id}:{self.outer_id}":
            raise ValueError("Outer cluster id must bind audit_unit_id and outer_id.")
        probability = float(self.posterior_draw_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Kernel posterior draw probability must lie in (0, 1].")
        if (
            isinstance(self.sampler_seed, bool)
            or not isinstance(self.sampler_seed, Integral)
            or not 0 <= int(self.sampler_seed) < 2**64
        ):
            raise ValueError("Kernel sampler_seed must be an unsigned 64-bit integer.")
        if self.source_episode_uid is not None and not str(self.source_episode_uid).strip():
            raise ValueError("Kernel source_episode_uid cannot be empty.")
        if self.posterior_mode not in {"exact", "approximate"}:
            raise ValueError("Kernel posterior_mode must be exact or approximate.")
        if not math.isfinite(float(self.rho_prune)) or not 0.0 <= float(
            self.rho_prune
        ) <= 1.0:
            raise ValueError("Kernel rho_prune must lie in [0, 1].")
        if self.posterior_mode == "exact" and float(self.rho_prune) != 0.0:
            raise ValueError("An exact kernel record cannot report positive pruning.")
        if any(
            not isinstance(item, PairedProbeResponsesV1)
            for item in self.probe_responses
        ):
            raise TypeError("Kernel probe_responses must be paired probe records.")
        probe_ids = [item.probe_id for item in self.probe_responses]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("Outer replica record repeats a probe id.")
        if not _is_sha256(self.snapshot_sha256) or not _is_sha256(self.battery_sha256):
            raise ValueError("Kernel record snapshot and battery hashes must be SHA-256 digests.")
        if not _is_sha256(self.hidden_state_sha256):
            raise ValueError("Kernel record hidden_state_sha256 must be a SHA-256 digest.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_unit_id": self.audit_unit_id,
            "outer_id": int(self.outer_id),
            "outer_cluster_id": self.outer_cluster_id,
            "snapshot_sha256": self.snapshot_sha256,
            "battery_sha256": self.battery_sha256,
            "theta_id": self.theta_id,
            "execution_state_key": self.execution_state_key,
            "hidden_state_sha256": self.hidden_state_sha256,
            "posterior_draw_probability": float(self.posterior_draw_probability),
            "sampler_seed": int(self.sampler_seed),
            "source_episode_uid": self.source_episode_uid,
            "posterior_mode": self.posterior_mode,
            "rho_prune": float(self.rho_prune),
            "probe_responses": [
                {
                    "probe_id": item.probe_id,
                    "sampling_probability": float(item.sampling_probability),
                    "responses": [
                        {
                            "inner_id": int(response.inner_id),
                            "response_token_id": int(response.response_token_id),
                            "support_violation": bool(response.support_violation),
                            "trajectory_sha256": response.trajectory_sha256,
                            "rng_keys": response.rng_keys.to_payload(),
                        }
                        for response in item.responses
                    ],
                }
                for item in self.probe_responses
            ],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "OuterReplicaKernelRecordV1":
        """Parse and revalidate one complete embedded kernel record."""

        expected_record_keys = {
            "schema_version",
            "audit_unit_id",
            "outer_id",
            "outer_cluster_id",
            "snapshot_sha256",
            "battery_sha256",
            "theta_id",
            "execution_state_key",
            "hidden_state_sha256",
            "posterior_draw_probability",
            "sampler_seed",
            "source_episode_uid",
            "posterior_mode",
            "rho_prune",
            "probe_responses",
        }
        record = _strict_mapping(payload, "embedded kernel record")
        _require_exact_keys(record, expected_record_keys, "embedded kernel record")
        raw_probe_responses = _strict_sequence(
            record["probe_responses"],
            "embedded kernel record probe_responses",
        )
        probe_responses = []
        for probe_index, raw_probe in enumerate(raw_probe_responses):
            probe = _strict_mapping(
                raw_probe,
                f"embedded kernel probe_responses[{probe_index}]",
            )
            _require_exact_keys(
                probe,
                {"probe_id", "sampling_probability", "responses"},
                f"embedded kernel probe_responses[{probe_index}]",
            )
            raw_responses = _strict_sequence(
                probe["responses"],
                f"embedded kernel probe_responses[{probe_index}].responses",
            )
            responses = []
            for response_index, raw_response in enumerate(raw_responses):
                response = _strict_mapping(
                    raw_response,
                    "embedded kernel "
                    f"probe_responses[{probe_index}].responses[{response_index}]",
                )
                _require_exact_keys(
                    response,
                    {
                        "inner_id",
                        "response_token_id",
                        "support_violation",
                        "trajectory_sha256",
                        "rng_keys",
                    },
                    "embedded kernel "
                    f"probe_responses[{probe_index}].responses[{response_index}]",
                )
                responses.append(
                    InnerForkResponseV1(
                        inner_id=_strict_nonbool_int(
                            response["inner_id"],
                            "embedded kernel inner_id",
                            minimum=0,
                        ),
                        response_token_id=_strict_nonbool_int(
                            response["response_token_id"],
                            "embedded kernel response_token_id",
                            minimum=0,
                        ),
                        support_violation=_strict_bool(
                            response["support_violation"],
                            "embedded kernel support_violation",
                        ),
                        trajectory_sha256=str(response["trajectory_sha256"]),
                        rng_keys=_named_rng_streams_from_payload(
                            response["rng_keys"]
                        ),
                    )
                )
            probe_responses.append(
                PairedProbeResponsesV1(
                    probe_id=_strict_text(
                        probe["probe_id"],
                        "embedded kernel probe_id",
                    ),
                    sampling_probability=_strict_finite_float(
                        probe["sampling_probability"],
                        "embedded kernel sampling_probability",
                    ),
                    responses=tuple(responses),
                )
            )
        source_episode_uid = record["source_episode_uid"]
        if source_episode_uid is not None:
            source_episode_uid = _strict_text(
                source_episode_uid,
                "embedded kernel source_episode_uid",
            )
        return cls(
            audit_unit_id=_strict_text(
                record["audit_unit_id"],
                "embedded kernel audit_unit_id",
            ),
            outer_id=_strict_nonbool_int(
                record["outer_id"],
                "embedded kernel outer_id",
                minimum=0,
            ),
            outer_cluster_id=_strict_text(
                record["outer_cluster_id"],
                "embedded kernel outer_cluster_id",
            ),
            snapshot_sha256=str(record["snapshot_sha256"]),
            battery_sha256=str(record["battery_sha256"]),
            theta_id=_strict_text(
                record["theta_id"],
                "embedded kernel theta_id",
            ),
            execution_state_key=_strict_text(
                record["execution_state_key"],
                "embedded kernel execution_state_key",
            ),
            hidden_state_sha256=str(record["hidden_state_sha256"]),
            posterior_draw_probability=_strict_finite_float(
                record["posterior_draw_probability"],
                "embedded kernel posterior_draw_probability",
            ),
            sampler_seed=_strict_nonbool_int(
                record["sampler_seed"],
                "embedded kernel sampler_seed",
                minimum=0,
            ),
            source_episode_uid=source_episode_uid,
            posterior_mode=_strict_text(
                record["posterior_mode"],
                "embedded kernel posterior_mode",
            ),
            rho_prune=_strict_finite_float(
                record["rho_prune"],
                "embedded kernel rho_prune",
            ),
            probe_responses=tuple(probe_responses),
            schema_version=_strict_text(
                record["schema_version"],
                "embedded kernel schema_version",
            ),
        )


def collect_paired_battery_kernel_records(
    snapshot: SnapshotV1,
    outer_draws: Sequence[OuterReplicaDrawV1],
    battery: FrozenAuditBatteryV1,
    runner: SnapshotForkRunner,
    *,
    role: AuditRole,
) -> tuple[OuterReplicaKernelRecordV1, ...]:
    """Collect one record per outer draw; inner forks never become outer samples."""

    if not outer_draws:
        raise ValueError("Kernel collection requires posterior outer draws.")
    scripts = battery.scripts_for_role(role)
    outer_ids = [draw.outer_id for draw in outer_draws]
    if len(outer_ids) != len(set(outer_ids)):
        raise ValueError("Posterior outer ids must be unique in one kernel table.")
    records = []
    for draw in sorted(outer_draws, key=lambda item: item.outer_id):
        if draw.audit_unit_id != snapshot.unit_id:
            raise ValueError("All posterior draws must match the snapshot audit unit.")
        paired = []
        for script in scripts:
            responses = []
            for inner_id in range(int(battery.L_inner)):
                execution = fork_from_snapshot(
                    snapshot,
                    draw,
                    script,
                    runner,
                    inner_id=inner_id,
                )
                responses.append(
                    InnerForkResponseV1(
                        inner_id=inner_id,
                        response_token_id=int(execution.result.response_token_id),
                        support_violation=bool(execution.result.support_violation),
                        trajectory_sha256=execution.result.trajectory_sha256,
                        rng_keys=execution.rng_keys,
                    )
                )
            paired.append(
                PairedProbeResponsesV1(
                    probe_id=script.script_id,
                    sampling_probability=float(script.sampling_probability),
                    responses=tuple(responses),
                )
            )
        records.append(
            OuterReplicaKernelRecordV1(
                audit_unit_id=draw.audit_unit_id,
                outer_id=draw.outer_id,
                outer_cluster_id=f"{draw.audit_unit_id}:{draw.outer_id}",
                snapshot_sha256=snapshot.sha256,
                battery_sha256=battery.sha256,
                theta_id=draw.hidden_state.theta_id,
                execution_state_key=draw.hidden_state.execution_state_key,
                hidden_state_sha256=draw.hidden_state.sha256,
                posterior_draw_probability=float(draw.posterior_draw_probability),
                sampler_seed=int(draw.sampler_seed),
                source_episode_uid=draw.source_episode_uid,
                posterior_mode=draw.posterior_mode,
                rho_prune=float(draw.rho_prune),
                probe_responses=tuple(paired),
            )
        )
    return tuple(records)


def first_inner_tokens_by_probe(
    records: Sequence[OuterReplicaKernelRecordV1],
    probe_id: str,
) -> tuple[int, ...]:
    """Return one theorem-aligned categorical token per independent outer replica.

    Additional inner forks remain conditional-noise diagnostics and are not counted
    as additional posterior samples.
    """

    if not records:
        raise ValueError("Token extraction requires outer replica records.")
    cluster_ids = [record.outer_cluster_id for record in records]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Outer cluster ids must be unique.")
    tokens = []
    for record in records:
        matches = [item for item in record.probe_responses if item.probe_id == probe_id]
        if len(matches) != 1:
            raise ValueError(f"Probe {probe_id!r} is missing or duplicated in a kernel row.")
        tokens.append(int(matches[0].responses[0].response_token_id))
    return tuple(tokens)


class HistoryHashCollisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExactHistoryKeyV1:
    sha256: str
    canonical_bytes: bytes
    schema_version: str = EXACT_HISTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXACT_HISTORY_SCHEMA_VERSION:
            raise ValueError(
                f"Exact-history schema_version must be {EXACT_HISTORY_SCHEMA_VERSION!r}."
            )
        if not isinstance(self.canonical_bytes, bytes) or not self.canonical_bytes:
            raise ValueError("Exact-history canonical_bytes must be non-empty bytes.")
        observed = hashlib.sha256(self.canonical_bytes).hexdigest()
        if self.sha256 != observed:
            raise ValueError("Exact-history SHA-256 does not match its canonical bytes.")


def make_exact_history_key(
    raw_observable_states: Sequence[Mapping[str, Any]],
    ego_primitive_actions: Sequence[int],
    partner_primitive_actions: Sequence[int],
    *,
    field_order: Sequence[str],
) -> ExactHistoryKeyV1:
    """Serialize a discrete raw-state and joint-action history byte-for-byte."""

    fields = tuple(map(str, field_order))
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Exact-history field_order must be non-empty and unique.")
    n_steps = len(raw_observable_states)
    if not (
        n_steps == len(ego_primitive_actions) == len(partner_primitive_actions)
    ):
        raise ValueError("Raw states and both primitive-action histories must align.")
    if n_steps == 0:
        raise ValueError("Exact history must contain at least one step.")

    output = bytearray()
    _append_blob(output, EXACT_HISTORY_SCHEMA_VERSION.encode("utf-8"))
    output.extend(struct.pack(">I", n_steps))
    output.extend(struct.pack(">I", len(fields)))
    for field in fields:
        _append_blob(output, field.encode("utf-8"))
    expected_fields = set(fields)
    for step_index, raw_state in enumerate(raw_observable_states):
        string_state = {str(key): value for key, value in raw_state.items()}
        if len(string_state) != len(raw_state):
            raise ValueError("Exact-history state contains keys that collide after string conversion.")
        observed_fields = set(string_state)
        if observed_fields != expected_fields:
            missing = sorted(expected_fields.difference(observed_fields))
            extra = sorted(observed_fields.difference(expected_fields))
            raise ValueError(
                f"Exact-history state {step_index} field mismatch; missing={missing}, extra={extra}."
            )
        output.extend(struct.pack(">I", step_index))
        for field in fields:
            dtype_text, shape, data = _canonical_discrete_array(string_state[field])
            _append_blob(output, dtype_text.encode("ascii"))
            output.extend(struct.pack(">I", len(shape)))
            for dimension in shape:
                output.extend(struct.pack(">Q", int(dimension)))
            _append_blob(output, data)
        raw_ego_action = ego_primitive_actions[step_index]
        raw_partner_action = partner_primitive_actions[step_index]
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (raw_ego_action, raw_partner_action)
        ):
            raise TypeError("Exact-history primitive action ids must be integers.")
        ego_action = int(raw_ego_action)
        partner_action = int(raw_partner_action)
        if (
            ego_action < 0
            or partner_action < 0
            or ego_action >= 2**63
            or partner_action >= 2**63
        ):
            raise ValueError(
                "Exact-history primitive action ids must be non-negative signed 64-bit values."
            )
        output.extend(struct.pack(">q", ego_action))
        output.extend(struct.pack(">q", partner_action))
    canonical = bytes(output)
    return ExactHistoryKeyV1(
        sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_bytes=canonical,
    )


def verify_exact_history_match(left: ExactHistoryKeyV1, right: ExactHistoryKeyV1) -> bool:
    """Check the hash first and then verify the complete canonical bytes."""

    return verify_history_digest_and_bytes(
        left.sha256,
        left.canonical_bytes,
        right.sha256,
        right.canonical_bytes,
    )


def verify_history_digest_and_bytes(
    left_digest: str,
    left_bytes: bytes,
    right_digest: str,
    right_bytes: bytes,
) -> bool:
    if left_digest != right_digest:
        return False
    if not hmac.compare_digest(left_bytes, right_bytes):
        raise HistoryHashCollisionError(
            "Exact-history SHA-256 collision or corrupted index: digest matches but bytes differ."
        )
    return True


@dataclass(frozen=True)
class Tier2HistoryCandidateV1:
    episode_uid: str
    decision_index: int
    history_key: ExactHistoryKeyV1
    snapshot_reference: str
    hidden_state: FullHiddenStateV1
    harvest_policy_id: str
    inclusion_probability: float

    def __post_init__(self) -> None:
        if not str(self.episode_uid).strip() or not str(self.snapshot_reference).strip():
            raise ValueError("Tier-2 episode_uid and snapshot_reference are required.")
        if not str(self.snapshot_reference).startswith("sha256:") or not _is_sha256(
            str(self.snapshot_reference).split(":", 1)[1]
        ):
            raise ValueError("Tier-2 snapshot_reference must be sha256:<digest>.")
        if not str(self.harvest_policy_id).strip():
            raise ValueError("Tier-2 harvest_policy_id is required for invariance telemetry.")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, Integral)
            or int(self.decision_index) < 0
        ):
            raise ValueError("Tier-2 decision_index must be non-negative.")
        if not isinstance(self.history_key, ExactHistoryKeyV1):
            raise TypeError("Tier-2 history_key must be an ExactHistoryKeyV1.")
        if not isinstance(self.hidden_state, FullHiddenStateV1):
            raise TypeError("Tier-2 hidden_state must be a complete hidden state.")
        if isinstance(self.inclusion_probability, bool) or not isinstance(
            self.inclusion_probability,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("Tier-2 inclusion_probability must be numeric.")
        probability = float(self.inclusion_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError(
                "Tier-2 inclusion_probability must be in (0, 1] to satisfy positivity."
            )


@dataclass(frozen=True)
class Tier2HistoryGroupV1:
    history_key: ExactHistoryKeyV1
    members: tuple[Tier2HistoryCandidateV1, ...]
    minimum_inclusion_probability: float
    inverse_probability_effective_sample_size: float

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("Tier-2 history group must be non-empty.")
        episodes = [member.episode_uid for member in self.members]
        if len(episodes) != len(set(episodes)):
            raise ValueError("Tier-2 history group may use each episode at most once.")
        for member in self.members:
            if not verify_exact_history_match(self.history_key, member.history_key):
                raise ValueError("Tier-2 group contains a different exact history.")
        probabilities = np.asarray(
            [member.inclusion_probability for member in self.members],
            dtype=np.float64,
        )
        observed_minimum = float(probabilities.min())
        if not math.isclose(
            float(self.minimum_inclusion_probability),
            observed_minimum,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Tier-2 group minimum inclusion probability is incorrect.")
        inverse_weights = 1.0 / probabilities
        observed_ess = float(
            inverse_weights.sum() ** 2 / np.square(inverse_weights).sum()
        )
        if not math.isclose(
            float(self.inverse_probability_effective_sample_size),
            observed_ess,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Tier-2 group inverse-probability effective size is incorrect.")


@dataclass(frozen=True)
class Tier2HistoryIndexV1:
    groups: tuple[Tier2HistoryGroupV1, ...]
    minimum_inclusion_probability: float
    minimum_unique_episodes: int

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("Tier-2 history index must contain groups.")
        if (
            not math.isfinite(float(self.minimum_inclusion_probability))
            or not 0.0 < float(self.minimum_inclusion_probability) <= 1.0
        ):
            raise ValueError("Tier-2 index positivity floor must be in (0, 1].")
        if (
            isinstance(self.minimum_unique_episodes, bool)
            or not isinstance(self.minimum_unique_episodes, Integral)
            or int(self.minimum_unique_episodes) <= 0
        ):
            raise ValueError("Tier-2 index minimum_unique_episodes must be positive.")
        digests = [group.history_key.sha256 for group in self.groups]
        if len(digests) != len(set(digests)):
            raise ValueError("Tier-2 index repeats an exact-history digest.")
        for group in self.groups:
            if len(group.members) < int(self.minimum_unique_episodes):
                raise ValueError("Tier-2 group is below the independent-episode floor.")
            if group.minimum_inclusion_probability < float(
                self.minimum_inclusion_probability
            ):
                raise ValueError("Tier-2 group violates the index positivity floor.")

    def lookup(self, key: ExactHistoryKeyV1) -> Tier2HistoryGroupV1 | None:
        for group in self.groups:
            if group.history_key.sha256 != key.sha256:
                continue
            if verify_exact_history_match(group.history_key, key):
                return group
        return None


def build_tier2_exact_history_index(
    candidates: Sequence[Tier2HistoryCandidateV1],
    *,
    minimum_inclusion_probability: float,
    minimum_unique_episodes: int = 1,
) -> Tier2HistoryIndexV1:
    """Group exact histories while enforcing positivity and episode uniqueness."""

    positivity_floor = float(minimum_inclusion_probability)
    if not math.isfinite(positivity_floor) or not 0.0 < positivity_floor <= 1.0:
        raise ValueError("Tier-2 positivity floor must be in (0, 1].")
    if isinstance(minimum_unique_episodes, bool) or int(minimum_unique_episodes) <= 0:
        raise ValueError("minimum_unique_episodes must be positive.")
    if not candidates:
        raise ValueError("Tier-2 history index requires candidates.")

    grouped: dict[str, list[Tier2HistoryCandidateV1]] = {}
    seen_episode_key: set[tuple[str, str]] = set()
    for candidate in candidates:
        if float(candidate.inclusion_probability) < positivity_floor:
            raise ValueError(
                f"Tier-2 candidate {candidate.episode_uid!r} violates the positivity floor."
            )
        episode_key = (candidate.episode_uid, candidate.history_key.sha256)
        if episode_key in seen_episode_key:
            raise ValueError(
                "Tier-2 index may retain at most one outer draw per episode and history key."
            )
        seen_episode_key.add(episode_key)
        bucket = grouped.setdefault(candidate.history_key.sha256, [])
        if bucket:
            verify_exact_history_match(bucket[0].history_key, candidate.history_key)
        bucket.append(candidate)

    groups = []
    for digest, members_list in sorted(grouped.items()):
        del digest
        members = tuple(
            sorted(members_list, key=lambda item: (item.episode_uid, item.decision_index))
        )
        if len(members) < int(minimum_unique_episodes):
            continue
        probabilities = np.asarray(
            [member.inclusion_probability for member in members],
            dtype=np.float64,
        )
        inverse_weights = 1.0 / probabilities
        effective_sample_size = float(
            inverse_weights.sum() ** 2 / np.square(inverse_weights).sum()
        )
        groups.append(
            Tier2HistoryGroupV1(
                history_key=members[0].history_key,
                members=members,
                minimum_inclusion_probability=float(probabilities.min()),
                inverse_probability_effective_sample_size=effective_sample_size,
            )
        )
    if not groups:
        raise ValueError(
            "No Tier-2 exact-history group meets the minimum unique-episode requirement."
        )
    return Tier2HistoryIndexV1(
        groups=tuple(groups),
        minimum_inclusion_probability=positivity_floor,
        minimum_unique_episodes=int(minimum_unique_episodes),
    )


@dataclass(frozen=True)
class Tier2MatchedFullStateSampler:
    """Exact empirical posterior sampler for an equal-propensity history group.

    Unequal inclusion probabilities require a separately registered weighted
    estimator. They are rejected here so an unweighted sample cannot silently be
    called an exact posterior draw.
    """

    group: Tier2HistoryGroupV1

    def __post_init__(self) -> None:
        probabilities = [member.inclusion_probability for member in self.group.members]
        if not all(
            math.isclose(value, probabilities[0], rel_tol=0.0, abs_tol=1.0e-12)
            for value in probabilities[1:]
        ):
            raise ValueError(
                "Tier-2 exact unweighted sampling requires equal inclusion probabilities."
            )

    def sample(
        self,
        audit_unit_id: str,
        n: int,
        seed: int,
        *,
        outer_id_start: int = 0,
    ) -> tuple[OuterReplicaDrawV1, ...]:
        if not str(audit_unit_id).strip():
            raise ValueError("audit_unit_id must be non-empty.")
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("Tier-2 posterior sampler n must be positive.")
        if (
            isinstance(outer_id_start, bool)
            or not isinstance(outer_id_start, Integral)
            or int(outer_id_start) < 0
        ):
            raise ValueError("outer_id_start must be non-negative.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError("Tier-2 sampler seed must be a non-negative integer.")
        stop = int(outer_id_start) + int(n)
        if stop > len(self.group.members):
            raise ValueError(
                "Tier-2 sampling without replacement cannot draw beyond the "
                "registered unique-episode support."
            )
        permutation_seed = int(
            canonical_sha256(
                {
                    "sampler": "tier2_exact_history_without_replacement_v1",
                    "base_seed": int(seed),
                    "audit_unit_id": str(audit_unit_id),
                    "history_sha256": self.group.history_key.sha256,
                }
            )[:16],
            16,
        )
        rng = np.random.default_rng(permutation_seed)
        selected = rng.permutation(len(self.group.members))[
            int(outer_id_start) : stop
        ]
        probability = 1.0 / float(len(self.group.members))
        return tuple(
            OuterReplicaDrawV1(
                audit_unit_id=str(audit_unit_id),
                outer_id=int(outer_id_start) + offset,
                hidden_state=self.group.members[int(index)].hidden_state,
                posterior_draw_probability=probability,
                sampler_seed=permutation_seed,
                posterior_mode="exact",
                rho_prune=0.0,
                source_episode_uid=self.group.members[int(index)].episode_uid,
            )
            for offset, index in enumerate(selected.tolist())
        )


@dataclass(frozen=True)
class PosteriorSupportEntryV1:
    """One frozen full-state support point for mechanical sampler verification."""

    theta_id: str
    execution_state_key: str
    execution_state_bytes_hex: str
    hidden_state_sha256: str
    posterior_probability: float

    def __post_init__(self) -> None:
        _strict_text(self.theta_id, "Posterior support theta_id")
        _strict_text(
            self.execution_state_key,
            "Posterior support execution_state_key",
        )
        encoded_state = _strict_text(
            self.execution_state_bytes_hex,
            "Posterior support execution_state_bytes_hex",
        )
        try:
            state_bytes = bytes.fromhex(encoded_state)
        except ValueError as error:
            raise ValueError(
                "Posterior support execution state must use canonical hexadecimal bytes."
            ) from error
        if not state_bytes or state_bytes.hex() != encoded_state:
            raise ValueError(
                "Posterior support execution-state hexadecimal encoding is not canonical."
            )
        if not _is_sha256(self.hidden_state_sha256):
            raise ValueError("Posterior support hidden-state hash must be SHA-256.")
        expected_hidden_state_sha256 = FullHiddenStateV1(
            theta_id=self.theta_id,
            execution_state_key=self.execution_state_key,
            execution_state_bytes=state_bytes,
        ).sha256
        if self.hidden_state_sha256 != expected_hidden_state_sha256:
            raise ValueError(
                "Posterior support hidden-state hash does not match its complete state bytes."
            )
        probability = float(self.posterior_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Posterior support probability must lie in (0, 1].")

    @classmethod
    def from_component(
        cls,
        component: PosteriorComponentV1,
    ) -> "PosteriorSupportEntryV1":
        if not isinstance(component, PosteriorComponentV1):
            raise TypeError("Posterior support requires PosteriorComponentV1 values.")
        return cls(
            theta_id=component.hidden_state.theta_id,
            execution_state_key=component.hidden_state.execution_state_key,
            execution_state_bytes_hex=(
                component.hidden_state.execution_state_bytes.hex()
            ),
            hidden_state_sha256=component.hidden_state.sha256,
            posterior_probability=float(component.posterior_probability),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "PosteriorSupportEntryV1":
        item = _strict_mapping(payload, "posterior support entry")
        _require_exact_keys(
            item,
            {
                "theta_id",
                "execution_state_key",
                "execution_state_bytes_hex",
                "hidden_state_sha256",
                "posterior_probability",
            },
            "posterior support entry",
        )
        return cls(
            theta_id=_strict_text(item["theta_id"], "posterior support theta_id"),
            execution_state_key=_strict_text(
                item["execution_state_key"],
                "posterior support execution_state_key",
            ),
            execution_state_bytes_hex=_strict_text(
                item["execution_state_bytes_hex"],
                "posterior support execution_state_bytes_hex",
            ),
            hidden_state_sha256=str(item["hidden_state_sha256"]),
            posterior_probability=_strict_finite_float(
                item["posterior_probability"],
                "posterior support probability",
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "theta_id": self.theta_id,
            "execution_state_key": self.execution_state_key,
            "execution_state_bytes_hex": self.execution_state_bytes_hex,
            "hidden_state_sha256": self.hidden_state_sha256,
            "posterior_probability": float(self.posterior_probability),
        }


@dataclass(frozen=True)
class FrozenInstrumentProbeV1:
    """Probe identity and sampling probability bound by the frozen battery."""

    probe_id: str
    sampling_probability: float

    def __post_init__(self) -> None:
        _strict_text(self.probe_id, "Frozen instrument probe_id")
        probability = float(self.sampling_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Frozen probe sampling probability must lie in (0, 1].")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "FrozenInstrumentProbeV1":
        item = _strict_mapping(payload, "frozen instrument probe")
        _require_exact_keys(
            item,
            {"probe_id", "sampling_probability"},
            "frozen instrument probe",
        )
        return cls(
            probe_id=_strict_text(item["probe_id"], "frozen instrument probe_id"),
            sampling_probability=_strict_finite_float(
                item["sampling_probability"],
                "frozen instrument probe sampling_probability",
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "sampling_probability": float(self.sampling_probability),
        }


@dataclass(frozen=True)
class InstrumentKernelCellV1:
    """One registered instrument cell with complete embedded outer records."""

    cell_id: str
    value_class_id: str
    audit_unit_id: str
    snapshot_sha256: str
    probe_id: str
    sampler_base_seed: int
    outer_id_start: int
    posterior_support: tuple[PosteriorSupportEntryV1, ...]
    kernel_records: tuple[OuterReplicaKernelRecordV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "posterior_support", tuple(self.posterior_support))
        object.__setattr__(self, "kernel_records", tuple(self.kernel_records))
        for name in ("cell_id", "value_class_id", "audit_unit_id", "probe_id"):
            _strict_text(getattr(self, name), f"Instrument cell {name}")
        if not _is_sha256(self.snapshot_sha256):
            raise ValueError("Instrument cell snapshot_sha256 must be SHA-256.")
        _strict_nonbool_int(
            self.sampler_base_seed,
            "Instrument cell sampler_base_seed",
            minimum=0,
        )
        _strict_nonbool_int(
            self.outer_id_start,
            "Instrument cell outer_id_start",
            minimum=0,
        )
        if not self.posterior_support or not self.kernel_records:
            raise ValueError(
                "Instrument cells need posterior support and embedded kernel records."
            )
        if any(
            not isinstance(item, PosteriorSupportEntryV1)
            for item in self.posterior_support
        ):
            raise TypeError("Instrument posterior support entries have the wrong type.")
        if any(
            not isinstance(item, OuterReplicaKernelRecordV1)
            for item in self.kernel_records
        ):
            raise TypeError("Instrument kernel records have the wrong type.")
        support_keys = [
            (
                item.theta_id,
                item.execution_state_key,
                item.hidden_state_sha256,
            )
            for item in self.posterior_support
        ]
        if len(support_keys) != len(set(support_keys)):
            raise ValueError("Instrument posterior support repeats a full hidden state.")
        if tuple(support_keys) != tuple(sorted(support_keys)):
            raise ValueError("Instrument posterior support must be canonically sorted.")
        if not math.isclose(
            math.fsum(item.posterior_probability for item in self.posterior_support),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Instrument posterior support probabilities must sum to one.")

    @classmethod
    def from_posterior(
        cls,
        *,
        cell_id: str,
        value_class_id: str,
        audit_unit_id: str,
        snapshot_sha256: str,
        probe_id: str,
        sampler_base_seed: int,
        outer_id_start: int,
        posterior: SparseForwardResultV1,
        kernel_records: Sequence[OuterReplicaKernelRecordV1],
    ) -> "InstrumentKernelCellV1":
        support = tuple(sorted(
            (
                PosteriorSupportEntryV1.from_component(component)
                for component in posterior.components
            ),
            key=lambda item: (
                item.theta_id,
                item.execution_state_key,
                item.hidden_state_sha256,
            ),
        ))
        return cls(
            cell_id=str(cell_id),
            value_class_id=str(value_class_id),
            audit_unit_id=str(audit_unit_id),
            snapshot_sha256=str(snapshot_sha256),
            probe_id=str(probe_id),
            sampler_base_seed=sampler_base_seed,
            outer_id_start=outer_id_start,
            posterior_support=support,
            kernel_records=tuple(kernel_records),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "InstrumentKernelCellV1":
        item = _strict_mapping(payload, "instrument kernel cell")
        _require_exact_keys(
            item,
            {
                "cell_id",
                "value_class_id",
                "audit_unit_id",
                "snapshot_sha256",
                "probe_id",
                "sampler_base_seed",
                "outer_id_start",
                "posterior_support",
                "kernel_records",
            },
            "instrument kernel cell",
        )
        return cls(
            cell_id=_strict_text(item["cell_id"], "instrument cell_id"),
            value_class_id=_strict_text(
                item["value_class_id"],
                "instrument value_class_id",
            ),
            audit_unit_id=_strict_text(
                item["audit_unit_id"],
                "instrument audit_unit_id",
            ),
            snapshot_sha256=str(item["snapshot_sha256"]),
            probe_id=_strict_text(item["probe_id"], "instrument probe_id"),
            sampler_base_seed=_strict_nonbool_int(
                item["sampler_base_seed"],
                "instrument sampler_base_seed",
                minimum=0,
            ),
            outer_id_start=_strict_nonbool_int(
                item["outer_id_start"],
                "instrument outer_id_start",
                minimum=0,
            ),
            posterior_support=tuple(
                PosteriorSupportEntryV1.from_payload(raw_support)
                for raw_support in _strict_sequence(
                    item["posterior_support"],
                    "instrument posterior_support",
                )
            ),
            kernel_records=tuple(
                OuterReplicaKernelRecordV1.from_payload(raw_record)
                for raw_record in _strict_sequence(
                    item["kernel_records"],
                    "instrument kernel_records",
                )
            ),
        )

    def registry_payload(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "value_class_id": self.value_class_id,
            "audit_unit_id": self.audit_unit_id,
            "snapshot_sha256": self.snapshot_sha256,
            "probe_id": self.probe_id,
            "sampler_base_seed": int(self.sampler_base_seed),
            "outer_id_start": int(self.outer_id_start),
            "outer_replicas_M": len(self.kernel_records),
        }

    def to_payload(self) -> dict[str, Any]:
        registry = self.registry_payload()
        registry.pop("outer_replicas_M")
        return {
            **registry,
            "posterior_support": [
                item.to_payload() for item in self.posterior_support
            ],
            "kernel_records": [
                record.to_payload() for record in self.kernel_records
            ],
        }


@dataclass(frozen=True)
class FiniteCategoricalCellV1:
    cell_id: str
    value_class_id: str
    probe_id: str
    summary_id: str
    counts: tuple[int, ...]
    probabilities: tuple[float, ...]
    outer_replicas_M: int
    vocabulary_size_q: int
    simultaneous_cell_count: int
    confidence_delta: float
    sampling_radius: float
    posterior_bias_bound: float
    reset_bias_bound: float

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (self.cell_id, self.value_class_id, self.probe_id, self.summary_id)
        ):
            raise ValueError("Finite categorical cell identifiers must be non-empty.")
        if len(self.counts) != self.vocabulary_size_q:
            raise ValueError("Categorical cell count vector does not match frozen q.")
        if len(self.probabilities) != self.vocabulary_size_q:
            raise ValueError("Categorical cell probability vector does not match frozen q.")
        if any(isinstance(value, bool) or int(value) < 0 for value in self.counts):
            raise ValueError("Categorical cell counts must be non-negative integers.")
        if sum(self.counts) != int(self.outer_replicas_M):
            raise ValueError("Categorical counts must contain one token per outer replica.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.probabilities):
            raise ValueError("Categorical cell probabilities must be finite and non-negative.")
        if not math.isclose(
            math.fsum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("Categorical cell probabilities must sum to one.")
        for name, value in (
            ("sampling_radius", self.sampling_radius),
            ("posterior_bias_bound", self.posterior_bias_bound),
            ("reset_bias_bound", self.reset_bias_bound),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Categorical cell {name} must be in [0, 1].")

    @property
    def total_error_radius(self) -> float:
        return float(
            self.sampling_radius + self.posterior_bias_bound + self.reset_bias_bound
        )


def estimate_finite_categorical_cell(
    *,
    cell_id: str,
    value_class_id: str,
    probe_id: str,
    summary_id: str,
    response_token_ids: Sequence[int],
    vocabulary_size_q: int,
    simultaneous_cell_count: int,
    confidence_delta: float,
    posterior_bias_bound: float = 0.0,
    reset_bias_bound: float = 0.0,
) -> FiniteCategoricalCellV1:
    """Estimate one cell with the registered finite categorical vocabulary."""

    if isinstance(vocabulary_size_q, bool) or not isinstance(
        vocabulary_size_q, Integral
    ):
        raise TypeError("vocabulary_size_q must be an integer.")
    if isinstance(simultaneous_cell_count, bool) or not isinstance(
        simultaneous_cell_count, Integral
    ):
        raise TypeError("simultaneous_cell_count must be an integer.")
    if isinstance(confidence_delta, bool) or not isinstance(
        confidence_delta, (int, float, np.integer, np.floating)
    ):
        raise TypeError("confidence_delta must be numeric.")
    q = int(vocabulary_size_q)
    total_cells = int(simultaneous_cell_count)
    delta = float(confidence_delta)
    rho = float(posterior_bias_bound)
    xi = float(reset_bias_bound)
    if q <= 1:
        raise ValueError("Frozen categorical vocabulary size q must exceed one.")
    if total_cells <= 0:
        raise ValueError("simultaneous_cell_count must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("confidence_delta must be in (0, 1).")
    for name, value in (("posterior_bias_bound", rho), ("reset_bias_bound", xi)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    raw_tokens = tuple(response_token_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw_tokens
    ):
        raise TypeError("Cell response tokens must be integers.")
    tokens = np.asarray(raw_tokens, dtype=np.int64)
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("Cell response tokens must be a non-empty vector.")
    if np.any(tokens < 0) or np.any(tokens >= q):
        raise ValueError("Cell response token lies outside the frozen vocabulary.")
    counts_array = np.bincount(tokens, minlength=q)
    M = int(tokens.size)
    radius = math.sqrt(
        (q * math.log(2.0) + math.log(total_cells / delta)) / (2.0 * M)
    )
    radius = min(1.0, float(radius))
    probabilities = counts_array.astype(np.float64) / float(M)
    return FiniteCategoricalCellV1(
        cell_id=str(cell_id),
        value_class_id=str(value_class_id),
        probe_id=str(probe_id),
        summary_id=str(summary_id),
        counts=tuple(int(value) for value in counts_array.tolist()),
        probabilities=tuple(float(value) for value in probabilities.tolist()),
        outer_replicas_M=M,
        vocabulary_size_q=q,
        simultaneous_cell_count=total_cells,
        confidence_delta=delta,
        sampling_radius=radius,
        posterior_bias_bound=rho,
        reset_bias_bound=xi,
    )


@dataclass(frozen=True)
class CellDistanceBoundV1:
    left_cell_id: str
    right_cell_id: str
    relation: Literal["within_class", "between_class"]
    observed_total_variation: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class AlignmentBoundsV1:
    alpha_upper: float
    beta_lower: float
    instrument_valid: bool
    within_class_bounds: tuple[CellDistanceBoundV1, ...]
    between_class_bounds: tuple[CellDistanceBoundV1, ...]
    maximum_sampling_radius: float
    maximum_posterior_bias_bound: float
    maximum_reset_bias_bound: float


def compute_alignment_bounds(
    cells: Sequence[FiniteCategoricalCellV1],
) -> AlignmentBoundsV1:
    """Compute ``alpha_upper=max UCB`` and ``beta_lower=min LCB``.

    All cells must belong to one frozen probe and summary family. Each pair receives
    its own uncertainty radius before the maximum or minimum is taken, avoiding a
    biased plug-in maximum/minimum decision.
    """

    if len(cells) < 3:
        raise ValueError("Alignment bounds require at least three categorical cells.")
    q_values = {cell.vocabulary_size_q for cell in cells}
    families = {(cell.probe_id, cell.summary_id) for cell in cells}
    if len(q_values) != 1:
        raise ValueError("Alignment cells use different frozen response vocabularies.")
    if len(families) != 1:
        raise ValueError("Compute alignment bounds separately for each probe and summary.")

    within: list[CellDistanceBoundV1] = []
    between: list[CellDistanceBoundV1] = []
    ordered = sorted(cells, key=lambda item: item.cell_id)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            observed = categorical_total_variation(left.probabilities, right.probabilities)
            pair_radius = left.total_error_radius + right.total_error_radius
            bound = CellDistanceBoundV1(
                left_cell_id=left.cell_id,
                right_cell_id=right.cell_id,
                relation=(
                    "within_class"
                    if left.value_class_id == right.value_class_id
                    else "between_class"
                ),
                observed_total_variation=observed,
                lower_bound=max(0.0, observed - pair_radius),
                upper_bound=min(1.0, observed + pair_radius),
            )
            (within if bound.relation == "within_class" else between).append(bound)
    if not within:
        raise ValueError("Alignment bounds require at least one within-class cell pair.")
    if not between:
        raise ValueError("Alignment bounds require at least one between-class cell pair.")
    alpha_upper = max(item.upper_bound for item in within)
    beta_lower = min(item.lower_bound for item in between)
    return AlignmentBoundsV1(
        alpha_upper=float(alpha_upper),
        beta_lower=float(beta_lower),
        instrument_valid=bool(beta_lower > alpha_upper),
        within_class_bounds=tuple(within),
        between_class_bounds=tuple(between),
        maximum_sampling_radius=max(cell.sampling_radius for cell in cells),
        maximum_posterior_bias_bound=max(cell.posterior_bias_bound for cell in cells),
        maximum_reset_bias_bound=max(cell.reset_bias_bound for cell in cells),
    )


def categorical_total_variation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Total variation distance under the one-half L1 convention."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1 or left_array.shape != right_array.shape:
        raise ValueError("Categorical probability vectors must be aligned one-dimensional arrays.")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("Categorical probability vectors must be finite.")
    if np.any(left_array < 0.0) or np.any(right_array < 0.0):
        raise ValueError("Categorical probabilities cannot be negative.")
    if not math.isclose(float(left_array.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("Left categorical probabilities must sum to one.")
    if not math.isclose(float(right_array.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("Right categorical probabilities must sum to one.")
    return float(0.5 * np.abs(left_array - right_array).sum())


@dataclass(frozen=True)
class SyntheticOuterSamplingDiagnosticV1:
    """Oracle-defined diagnostic for the full-hidden-state outer sampler.

    The point-mass arm is selected from the frozen synthetic response table, not
    by the implementation being tested.  The naive arm is summarized by its
    analytic expected discrepancy and failure probability, so a lucky state draw
    cannot turn software conformance into a random pass/fail event.
    """

    target_distribution: tuple[float, ...]
    oracle_point_mass_index: int
    oracle_point_mass_tv: float
    expected_naive_single_draw_tv: float
    naive_failure_probability: float
    failure_threshold: float
    outer_replicas_M: int
    repetitions: int
    full_u_mean_empirical_tv: float
    full_u_tv_quantile_95: float


def diagnose_synthetic_outer_sampling(
    posterior_weights: Sequence[float],
    response_probabilities_by_full_state: Sequence[Sequence[float]],
    *,
    outer_replicas_M: int,
    repetitions: int,
    failure_threshold: float,
    seed: int,
) -> SyntheticOuterSamplingDiagnosticV1:
    """Compare theorem-aligned full-state draws with the single-state shortcut."""

    weights = np.asarray(posterior_weights, dtype=np.float64)
    conditional = np.asarray(
        response_probabilities_by_full_state,
        dtype=np.float64,
    )
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("Synthetic posterior must contain at least two full states.")
    if conditional.ndim != 2 or conditional.shape[0] != weights.size:
        raise ValueError(
            "Synthetic response table must have shape [full_state, response_token]."
        )
    if conditional.shape[1] <= 1:
        raise ValueError("Synthetic response vocabulary must contain at least two tokens.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Synthetic posterior weights must be finite and non-negative.")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Synthetic posterior weights must sum to one.")
    if not np.all(np.isfinite(conditional)) or np.any(conditional < 0.0):
        raise ValueError("Synthetic conditional responses must be finite and non-negative.")
    if not np.allclose(conditional.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Every synthetic conditional response row must sum to one.")
    M = int(outer_replicas_M)
    repeats = int(repetitions)
    threshold = float(failure_threshold)
    if isinstance(outer_replicas_M, bool) or M <= 0:
        raise ValueError("outer_replicas_M must be positive.")
    if isinstance(repetitions, bool) or repeats <= 0:
        raise ValueError("repetitions must be positive.")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("failure_threshold must be finite and non-negative.")

    target = weights @ conditional
    state_tv = 0.5 * np.abs(conditional - target[None, :]).sum(axis=1)
    oracle_index = int(np.argmax(state_tv))
    expected_naive_tv = float(np.dot(weights, state_tv))
    failure_probability = float(weights[state_tv > threshold].sum())

    rng = np.random.default_rng(int(seed))
    empirical_tv = np.empty(repeats, dtype=np.float64)
    token_ids = np.arange(conditional.shape[1], dtype=np.int64)
    for repeat in range(repeats):
        state_ids = rng.choice(weights.size, size=M, replace=True, p=weights)
        counts = np.zeros(conditional.shape[1], dtype=np.int64)
        for state_id in state_ids:
            token_id = int(rng.choice(token_ids, p=conditional[int(state_id)]))
            counts[token_id] += 1
        empirical = counts.astype(np.float64) / float(M)
        empirical_tv[repeat] = 0.5 * np.abs(empirical - target).sum()

    return SyntheticOuterSamplingDiagnosticV1(
        target_distribution=tuple(float(value) for value in target.tolist()),
        oracle_point_mass_index=oracle_index,
        oracle_point_mass_tv=float(state_tv[oracle_index]),
        expected_naive_single_draw_tv=expected_naive_tv,
        naive_failure_probability=failure_probability,
        failure_threshold=threshold,
        outer_replicas_M=M,
        repetitions=repeats,
        full_u_mean_empirical_tv=float(np.mean(empirical_tv, dtype=np.float64)),
        full_u_tv_quantile_95=float(np.quantile(empirical_tv, 0.95)),
    )


@dataclass(frozen=True)
class TierAgreementEquivalenceV1:
    difference_estimate: float
    difference_confidence_interval: tuple[float, float]
    equivalence_bound: float
    equivalent: bool


def assess_tier_agreement_equivalence(
    difference_estimate: float,
    difference_confidence_interval: Sequence[float],
    *,
    equivalence_bound: float,
) -> TierAgreementEquivalenceV1:
    """Require the complete tier-difference interval inside a frozen bound."""

    estimate = float(difference_estimate)
    interval = tuple(float(value) for value in difference_confidence_interval)
    bound = float(equivalence_bound)
    if len(interval) != 2:
        raise ValueError("Tier agreement confidence interval must have two endpoints.")
    if not all(math.isfinite(value) for value in (estimate, *interval, bound)):
        raise ValueError("Tier agreement inputs must be finite.")
    if interval[1] < interval[0]:
        raise ValueError("Tier agreement confidence interval is reversed.")
    if bound <= 0.0:
        raise ValueError("Tier agreement equivalence_bound must be positive.")
    return TierAgreementEquivalenceV1(
        difference_estimate=estimate,
        difference_confidence_interval=(interval[0], interval[1]),
        equivalence_bound=bound,
        equivalent=bool(interval[0] > -bound and interval[1] < bound),
    )


def build_instrument_measurement_v1(
    alignment: AlignmentBoundsV1,
    *,
    exact_or_approximate: Literal["exact", "approximate_bounded"],
    rho_prune: float,
    minimum_outer_replicas: int,
    response_vocabulary_sha256: str,
) -> dict[str, Any]:
    """Build the numeric summary; evidence checks are added only by recomputation."""

    if exact_or_approximate not in {"exact", "approximate_bounded"}:
        raise ValueError("Unknown instrument approximation status.")
    rho = float(rho_prune)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho_prune must be in [0, 1].")
    if exact_or_approximate == "exact" and rho != 0.0:
        raise ValueError("Exact instrument measurements cannot report pruning.")
    if (
        exact_or_approximate == "approximate_bounded"
        and alignment.maximum_posterior_bias_bound < rho
    ):
        raise ValueError("Posterior bias bound must cover rho_prune.")
    if isinstance(minimum_outer_replicas, bool) or int(minimum_outer_replicas) <= 0:
        raise ValueError("minimum_outer_replicas must be positive.")
    if not _is_sha256(response_vocabulary_sha256):
        raise ValueError("response_vocabulary_sha256 must be a SHA-256 digest.")
    return {
        "schema_version": "path_c_instrument_measurement_v1",
        "alpha_upper": float(alignment.alpha_upper),
        "beta_lower": float(alignment.beta_lower),
        "sampling_radius": float(alignment.maximum_sampling_radius),
        "rho_prune": rho,
        "posterior_bias_bound": float(alignment.maximum_posterior_bias_bound),
        "reset_bias_bound": float(alignment.maximum_reset_bias_bound),
        "exact_or_approximate": str(exact_or_approximate),
        "minimum_outer_replicas": int(minimum_outer_replicas),
        "response_vocabulary_sha256": str(response_vocabulary_sha256),
        "distance_convention": "total_variation_half_l1",
        "outer_sampling_unit": "independent_full_hidden_state_posterior_draw",
        "inner_forks_are_outer_samples": False,
    }


def build_instrument_evidence_from_kernel_records_v2(
    cells: Sequence[InstrumentKernelCellV1],
    *,
    response_vocabulary_sha256: str,
    vocabulary_size_q: int,
    confidence_delta: float,
    exact_or_approximate: Literal["exact", "approximate_bounded"],
    rho_prune: float,
    posterior_bias_bound: float,
    reset_bias_bound: float,
    battery_sha256: str,
    frozen_probes: Sequence[FrozenInstrumentProbeV1],
    summary_id: str,
    inner_forks_L_inner: int,
    support_violation_token_id: int,
    frozen_cell_registry_sha256: str,
) -> dict[str, Any]:
    """Build version-2 evidence only from complete embedded kernel records."""

    raw_cells = tuple(cells)
    if not raw_cells or any(
        not isinstance(item, InstrumentKernelCellV1) for item in raw_cells
    ):
        raise TypeError("Instrument evidence cells must be InstrumentKernelCellV1 values.")
    cell_tuple = tuple(sorted(raw_cells, key=lambda item: item.cell_id))
    probe_tuple = tuple(frozen_probes)
    if not probe_tuple or any(
        not isinstance(item, FrozenInstrumentProbeV1) for item in probe_tuple
    ):
        raise TypeError("Instrument evidence probes must be FrozenInstrumentProbeV1 values.")
    q = _strict_nonbool_int(
        vocabulary_size_q,
        "Instrument evidence vocabulary_size_q",
        minimum=2,
    )
    delta = _strict_finite_float(
        confidence_delta,
        "Instrument evidence confidence_delta",
    )
    if not 0.0 < delta < 1.0:
        raise ValueError("Instrument evidence confidence_delta must be in (0, 1).")
    mode = _strict_text(
        exact_or_approximate,
        "Instrument evidence exact_or_approximate",
    )
    rho = _strict_finite_float(rho_prune, "Instrument evidence rho_prune")
    posterior_bias = _strict_finite_float(
        posterior_bias_bound,
        "Instrument evidence posterior_bias_bound",
    )
    reset_bias = _strict_finite_float(
        reset_bias_bound,
        "Instrument evidence reset_bias_bound",
    )
    inner_forks = _strict_nonbool_int(
        inner_forks_L_inner,
        "Instrument evidence inner_forks_L_inner",
        minimum=1,
    )
    support_token = _strict_nonbool_int(
        support_violation_token_id,
        "Instrument evidence support_violation_token_id",
        minimum=0,
    )
    if support_token >= q:
        raise ValueError("Support-violation token must lie in the frozen vocabulary.")
    if not _is_sha256(frozen_cell_registry_sha256):
        raise ValueError("Frozen instrument cell registry must be a SHA-256 digest.")
    _strict_text(summary_id, "Instrument evidence summary_id")
    if not _is_sha256(response_vocabulary_sha256):
        raise ValueError("Instrument evidence must bind a response-vocabulary SHA-256.")
    if not _is_sha256(battery_sha256):
        raise ValueError("Instrument evidence must bind a frozen battery SHA-256.")
    registry_payload = _instrument_cell_registry_payload(
        cells=cell_tuple,
        response_vocabulary_sha256=response_vocabulary_sha256,
        battery_sha256=battery_sha256,
        frozen_probes=probe_tuple,
        summary_id=summary_id,
        inner_forks_L_inner=inner_forks,
        support_violation_token_id=support_token,
    )
    if canonical_sha256(registry_payload) != frozen_cell_registry_sha256:
        raise ValueError(
            "Instrument cells do not match the cell registry frozen before evidence collection."
        )
    evidence = {
        "schema_version": INSTRUMENT_EVIDENCE_SCHEMA_VERSION,
        "response_vocabulary_sha256": str(response_vocabulary_sha256),
        "vocabulary_size_q": q,
        "simultaneous_cell_count": len(cell_tuple),
        "confidence_delta": delta,
        "exact_or_approximate": mode,
        "rho_prune": rho,
        "posterior_bias_bound": posterior_bias,
        "reset_bias_bound": reset_bias,
        "sampling_design": INSTRUMENT_SAMPLING_DESIGN,
        "sampling_source": INSTRUMENT_SAMPLING_SOURCE,
        "battery_sha256": str(battery_sha256),
        "frozen_probes": [item.to_payload() for item in probe_tuple],
        "summary_id": str(summary_id),
        "inner_forks_L_inner": inner_forks,
        "support_violation_token_id": support_token,
        "cell_registry_sha256": str(frozen_cell_registry_sha256),
        "kernel_records_sha256": canonical_sha256([
            record.to_payload()
            for cell in cell_tuple
            for record in cell.kernel_records
        ]),
        "cells": [item.to_payload() for item in cell_tuple],
    }
    recompute_instrument_measurement_from_evidence_v2(evidence)
    return evidence


def recompute_instrument_measurement_from_evidence_v2(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the instrument from full version-2 outer-replica records.

    Version-1 evidence containing self-reported token arrays or validity booleans is
    deliberately rejected.  Every theorem-level token is extracted from the first
    inner response of the selected frozen probe in a strict
    :class:`OuterReplicaKernelRecordV1`.  The categorical support point and sampler
    seed are independently reconstructed from the frozen base seed and outer id.
    """

    expected_keys = {
        "schema_version",
        "response_vocabulary_sha256",
        "vocabulary_size_q",
        "simultaneous_cell_count",
        "confidence_delta",
        "exact_or_approximate",
        "rho_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "sampling_design",
        "sampling_source",
        "battery_sha256",
        "frozen_probes",
        "summary_id",
        "inner_forks_L_inner",
        "support_violation_token_id",
        "cell_registry_sha256",
        "kernel_records_sha256",
        "cells",
    }
    evidence_mapping = _strict_mapping(evidence, "instrument evidence")
    _require_exact_keys(evidence_mapping, expected_keys, "instrument evidence")
    if evidence_mapping["schema_version"] != INSTRUMENT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "Instrument evidence must use path_c_instrument_evidence_v2; "
            "self-reported version-1 token tables are unsupported."
        )
    response_vocabulary_sha256 = str(
        evidence_mapping["response_vocabulary_sha256"]
    )
    battery_sha256 = str(evidence_mapping["battery_sha256"])
    if not _is_sha256(response_vocabulary_sha256):
        raise ValueError("Instrument evidence must bind a response-vocabulary SHA-256.")
    if not _is_sha256(battery_sha256):
        raise ValueError("Instrument evidence must bind a frozen battery SHA-256.")
    if evidence_mapping["sampling_design"] != INSTRUMENT_SAMPLING_DESIGN:
        raise ValueError(
            "Primary instrument requires independent full-posterior draws with replacement."
        )
    if evidence_mapping["sampling_source"] != INSTRUMENT_SAMPLING_SOURCE:
        raise ValueError(
            "Primary instrument rejects matched-history and other non-posterior sources."
        )
    q = _strict_nonbool_int(
        evidence_mapping["vocabulary_size_q"],
        "Instrument evidence vocabulary_size_q",
        minimum=2,
    )
    simultaneous_cell_count = _strict_nonbool_int(
        evidence_mapping["simultaneous_cell_count"],
        "Instrument evidence simultaneous_cell_count",
        minimum=3,
    )
    confidence_delta = _strict_finite_float(
        evidence_mapping["confidence_delta"],
        "Instrument evidence confidence_delta",
    )
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("Instrument evidence confidence_delta must be in (0, 1).")
    inner_forks = _strict_nonbool_int(
        evidence_mapping["inner_forks_L_inner"],
        "Instrument evidence inner_forks_L_inner",
        minimum=1,
    )
    support_violation_token_id = _strict_nonbool_int(
        evidence_mapping["support_violation_token_id"],
        "Instrument evidence support_violation_token_id",
        minimum=0,
    )
    if support_violation_token_id >= q:
        raise ValueError("Support-violation token lies outside the frozen vocabulary.")
    summary_id = _strict_text(
        evidence_mapping["summary_id"],
        "Instrument evidence summary_id",
    )
    mode = _strict_text(
        evidence_mapping["exact_or_approximate"],
        "Instrument evidence exact_or_approximate",
    )
    if mode not in {"exact", "approximate_bounded"}:
        raise ValueError("Instrument evidence approximation mode is unsupported.")
    rho_prune = _strict_finite_float(
        evidence_mapping["rho_prune"],
        "Instrument evidence rho_prune",
    )
    posterior_bias_bound = _strict_finite_float(
        evidence_mapping["posterior_bias_bound"],
        "Instrument evidence posterior_bias_bound",
    )
    reset_bias_bound = _strict_finite_float(
        evidence_mapping["reset_bias_bound"],
        "Instrument evidence reset_bias_bound",
    )
    for name, value in (
        ("rho_prune", rho_prune),
        ("posterior_bias_bound", posterior_bias_bound),
        ("reset_bias_bound", reset_bias_bound),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Instrument evidence {name} must be in [0, 1].")
    if mode == "exact" and any(
        value != 0.0
        for value in (rho_prune, posterior_bias_bound, reset_bias_bound)
    ):
        raise ValueError("Exact instrument evidence cannot report approximation bias.")
    if mode == "approximate_bounded" and posterior_bias_bound < rho_prune:
        raise ValueError("Instrument posterior-bias bound must cover rho_prune.")

    frozen_probes = tuple(
        FrozenInstrumentProbeV1.from_payload(raw_probe)
        for raw_probe in _strict_sequence(
            evidence_mapping["frozen_probes"],
            "Instrument evidence frozen_probes",
        )
    )
    if not frozen_probes:
        raise ValueError("Instrument evidence must contain frozen battery probes.")
    frozen_probe_ids = tuple(item.probe_id for item in frozen_probes)
    if len(frozen_probe_ids) != len(set(frozen_probe_ids)):
        raise ValueError("Instrument evidence repeats a frozen probe_id.")
    probe_probability = {
        item.probe_id: float(item.sampling_probability) for item in frozen_probes
    }
    cells = tuple(
        InstrumentKernelCellV1.from_payload(raw_cell)
        for raw_cell in _strict_sequence(
            evidence_mapping["cells"],
            "Instrument evidence cells",
        )
    )
    if len(cells) != simultaneous_cell_count:
        raise ValueError(
            "Instrument evidence cell count differs from simultaneous_cell_count."
        )
    cell_ids = tuple(item.cell_id for item in cells)
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Instrument evidence repeats a cell_id.")
    if cell_ids != tuple(sorted(cell_ids)):
        raise ValueError("Instrument evidence cells must be canonically sorted.")
    audit_units = tuple(item.audit_unit_id for item in cells)
    snapshots = tuple(item.snapshot_sha256 for item in cells)
    if len(audit_units) != len(set(audit_units)):
        raise ValueError("Instrument evidence reuses an audit unit across cells.")
    if len(snapshots) != len(set(snapshots)):
        raise ValueError("Instrument evidence reuses a snapshot across cells.")
    if not _is_sha256(evidence_mapping["cell_registry_sha256"]):
        raise ValueError("Instrument evidence cell_registry_sha256 must be SHA-256.")
    if not _is_sha256(evidence_mapping["kernel_records_sha256"]):
        raise ValueError("Instrument evidence kernel_records_sha256 must be SHA-256.")
    registry_payload = _instrument_cell_registry_payload(
        cells=cells,
        response_vocabulary_sha256=response_vocabulary_sha256,
        battery_sha256=battery_sha256,
        frozen_probes=frozen_probes,
        summary_id=summary_id,
        inner_forks_L_inner=inner_forks,
        support_violation_token_id=support_violation_token_id,
    )
    if evidence_mapping["cell_registry_sha256"] != canonical_sha256(registry_payload):
        raise ValueError("Instrument cell registry SHA-256 does not match its content.")

    kernel_payloads = [
        record.to_payload()
        for cell in cells
        for record in cell.kernel_records
    ]
    for cell in cells:
        for record in cell.kernel_records:
            expected_seed = _categorical_full_state_draw_seed(
                base_seed=cell.sampler_base_seed,
                audit_unit_id=cell.audit_unit_id,
                outer_id=record.outer_id,
            )
            if int(record.sampler_seed) != expected_seed:
                raise ValueError(
                    "Kernel record sampler seed does not match its outer coordinate."
                )
    kernel_hash_valid = (
        evidence_mapping["kernel_records_sha256"]
        == canonical_sha256(kernel_payloads)
    )

    expected_record_mode = "exact" if mode == "exact" else "approximate"
    finite_cells: list[FiniteCategoricalCellV1] = []
    cell_diagnostics: list[dict[str, Any]] = []
    all_outer_cluster_ids: set[str] = set()
    all_sampler_seeds: set[int] = set()
    all_fork_seed_tuples: set[tuple[int, int, int, int]] = set()
    selected_probe_ids: set[str] = set()
    support_violation_count = 0
    for cell in cells:
        if cell.probe_id not in probe_probability:
            raise ValueError("Instrument cell selects a probe outside the frozen battery.")
        selected_probe_ids.add(cell.probe_id)
        records = cell.kernel_records
        observed_outer_ids = tuple(record.outer_id for record in records)
        expected_outer_ids = tuple(
            range(cell.outer_id_start, cell.outer_id_start + len(records))
        )
        if observed_outer_ids != expected_outer_ids:
            raise ValueError(
                "Instrument kernel records must be ordered contiguous outer replicas."
            )
        support_probabilities = np.asarray(
            [item.posterior_probability for item in cell.posterior_support],
            dtype=np.float64,
        )
        observed_support: set[tuple[str, str, str]] = set()
        response_token_ids: list[int] = []
        cell_support_violation_count = 0
        for record in records:
            if record.outer_cluster_id in all_outer_cluster_ids:
                raise ValueError(
                    "Instrument evidence reuses an outer cluster across cells."
                )
            all_outer_cluster_ids.add(record.outer_cluster_id)
            if record.audit_unit_id != cell.audit_unit_id:
                raise ValueError("Kernel record changed the registered audit unit.")
            if record.snapshot_sha256 != cell.snapshot_sha256:
                raise ValueError("Kernel record changed the registered snapshot.")
            if record.battery_sha256 != battery_sha256:
                raise ValueError("Kernel record changed the frozen battery hash.")
            if record.posterior_mode != expected_record_mode:
                raise ValueError("Kernel record changed the frozen posterior mode.")
            if not math.isclose(
                float(record.rho_prune),
                rho_prune,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("Kernel record changed the frozen pruning bound.")
            if record.source_episode_uid is not None:
                raise ValueError(
                    "Primary instrument rejects matched-history and other episode sources."
                )
            expected_seed = _categorical_full_state_draw_seed(
                base_seed=cell.sampler_base_seed,
                audit_unit_id=cell.audit_unit_id,
                outer_id=record.outer_id,
            )
            if int(record.sampler_seed) != expected_seed:
                raise ValueError("Kernel record sampler seed does not match its outer coordinate.")
            if expected_seed in all_sampler_seeds:
                raise ValueError("Primary instrument repeats a full-state sampler seed.")
            all_sampler_seeds.add(expected_seed)
            support_index = int(
                np.random.default_rng(expected_seed).choice(
                    len(cell.posterior_support),
                    size=None,
                    replace=True,
                    p=support_probabilities,
                )
            )
            support_item = cell.posterior_support[support_index]
            observed_identity = (
                record.theta_id,
                record.execution_state_key,
                record.hidden_state_sha256,
            )
            expected_identity = (
                support_item.theta_id,
                support_item.execution_state_key,
                support_item.hidden_state_sha256,
            )
            if observed_identity != expected_identity:
                raise ValueError(
                    "Kernel record hidden state differs from the deterministic posterior draw."
                )
            if not math.isclose(
                float(record.posterior_draw_probability),
                float(support_item.posterior_probability),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("Kernel record changed its posterior draw probability.")
            observed_support.add(observed_identity)
            record_probe_ids = tuple(item.probe_id for item in record.probe_responses)
            if record_probe_ids != frozen_probe_ids:
                raise ValueError("Kernel record probe set differs from the frozen battery.")
            selected_probe = None
            for probe in record.probe_responses:
                if not math.isclose(
                    float(probe.sampling_probability),
                    probe_probability[probe.probe_id],
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        "Kernel record changed a frozen probe sampling probability."
                    )
                if len(probe.responses) != inner_forks:
                    raise ValueError("Kernel record changed frozen L_inner.")
                for response in probe.responses:
                    expected_coordinate = (
                        cell.audit_unit_id,
                        str(record.outer_id),
                        probe.probe_id,
                        str(response.inner_id),
                    )
                    if response.rng_keys.coordinate != expected_coordinate:
                        raise ValueError(
                            "Kernel response RNG coordinate differs from its outer record."
                        )
                    fork_seeds = tuple(
                        response.rng_keys.seed(name) for name in RNG_STREAM_NAMES
                    )
                    if fork_seeds in all_fork_seed_tuples:
                        raise ValueError("Kernel evidence repeats a fork RNG seed tuple.")
                    all_fork_seed_tuples.add(fork_seeds)
                    if bool(response.support_violation) != (
                        int(response.response_token_id)
                        == support_violation_token_id
                    ):
                        raise ValueError(
                            "Kernel support_violation must be represented only by the "
                            "frozen special response token; the embedded kernel-record "
                            "SHA-256 also does not match its content."
                        )
                    if not kernel_hash_valid:
                        raise ValueError(
                            "Embedded kernel-record SHA-256 does not match its content."
                        )
                    if int(response.response_token_id) >= q:
                        raise ValueError(
                            "Kernel response token lies outside the frozen vocabulary."
                        )
                    if response.support_violation:
                        support_violation_count += 1
                        cell_support_violation_count += 1
                if probe.probe_id == cell.probe_id:
                    selected_probe = probe
            if selected_probe is None:
                raise ValueError("Kernel record lacks the registered instrument probe.")
            selected_response = selected_probe.responses[0]
            if selected_response.support_violation:
                pass
            else:
                response_token_ids.append(int(selected_response.response_token_id))
        if not response_token_ids:
            raise ValueError(
                "Instrument cell has no ordinary response after support violations are removed."
            )
        posterior_mass_covered = math.fsum(
            item.posterior_probability
            for item in cell.posterior_support
            if (
                item.theta_id,
                item.execution_state_key,
                item.hidden_state_sha256,
            ) in observed_support
        )
        finite_cells.append(
            estimate_finite_categorical_cell(
                cell_id=cell.cell_id,
                value_class_id=cell.value_class_id,
                probe_id=cell.probe_id,
                summary_id=summary_id,
                response_token_ids=response_token_ids,
                vocabulary_size_q=q,
                simultaneous_cell_count=simultaneous_cell_count,
                confidence_delta=confidence_delta,
                posterior_bias_bound=posterior_bias_bound,
                reset_bias_bound=reset_bias_bound,
            )
        )
        cell_diagnostics.append(
            {
                "cell_id": cell.cell_id,
                "audit_unit_id": cell.audit_unit_id,
                "snapshot_sha256": cell.snapshot_sha256,
                "probe_id": cell.probe_id,
                "outer_replicas_M": len(records),
                "effective_outer_sample_size": float(len(response_token_ids)),
                "support_violation_count": int(
                    cell_support_violation_count
                ),
                "posterior_support_size": len(cell.posterior_support),
                "posterior_support_sha256": canonical_sha256([
                    item.to_payload() for item in cell.posterior_support
                ]),
                "observed_support_size": len(observed_support),
                "observed_support_fraction": float(
                    len(observed_support) / len(cell.posterior_support)
                ),
                "observed_posterior_mass": float(posterior_mass_covered),
                "sampler_base_seed": int(cell.sampler_base_seed),
                "outer_id_start": int(cell.outer_id_start),
                "posterior_mode": expected_record_mode,
                "source_episode_uids": [],
            }
        )
    if len(selected_probe_ids) != 1:
        raise ValueError("Alignment cells must use one frozen probe family.")
    if evidence_mapping["kernel_records_sha256"] != canonical_sha256(kernel_payloads):
        raise ValueError("Embedded kernel-record SHA-256 does not match its content.")
    alignment = compute_alignment_bounds(finite_cells)
    mechanical_checks = {
        "embedded_kernel_records_schema_valid": True,
        "battery_snapshot_audit_unit_probe_binding_valid": True,
        "iid_full_posterior_with_replacement_valid": True,
        "sampler_seed_schedule_valid": True,
        "posterior_support_draws_valid": True,
        "outer_cluster_and_fork_coordinates_unique": True,
        "no_primary_support_violations": support_violation_count == 0,
    }
    result = build_instrument_measurement_v1(
        alignment,
        exact_or_approximate=mode,
        rho_prune=rho_prune,
        minimum_outer_replicas=min(
            cell.outer_replicas_M for cell in finite_cells
        ),
        response_vocabulary_sha256=response_vocabulary_sha256,
    )
    result.update({
        "instrument_evidence_schema_version": INSTRUMENT_EVIDENCE_SCHEMA_VERSION,
        "sampling_design": INSTRUMENT_SAMPLING_DESIGN,
        "sampling_source": INSTRUMENT_SAMPLING_SOURCE,
        "battery_sha256": battery_sha256,
        "cell_registry_sha256": str(evidence_mapping["cell_registry_sha256"]),
        "kernel_records_sha256": canonical_sha256(kernel_payloads),
        "validity_checks": mechanical_checks,
        "inner_forks_L_inner": inner_forks,
        "support_violation_token_id": support_violation_token_id,
        "support_violation_count": int(support_violation_count),
        "minimum_effective_outer_sample_size": float(min(
            item["effective_outer_sample_size"] for item in cell_diagnostics
        )),
        "minimum_observed_support_fraction": float(min(
            item["observed_support_fraction"] for item in cell_diagnostics
        )),
        "minimum_observed_posterior_mass": float(min(
            item["observed_posterior_mass"] for item in cell_diagnostics
        )),
        "cell_diagnostics": cell_diagnostics,
    })
    return result


def _instrument_cell_registry_payload(
    *,
    cells: Sequence[InstrumentKernelCellV1 | Mapping[str, Any]],
    response_vocabulary_sha256: str,
    battery_sha256: str,
    frozen_probes: Sequence[FrozenInstrumentProbeV1],
    summary_id: str,
    inner_forks_L_inner: int,
    support_violation_token_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": INSTRUMENT_CELL_REGISTRY_SCHEMA_VERSION,
        "response_vocabulary_sha256": str(response_vocabulary_sha256),
        "battery_sha256": str(battery_sha256),
        "frozen_probes": [item.to_payload() for item in frozen_probes],
        "summary_id": str(summary_id),
        "inner_forks_L_inner": int(inner_forks_L_inner),
        "support_violation_token_id": int(support_violation_token_id),
        "cells": [_instrument_cell_descriptor(item) for item in cells],
    }


def _instrument_cell_descriptor(
    item: InstrumentKernelCellV1 | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(item, InstrumentKernelCellV1):
        return item.registry_payload()
    descriptor = _strict_mapping(item, "instrument cell registry entry")
    expected = {
        "cell_id",
        "value_class_id",
        "audit_unit_id",
        "snapshot_sha256",
        "probe_id",
        "sampler_base_seed",
        "outer_id_start",
        "outer_replicas_M",
    }
    _require_exact_keys(descriptor, expected, "instrument cell registry entry")
    normalized = {
        "cell_id": _strict_text(descriptor["cell_id"], "instrument registry cell_id"),
        "value_class_id": _strict_text(
            descriptor["value_class_id"],
            "instrument registry value_class_id",
        ),
        "audit_unit_id": _strict_text(
            descriptor["audit_unit_id"],
            "instrument registry audit_unit_id",
        ),
        "snapshot_sha256": str(descriptor["snapshot_sha256"]),
        "probe_id": _strict_text(
            descriptor["probe_id"],
            "instrument registry probe_id",
        ),
        "sampler_base_seed": _strict_nonbool_int(
            descriptor["sampler_base_seed"],
            "instrument registry sampler_base_seed",
            minimum=0,
        ),
        "outer_id_start": _strict_nonbool_int(
            descriptor["outer_id_start"],
            "instrument registry outer_id_start",
            minimum=0,
        ),
        "outer_replicas_M": _strict_nonbool_int(
            descriptor["outer_replicas_M"],
            "instrument registry outer_replicas_M",
            minimum=1,
        ),
    }
    if not _is_sha256(normalized["snapshot_sha256"]):
        raise ValueError("Instrument registry snapshot_sha256 must be SHA-256.")
    return normalized


def build_instrument_cell_registry_v2(
    cells: Sequence[InstrumentKernelCellV1 | Mapping[str, Any]],
    *,
    response_vocabulary_sha256: str,
    battery_sha256: str,
    frozen_probes: Sequence[FrozenInstrumentProbeV1],
    summary_id: str,
    inner_forks_L_inner: int,
    support_violation_token_id: int,
) -> dict[str, Any]:
    """Return the exact content-addressed cell registry frozen before collection."""

    normalized_cells = tuple(_instrument_cell_descriptor(item) for item in cells)
    if not normalized_cells:
        raise ValueError("Instrument cell registry must contain at least one cell.")
    ordered_cells = tuple(sorted(normalized_cells, key=lambda item: item["cell_id"]))
    for field_name in ("cell_id", "audit_unit_id", "snapshot_sha256"):
        values = tuple(str(item[field_name]) for item in ordered_cells)
        if len(values) != len(set(values)):
            raise ValueError(
                f"Instrument cell registry repeats {field_name}."
            )
    payload = _instrument_cell_registry_payload(
        cells=ordered_cells,
        response_vocabulary_sha256=response_vocabulary_sha256,
        battery_sha256=battery_sha256,
        frozen_probes=tuple(frozen_probes),
        summary_id=summary_id,
        inner_forks_L_inner=inner_forks_L_inner,
        support_violation_token_id=support_violation_token_id,
    )
    return {**payload, "sha256": canonical_sha256(payload)}


def write_kernel_records_parquet(
    requested_path: str | Path,
    records: Sequence[OuterReplicaKernelRecordV1],
    *,
    semantic_bindings: Mapping[str, Any],
    response_vocabulary_sha256: str,
) -> tuple[Path, str]:
    """Write an immutable content-addressed outer-replica table.

    One Parquet row is one independent posterior outer draw. Paired probe and
    conditional inner-fork records remain nested JSON in that row and therefore
    cannot be mistaken for additional outer samples.
    """

    path = Path(requested_path)
    if path.suffix.lower() != ".parquet":
        raise ValueError("Kernel record output must end in .parquet.")
    if path.exists():
        raise FileExistsError("Append-only kernel storage refuses overwrite.")
    if not records:
        raise ValueError("Kernel storage requires at least one outer record.")
    cluster_ids = [record.outer_cluster_id for record in records]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Kernel storage requires unique outer cluster ids.")
    if not _is_sha256(response_vocabulary_sha256):
        raise ValueError("response_vocabulary_sha256 must be a SHA-256 digest.")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Writing kernel Parquet tables requires pyarrow.") from exc

    bindings_json = json.dumps(
        dict(semantic_bindings), sort_keys=True, separators=(",", ":")
    )
    rows = [record.to_payload() for record in records]
    table = pa.table({
        "schema_version": [row["schema_version"] for row in rows],
        "audit_unit_id": [row["audit_unit_id"] for row in rows],
        "outer_id": [row["outer_id"] for row in rows],
        "outer_cluster_id": [row["outer_cluster_id"] for row in rows],
        "snapshot_sha256": [row["snapshot_sha256"] for row in rows],
        "battery_sha256": [row["battery_sha256"] for row in rows],
        "theta_id": [row["theta_id"] for row in rows],
        "execution_state_key": [row["execution_state_key"] for row in rows],
        "hidden_state_sha256": [row["hidden_state_sha256"] for row in rows],
        "posterior_draw_probability": [
            row["posterior_draw_probability"] for row in rows
        ],
        "sampler_seed": pa.array(
            [row["sampler_seed"] for row in rows],
            type=pa.uint64(),
        ),
        "source_episode_uid": [row["source_episode_uid"] for row in rows],
        "posterior_mode": [row["posterior_mode"] for row in rows],
        "rho_prune": [row["rho_prune"] for row in rows],
        "probe_responses_json": [
            json.dumps(
                row["probe_responses"], sort_keys=True, separators=(",", ":")
            )
            for row in rows
        ],
        "semantic_bindings_json": [bindings_json] * len(rows),
        "response_vocabulary_sha256": [response_vocabulary_sha256] * len(rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    content_path = path.with_name(f"{path.stem}.{digest[:16]}{path.suffix}")
    if content_path.exists():
        raise FileExistsError(f"Content-addressed kernel shard exists: {content_path}")
    path.replace(content_path)
    return content_path, digest


def _merge_log_hypothesis(
    masses: dict[tuple[str, str], float],
    payloads: dict[tuple[str, str], bytes],
    key: tuple[str, str],
    payload: bytes,
    log_mass: float,
) -> None:
    if key in payloads and payloads[key] != payload:
        raise ValueError(
            "Sparse forward states share a merge key but have different state bytes."
        )
    payloads[key] = payload
    masses[key] = _logaddexp(masses.get(key, -math.inf), log_mass)


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _logsumexp(values: Iterable[float]) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return -math.inf
    maximum = max(values_tuple)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values_tuple))


def _append_blob(output: bytearray, payload: bytes) -> None:
    output.extend(struct.pack(">Q", len(payload)))
    output.extend(payload)


def _canonical_discrete_array(value: Any) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(
            "Exact-history raw observable state fields must be discrete bool/integer arrays."
        )
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    return canonical.dtype.str, tuple(map(int, canonical.shape)), canonical.tobytes(order="C")


def _strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(value)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch; missing={missing}, unknown={unknown}.")


def _strict_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean.")
    return value


def _strict_nonbool_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _strict_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _named_rng_streams_from_payload(payload: Any) -> NamedRNGStreamsV1:
    item = _strict_mapping(payload, "embedded kernel rng_keys")
    _require_exact_keys(
        item,
        {
            "jax",
            "numpy",
            "python",
            "torch",
            "mode",
            "schedule_version",
            "coordinate",
        },
        "embedded kernel rng_keys",
    )
    coordinate = tuple(
        _strict_text(value, "embedded kernel RNG coordinate")
        for value in _strict_sequence(
            item["coordinate"],
            "embedded kernel RNG coordinate",
        )
    )
    return NamedRNGStreamsV1(
        jax=_strict_nonbool_int(item["jax"], "embedded kernel JAX seed", minimum=0),
        numpy=_strict_nonbool_int(
            item["numpy"],
            "embedded kernel NumPy seed",
            minimum=0,
        ),
        python=_strict_nonbool_int(
            item["python"],
            "embedded kernel Python seed",
            minimum=0,
        ),
        torch=_strict_nonbool_int(
            item["torch"],
            "embedded kernel Torch seed",
            minimum=0,
        ),
        mode=_strict_text(item["mode"], "embedded kernel RNG mode"),
        schedule_version=_strict_text(
            item["schedule_version"],
            "embedded kernel RNG schedule_version",
        ),
        coordinate=coordinate,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
