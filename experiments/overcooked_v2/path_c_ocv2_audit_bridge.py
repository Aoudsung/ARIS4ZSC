from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.overcooked_v2.partner_pool import (
    PARTNER_REGISTRIES,
    ProtocolSpec,
    ScriptedProtocolPartner,
    option_distribution,
)
from experiments.overcooked_v2.partner_modes import (
    LatentModeController,
    LatentPartnerSpec,
)
from experiments.overcooked_v2.path_c_audit_battery import (
    AuditRole,
    ProbeScriptV1,
)
from experiments.overcooked_v2.path_c_belief_audit import (
    CategoricalPosteriorFullStateSampler,
    CategoricalRolloutResultV1,
    ForwardBranchV1,
    ForwardHypothesisV1,
    FullHiddenStateV1,
    NamedRNGStreamsV1,
    RNGKeyScheduleV1,
    SnapshotV1,
    SparseForwardResultV1,
    SparseForwardStateV1,
    StateCodec,
    canonical_sha256,
    sparse_state_merging_forward_recursion,
)
from experiments.overcooked_v2.path_c_sequence import EgoEvidenceSpecV1


OCV2_AUDIT_BRIDGE_VERSION = "path_c_ocv2_audit_bridge_v1"
OCV2_AUDIT_CAPTURE_VERSION = "path_c_ocv2_audit_capture_v1"
OCV2_AUDIT_HARVEST_VERSION = "path_c_ocv2_audit_harvest_v1"
OCV2_AUDIT_BUNDLE_VERSION = "path_c_ocv2_audit_bundle_v1"
OCV2_ROLLOUT_HOOK_CONTRACT_VERSION = "path_c_ocv2_restored_rollout_hook_v1"
CANONICAL_JSON_CODEC_VERSION = "path_c_canonical_json_state_v1"
OPTION_DISTRIBUTION_IDENTITY = (
    "experiments.overcooked_v2.partner_pool.option_distribution"
)
_PYTREE_CODEC_MAGIC = b"path_c_ocv2_pytree_v1\x00"
_AUDIT_BUNDLE_MAGIC = b"path_c_ocv2_audit_bundle_v1\x00"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Canonical audit state cannot contain non-finite numbers.")
        return number
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": _canonical_value(asdict(value)),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Canonical audit state cannot contain object arrays.")
        return {
            "ndarray_dtype": np.dtype(value.dtype).str,
            "ndarray_shape": list(map(int, value.shape)),
            "ndarray_values": _canonical_value(value.tolist()),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical audit mappings require string keys.")
        return {
            str(key): _canonical_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(
        "Canonical audit state does not support values of type "
        f"{type(value).__name__}."
    )


@dataclass(frozen=True)
class CanonicalJSONStateCodecV1:
    """Strict immutable codec for partner and ego runtime mappings."""

    require_mapping_root: bool = True

    def encode(self, value: Any) -> bytes:
        canonical = _canonical_value(value)
        if self.require_mapping_root and not isinstance(canonical, Mapping):
            raise TypeError("This audit runtime codec requires a mapping root.")
        return json.dumps(
            {
                "codec_version": CANONICAL_JSON_CODEC_VERSION,
                "require_mapping_root": bool(self.require_mapping_root),
                "value": canonical,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def decode(self, payload: bytes) -> Any:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Canonical JSON audit payload must be non-empty bytes.")
        try:
            envelope = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Canonical JSON audit payload is invalid.") from error
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "codec_version",
            "require_mapping_root",
            "value",
        }:
            raise ValueError("Canonical JSON audit payload has the wrong envelope.")
        if envelope["codec_version"] != CANONICAL_JSON_CODEC_VERSION:
            raise ValueError("Canonical JSON audit codec version changed.")
        if envelope["require_mapping_root"] is not bool(self.require_mapping_root):
            raise ValueError("Canonical JSON audit root contract changed.")
        value = _restore_canonical_value(envelope["value"])
        if self.require_mapping_root and not isinstance(value, Mapping):
            raise ValueError("Canonical JSON audit payload must decode to a mapping.")
        if self.encode(value) != payload:
            raise ValueError("Canonical JSON audit payload is not canonically encoded.")
        return value


class JaxPytreeStateCodecV1:
    """Deterministic numeric-pytree codec for OCV2 state and raw observations.

    The OCV2 adapter exposes detached host NumPy leaves through its snapshot API.
    This codec binds one frozen JAX pytree definition from a template and stores
    only numeric leaf dtype, shape, and bytes. It does not use pickle and rejects
    a changed tree definition or leaf signature.
    """

    def __init__(self, template: Any) -> None:
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("JaxPytreeStateCodecV1 requires JAX.") from error
        leaves, tree_definition = jax.tree_util.tree_flatten(template)
        if not leaves:
            raise ValueError("OCV2 pytree codec template must contain numeric leaves.")
        self._jax = jax
        self._tree_definition = tree_definition
        self._tree_definition_sha256 = hashlib.sha256(
            repr(tree_definition).encode("utf-8")
        ).digest()
        self._signatures = tuple(self._leaf_signature(leaf) for leaf in leaves)

    @staticmethod
    def _leaf_signature(value: Any) -> tuple[str, tuple[int, ...]]:
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TypeError("OCV2 pytree leaves must be numeric or boolean arrays.")
        dtype = (
            array.dtype.newbyteorder("<")
            if int(array.dtype.itemsize) > 1
            else array.dtype
        )
        return np.dtype(dtype).str, tuple(map(int, array.shape))

    def encode(self, value: Any) -> bytes:
        leaves, tree_definition = self._jax.tree_util.tree_flatten(value)
        if tree_definition != self._tree_definition:
            raise ValueError("OCV2 state changed its frozen JAX pytree definition.")
        signatures = tuple(self._leaf_signature(leaf) for leaf in leaves)
        if signatures != self._signatures:
            raise ValueError("OCV2 state changed a frozen leaf dtype or shape.")
        output = bytearray(_PYTREE_CODEC_MAGIC)
        output.extend(self._tree_definition_sha256)
        output.extend(struct.pack(">I", len(leaves)))
        for leaf, (dtype_text, shape) in zip(leaves, signatures, strict=True):
            dtype = np.dtype(dtype_text)
            array = np.ascontiguousarray(np.asarray(leaf).astype(dtype, copy=False))
            _append_blob(output, dtype_text.encode("ascii"))
            output.extend(struct.pack(">I", len(shape)))
            for dimension in shape:
                output.extend(struct.pack(">Q", int(dimension)))
            _append_blob(output, array.tobytes(order="C"))
        return bytes(output)

    def decode(self, payload: bytes) -> Any:
        if not isinstance(payload, bytes) or not payload.startswith(_PYTREE_CODEC_MAGIC):
            raise ValueError("OCV2 pytree payload has the wrong codec header.")
        cursor = len(_PYTREE_CODEC_MAGIC)
        tree_digest_end = cursor + 32
        if tree_digest_end > len(payload):
            raise ValueError("Truncated OCV2 pytree definition hash.")
        if payload[cursor:tree_digest_end] != self._tree_definition_sha256:
            raise ValueError("OCV2 pytree payload changed its frozen tree definition.")
        cursor = tree_digest_end
        leaf_count, cursor = _read_u32(payload, cursor)
        if leaf_count != len(self._signatures):
            raise ValueError("OCV2 pytree payload changed its frozen leaf count.")
        leaves = []
        for expected_dtype, expected_shape in self._signatures:
            dtype_bytes, cursor = _read_blob(payload, cursor)
            try:
                dtype_text = dtype_bytes.decode("ascii")
                dtype = np.dtype(dtype_text)
            except (UnicodeDecodeError, TypeError) as error:
                raise ValueError("OCV2 pytree payload contains an invalid dtype.") from error
            rank, cursor = _read_u32(payload, cursor)
            shape = []
            for _ in range(rank):
                dimension, cursor = _read_u64(payload, cursor)
                shape.append(int(dimension))
            raw, cursor = _read_blob(payload, cursor)
            signature = (dtype.str, tuple(shape))
            if signature != (expected_dtype, expected_shape):
                raise ValueError("OCV2 pytree payload changed a frozen leaf signature.")
            expected_bytes = int(dtype.itemsize) * int(math.prod(shape, start=1))
            if len(raw) != expected_bytes:
                raise ValueError("OCV2 pytree leaf byte length does not match its shape.")
            leaves.append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
        if cursor != len(payload):
            raise ValueError("OCV2 pytree payload contains trailing bytes.")
        return self._jax.tree_util.tree_unflatten(self._tree_definition, leaves)


class RegisteredPartnerStateCodecV1:
    """Encode registered controller state plus embedded OCV2 public states.

    Scripted and latent controllers retain the previous public state because option
    termination and latent opportunity updates depend on it. Those pytree values use
    the supplied public-state codec; the remaining finite controller metadata uses the
    strict canonical JSON codec.
    """

    _EMBEDDED_STATE_KEY = "encoded_public_state_hex"

    def __init__(self, public_state_codec: StateCodec) -> None:
        self.public_state_codec = public_state_codec
        self.metadata_codec = CanonicalJSONStateCodecV1()

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, Mapping):
            raise TypeError("Registered partner state must be a mapping.")
        prepared = copy.deepcopy(dict(value))
        for container, key in self._embedded_state_slots(prepared):
            public_state = container[key]
            if public_state is None:
                continue
            encoded = self.public_state_codec.encode(public_state)
            if not isinstance(encoded, bytes) or not encoded:
                raise TypeError("Public-state codec must return non-empty bytes.")
            if self.public_state_codec.encode(
                self.public_state_codec.decode(encoded)
            ) != encoded:
                raise ValueError("Public-state codec is not canonical.")
            container[key] = {self._EMBEDDED_STATE_KEY: encoded.hex()}
        return self.metadata_codec.encode(prepared)

    def decode(self, payload: bytes) -> Mapping[str, Any]:
        decoded = self.metadata_codec.decode(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("Registered partner state must decode to a mapping.")
        restored = copy.deepcopy(dict(decoded))
        for container, key in self._embedded_state_slots(restored):
            encoded_wrapper = container[key]
            if encoded_wrapper is None:
                continue
            if not isinstance(encoded_wrapper, Mapping) or set(encoded_wrapper) != {
                self._EMBEDDED_STATE_KEY
            }:
                raise ValueError("Embedded partner public state has the wrong envelope.")
            try:
                encoded = bytes.fromhex(str(
                    encoded_wrapper[self._EMBEDDED_STATE_KEY]
                ))
            except ValueError as error:
                raise ValueError("Embedded partner public-state bytes are invalid.") from error
            public_state = self.public_state_codec.decode(encoded)
            if self.public_state_codec.encode(public_state) != encoded:
                raise ValueError("Embedded partner public state is not canonical.")
            container[key] = public_state
        if self.encode(restored) != payload:
            raise ValueError("Registered partner state is not canonically encoded.")
        return restored

    @staticmethod
    def _embedded_state_slots(
        state: dict[str, Any],
    ) -> tuple[tuple[dict[str, Any], str], ...]:
        schema = state.get("schema_version")
        if schema == "path_c_scripted_partner_state_v1":
            if "last_state" not in state:
                raise ValueError("Scripted partner state lacks last_state.")
            return ((state, "last_state"),)
        if schema == "path_c_latent_partner_state_v1":
            inner = state.get("inner")
            if not isinstance(inner, dict) or "last_state" not in inner:
                raise ValueError("Latent partner state lacks inner last_state.")
            if "last_state" not in state:
                raise ValueError("Latent partner state lacks wrapper last_state.")
            return ((state, "last_state"), (inner, "last_state"))
        raise ValueError("Registered partner state has an unknown schema version.")


@dataclass(frozen=True)
class FrozenPartnerThetaV1:
    theta_id: str
    specification: Any

    def __post_init__(self) -> None:
        if not str(self.theta_id).strip():
            raise ValueError("Frozen partner theta_id must be non-empty.")
        _canonical_value(self.specification)

    def to_payload(self) -> dict[str, Any]:
        return {
            "theta_id": str(self.theta_id),
            "specification": _canonical_value(self.specification),
        }


@dataclass(frozen=True)
class FrozenPartnerRegistryV1:
    """Immutable view of one named partner registry used by generation."""

    partner_set: str
    entries: tuple[FrozenPartnerThetaV1, ...]
    option_distribution_identity: str = OPTION_DISTRIBUTION_IDENTITY

    def __post_init__(self) -> None:
        if not str(self.partner_set).strip() or not self.entries:
            raise ValueError("Frozen partner registry id and entries are required.")
        ids = [entry.theta_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Frozen partner registry theta ids must be unique.")
        if self.option_distribution_identity != OPTION_DISTRIBUTION_IDENTITY:
            raise ValueError("Frozen registry must bind the shared option_distribution.")

    @classmethod
    def from_registered(cls, partner_set: str) -> "FrozenPartnerRegistryV1":
        try:
            registered = PARTNER_REGISTRIES[str(partner_set)]
        except KeyError as error:
            raise ValueError(f"Unknown frozen partner registry {partner_set!r}.") from error
        return cls(
            partner_set=str(partner_set),
            entries=tuple(
                FrozenPartnerThetaV1(theta_id=str(name), specification=specification)
                for name, specification in registered
            ),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "path_c_frozen_partner_registry_v1",
            "partner_set": str(self.partner_set),
            "option_distribution_identity": self.option_distribution_identity,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    def theta(self, theta_id: str) -> FrozenPartnerThetaV1:
        matches = [entry for entry in self.entries if entry.theta_id == str(theta_id)]
        if len(matches) != 1:
            raise KeyError(f"Unknown frozen partner theta {theta_id!r}.")
        return matches[0]


@dataclass(frozen=True)
class EncodedOCV2AuditCaptureInputsV1:
    """Immutable, already encoded inputs for one audit-only snapshot capture."""

    unit_id: str
    episode_uid: str
    decision_index: int
    audit_role: AuditRole
    harvest_run_id: str
    harvest_policy_id: str
    inclusion_probability: float
    env_state_bytes: bytes
    raw_observation_bytes: bytes
    partner_state_bytes: bytes
    ego_state_bytes: bytes
    rng_key_schedule: RNGKeyScheduleV1
    source_kind: str = "audit_harvest"
    training_snapshot_reference: None = None
    schema_version: str = OCV2_AUDIT_CAPTURE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OCV2_AUDIT_CAPTURE_VERSION:
            raise ValueError("OCV2 audit capture schema version changed.")
        if self.audit_role not in {"design", "calibration", "locked_audit"}:
            raise ValueError("OCV2 audit capture has an invalid audit role.")
        for name in ("unit_id", "episode_uid", "harvest_run_id", "harvest_policy_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OCV2 audit capture {name} must be non-empty.")
        if isinstance(self.decision_index, bool) or int(self.decision_index) < 0:
            raise ValueError("OCV2 audit capture decision_index must be non-negative.")
        if isinstance(self.inclusion_probability, (bool, np.bool_)):
            raise ValueError("Audit-harvest inclusion_probability cannot be boolean.")
        probability = float(self.inclusion_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Audit-harvest inclusion_probability must be in (0, 1].")
        if self.source_kind != "audit_harvest":
            raise ValueError("Path C snapshots must come from a separate audit harvest.")
        if self.training_snapshot_reference is not None:
            raise ValueError("Training snapshots cannot be reused as audit snapshots.")
        for name in (
            "env_state_bytes",
            "raw_observation_bytes",
            "partner_state_bytes",
            "ego_state_bytes",
        ):
            if not isinstance(getattr(self, name), bytes) or not getattr(self, name):
                raise ValueError(f"Encoded audit capture {name} must be non-empty bytes.")

    @classmethod
    def from_runtime(
        cls,
        *,
        unit_id: str,
        episode_uid: str,
        decision_index: int,
        audit_role: AuditRole,
        harvest_run_id: str,
        harvest_policy_id: str,
        inclusion_probability: float,
        env_state: Any,
        raw_observation: Any,
        partner_state: Any,
        ego_state: Any,
        env_state_codec: StateCodec,
        observation_codec: StateCodec,
        partner_state_codec: StateCodec,
        ego_state_codec: StateCodec,
        rng_key_schedule: RNGKeyScheduleV1,
    ) -> "EncodedOCV2AuditCaptureInputsV1":
        if isinstance(inclusion_probability, (bool, np.bool_)):
            raise ValueError("Audit-harvest inclusion_probability cannot be boolean.")
        return cls(
            unit_id=str(unit_id),
            episode_uid=str(episode_uid),
            decision_index=int(decision_index),
            audit_role=audit_role,
            harvest_run_id=str(harvest_run_id),
            harvest_policy_id=str(harvest_policy_id),
            inclusion_probability=float(inclusion_probability),
            env_state_bytes=_encoded_copy(env_state_codec, env_state, "env_state"),
            raw_observation_bytes=_encoded_copy(
                observation_codec, raw_observation, "raw_observation"
            ),
            partner_state_bytes=_encoded_copy(
                partner_state_codec, partner_state, "partner_state"
            ),
            ego_state_bytes=_encoded_copy(ego_state_codec, ego_state, "ego_state"),
            rng_key_schedule=rng_key_schedule,
        )

    def to_snapshot(self) -> SnapshotV1:
        return SnapshotV1(
            unit_id=str(self.unit_id),
            episode_uid=str(self.episode_uid),
            decision_index=int(self.decision_index),
            env_state_bytes=bytes(self.env_state_bytes),
            raw_observation_bytes=bytes(self.raw_observation_bytes),
            partner_state_bytes=bytes(self.partner_state_bytes),
            ego_state_bytes=bytes(self.ego_state_bytes),
            rng_key_schedule=self.rng_key_schedule,
        )


@dataclass(frozen=True)
class OCV2AuditHarvestRecordV1:
    """Audit-only metadata kept separately from ordinary training trajectories."""

    audit_role: AuditRole
    harvest_run_id: str
    harvest_policy_id: str
    inclusion_probability: float
    snapshot: SnapshotV1
    source_kind: str = "audit_harvest"
    training_snapshot_reference: None = None
    schema_version: str = OCV2_AUDIT_HARVEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OCV2_AUDIT_HARVEST_VERSION:
            raise ValueError("OCV2 audit-harvest schema version changed.")
        if self.audit_role not in {"design", "calibration", "locked_audit"}:
            raise ValueError("OCV2 audit harvest has an invalid audit role.")
        if not str(self.harvest_run_id).strip() or not str(self.harvest_policy_id).strip():
            raise ValueError("Audit harvest run and policy ids are required.")
        if isinstance(self.inclusion_probability, (bool, np.bool_)):
            raise ValueError("Audit-harvest inclusion_probability cannot be boolean.")
        probability = float(self.inclusion_probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Audit-harvest inclusion_probability must be in (0, 1].")
        if self.source_kind != "audit_harvest":
            raise ValueError("Audit harvest source_kind must be audit_harvest.")
        if self.training_snapshot_reference is not None:
            raise ValueError("Audit harvest cannot reference a training snapshot.")

    @classmethod
    def from_capture(
        cls, capture: EncodedOCV2AuditCaptureInputsV1
    ) -> "OCV2AuditHarvestRecordV1":
        return cls(
            audit_role=capture.audit_role,
            harvest_run_id=capture.harvest_run_id,
            harvest_policy_id=capture.harvest_policy_id,
            inclusion_probability=float(capture.inclusion_probability),
            snapshot=capture.to_snapshot(),
        )

    @property
    def snapshot_reference(self) -> str:
        return f"sha256:{self.snapshot.sha256}"

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._unsigned_payload())

    @property
    def bundle_sha256(self) -> str:
        return hashlib.sha256(self.to_bundle_bytes()).hexdigest()

    @property
    def bundle_reference(self) -> str:
        return f"sha256:{self.bundle_sha256}"

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "audit_role": self.audit_role,
            "harvest_run_id": self.harvest_run_id,
            "harvest_policy_id": self.harvest_policy_id,
            "inclusion_probability": float(self.inclusion_probability),
            "training_snapshot_reference": None,
            "snapshot_reference": self.snapshot_reference,
            "snapshot_manifest": self.snapshot.to_manifest(),
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self._unsigned_payload(),
            "sha256": self.sha256,
            "bundle_sha256": self.bundle_sha256,
            "bundle_reference": self.bundle_reference,
        }

    def to_bundle_bytes(self) -> bytes:
        """Serialize metadata and all four immutable snapshot components."""

        metadata = {
            "bundle_version": OCV2_AUDIT_BUNDLE_VERSION,
            "record_sha256": self.sha256,
            "snapshot_sha256": self.snapshot.sha256,
            "audit_role": self.audit_role,
            "harvest_run_id": self.harvest_run_id,
            "harvest_policy_id": self.harvest_policy_id,
            "inclusion_probability": float(self.inclusion_probability),
            "source_kind": self.source_kind,
            "training_snapshot_reference": None,
            "unit_id": self.snapshot.unit_id,
            "episode_uid": self.snapshot.episode_uid,
            "decision_index": int(self.snapshot.decision_index),
            "rng_key_schedule": self.snapshot.rng_key_schedule.to_payload(),
            "component_sha256": {
                "env_state": hashlib.sha256(
                    self.snapshot.env_state_bytes
                ).hexdigest(),
                "raw_observation": hashlib.sha256(
                    self.snapshot.raw_observation_bytes
                ).hexdigest(),
                "partner_state": hashlib.sha256(
                    self.snapshot.partner_state_bytes
                ).hexdigest(),
                "ego_state": hashlib.sha256(
                    self.snapshot.ego_state_bytes
                ).hexdigest(),
            },
        }
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        output = bytearray(_AUDIT_BUNDLE_MAGIC)
        _append_blob(output, metadata_bytes)
        for component in (
            self.snapshot.env_state_bytes,
            self.snapshot.raw_observation_bytes,
            self.snapshot.partner_state_bytes,
            self.snapshot.ego_state_bytes,
        ):
            _append_blob(output, component)
        return bytes(output)

    @classmethod
    def from_bundle_bytes(cls, payload: bytes) -> "OCV2AuditHarvestRecordV1":
        if not isinstance(payload, bytes) or not payload.startswith(_AUDIT_BUNDLE_MAGIC):
            raise ValueError("OCV2 audit bundle has the wrong header.")
        cursor = len(_AUDIT_BUNDLE_MAGIC)
        metadata_bytes, cursor = _read_blob(payload, cursor)
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("OCV2 audit bundle metadata is invalid.") from error
        expected_metadata_keys = {
            "bundle_version",
            "record_sha256",
            "snapshot_sha256",
            "audit_role",
            "harvest_run_id",
            "harvest_policy_id",
            "inclusion_probability",
            "source_kind",
            "training_snapshot_reference",
            "unit_id",
            "episode_uid",
            "decision_index",
            "rng_key_schedule",
            "component_sha256",
        }
        if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_keys:
            raise ValueError("OCV2 audit bundle metadata schema changed.")
        canonical_metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if canonical_metadata_bytes != metadata_bytes:
            raise ValueError("OCV2 audit bundle metadata is not canonical.")
        if metadata["bundle_version"] != OCV2_AUDIT_BUNDLE_VERSION:
            raise ValueError("OCV2 audit bundle version changed.")
        components = []
        for _ in range(4):
            component, cursor = _read_blob(payload, cursor)
            if not component:
                raise ValueError("OCV2 audit bundle contains an empty state component.")
            components.append(component)
        if cursor != len(payload):
            raise ValueError("OCV2 audit bundle contains trailing bytes.")
        component_names = (
            "env_state",
            "raw_observation",
            "partner_state",
            "ego_state",
        )
        component_hashes = metadata["component_sha256"]
        if not isinstance(component_hashes, Mapping) or set(component_hashes) != set(
            component_names
        ):
            raise ValueError("OCV2 audit bundle component hash table changed.")
        for name, component in zip(component_names, components, strict=True):
            if hashlib.sha256(component).hexdigest() != component_hashes[name]:
                raise ValueError(f"OCV2 audit bundle {name} hash mismatch.")
        rng_schedule = _rng_schedule_from_payload(metadata["rng_key_schedule"])
        snapshot = SnapshotV1(
            unit_id=str(metadata["unit_id"]),
            episode_uid=str(metadata["episode_uid"]),
            decision_index=int(metadata["decision_index"]),
            env_state_bytes=components[0],
            raw_observation_bytes=components[1],
            partner_state_bytes=components[2],
            ego_state_bytes=components[3],
            rng_key_schedule=rng_schedule,
        )
        if snapshot.sha256 != metadata["snapshot_sha256"]:
            raise ValueError("OCV2 audit bundle snapshot SHA-256 mismatch.")
        record = cls(
            audit_role=str(metadata["audit_role"]),  # type: ignore[arg-type]
            harvest_run_id=str(metadata["harvest_run_id"]),
            harvest_policy_id=str(metadata["harvest_policy_id"]),
            inclusion_probability=metadata["inclusion_probability"],
            snapshot=snapshot,
            source_kind=str(metadata["source_kind"]),
            training_snapshot_reference=metadata["training_snapshot_reference"],
        )
        if record.sha256 != metadata["record_sha256"]:
            raise ValueError("OCV2 audit bundle record SHA-256 mismatch.")
        return record


@dataclass(frozen=True)
class OCV2AuditContentAddressedStoreV1:
    """Minimal immutable blob store for audit bundles, separate from training data."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def write(self, record: OCV2AuditHarvestRecordV1) -> Path:
        bundle = record.to_bundle_bytes()
        digest = hashlib.sha256(bundle).hexdigest()
        directory = self.root / digest[:2]
        path = directory / f"{digest}.ocv2audit"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(bundle)
                handle.flush()
        except FileExistsError:
            if path.read_bytes() != bundle:
                raise RuntimeError(
                    "Content-addressed audit path contains different bytes."
                )
        return path

    def read(self, bundle_reference: str) -> OCV2AuditHarvestRecordV1:
        prefix = "sha256:"
        if not str(bundle_reference).startswith(prefix):
            raise ValueError("Audit bundle reference must begin with sha256:.")
        digest = str(bundle_reference)[len(prefix):]
        if not _is_sha256(digest):
            raise ValueError("Audit bundle reference contains an invalid SHA-256.")
        path = self.root / digest[:2] / f"{digest}.ocv2audit"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Stored audit bundle SHA-256 mismatch.")
        return OCV2AuditHarvestRecordV1.from_bundle_bytes(payload)


@dataclass(frozen=True)
class OCV2AuditRolloutTraceV1:
    snapshot_sha256: str
    rng_keys: NamedRNGStreamsV1
    response_token_id: int
    support_violation: bool
    trajectory_bytes: bytes
    primitive_steps: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.snapshot_sha256):
            raise ValueError("OCV2 audit trace snapshot hash must be SHA-256.")
        if isinstance(self.response_token_id, bool) or int(self.response_token_id) < 0:
            raise ValueError("OCV2 audit response token id must be non-negative.")
        if not isinstance(self.trajectory_bytes, bytes) or not self.trajectory_bytes:
            raise ValueError("OCV2 audit trajectory must be non-empty immutable bytes.")
        if isinstance(self.primitive_steps, bool) or int(self.primitive_steps) <= 0:
            raise ValueError("OCV2 audit trajectory must contain primitive steps.")


@runtime_checkable
class OCV2AuditRolloutHooks(Protocol):
    """Required restoration hook absent from the current imperative adapter API."""

    contract_version: str
    fresh_runtime_per_call: bool
    shared_option_distribution: Callable[..., np.ndarray]

    def rollout_from(
        self,
        *,
        snapshot_sha256: str,
        env_state: Any,
        raw_observation: Any,
        partner_state: Any,
        ego_state: Any,
        theta_id: str | None,
        probe_script: ProbeScriptV1,
        rng_keys: NamedRNGStreamsV1,
    ) -> OCV2AuditRolloutTraceV1:
        ...


@dataclass(frozen=True)
class FreshOCV2AdapterRolloutHooks:
    """Fail-closed wrapper that creates one unused OCV2Adapter per continuation.

    OCV2Adapter.step mutates key/state/observation, while partner controllers also
    mutate option and latent-mode state. The callback therefore receives a newly
    constructed adapter and freshly decoded state for every call. The callback is
    the explicit integration hook that must restore partner/ego controllers and
    execute the frozen probe; no default restoration is guessed here.
    """

    adapter_factory: Callable[[], Any]
    restored_rollout: Callable[..., OCV2AuditRolloutTraceV1]
    restoration_contract_id: str
    contract_version: str = OCV2_ROLLOUT_HOOK_CONTRACT_VERSION
    fresh_runtime_per_call: bool = True

    def __post_init__(self) -> None:
        if not str(self.restoration_contract_id).strip():
            raise ValueError("A partner/ego restoration contract id is required.")
        if self.contract_version != OCV2_ROLLOUT_HOOK_CONTRACT_VERSION:
            raise ValueError("OCV2 restored-rollout hook contract version changed.")
        if self.fresh_runtime_per_call is not True:
            raise ValueError("OCV2 audit continuations require a fresh runtime per call.")

    @property
    def shared_option_distribution(self) -> Callable[..., np.ndarray]:
        return option_distribution

    def rollout_from(self, **kwargs: Any) -> OCV2AuditRolloutTraceV1:
        from experiments.overcooked_v2.env_adapter import OCV2Adapter

        adapter = self.adapter_factory()
        if not isinstance(adapter, OCV2Adapter):
            raise TypeError("adapter_factory must return a concrete OCV2Adapter.")
        if any(getattr(adapter, name, None) is not None for name in ("key", "state", "obs")):
            raise ValueError("Fresh audit OCV2Adapter must not already hold episode state.")
        result = self.restored_rollout(
            adapter=adapter,
            shared_option_distribution=option_distribution,
            restoration_contract_id=self.restoration_contract_id,
            **kwargs,
        )
        if not isinstance(result, OCV2AuditRolloutTraceV1):
            raise TypeError("Restored OCV2 rollout must return OCV2AuditRolloutTraceV1.")
        return result


@runtime_checkable
class RestorableRuntimeController(Protocol):
    """Controller whose complete future-affecting state can be injected."""

    def get_state(self) -> Mapping[str, Any]:
        ...

    def set_state(self, state: Mapping[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class NativeOCV2AdapterRolloutHooks:
    """Production restoration path using OCV2Adapter and restorable controllers."""

    adapter_factory: Callable[[], Any]
    partner_factory: Callable[[str | None], RestorableRuntimeController]
    ego_factory: Callable[[], RestorableRuntimeController]
    probe_executor: Callable[..., OCV2AuditRolloutTraceV1]
    restoration_contract_id: str
    contract_version: str = OCV2_ROLLOUT_HOOK_CONTRACT_VERSION
    fresh_runtime_per_call: bool = True

    def __post_init__(self) -> None:
        if not str(self.restoration_contract_id).strip():
            raise ValueError("Native OCV2 restoration contract id is required.")
        if self.contract_version != OCV2_ROLLOUT_HOOK_CONTRACT_VERSION:
            raise ValueError("Native OCV2 rollout hook contract version changed.")
        if self.fresh_runtime_per_call is not True:
            raise ValueError("Native OCV2 audit requires fresh runtime per call.")

    @property
    def shared_option_distribution(self) -> Callable[..., np.ndarray]:
        return option_distribution

    def rollout_from(
        self,
        *,
        snapshot_sha256: str,
        env_state: Any,
        raw_observation: Any,
        partner_state: Any,
        ego_state: Any,
        theta_id: str | None,
        probe_script: ProbeScriptV1,
        rng_keys: NamedRNGStreamsV1,
    ) -> OCV2AuditRolloutTraceV1:
        import jax.numpy as jnp
        from experiments.overcooked_v2.env_adapter import (
            OCV2Adapter,
            OCV2AdapterSnapshot,
        )

        adapter = self.adapter_factory()
        if not isinstance(adapter, OCV2Adapter):
            raise TypeError("Native adapter_factory must return OCV2Adapter.")
        if any(
            getattr(adapter, name, None) is not None
            for name in ("key", "state", "raw_obs", "obs")
        ):
            raise ValueError("Native audit adapter must be unused before restoration.")
        if not isinstance(raw_observation, Mapping) or not raw_observation:
            raise TypeError("Native audit raw_observation must be a mapping.")
        if not isinstance(partner_state, Mapping) or not isinstance(ego_state, Mapping):
            raise TypeError("Native audit partner and ego states must be mappings.")
        adapter.restore_state(OCV2AdapterSnapshot(
            layout_name=str(adapter.layout_name),
            max_steps=int(adapter.max_steps),
            key=jnp.asarray(
                rng_keys.jax_key_words_uint32(),
                dtype=jnp.uint32,
            ),
            state=env_state,
            raw_obs={
                str(name): np.asarray(value).copy()
                for name, value in raw_observation.items()
            },
        ))
        partner = self.partner_factory(theta_id)
        ego = self.ego_factory()
        if not isinstance(partner, RestorableRuntimeController):
            raise TypeError("Native partner controller does not implement state injection.")
        if not isinstance(ego, RestorableRuntimeController):
            raise TypeError("Native ego controller does not implement state injection.")
        partner.set_state(dict(partner_state))
        ego.set_state(dict(ego_state))
        if _canonical_value(partner.get_state()) != _canonical_value(partner_state):
            raise ValueError("Native partner state did not round-trip through set_state.")
        if _canonical_value(ego.get_state()) != _canonical_value(ego_state):
            raise ValueError("Native ego state did not round-trip through set_state.")
        result = self.probe_executor(
            adapter=adapter,
            partner=partner,
            ego=ego,
            probe_script=probe_script,
            rng_keys=rng_keys,
            snapshot_sha256=snapshot_sha256,
            shared_option_distribution=option_distribution,
            restoration_contract_id=self.restoration_contract_id,
        )
        if not isinstance(result, OCV2AuditRolloutTraceV1):
            raise TypeError("Native probe executor must return OCV2AuditRolloutTraceV1.")
        if result.snapshot_sha256 != snapshot_sha256 or result.rng_keys != rng_keys:
            raise ValueError("Native probe trace changed snapshot or RNG binding.")
        return result


class OCV2SnapshotForkRunner:
    """SnapshotForkRunner bridge with immutable decode and RNG-mode enforcement."""

    def __init__(
        self,
        *,
        env_state_codec: StateCodec,
        observation_codec: StateCodec,
        partner_state_codec: StateCodec,
        ego_state_codec: StateCodec,
        rollout_hooks: OCV2AuditRolloutHooks,
    ) -> None:
        if getattr(rollout_hooks, "contract_version", None) != OCV2_ROLLOUT_HOOK_CONTRACT_VERSION:
            raise ValueError("OCV2 rollout hooks do not implement the frozen contract version.")
        if getattr(rollout_hooks, "fresh_runtime_per_call", None) is not True:
            raise ValueError("OCV2 rollout hooks must create fresh runtime state per call.")
        if getattr(rollout_hooks, "shared_option_distribution", None) is not option_distribution:
            raise ValueError("OCV2 rollout hooks must use the shared option_distribution.")
        self.env_state_codec = env_state_codec
        self.observation_codec = observation_codec
        self.partner_state_codec = partner_state_codec
        self.ego_state_codec = ego_state_codec
        self.rollout_hooks = rollout_hooks

    def run(
        self,
        *,
        snapshot: SnapshotV1,
        hidden_state: FullHiddenStateV1 | None,
        probe_script: ProbeScriptV1,
        rng_keys: NamedRNGStreamsV1,
    ) -> CategoricalRolloutResultV1:
        source = snapshot.snapshot_copy()
        before = source.sha256
        self._validate_rng_binding(source, hidden_state, probe_script, rng_keys)
        if hidden_state is not None and hidden_state.execution_state_key != hashlib.sha256(
            hidden_state.execution_state_bytes
        ).hexdigest():
            raise ValueError(
                "OCV2 posterior execution-state key must hash its immutable bytes."
            )
        partner_payload = (
            source.partner_state_bytes
            if hidden_state is None
            else hidden_state.execution_state_bytes
        )
        theta_id = None if hidden_state is None else str(hidden_state.theta_id)
        env_state = _decode_round_trip(
            self.env_state_codec, source.env_state_bytes, "env_state"
        )
        raw_observation = _decode_round_trip(
            self.observation_codec,
            source.raw_observation_bytes,
            "raw_observation",
        )
        partner_state = _decode_round_trip(
            self.partner_state_codec, partner_payload, "partner_state"
        )
        ego_state = _decode_round_trip(
            self.ego_state_codec, source.ego_state_bytes, "ego_state"
        )
        trace = self.rollout_hooks.rollout_from(
            snapshot_sha256=before,
            env_state=env_state,
            raw_observation=raw_observation,
            partner_state=partner_state,
            ego_state=ego_state,
            theta_id=theta_id,
            probe_script=probe_script,
            rng_keys=rng_keys,
        )
        if not isinstance(trace, OCV2AuditRolloutTraceV1):
            raise TypeError("OCV2 rollout hook returned the wrong trace type.")
        if trace.snapshot_sha256 != before:
            raise RuntimeError("OCV2 rollout trace does not bind its source snapshot.")
        if trace.rng_keys != rng_keys:
            raise RuntimeError("OCV2 rollout hook did not report the supplied named RNG streams.")
        if source.sha256 != before or snapshot.sha256 != before:
            raise RuntimeError("OCV2 rollout hook mutated the immutable source snapshot.")
        trajectory_sha256 = canonical_sha256(
            {
                "schema_version": OCV2_AUDIT_BRIDGE_VERSION,
                "snapshot_sha256": before,
                "theta_id": theta_id,
                "probe_script": probe_script.to_payload(),
                "rng_keys": rng_keys.to_payload(),
                "response_token_id": int(trace.response_token_id),
                "support_violation": bool(trace.support_violation),
                "primitive_steps": int(trace.primitive_steps),
                "trajectory_bytes_sha256": hashlib.sha256(
                    trace.trajectory_bytes
                ).hexdigest(),
            }
        )
        return CategoricalRolloutResultV1(
            response_token_id=int(trace.response_token_id),
            support_violation=bool(trace.support_violation),
            trajectory_sha256=trajectory_sha256,
        )

    @staticmethod
    def _validate_rng_binding(
        snapshot: SnapshotV1,
        hidden_state: FullHiddenStateV1 | None,
        probe_script: ProbeScriptV1,
        rng_keys: NamedRNGStreamsV1,
    ) -> None:
        if hidden_state is None:
            if rng_keys.mode != "replay":
                raise ValueError("Snapshot replay requires original replay RNG streams.")
            if rng_keys != snapshot.rng_key_schedule.replay_keys():
                raise ValueError("Snapshot replay RNG streams differ from generation keys.")
            return
        if rng_keys.mode != "fork":
            raise ValueError("Posterior continuation requires fresh fork RNG streams.")
        if len(rng_keys.coordinate) != 4 or rng_keys.coordinate[0] != snapshot.unit_id:
            raise ValueError("Fork RNG coordinate does not bind the source audit unit.")
        if rng_keys.coordinate[2] != probe_script.script_id:
            raise ValueError("Fork RNG coordinate does not bind the frozen probe script.")
        try:
            outer_id = int(rng_keys.coordinate[1])
            inner_id = int(rng_keys.coordinate[3])
        except ValueError as error:
            raise ValueError("Fork RNG coordinate contains a non-integer replica id.") from error
        expected = snapshot.rng_key_schedule.fork_keys(
            unit_id=snapshot.unit_id,
            outer_id=outer_id,
            probe_id=probe_script.script_id,
            inner_id=inner_id,
        )
        if rng_keys != expected:
            raise ValueError("Fork RNG streams do not match the frozen derivation schedule.")
        replay = snapshot.rng_key_schedule.replay_keys()
        if any(rng_keys.seed(name) == replay.seed(name) for name in ("jax", "numpy", "python", "torch")):
            raise ValueError("A fork reused a generation-time RNG stream.")


@dataclass(frozen=True)
class PartnerEvidenceObservationV1:
    """One complete registered evidence row for an option-decision transition.

    The partner option is deliberately not stored as observed evidence. Exact
    inference marginalizes it through the shared generation distribution. The
    evidence bytes contain the complete EgoEvidenceSpecV1 row: public observation,
    both ordered primitive-action traces, ego option, duration, reward, progress,
    valid-option mask, and termination flags.
    """

    evidence_spec_sha256: str
    evidence_row_bytes: bytes
    option_choice_public_state_bytes: bytes
    decision_public_state_bytes: bytes
    intermediate_public_state_bytes: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if not _is_sha256(self.evidence_spec_sha256):
            raise ValueError("Evidence observation must bind EgoEvidenceSpecV1 SHA-256.")
        for name in (
            "evidence_row_bytes",
            "option_choice_public_state_bytes",
            "decision_public_state_bytes",
        ):
            if not isinstance(getattr(self, name), bytes) or not getattr(self, name):
                raise ValueError(f"Evidence observation {name} must be non-empty bytes.")
        if not isinstance(self.intermediate_public_state_bytes, tuple) or any(
            not isinstance(value, bytes) or not value
            for value in self.intermediate_public_state_bytes
        ):
            raise ValueError(
                "Evidence intermediate public states must be a tuple of non-empty bytes."
            )

    @classmethod
    def from_evidence_row(
        cls,
        *,
        evidence_spec: EgoEvidenceSpecV1,
        evidence_row: Any,
        option_choice_public_state_bytes: bytes,
        decision_public_state_bytes: bytes,
        intermediate_public_state_bytes: Sequence[bytes] = (),
    ) -> "PartnerEvidenceObservationV1":
        row = np.asarray(evidence_row, dtype=np.dtype("<f4"))
        if row.shape != (int(evidence_spec.evidence_dim),):
            raise ValueError(
                "Complete evidence row shape does not match frozen EgoEvidenceSpecV1."
            )
        if not np.all(np.isfinite(row)):
            raise ValueError("Complete evidence row must contain only finite values.")
        _validate_complete_evidence_row(evidence_spec, row)
        duration_value = float(row[evidence_spec.field_slices()["duration"]][0])
        duration = int(duration_value)
        intermediate = tuple(bytes(value) for value in intermediate_public_state_bytes)
        if len(intermediate) != max(0, duration - 1):
            raise ValueError(
                "Exact partner evidence requires one intermediate public state "
                "between every adjacent primitive action."
            )
        if duration == 0 and bytes(option_choice_public_state_bytes) != bytes(
            decision_public_state_bytes
        ):
            raise ValueError(
                "A zero-duration evidence row must reference one identical public state."
            )
        return cls(
            evidence_spec_sha256=evidence_spec.sha256(),
            evidence_row_bytes=np.ascontiguousarray(row).tobytes(order="C"),
            option_choice_public_state_bytes=bytes(option_choice_public_state_bytes),
            decision_public_state_bytes=bytes(decision_public_state_bytes),
            intermediate_public_state_bytes=intermediate,
        )


@dataclass(frozen=True)
class PartnerRuntimeBranchV1:
    state_key: str
    state_bytes: bytes
    conditional_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.state_bytes, bytes) or not self.state_bytes:
            raise ValueError("Partner runtime branch state must be non-empty bytes.")
        expected_key = hashlib.sha256(self.state_bytes).hexdigest()
        if self.state_key != expected_key:
            raise ValueError("Partner runtime branch key must hash its encoded state.")
        if isinstance(self.conditional_probability, (bool, np.bool_)):
            raise ValueError("Partner runtime branch probability cannot be boolean.")
        probability = float(self.conditional_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Partner runtime branch probability must be in [0, 1].")

    @classmethod
    def from_state(
        cls,
        state: Any,
        probability: float,
        *,
        state_codec: StateCodec,
    ) -> "PartnerRuntimeBranchV1":
        payload = _encoded_copy(state_codec, state, "partner_runtime_branch")
        return cls(
            state_key=hashlib.sha256(payload).hexdigest(),
            state_bytes=payload,
            conditional_probability=float(probability),
        )


@runtime_checkable
class ExactPartnerRuntimeTransition(Protocol):
    """Explicit hook for latent/runtime updates not exposed as pure partner APIs."""

    contract_version: str
    shared_option_distribution: Callable[..., np.ndarray]

    def advance(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        runtime_state: Mapping[str, Any],
        runtime_state_bytes: bytes,
        option_choice_public_state: Any,
        decision_public_state: Any,
        public_state_path: Sequence[Any],
        evidence_row: np.ndarray,
        evidence_spec: EgoEvidenceSpecV1,
        step_index: int,
    ) -> Sequence[PartnerRuntimeBranchV1]:
        ...


@runtime_checkable
class ExactPartnerEvidenceLikelihood(Protocol):
    """Exact likelihood kernel for the complete registered evidence transition.

    It must marginalize every unobserved partner option choice in the primitive
    window, including the case where an option persists across an ego boundary.
    The shared generation function is both identity-bound and passed explicitly.
    """

    contract_version: str
    shared_option_distribution: Callable[..., np.ndarray]

    def probability(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        runtime_state: Mapping[str, Any],
        next_runtime_state_bytes: bytes,
        option_choice_public_state: Any,
        decision_public_state: Any,
        public_state_path: Sequence[Any],
        evidence_row: np.ndarray,
        evidence_spec: EgoEvidenceSpecV1,
        option_library: Any,
        option_distribution_fn: Callable[..., np.ndarray],
        step_index: int,
    ) -> float:
        ...


@runtime_checkable
class ExactObservedPartnerController(Protocol):
    """Registered controller that can enumerate one observed primitive action."""

    def get_state(self) -> Mapping[str, Any]:
        ...

    def set_state(self, state: Mapping[str, Any]) -> None:
        ...

    def exact_observed_action_branches(
        self,
        state: Any,
        observed_primitive_action: int,
    ) -> Sequence[tuple[Mapping[str, Any], float]]:
        ...


@dataclass(frozen=True)
class RegisteredPartnerControllerFactoryV1:
    """Construct the same scripted or latent controller used by generation."""

    option_library: Any

    def __post_init__(self) -> None:
        if not getattr(self.option_library, "options", None):
            raise ValueError("Registered partner factory requires an option library.")

    def __call__(
        self,
        theta_id: str,
        theta_specification: Any,
    ) -> ExactObservedPartnerController:
        if isinstance(theta_specification, LatentPartnerSpec):
            controller: Any = LatentModeController(
                name=str(theta_id),
                option_library=self.option_library,
                spec=theta_specification,
                partner_cls=ScriptedProtocolPartner,
                partner_id=0,
            )
        elif isinstance(theta_specification, ProtocolSpec):
            controller = ScriptedProtocolPartner(
                name=str(theta_id),
                option_library=self.option_library,
                protocol=theta_specification,
                partner_id=0,
            )
        else:
            raise TypeError("Registered partner theta has an unknown specification type.")
        if not isinstance(controller, ExactObservedPartnerController):
            raise TypeError("Registered partner controller lacks exact action enumeration.")
        return controller


class ExactRegisteredPartnerKernelV1:
    """Concrete transition and likelihood hooks for registered partner controllers.

    The transition enumerates every hidden option draw that is compatible with the
    complete observed primitive-action window. It normalizes those compatible
    branches; the likelihood hook returns their pre-normalization probability, so
    their product is the exact joint transition-and-evidence mass.
    """

    contract_version = OCV2_AUDIT_BRIDGE_VERSION
    shared_option_distribution = staticmethod(option_distribution)

    def __init__(
        self,
        *,
        option_library: Any,
        runtime_state_codec: StateCodec,
        controller_factory: Callable[
            [str, Any],
            ExactObservedPartnerController,
        ] | None = None,
    ) -> None:
        if not getattr(option_library, "options", None):
            raise ValueError("Exact registered partner kernel requires an option library.")
        self.option_library = option_library
        self.runtime_state_codec = runtime_state_codec
        self.controller_factory = (
            RegisteredPartnerControllerFactoryV1(option_library)
            if controller_factory is None
            else controller_factory
        )

    def initial_runtime_state_bytes(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        seed: int,
    ) -> bytes:
        """Reset one registered controller and encode its complete initial state."""

        controller = self.controller_factory(str(theta_id), theta_specification)
        reset = getattr(controller, "reset", None)
        if not callable(reset):
            raise TypeError("Exact partner controller lacks reset(seed).")
        reset(int(seed))
        return _encoded_copy(
            self.runtime_state_codec,
            controller.get_state(),
            "exact_partner_initial_runtime",
        )

    def advance(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        runtime_state: Mapping[str, Any],
        runtime_state_bytes: bytes,
        option_choice_public_state: Any,
        decision_public_state: Any,
        public_state_path: Sequence[Any],
        evidence_row: np.ndarray,
        evidence_spec: EgoEvidenceSpecV1,
        step_index: int,
    ) -> Sequence[PartnerRuntimeBranchV1]:
        del option_choice_public_state, decision_public_state, step_index
        encoded_runtime = _encoded_copy(
            self.runtime_state_codec,
            runtime_state,
            "exact_partner_runtime",
        )
        if encoded_runtime != runtime_state_bytes:
            raise ValueError("Exact partner runtime bytes changed after decoding.")
        branch_mass = self._enumerate_observed_window(
            theta_id=theta_id,
            theta_specification=theta_specification,
            runtime_state=runtime_state,
            public_state_path=public_state_path,
            evidence_row=evidence_row,
            evidence_spec=evidence_spec,
        )
        total = math.fsum(branch_mass.values())
        if total == 0.0:
            return (
                PartnerRuntimeBranchV1(
                    state_key=hashlib.sha256(runtime_state_bytes).hexdigest(),
                    state_bytes=bytes(runtime_state_bytes),
                    conditional_probability=1.0,
                ),
            )
        return tuple(
            PartnerRuntimeBranchV1(
                state_key=hashlib.sha256(payload).hexdigest(),
                state_bytes=payload,
                conditional_probability=float(mass / total),
            )
            for payload, mass in sorted(branch_mass.items())
        )

    def probability(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        runtime_state: Mapping[str, Any],
        next_runtime_state_bytes: bytes,
        option_choice_public_state: Any,
        decision_public_state: Any,
        public_state_path: Sequence[Any],
        evidence_row: np.ndarray,
        evidence_spec: EgoEvidenceSpecV1,
        option_library: Any,
        option_distribution_fn: Callable[..., np.ndarray],
        step_index: int,
    ) -> float:
        del option_choice_public_state, decision_public_state, step_index
        if option_library is not self.option_library:
            raise ValueError("Exact partner kernel changed the frozen option library.")
        if option_distribution_fn is not option_distribution:
            raise ValueError("Exact partner kernel changed shared option_distribution.")
        branch_mass = self._enumerate_observed_window(
            theta_id=theta_id,
            theta_specification=theta_specification,
            runtime_state=runtime_state,
            public_state_path=public_state_path,
            evidence_row=evidence_row,
            evidence_spec=evidence_spec,
        )
        total = math.fsum(branch_mass.values())
        if total > 0.0 and bytes(next_runtime_state_bytes) not in branch_mass:
            raise ValueError(
                "Exact likelihood was requested for an unreachable runtime branch."
            )
        return float(total)

    def _enumerate_observed_window(
        self,
        *,
        theta_id: str,
        theta_specification: Any,
        runtime_state: Mapping[str, Any],
        public_state_path: Sequence[Any],
        evidence_row: np.ndarray,
        evidence_spec: EgoEvidenceSpecV1,
    ) -> dict[bytes, float]:
        _validate_complete_evidence_row(evidence_spec, evidence_row)
        slices = evidence_spec.field_slices()
        duration = int(float(evidence_row[slices["duration"]][0]))
        states = tuple(public_state_path)
        if len(states) != duration + 1:
            raise ValueError(
                "Exact partner public-state path must contain duration plus one states."
            )
        action_width = (
            int(evidence_spec.max_primitive_steps_per_decision),
            int(evidence_spec.primitive_action_vocab_dim),
        )
        encoded_actions = np.asarray(
            evidence_row[slices["partner_primitive_actions"]],
            dtype=np.float32,
        ).reshape(action_width)
        observed_actions = np.argmax(encoded_actions, axis=1)[:duration]
        components: list[tuple[Mapping[str, Any], float]] = [
            (runtime_state, 1.0)
        ]
        for public_state, observed_action in zip(
            states[:-1],
            observed_actions.tolist(),
            strict=True,
        ):
            next_components: list[tuple[Mapping[str, Any], float]] = []
            for component_state, component_mass in components:
                controller = self.controller_factory(
                    str(theta_id),
                    theta_specification,
                )
                if not isinstance(controller, ExactObservedPartnerController):
                    raise TypeError(
                        "Exact partner controller factory returned an incompatible controller."
                    )
                controller.set_state(component_state)
                for next_state, conditional_mass in (
                    controller.exact_observed_action_branches(
                        public_state,
                        int(observed_action),
                    )
                ):
                    mass = float(component_mass) * float(conditional_mass)
                    if not math.isfinite(mass) or mass < 0.0:
                        raise ValueError("Exact partner branch mass is invalid.")
                    if mass > 0.0:
                        next_components.append((next_state, mass))
            components = next_components
            if not components:
                break
        merged: dict[bytes, float] = {}
        for state, mass in components:
            payload = _encoded_copy(
                self.runtime_state_codec,
                state,
                "exact_partner_next_runtime",
            )
            merged[payload] = merged.get(payload, 0.0) + float(mass)
        total = math.fsum(merged.values())
        if not math.isfinite(total) or not 0.0 <= total <= 1.0 + 1.0e-12:
            raise ValueError("Exact partner evidence probability is outside [0, 1].")
        return merged


@dataclass(frozen=True)
class ExactPartnerPosteriorResultV1:
    forward: SparseForwardResultV1
    partner_registry_sha256: str
    option_library_sha256: str
    ego_evidence_spec_sha256: str
    option_distribution_identity: str = OPTION_DISTRIBUTION_IDENTITY

    def __post_init__(self) -> None:
        if self.forward.mode != "exact" or self.forward.rho_prune != 0.0:
            raise ValueError("Exact partner posterior result cannot prune posterior mass.")
        if not _is_sha256(self.partner_registry_sha256):
            raise ValueError("Exact posterior must bind a frozen partner registry hash.")
        if not _is_sha256(self.option_library_sha256):
            raise ValueError("Exact posterior must bind a frozen option-library hash.")
        if not _is_sha256(self.ego_evidence_spec_sha256):
            raise ValueError("Exact posterior must bind frozen EgoEvidenceSpecV1.")
        if self.option_distribution_identity != OPTION_DISTRIBUTION_IDENTITY:
            raise ValueError("Exact posterior must bind shared option_distribution.")

    def full_state_sampler(self) -> CategoricalPosteriorFullStateSampler:
        """Return the theorem-aligned sampler over complete U=(theta, runtime state)."""

        return CategoricalPosteriorFullStateSampler(self.forward)


class ExactPartnerPosteriorBridgeV1:
    """Exact sparse posterior recursion using the generation policy distribution.

    The concrete ``ExactRegisteredPartnerKernelV1`` restores registered controllers
    and enumerates every hidden option draw compatible with the complete primitive
    action and public-state path. Custom hooks remain admissible only when they bind
    the same shared ``partner_pool.option_distribution`` identity. The bridge forbids
    pruning, merges encoded runtime states, and exposes complete-state sampling.
    """

    def __init__(
        self,
        *,
        registry: FrozenPartnerRegistryV1,
        option_library: Any,
        option_library_sha256: str,
        ego_evidence_spec: EgoEvidenceSpecV1,
        runtime_state_codec: StateCodec,
        public_state_codec: StateCodec,
        runtime_transition: ExactPartnerRuntimeTransition,
        evidence_likelihood: ExactPartnerEvidenceLikelihood,
    ) -> None:
        if not _is_sha256(option_library_sha256):
            raise ValueError("Exact posterior requires a frozen option-library SHA-256.")
        if getattr(runtime_transition, "contract_version", None) != OCV2_AUDIT_BRIDGE_VERSION:
            raise ValueError("Exact partner runtime transition hook version is missing or stale.")
        if getattr(evidence_likelihood, "contract_version", None) != OCV2_AUDIT_BRIDGE_VERSION:
            raise ValueError("Exact partner evidence likelihood hook version is missing or stale.")
        if getattr(runtime_transition, "shared_option_distribution", None) is not option_distribution:
            raise ValueError(
                "Exact runtime transition must bind shared option_distribution."
            )
        if getattr(evidence_likelihood, "shared_option_distribution", None) is not option_distribution:
            raise ValueError(
                "Exact evidence likelihood must bind shared option_distribution."
            )
        if not getattr(option_library, "options", None):
            raise ValueError("Exact posterior requires the frozen non-empty option library.")
        self.registry = registry
        self.option_library = option_library
        self.option_library_sha256 = str(option_library_sha256)
        self.ego_evidence_spec = ego_evidence_spec
        self.ego_evidence_spec_sha256 = ego_evidence_spec.sha256()
        self.runtime_state_codec = runtime_state_codec
        self.public_state_codec = public_state_codec
        self.runtime_transition = runtime_transition
        self.evidence_likelihood = evidence_likelihood

    def infer(
        self,
        *,
        initial_runtime_state_bytes_by_theta: Mapping[str, bytes],
        prior_by_theta: Mapping[str, float],
        observations: Sequence[PartnerEvidenceObservationV1],
    ) -> ExactPartnerPosteriorResultV1:
        theta_ids = tuple(entry.theta_id for entry in self.registry.entries)
        if set(initial_runtime_state_bytes_by_theta) != set(theta_ids):
            raise ValueError("Initial runtime states must exactly cover the frozen registry.")
        if set(prior_by_theta) != set(theta_ids):
            raise ValueError("Partner prior must exactly cover the frozen registry.")
        probabilities = []
        initial = []
        for theta_id in theta_ids:
            raw_probability = prior_by_theta[theta_id]
            if isinstance(raw_probability, (bool, np.bool_)):
                raise ValueError("Frozen partner prior mass cannot be boolean.")
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability <= 0.0:
                raise ValueError("Every frozen partner theta requires positive finite prior mass.")
            payload = initial_runtime_state_bytes_by_theta[theta_id]
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("Initial partner runtime states must be non-empty bytes.")
            _decode_round_trip(self.runtime_state_codec, payload, "initial_partner_runtime")
            probabilities.append(probability)
            initial.append(
                ForwardHypothesisV1(
                    theta_id=theta_id,
                    state_key=hashlib.sha256(payload).hexdigest(),
                    state_bytes=payload,
                    mass=probability,
                )
            )
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Frozen partner prior probabilities must sum to one.")

        def transition(
            state: SparseForwardStateV1,
            observation: PartnerEvidenceObservationV1,
            step_index: int,
        ) -> tuple[ForwardBranchV1, ...]:
            theta = self.registry.theta(state.theta_id)
            runtime_state = _decode_round_trip(
                self.runtime_state_codec, state.state_bytes, "partner_runtime"
            )
            if not isinstance(runtime_state, Mapping):
                raise TypeError("Shared option_distribution requires mapping runtime state.")
            if observation.evidence_spec_sha256 != self.ego_evidence_spec_sha256:
                raise ValueError(
                    "History row does not bind this frozen EgoEvidenceSpecV1."
                )
            expected_evidence_bytes = int(self.ego_evidence_spec.evidence_dim) * 4
            if len(observation.evidence_row_bytes) != expected_evidence_bytes:
                raise ValueError(
                    "History row byte length does not match frozen EgoEvidenceSpecV1."
                )
            evidence_row = np.frombuffer(
                observation.evidence_row_bytes,
                dtype=np.dtype("<f4"),
            ).copy()
            if not np.all(np.isfinite(evidence_row)):
                raise ValueError("Complete EgoEvidenceSpecV1 history row must be finite.")
            _validate_complete_evidence_row(self.ego_evidence_spec, evidence_row)
            option_choice_public_state = _decode_round_trip(
                self.public_state_codec,
                observation.option_choice_public_state_bytes,
                "option_choice_public_state",
            )
            decision_public_state = _decode_round_trip(
                self.public_state_codec,
                observation.decision_public_state_bytes,
                "decision_public_state",
            )
            intermediate_public_states = tuple(
                _decode_round_trip(
                    self.public_state_codec,
                    payload,
                    "intermediate_public_state",
                )
                for payload in observation.intermediate_public_state_bytes
            )
            duration = int(float(
                evidence_row[self.ego_evidence_spec.field_slices()["duration"]][0]
            ))
            if len(intermediate_public_states) != max(0, duration - 1):
                raise ValueError(
                    "Exact posterior evidence lacks its complete public-state path."
                )
            if duration == 0 and (
                observation.option_choice_public_state_bytes
                != observation.decision_public_state_bytes
            ):
                raise ValueError(
                    "Zero-duration exact evidence changed its public state."
                )
            public_state_path = (
                (option_choice_public_state,)
                if duration == 0
                else (
                    option_choice_public_state,
                    *intermediate_public_states,
                    decision_public_state,
                )
            )
            branches = tuple(
                self.runtime_transition.advance(
                    theta_id=state.theta_id,
                    theta_specification=theta.specification,
                    runtime_state=runtime_state,
                    runtime_state_bytes=state.state_bytes,
                    option_choice_public_state=option_choice_public_state,
                    decision_public_state=decision_public_state,
                    public_state_path=public_state_path,
                    evidence_row=evidence_row.copy(),
                    evidence_spec=self.ego_evidence_spec,
                    step_index=int(step_index),
                )
            )
            if not branches:
                raise ValueError("Exact runtime transition returned no state branch.")
            branch_total = math.fsum(
                float(branch.conditional_probability) for branch in branches
            )
            if not math.isclose(branch_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    "Exact runtime transition probabilities must sum to one."
                )
            emitted = []
            for branch in branches:
                raw_evidence_probability = self.evidence_likelihood.probability(
                    theta_id=state.theta_id,
                    theta_specification=theta.specification,
                    runtime_state=runtime_state,
                    next_runtime_state_bytes=branch.state_bytes,
                    option_choice_public_state=option_choice_public_state,
                    decision_public_state=decision_public_state,
                    public_state_path=public_state_path,
                    evidence_row=evidence_row.copy(),
                    evidence_spec=self.ego_evidence_spec,
                    option_library=self.option_library,
                    option_distribution_fn=option_distribution,
                    step_index=int(step_index),
                )
                if isinstance(raw_evidence_probability, (bool, np.bool_)):
                    raise ValueError("Exact evidence likelihood cannot return boolean.")
                evidence_probability = float(raw_evidence_probability)
                if (
                    not math.isfinite(evidence_probability)
                    or not 0.0 <= evidence_probability <= 1.0
                ):
                    raise ValueError(
                        "Exact evidence likelihood must return a probability in [0, 1]."
                    )
                probability = (
                    float(branch.conditional_probability) * evidence_probability
                )
                if probability == 0.0:
                    continue
                emitted.append(
                    ForwardBranchV1(
                        state_key=branch.state_key,
                        state_bytes=branch.state_bytes,
                        probability=probability,
                    )
                )
            return tuple(emitted)

        forward = sparse_state_merging_forward_recursion(
            initial,
            observations,
            transition,
            mode="exact",
            prune_below=0.0,
        )
        return ExactPartnerPosteriorResultV1(
            forward=forward,
            partner_registry_sha256=self.registry.sha256,
            option_library_sha256=self.option_library_sha256,
            ego_evidence_spec_sha256=self.ego_evidence_spec_sha256,
        )


def build_registered_exact_partner_posterior_bridge_v1(
    *,
    partner_set: str,
    option_library: Any,
    option_library_sha256: str,
    ego_evidence_spec: EgoEvidenceSpecV1,
    public_state_codec: StateCodec,
) -> ExactPartnerPosteriorBridgeV1:
    """Bind the production registry, controller state codec, and exact kernel."""

    runtime_state_codec = RegisteredPartnerStateCodecV1(public_state_codec)
    kernel = ExactRegisteredPartnerKernelV1(
        option_library=option_library,
        runtime_state_codec=runtime_state_codec,
    )
    return ExactPartnerPosteriorBridgeV1(
        registry=FrozenPartnerRegistryV1.from_registered(partner_set),
        option_library=option_library,
        option_library_sha256=option_library_sha256,
        ego_evidence_spec=ego_evidence_spec,
        runtime_state_codec=runtime_state_codec,
        public_state_codec=public_state_codec,
        runtime_transition=kernel,
        evidence_likelihood=kernel,
    )


def _encoded_copy(codec: StateCodec, value: Any, label: str) -> bytes:
    payload = codec.encode(value)
    if not isinstance(payload, bytes) or not payload:
        raise TypeError(f"{label} codec must return non-empty immutable bytes.")
    decoded = codec.decode(bytes(payload))
    if codec.encode(decoded) != payload:
        raise ValueError(f"{label} codec does not round-trip canonically.")
    return bytes(payload)


def _validate_complete_evidence_row(
    evidence_spec: EgoEvidenceSpecV1,
    evidence_row: np.ndarray,
) -> None:
    row = np.asarray(evidence_row, dtype=np.float32)
    if row.shape != (int(evidence_spec.evidence_dim),) or not np.all(np.isfinite(row)):
        raise ValueError("Complete evidence row violates its frozen shape or finiteness.")
    slices = evidence_spec.field_slices()
    action_shape = (
        int(evidence_spec.max_primitive_steps_per_decision),
        int(evidence_spec.primitive_action_vocab_dim),
    )
    action_indices = []
    for field_name in ("ego_primitive_actions", "partner_primitive_actions"):
        encoded = row[slices[field_name]].reshape(action_shape)
        if not np.all((encoded == 0.0) | (encoded == 1.0)) or not np.all(
            encoded.sum(axis=1) == 1.0
        ):
            raise ValueError(
                f"Complete evidence {field_name} must contain canonical one-hot rows."
            )
        action_indices.append(np.argmax(encoded, axis=1))
    duration_value = float(row[slices["duration"]][0])
    if (
        not duration_value.is_integer()
        or not 0 <= int(duration_value) <= int(evidence_spec.max_primitive_steps_per_decision)
    ):
        raise ValueError("Complete evidence duration is outside the frozen action window.")
    duration = int(duration_value)
    padding_id = int(evidence_spec.num_primitive_actions)
    for indices in action_indices:
        if np.any(indices[:duration] == padding_id) or np.any(
            indices[duration:] != padding_id
        ):
            raise ValueError(
                "Complete evidence action padding must begin exactly at duration."
            )
    option_code = row[slices["ego_option_id"]]
    if (
        not np.all((option_code == 0.0) | (option_code == 1.0))
        or float(option_code.sum()) != 1.0
    ):
        raise ValueError("Complete evidence ego option must be canonical one-hot.")
    option_id = int(np.argmax(option_code))
    if (duration == 0) != (option_id == int(evidence_spec.num_options)):
        raise ValueError("Complete evidence option padding must match zero duration.")
    valid_actions = row[slices["valid_actions"]]
    if not np.all((valid_actions == 0.0) | (valid_actions == 1.0)):
        raise ValueError("Complete evidence valid-action mask must be binary.")
    terminated = float(row[slices["terminated"]][0])
    truncated = float(row[slices["truncated"]][0])
    if terminated not in {0.0, 1.0} or truncated not in {0.0, 1.0}:
        raise ValueError("Complete evidence termination flags must be binary.")
    if terminated == 1.0 and truncated == 1.0:
        raise ValueError("Complete evidence cannot be both terminated and truncated.")
    if terminated == 0.0 and truncated == 0.0 and not bool(
        np.any(valid_actions == 1.0)
    ):
        raise ValueError("Complete non-terminal evidence requires a valid action.")


def _restore_canonical_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_canonical_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if set(value) == {"ndarray_dtype", "ndarray_shape", "ndarray_values"}:
        try:
            dtype = np.dtype(str(value["ndarray_dtype"]))
            shape = tuple(int(item) for item in value["ndarray_shape"])
            array = np.asarray(value["ndarray_values"], dtype=dtype).reshape(shape)
        except (TypeError, ValueError) as error:
            raise ValueError("Canonical audit ndarray payload is invalid.") from error
        return array.copy()
    return {
        str(key): _restore_canonical_value(item)
        for key, item in value.items()
    }


def _decode_round_trip(codec: StateCodec, payload: bytes, label: str) -> Any:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"Encoded {label} must be non-empty bytes.")
    decoded = codec.decode(bytes(payload))
    if codec.encode(decoded) != payload:
        raise ValueError(f"Encoded {label} failed canonical codec round-trip.")
    return decoded


def _rng_schedule_from_payload(payload: Any) -> RNGKeyScheduleV1:
    if not isinstance(payload, Mapping) or set(payload) != {
        "version",
        "manifest_seed",
        "original_replay_keys",
    }:
        raise ValueError("Audit bundle RNG schedule schema changed.")
    original = payload["original_replay_keys"]
    expected_original_keys = {
        "jax",
        "numpy",
        "python",
        "torch",
        "mode",
        "schedule_version",
        "coordinate",
    }
    if not isinstance(original, Mapping) or set(original) != expected_original_keys:
        raise ValueError("Audit bundle replay RNG schema changed.")
    coordinate = original["coordinate"]
    if not isinstance(coordinate, list):
        raise ValueError("Audit bundle replay RNG coordinate must be a list.")
    replay_keys = NamedRNGStreamsV1(
        jax=int(original["jax"]),
        numpy=int(original["numpy"]),
        python=int(original["python"]),
        torch=int(original["torch"]),
        mode=str(original["mode"]),  # type: ignore[arg-type]
        schedule_version=str(original["schedule_version"]),
        coordinate=tuple(str(item) for item in coordinate),
    )
    return RNGKeyScheduleV1(
        manifest_seed=int(payload["manifest_seed"]),
        original_replay_keys=replay_keys,
        version=str(payload["version"]),
    )


def _append_blob(output: bytearray, payload: bytes) -> None:
    output.extend(struct.pack(">Q", len(payload)))
    output.extend(payload)


def _read_u32(payload: bytes, cursor: int) -> tuple[int, int]:
    end = cursor + 4
    if end > len(payload):
        raise ValueError("Truncated OCV2 pytree payload.")
    return int(struct.unpack(">I", payload[cursor:end])[0]), end


def _read_u64(payload: bytes, cursor: int) -> tuple[int, int]:
    end = cursor + 8
    if end > len(payload):
        raise ValueError("Truncated OCV2 pytree payload.")
    return int(struct.unpack(">Q", payload[cursor:end])[0]), end


def _read_blob(payload: bytes, cursor: int) -> tuple[bytes, int]:
    length, cursor = _read_u64(payload, cursor)
    end = cursor + int(length)
    if end > len(payload):
        raise ValueError("Truncated OCV2 pytree blob.")
    return payload[cursor:end], end
