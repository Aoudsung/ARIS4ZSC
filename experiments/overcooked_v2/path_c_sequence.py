from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


EGO_EVIDENCE_SCHEMA_VERSION = "ego_evidence_spec_v1"
EPISODE_BOOTSTRAP_SCHEMA_VERSION = "episode_bootstrap_v1"

__all__ = (
    "EGO_EVIDENCE_SCHEMA_VERSION",
    "EPISODE_BOOTSTRAP_SCHEMA_VERSION",
    "EgoEvidenceSpecV1",
    "EvidenceBatch",
    "SequenceTDBatch",
    "EpisodeBootstrapRecord",
    "DecisionEvidenceBuffer",
    "EpisodeEvidenceBuffer",
    "EpisodeSequenceRecord",
    "collate_episode_records",
    "valid_action_dueling",
    "ensemble_diversity_telemetry",
    "RecurrentEnsembleQ",
    "SequenceTDLossOutput",
    "per_head_masked_mean",
    "sequence_td_loss",
)

_NUISANCE_KEY_FRAGMENTS = (
    "identity",
    "mechanism",
    "style",
    "fingerprint",
    "partner_id",
    "seed",
    "trajectory_source",
    "latent",
    "oracle",
    "protocol",
    "role_label",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_nuisance_keys(payload: Mapping[str, Any], *, context: str) -> None:
    for raw_key in payload:
        key = str(raw_key).strip().lower()
        if any(fragment in key for fragment in _NUISANCE_KEY_FRAGMENTS):
            raise ValueError(
                f"{context} contains forbidden nuisance field {raw_key!r}. "
                "EgoEvidenceSpecV1 accepts observable control evidence only."
            )


def _require_sha256(value: str | None, *, name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not a boolean.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, not a boolean.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


@dataclass(frozen=True)
class EgoEvidenceSpecV1:
    """Legacy option-level evidence contract for the predecessor instrument.

    This schema includes partner primitive actions and therefore is not the
    official-information contract for the decision-focused proposal or R015.
    Those formal paths use local observations plus the registered response
    summary and reject this schema. One encoded row describes the public state at
    an option-decision boundary and
    the complete primitive-action window that ended at that boundary. Exact action
    order is retained with an explicit padding token. Identity, mechanism, style,
    fingerprint, seed-group, and trajectory-source fields are not admissible.
    Encoding returns a CPU float32 tensor; device transfer happens only after
    episode collation.
    """

    observation_dim: int
    num_primitive_actions: int
    num_options: int
    max_primitive_steps_per_decision: int
    progress_event_dim: int
    observation_schema: str
    primitive_action_names: tuple[str, ...]
    option_names: tuple[str, ...]
    progress_event_names: tuple[str, ...]
    schema_version: str = EGO_EVIDENCE_SCHEMA_VERSION

    ENCODE_FIELDS: ClassVar[tuple[str, ...]] = (
        "observation",
        "ego_primitive_actions",
        "partner_primitive_actions",
        "ego_option_id",
        "duration",
        "reward",
        "progress_events",
        "valid_actions",
        "terminated",
        "truncated",
    )

    def __post_init__(self) -> None:
        if self.schema_version != EGO_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EGO_EVIDENCE_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}."
            )
        for name in (
            "observation_dim",
            "num_primitive_actions",
            "num_options",
            "max_primitive_steps_per_decision",
            "progress_event_dim",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        if not str(self.observation_schema).strip():
            raise ValueError("observation_schema must be non-empty.")
        if isinstance(self.primitive_action_names, (str, bytes)):
            raise ValueError("primitive_action_names must be a sequence of names.")
        if isinstance(self.option_names, (str, bytes)):
            raise ValueError("option_names must be a sequence of names.")
        if isinstance(self.progress_event_names, (str, bytes)):
            raise ValueError("progress_event_names must be a sequence of names.")
        object.__setattr__(
            self,
            "primitive_action_names",
            tuple(str(name) for name in self.primitive_action_names),
        )
        object.__setattr__(self, "option_names", tuple(str(name) for name in self.option_names))
        object.__setattr__(
            self,
            "progress_event_names",
            tuple(str(name) for name in self.progress_event_names),
        )
        for names, expected, label in (
            (self.primitive_action_names, self.num_primitive_actions, "primitive_action_names"),
            (self.option_names, self.num_options, "option_names"),
            (self.progress_event_names, self.progress_event_dim, "progress_event_names"),
        ):
            if len(names) != int(expected):
                raise ValueError(f"{label} must contain exactly {expected} names.")
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(f"{label} must contain unique, non-empty names.")
            for name in names:
                _reject_nuisance_keys({name: None}, context=label)
        _reject_nuisance_keys(
            {str(self.observation_schema): None},
            context="observation_schema",
        )

    @property
    def primitive_action_vocab_dim(self) -> int:
        return int(self.num_primitive_actions) + 1

    @property
    def option_vocab_dim(self) -> int:
        return int(self.num_options) + 1

    @property
    def evidence_dim(self) -> int:
        return int(
            self.observation_dim
            + 2
            * self.max_primitive_steps_per_decision
            * self.primitive_action_vocab_dim
            + self.option_vocab_dim
            + 2  # duration and reward
            + self.progress_event_dim
            + self.num_options
            + 2  # terminated and truncated
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_dim": int(self.observation_dim),
            "num_primitive_actions": int(self.num_primitive_actions),
            "num_options": int(self.num_options),
            "max_primitive_steps_per_decision": int(
                self.max_primitive_steps_per_decision
            ),
            "progress_event_dim": int(self.progress_event_dim),
            "observation_schema": str(self.observation_schema),
            "primitive_action_names": list(self.primitive_action_names),
            "option_names": list(self.option_names),
            "progress_event_names": list(self.progress_event_names),
            "encoding_order": list(self.ENCODE_FIELDS),
            "forbidden_nuisance_key_fragments": list(_NUISANCE_KEY_FRAGMENTS),
            "dtype": "float32",
            "reward_aggregation": (
                "sum_over_previous_primitive_window_plus_registered_probe_cost"
            ),
            "progress_event_aggregation": "sum_over_previous_primitive_window",
            "primitive_action_padding_id": int(self.num_primitive_actions),
            "option_padding_id": int(self.num_options),
            "evidence_dim": int(self.evidence_dim),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EgoEvidenceSpecV1":
        if not isinstance(payload, Mapping):
            raise TypeError("EgoEvidenceSpecV1 payload must be a mapping.")
        _reject_nuisance_keys(payload, context="EgoEvidenceSpecV1 payload")
        constructor_fields = {field.name for field in fields(cls)}
        derived_fields = {
            "encoding_order",
            "primitive_action_padding_id",
            "option_padding_id",
            "evidence_dim",
            "forbidden_nuisance_key_fragments",
            "dtype",
            "reward_aggregation",
            "progress_event_aggregation",
        }
        unknown = set(payload) - constructor_fields - derived_fields
        if unknown:
            raise ValueError(
                "EgoEvidenceSpecV1 rejects unknown field(s): "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        missing = constructor_fields - set(payload)
        if missing:
            raise ValueError(
                "EgoEvidenceSpecV1 is missing field(s): "
                + ", ".join(sorted(missing))
            )
        spec = cls(**{name: payload[name] for name in constructor_fields})
        if "encoding_order" in payload and tuple(payload["encoding_order"]) != cls.ENCODE_FIELDS:
            raise ValueError("EgoEvidenceSpecV1 encoding_order does not match version 1.")
        if "forbidden_nuisance_key_fragments" in payload and tuple(
            payload["forbidden_nuisance_key_fragments"]
        ) != _NUISANCE_KEY_FRAGMENTS:
            raise ValueError("EgoEvidenceSpecV1 nuisance-field policy does not match version 1.")
        if "dtype" in payload and str(payload["dtype"]) != "float32":
            raise ValueError("EgoEvidenceSpecV1 dtype must be float32.")
        expected_aggregation = {
            "reward_aggregation": (
                "sum_over_previous_primitive_window_plus_registered_probe_cost"
            ),
            "progress_event_aggregation": "sum_over_previous_primitive_window",
        }
        for name, expected in expected_aggregation.items():
            if name in payload and str(payload[name]) != expected:
                raise ValueError(f"EgoEvidenceSpecV1 {name} does not match version 1.")
        for name, expected in (
            ("primitive_action_padding_id", spec.num_primitive_actions),
            ("option_padding_id", spec.num_options),
            ("evidence_dim", spec.evidence_dim),
        ):
            if name in payload and int(payload[name]) != int(expected):
                raise ValueError(f"EgoEvidenceSpecV1 derived field {name} is inconsistent.")
        return spec

    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def field_slices(self) -> dict[str, slice]:
        cursor = 0
        out: dict[str, slice] = {}

        def take(name: str, width: int) -> None:
            nonlocal cursor
            out[name] = slice(cursor, cursor + int(width))
            cursor += int(width)

        take("observation", self.observation_dim)
        action_width = (
            self.max_primitive_steps_per_decision * self.primitive_action_vocab_dim
        )
        take("ego_primitive_actions", action_width)
        take("partner_primitive_actions", action_width)
        take("ego_option_id", self.option_vocab_dim)
        take("duration", 1)
        take("reward", 1)
        take("progress_events", self.progress_event_dim)
        take("valid_actions", self.num_options)
        take("terminated", 1)
        take("truncated", 1)
        if cursor != self.evidence_dim:
            raise RuntimeError("EgoEvidenceSpecV1 field layout is internally inconsistent.")
        return out

    def encode_mapping(self, payload: Mapping[str, Any]) -> torch.Tensor:
        if not isinstance(payload, Mapping):
            raise TypeError("Decision evidence must be supplied as a mapping.")
        _reject_nuisance_keys(payload, context="Decision evidence")
        unknown = set(payload) - set(self.ENCODE_FIELDS)
        missing = set(self.ENCODE_FIELDS) - set(payload)
        if unknown:
            raise ValueError(
                "Decision evidence rejects unknown field(s): "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        if missing:
            raise ValueError(
                "Decision evidence is missing field(s): "
                + ", ".join(sorted(missing))
            )
        return self.encode_decision(**{name: payload[name] for name in self.ENCODE_FIELDS})

    def encode_decision(
        self,
        *,
        observation: Any,
        ego_primitive_actions: Sequence[int] | torch.Tensor,
        partner_primitive_actions: Sequence[int] | torch.Tensor,
        ego_option_id: int | None,
        duration: int,
        reward: float,
        progress_events: Any,
        valid_actions: Any,
        terminated: bool,
        truncated: bool,
    ) -> torch.Tensor:
        observation_tensor = _finite_vector(
            observation,
            self.observation_dim,
            name="observation",
        )
        ego_actions = _action_list(ego_primitive_actions, name="ego_primitive_actions")
        partner_actions = _action_list(
            partner_primitive_actions,
            name="partner_primitive_actions",
        )
        if len(ego_actions) != len(partner_actions):
            raise ValueError(
                "ego_primitive_actions and partner_primitive_actions must have equal length."
            )
        if len(ego_actions) > self.max_primitive_steps_per_decision:
            raise ValueError(
                "Primitive-action window exceeds max_primitive_steps_per_decision; "
                "refusing to truncate observable history."
            )
        if not float(duration).is_integer() or int(duration) != len(ego_actions):
            raise ValueError(
                "duration must equal the number of paired primitive-action steps."
            )
        if int(duration) < 0:
            raise ValueError("duration must be non-negative.")
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite.")
        terminated_bool = _strict_bool_scalar(terminated, name="terminated")
        truncated_bool = _strict_bool_scalar(truncated, name="truncated")
        if terminated_bool and truncated_bool:
            raise ValueError("A decision boundary cannot be both terminated and truncated.")

        progress = _finite_vector(
            progress_events,
            self.progress_event_dim,
            name="progress_events",
        )
        valid = _strict_bool_tensor(
            valid_actions,
            name="valid_actions",
            device="cpu",
        ).reshape(-1)
        if valid.numel() != self.num_options:
            raise ValueError(
                f"valid_actions must contain {self.num_options} entries; got {valid.numel()}."
            )
        if not terminated_bool and not bool(valid.any()):
            raise ValueError("A non-terminal decision state must expose a valid action.")

        ego_action_code = self._encode_action_window(ego_actions)
        partner_action_code = self._encode_action_window(partner_actions)
        option_code = torch.zeros(self.option_vocab_dim, dtype=torch.float32)
        if ego_option_id is not None and (
            not math.isfinite(float(ego_option_id))
            or not float(ego_option_id).is_integer()
        ):
            raise ValueError("ego_option_id must be an integer or None.")
        option_index = self.num_options if ego_option_id is None else int(ego_option_id)
        if not 0 <= option_index <= self.num_options:
            raise ValueError(
                f"ego_option_id must be in [0,{self.num_options - 1}] or None."
            )
        if (int(duration) == 0) != (ego_option_id is None):
            raise ValueError("ego_option_id must be None exactly at the initial zero-duration row.")
        option_code[option_index] = 1.0

        encoded = torch.cat(
            (
                observation_tensor,
                ego_action_code,
                partner_action_code,
                option_code,
                torch.tensor([float(duration), float(reward)], dtype=torch.float32),
                progress,
                valid.to(dtype=torch.float32),
                torch.tensor(
                    [float(terminated_bool), float(truncated_bool)],
                    dtype=torch.float32,
                ),
            ),
            dim=0,
        )
        if encoded.shape != (self.evidence_dim,):
            raise RuntimeError(
                f"Encoded evidence has shape {tuple(encoded.shape)}; "
                f"expected {(self.evidence_dim,)}."
            )
        return encoded

    def extract_valid_actions(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.shape[-1] != self.evidence_dim:
            raise ValueError(
                f"Evidence last dimension must be {self.evidence_dim}; "
                f"got {evidence.shape[-1]}."
            )
        return evidence[..., self.field_slices()["valid_actions"]].bool()

    def extract_option_id(self, evidence: torch.Tensor) -> int | None:
        row = self.validate_encoded(evidence)
        option_code = row[self.field_slices()["ego_option_id"]]
        option_id = int(option_code.argmax().item())
        return None if option_id == self.num_options else option_id

    def extract_reward(self, evidence: torch.Tensor) -> float:
        row = self.validate_encoded(evidence)
        return float(row[self.field_slices()["reward"]].item())

    def extract_boundary_flags(self, evidence: torch.Tensor) -> tuple[bool, bool]:
        row = self.validate_encoded(evidence)
        terminated = bool(row[self.field_slices()["terminated"]].item())
        truncated = bool(row[self.field_slices()["truncated"]].item())
        return terminated, truncated

    def validate_encoded(self, evidence: torch.Tensor) -> torch.Tensor:
        """Validate one encoded row without silently repairing its contents."""

        row = torch.as_tensor(evidence, dtype=torch.float32).reshape(-1).detach().cpu()
        if row.shape != (self.evidence_dim,):
            raise ValueError(
                f"Encoded evidence must have shape {(self.evidence_dim,)}; "
                f"got {tuple(row.shape)}."
            )
        if not bool(torch.isfinite(row).all()):
            raise ValueError("Encoded evidence must contain only finite values.")
        slices = self.field_slices()
        duration_value = float(row[slices["duration"]].item())
        if (
            not duration_value.is_integer()
            or not 0 <= int(duration_value) <= self.max_primitive_steps_per_decision
        ):
            raise ValueError("Encoded duration is invalid.")
        duration = int(duration_value)

        for field_name in ("ego_primitive_actions", "partner_primitive_actions"):
            code = row[slices[field_name]].reshape(
                self.max_primitive_steps_per_decision,
                self.primitive_action_vocab_dim,
            )
            if not _is_one_hot(code):
                raise ValueError(f"Encoded {field_name} must be one-hot at every step.")
            ids = code.argmax(dim=-1)
            padding_id = self.num_primitive_actions
            if bool((ids[:duration] == padding_id).any()) or bool(
                (ids[duration:] != padding_id).any()
            ):
                raise ValueError(f"Encoded {field_name} padding disagrees with duration.")

        option_code = row[slices["ego_option_id"]]
        if not _is_one_hot(option_code.unsqueeze(0)):
            raise ValueError("Encoded ego_option_id must be one-hot.")
        option_id = int(option_code.argmax().item())
        if (duration == 0) != (option_id == self.num_options):
            raise ValueError("Encoded ego option and duration disagree.")

        valid_values = row[slices["valid_actions"]]
        if not _contains_only_zero_one(valid_values):
            raise ValueError("Encoded valid_actions must contain only 0/1 values.")
        terminated_value = row[slices["terminated"]]
        truncated_value = row[slices["truncated"]]
        if not _contains_only_zero_one(terminated_value) or not _contains_only_zero_one(
            truncated_value
        ):
            raise ValueError("Encoded boundary flags must contain only 0/1 values.")
        terminated = bool(terminated_value.item())
        truncated = bool(truncated_value.item())
        if terminated and truncated:
            raise ValueError("Encoded state cannot be both terminal and truncated.")
        if not terminated and not bool(valid_values.bool().any()):
            raise ValueError("Encoded non-terminal state must expose a valid action.")
        return row

    def _encode_action_window(self, actions: Sequence[int]) -> torch.Tensor:
        padding_id = int(self.num_primitive_actions)
        encoded = torch.zeros(
            self.max_primitive_steps_per_decision,
            self.primitive_action_vocab_dim,
            dtype=torch.float32,
        )
        encoded[:, padding_id] = 1.0
        for step, raw_action in enumerate(actions):
            action = int(raw_action)
            if not 0 <= action < self.num_primitive_actions:
                raise ValueError(
                    f"Primitive action {action} is outside [0,{self.num_primitive_actions - 1}]."
                )
            encoded[step].zero_()
            encoded[step, action] = 1.0
        return encoded.reshape(-1)


def _finite_vector(value: Any, size: int, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).detach().cpu()
    if tensor.numel() != int(size):
        raise ValueError(f"{name} must contain {size} values; got {tensor.numel()}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _contains_only_zero_one(tensor: torch.Tensor) -> bool:
    return bool(((tensor == 0.0) | (tensor == 1.0)).all())


def _is_one_hot(tensor: torch.Tensor) -> bool:
    return _contains_only_zero_one(tensor) and bool((tensor.sum(dim=-1) == 1.0).all())


def _strict_bool_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    raw = torch.as_tensor(value, device=device)
    if raw.dtype == torch.bool:
        return raw
    numeric = raw.to(dtype=torch.float64)
    if not bool(torch.isfinite(numeric).all()) or bool(
        ((numeric != 0.0) & (numeric != 1.0)).any()
    ):
        raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return numeric.bool()


def _strict_bool_scalar(value: Any, *, name: str) -> bool:
    tensor = _strict_bool_tensor(value, name=name).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar boolean.")
    return bool(tensor.item())


def _action_list(value: Sequence[int] | torch.Tensor, *, name: str) -> list[int]:
    raw = torch.as_tensor(value).reshape(-1).detach().cpu()
    if raw.numel() == 0:
        return []
    numeric = raw.to(dtype=torch.float64)
    if not bool(torch.isfinite(numeric).all()):
        raise ValueError(f"{name} must contain finite integer action ids.")
    if not bool(torch.equal(numeric, numeric.round())):
        raise ValueError(f"{name} must contain integer action ids.")
    tensor = numeric.to(dtype=torch.long)
    return [int(item) for item in tensor.tolist()]


@dataclass(frozen=True)
class EvidenceBatch:
    """Padded sequence of decision-state evidence.

    ``evidence`` is ``[B,S,D]`` and ``valid_actions`` is ``[B,S,A]``. ``S`` is
    the number of decision states, which is one larger than the number of TD
    transitions in a :class:`SequenceTDBatch`.
    """

    evidence: torch.Tensor
    valid_actions: torch.Tensor
    lengths: torch.Tensor
    spec_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, torch.Tensor) or self.evidence.ndim != 3:
            raise ValueError("evidence must be a tensor with shape [B,S,D].")
        if self.evidence.dtype != torch.float32:
            raise TypeError("evidence must use the frozen float32 dtype.")
        valid = _strict_bool_tensor(
            self.valid_actions,
            name="valid_actions",
            device=self.evidence.device,
        )
        lengths = torch.as_tensor(
            self.lengths,
            dtype=torch.long,
            device=self.evidence.device,
        )
        object.__setattr__(self, "valid_actions", valid)
        object.__setattr__(self, "lengths", lengths)
        if valid.ndim != 3:
            raise ValueError("valid_actions must have shape [B,S,A].")
        if valid.shape[:2] != self.evidence.shape[:2]:
            raise ValueError("evidence and valid_actions must share [B,S].")
        if valid.shape[-1] <= 0 or self.evidence.shape[-1] <= 0:
            raise ValueError("Evidence and action dimensions must be positive.")
        if lengths.shape != (self.evidence.shape[0],):
            raise ValueError("lengths must have shape [B].")
        if bool((lengths < 1).any()) or bool((lengths > self.evidence.shape[1]).any()):
            raise ValueError("Each evidence length must be in [1,S].")
        _require_sha256(self.spec_sha256, name="spec_sha256")
        mask = self.time_mask
        if not bool(torch.isfinite(self.evidence[mask]).all()):
            raise ValueError("Active evidence rows must contain finite values.")
        if bool(valid[~mask].any()):
            raise ValueError("Padded evidence rows must not expose valid actions.")

    @property
    def time_mask(self) -> torch.Tensor:
        steps = torch.arange(self.evidence.shape[1], device=self.evidence.device)
        return steps.unsqueeze(0) < self.lengths.unsqueeze(1)

    def to(self, device: torch.device | str) -> "EvidenceBatch":
        return EvidenceBatch(
            evidence=self.evidence.to(device),
            valid_actions=self.valid_actions.to(device),
            lengths=self.lengths.to(device),
            spec_sha256=self.spec_sha256,
        )


@dataclass(frozen=True)
class SequenceTDBatch:
    """Full-episode semi-Markov TD batch with one bootstrap mask per episode.

    ``rewards`` must already be the Bellman immediate return on the registered
    reward scale, including any step or probe cost. No auxiliary target is added
    by :func:`sequence_td_loss`.
    """

    evidence: EvidenceBatch
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    dones: torch.Tensor
    lengths: torch.Tensor
    bootstrap_mask: torch.Tensor
    truncated: torch.Tensor | None = None
    episode_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceBatch):
            raise TypeError("evidence must be an EvidenceBatch.")
        device = self.evidence.evidence.device
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(
            self.rewards,
            dtype=self.evidence.evidence.dtype,
            device=device,
        )
        discounts = torch.as_tensor(
            self.discounts,
            dtype=self.evidence.evidence.dtype,
            device=device,
        )
        dones = _strict_bool_tensor(self.dones, name="dones", device=device)
        lengths = torch.as_tensor(self.lengths, dtype=torch.long, device=device)
        bootstrap_mask = _strict_bool_tensor(
            self.bootstrap_mask,
            name="bootstrap_mask",
            device=device,
        )
        truncated = (
            torch.zeros_like(dones)
            if self.truncated is None
            else _strict_bool_tensor(self.truncated, name="truncated", device=device)
        )
        for name, value in (
            ("actions", actions),
            ("rewards", rewards),
            ("discounts", discounts),
            ("dones", dones),
            ("lengths", lengths),
            ("bootstrap_mask", bootstrap_mask),
            ("truncated", truncated),
        ):
            object.__setattr__(self, name, value)

        batch_size, state_steps = self.evidence.evidence.shape[:2]
        transition_steps = state_steps - 1
        expected = (batch_size, transition_steps)
        for name, tensor in (
            ("actions", actions),
            ("rewards", rewards),
            ("discounts", discounts),
            ("dones", dones),
            ("truncated", truncated),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must have shape {expected}; got {tuple(tensor.shape)}.")
        if transition_steps <= 0:
            raise ValueError("SequenceTDBatch requires at least one TD transition.")
        if lengths.shape != (batch_size,):
            raise ValueError("lengths must have shape [B].")
        if bool((lengths < 1).any()) or bool((lengths > transition_steps).any()):
            raise ValueError("Each transition length must be in [1,S-1].")
        if not torch.equal(self.evidence.lengths, lengths + 1):
            raise ValueError("Evidence lengths must equal transition lengths plus one.")
        if bootstrap_mask.ndim != 2 or bootstrap_mask.shape[0] != batch_size:
            raise ValueError("bootstrap_mask must have shape [B,K].")
        if bootstrap_mask.shape[1] <= 0:
            raise ValueError("bootstrap_mask must contain at least one head.")
        if self.episode_ids and len(self.episode_ids) != batch_size:
            raise ValueError("episode_ids must be empty or contain one id per batch row.")
        if bool((dones & truncated).any()):
            raise ValueError("A transition cannot be both terminal and truncated.")
        if not bool(torch.isfinite(rewards[self.transition_mask]).all()):
            raise ValueError("Active rewards must be finite.")
        active_discounts = discounts[self.transition_mask]
        if not bool(torch.isfinite(active_discounts).all()):
            raise ValueError("Active discounts must be finite.")
        if bool((active_discounts < 0.0).any()) or bool((active_discounts > 1.0).any()):
            raise ValueError("SMDP discounts must be in [0,1].")

        time = torch.arange(transition_steps, device=device).unsqueeze(0)
        is_last = time == (lengths - 1).unsqueeze(1)
        if bool(((dones | truncated) & self.transition_mask & ~is_last).any()):
            raise ValueError("Terminal or truncated flags may occur only at the last transition.")
        if bool((actions[~self.transition_mask] != -1).any()):
            raise ValueError("Padded actions must use -1.")
        if bool(rewards[~self.transition_mask].ne(0).any()) or bool(
            discounts[~self.transition_mask].ne(0).any()
        ):
            raise ValueError("Padded rewards and discounts must be zero.")
        if bool(dones[~self.transition_mask].any()) or bool(truncated[~self.transition_mask].any()):
            raise ValueError("Padded transitions must not carry boundary flags.")

        n_actions = self.evidence.valid_actions.shape[-1]
        invalid_active_action = self.transition_mask & (
            (actions < 0) | (actions >= n_actions)
        )
        if bool(invalid_active_action.any()):
            raise ValueError(f"Active actions must be in [0,{n_actions - 1}].")
        safe_actions = actions.clamp(min=0, max=n_actions - 1)
        chosen_valid = self.evidence.valid_actions[:, :-1].gather(
            -1,
            safe_actions.unsqueeze(-1),
        ).squeeze(-1)
        if bool((self.transition_mask & ~chosen_valid).any()):
            raise ValueError("Every active action must be valid in its current state.")
        next_has_action = self.evidence.valid_actions[:, 1:].any(dim=-1)
        needs_bootstrap = self.transition_mask & ~dones
        if bool((needs_bootstrap & ~next_has_action).any()):
            raise ValueError("Every non-terminal next state must expose a valid action.")

    @property
    def transition_mask(self) -> torch.Tensor:
        steps = torch.arange(self.actions.shape[1], device=self.actions.device)
        return steps.unsqueeze(0) < self.lengths.unsqueeze(1)

    @property
    def n_heads(self) -> int:
        return int(self.bootstrap_mask.shape[1])

    @property
    def current_evidence(self) -> torch.Tensor:
        return self.evidence.evidence[:, :-1]

    @property
    def next_evidence(self) -> torch.Tensor:
        return self.evidence.evidence[:, 1:]

    @property
    def valid_actions(self) -> torch.Tensor:
        return self.evidence.valid_actions[:, :-1]

    @property
    def next_valid_actions(self) -> torch.Tensor:
        return self.evidence.valid_actions[:, 1:]

    def to(self, device: torch.device | str) -> "SequenceTDBatch":
        return SequenceTDBatch(
            evidence=self.evidence.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            discounts=self.discounts.to(device),
            dones=self.dones.to(device),
            lengths=self.lengths.to(device),
            bootstrap_mask=self.bootstrap_mask.to(device),
            truncated=self.truncated.to(device),
            episode_ids=self.episode_ids,
        )


@dataclass(frozen=True)
class EpisodeBootstrapRecord:
    """Deterministic episode-level bootstrap assignment from a dedicated stream."""

    episode_id: str
    mask: tuple[bool, ...]
    bootstrap_p: float
    draw_seed: int
    schema_version: str = EPISODE_BOOTSTRAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EPISODE_BOOTSTRAP_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EPISODE_BOOTSTRAP_SCHEMA_VERSION!r}."
            )
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be non-empty.")
        if isinstance(self.mask, (str, bytes)):
            raise ValueError("Episode bootstrap mask must be a boolean sequence.")
        object.__setattr__(self, "mask", tuple(self.mask))
        if not self.mask:
            raise ValueError("Episode bootstrap mask must contain at least one head.")
        if any(type(value) is not bool for value in self.mask):
            raise ValueError("Episode bootstrap mask entries must be booleans.")
        if not 0.0 < float(self.bootstrap_p) <= 1.0:
            raise ValueError("bootstrap_p must be in (0,1].")
        object.__setattr__(
            self,
            "draw_seed",
            _nonnegative_int(self.draw_seed, name="draw_seed"),
        )

    @classmethod
    def sample(
        cls,
        *,
        episode_id: str,
        n_heads: int,
        bootstrap_p: float,
        manifest_seed: int,
        ensure_nonempty: bool = True,
    ) -> "EpisodeBootstrapRecord":
        n_heads = _positive_int(n_heads, name="n_heads")
        if not 0.0 < float(bootstrap_p) <= 1.0:
            raise ValueError("bootstrap_p must be in (0,1].")
        manifest_seed = _nonnegative_int(manifest_seed, name="manifest_seed")
        material = (
            f"{EPISODE_BOOTSTRAP_SCHEMA_VERSION}\0{manifest_seed}\0{episode_id}"
        ).encode("utf-8")
        draw_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(draw_seed)
        mask_tensor = torch.rand(n_heads, generator=generator) < float(bootstrap_p)
        if ensure_nonempty and not bool(mask_tensor.any()):
            selected = int(torch.randint(n_heads, (1,), generator=generator).item())
            mask_tensor[selected] = True
        return cls(
            episode_id=str(episode_id),
            mask=tuple(bool(value) for value in mask_tensor.tolist()),
            bootstrap_p=float(bootstrap_p),
            draw_seed=int(draw_seed),
        )

    def as_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.mask, dtype=torch.bool, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "mask": [bool(value) for value in self.mask],
            "bootstrap_p": float(self.bootstrap_p),
            "draw_seed": int(self.draw_seed),
        }

    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


class DecisionEvidenceBuffer:
    """Accumulate the exact paired primitive trace for one option decision."""

    def __init__(self, spec: EgoEvidenceSpecV1):
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self._ego_actions: list[int] = []
        self._partner_actions: list[int] = []
        self._reward = 0.0
        self._progress = torch.zeros(self.spec.progress_event_dim, dtype=torch.float32)

    @property
    def primitive_steps(self) -> int:
        return len(self._ego_actions)

    def append_primitive(
        self,
        ego_action: int,
        partner_action: int,
        *,
        reward: float = 0.0,
        progress_event: Any | None = None,
    ) -> None:
        if self.primitive_steps >= self.spec.max_primitive_steps_per_decision:
            raise ValueError(
                "Primitive-action window exceeds EgoEvidenceSpecV1; refusing to truncate."
            )
        for name, action in (("ego_action", ego_action), ("partner_action", partner_action)):
            if (
                not math.isfinite(float(action))
                or not float(action).is_integer()
                or not 0 <= int(action) < self.spec.num_primitive_actions
            ):
                raise ValueError(
                    f"{name} must be in [0,{self.spec.num_primitive_actions - 1}]."
                )
        if not math.isfinite(float(reward)):
            raise ValueError("Primitive reward must be finite.")
        self._ego_actions.append(int(ego_action))
        self._partner_actions.append(int(partner_action))
        self._reward += float(reward)
        if progress_event is not None:
            self._progress += _finite_vector(
                progress_event,
                self.spec.progress_event_dim,
                name="progress_event",
            )

    def encode_boundary(
        self,
        *,
        observation: Any,
        ego_option_id: int | None,
        valid_actions: Any,
        terminated: bool = False,
        truncated: bool = False,
    ) -> torch.Tensor:
        return self.spec.encode_decision(
            observation=observation,
            ego_primitive_actions=self._ego_actions,
            partner_primitive_actions=self._partner_actions,
            ego_option_id=ego_option_id,
            duration=self.primitive_steps,
            reward=self._reward,
            progress_events=self._progress,
            valid_actions=valid_actions,
            terminated=terminated,
            truncated=truncated,
        )


@dataclass(frozen=True)
class EpisodeSequenceRecord:
    episode_id: str
    evidence_spec_sha256: str
    evidence: torch.Tensor
    valid_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor
    bootstrap: EpisodeBootstrapRecord

    def __post_init__(self) -> None:
        if not self.evidence_spec_sha256:
            raise ValueError("evidence_spec_sha256 is required for an episode record.")
        _require_sha256(self.evidence_spec_sha256, name="evidence_spec_sha256")
        if self.bootstrap.episode_id != self.episode_id:
            raise ValueError("Bootstrap record episode_id does not match episode record.")
        state_steps = int(self.evidence.shape[0])
        batch = SequenceTDBatch(
            evidence=EvidenceBatch(
                self.evidence.unsqueeze(0),
                self.valid_actions.unsqueeze(0),
                torch.tensor([state_steps], dtype=torch.long),
                spec_sha256=self.evidence_spec_sha256,
            ),
            actions=self.actions.unsqueeze(0),
            rewards=self.rewards.unsqueeze(0),
            discounts=self.discounts.unsqueeze(0),
            dones=self.dones.unsqueeze(0),
            truncated=self.truncated.unsqueeze(0),
            lengths=torch.tensor([state_steps - 1], dtype=torch.long),
            bootstrap_mask=self.bootstrap.as_tensor().unsqueeze(0),
            episode_ids=(self.episode_id,),
        )
        del batch

    @property
    def transition_length(self) -> int:
        return int(self.actions.shape[0])


class EpisodeEvidenceBuffer:
    """Build one ordered episode without allowing transition-level resampling."""

    def __init__(
        self,
        spec: EgoEvidenceSpecV1,
        bootstrap: EpisodeBootstrapRecord,
    ):
        self.spec = spec
        self.bootstrap = bootstrap
        self._evidence: list[torch.Tensor] = []
        self._valid_actions: list[torch.Tensor] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._discounts: list[float] = []
        self._dones: list[bool] = []
        self._truncated: list[bool] = []
        self._closed = False

    @property
    def episode_id(self) -> str:
        return self.bootstrap.episode_id

    @property
    def started(self) -> bool:
        return bool(self._evidence)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def transition_length(self) -> int:
        return len(self._actions)

    def evidence_batch(
        self,
        device: torch.device | str | None = None,
    ) -> EvidenceBatch:
        """Return the complete decision prefix used for online action selection.

        The prefix starts at the episode's initial decision state and therefore
        resets recurrent state only at a real episode boundary.  No rolling
        window or transition-level reconstruction is permitted here.
        """

        if not self.started:
            raise RuntimeError("EpisodeEvidenceBuffer has not been started.")
        evidence = torch.stack(self._evidence, dim=0).unsqueeze(0)
        valid_actions = torch.stack(self._valid_actions, dim=0).unsqueeze(0)
        batch = EvidenceBatch(
            evidence=evidence,
            valid_actions=valid_actions,
            lengths=torch.tensor([len(self._evidence)], dtype=torch.long),
            spec_sha256=self.spec.sha256(),
        )
        return batch if device is None else batch.to(device)

    def start(self, evidence: torch.Tensor, valid_actions: Any) -> None:
        if self.started:
            raise RuntimeError("EpisodeEvidenceBuffer.start() may be called only once.")
        row, valid = self._validate_state(evidence, valid_actions)
        terminated, truncated = self.spec.extract_boundary_flags(row)
        if terminated or truncated or self.spec.extract_option_id(row) is not None:
            raise ValueError("Initial episode evidence must be an open pre-action state.")
        if not math.isclose(self.spec.extract_reward(row), 0.0, abs_tol=1.0e-8):
            raise ValueError("Initial episode evidence must have zero previous reward.")
        progress = row[self.spec.field_slices()["progress_events"]]
        if bool(progress.ne(0.0).any()):
            raise ValueError("Initial episode evidence must have no previous progress events.")
        self._evidence.append(row)
        self._valid_actions.append(valid)

    def append_transition(
        self,
        *,
        action: int,
        reward: float,
        discount: float,
        done: bool,
        truncated: bool,
        next_evidence: torch.Tensor,
        next_valid_actions: Any,
    ) -> None:
        if not self.started:
            raise RuntimeError("EpisodeEvidenceBuffer must be started before append_transition().")
        if self.closed:
            raise RuntimeError("Cannot append to a closed episode.")
        done_bool = _strict_bool_scalar(done, name="done")
        truncated_bool = _strict_bool_scalar(truncated, name="truncated")
        if done_bool and truncated_bool:
            raise ValueError("A transition cannot be both terminal and truncated.")
        current_valid = self._valid_actions[-1]
        if (
            not math.isfinite(float(action))
            or not float(action).is_integer()
            or not 0 <= int(action) < self.spec.num_options
            or not bool(current_valid[int(action)])
        ):
            raise ValueError("action must select a currently valid option.")
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite.")
        if not math.isfinite(float(discount)) or not 0.0 <= float(discount) <= 1.0:
            raise ValueError("discount must be a finite SMDP discount in [0,1].")
        row, valid = self._validate_state(next_evidence, next_valid_actions)
        encoded_done, encoded_truncated = self.spec.extract_boundary_flags(row)
        if encoded_done != done_bool or encoded_truncated != truncated_bool:
            raise ValueError("Encoded boundary flags disagree with transition metadata.")
        if self.spec.extract_option_id(row) != int(action):
            raise ValueError("Encoded ego option disagrees with the executed action.")
        if not math.isclose(
            self.spec.extract_reward(row),
            float(reward),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError("Encoded reward disagrees with transition metadata.")
        if not done_bool and not bool(valid.any()):
            raise ValueError("A non-terminal next state must expose a valid option.")
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._discounts.append(float(discount))
        self._dones.append(done_bool)
        self._truncated.append(truncated_bool)
        self._evidence.append(row)
        self._valid_actions.append(valid)
        self._closed = done_bool or truncated_bool

    def to_record(self, *, require_closed: bool = True) -> EpisodeSequenceRecord:
        if not self._actions:
            raise ValueError("EpisodeEvidenceBuffer contains no transitions.")
        if require_closed and not self.closed:
            raise ValueError("Full-episode replay requires an explicit terminal or truncated boundary.")
        return EpisodeSequenceRecord(
            episode_id=self.episode_id,
            evidence_spec_sha256=self.spec.sha256(),
            evidence=torch.stack(self._evidence, dim=0),
            valid_actions=torch.stack(self._valid_actions, dim=0),
            actions=torch.tensor(self._actions, dtype=torch.long),
            rewards=torch.tensor(self._rewards, dtype=torch.float32),
            discounts=torch.tensor(self._discounts, dtype=torch.float32),
            dones=torch.tensor(self._dones, dtype=torch.bool),
            truncated=torch.tensor(self._truncated, dtype=torch.bool),
            bootstrap=self.bootstrap,
        )

    def _validate_state(
        self,
        evidence: torch.Tensor,
        valid_actions: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.spec.validate_encoded(evidence)
        valid = _strict_bool_tensor(
            valid_actions,
            name="valid_actions",
            device="cpu",
        ).reshape(-1)
        if valid.shape != (self.spec.num_options,):
            raise ValueError(
                f"valid_actions must have shape {(self.spec.num_options,)}; "
                f"got {tuple(valid.shape)}."
            )
        encoded_valid = self.spec.extract_valid_actions(row)
        if not torch.equal(valid, encoded_valid):
            raise ValueError("Explicit valid_actions disagree with the encoded evidence row.")
        return row.clone(), valid.clone()


def collate_episode_records(records: Sequence[EpisodeSequenceRecord]) -> SequenceTDBatch:
    if not records:
        raise ValueError("At least one episode record is required.")
    spec_hashes = {record.evidence_spec_sha256 for record in records}
    if len(spec_hashes) != 1:
        raise ValueError("Cannot collate episodes encoded with different evidence specs.")
    evidence_dims = {int(record.evidence.shape[-1]) for record in records}
    action_dims = {int(record.valid_actions.shape[-1]) for record in records}
    head_counts = {len(record.bootstrap.mask) for record in records}
    if len(evidence_dims) != 1 or len(action_dims) != 1 or len(head_counts) != 1:
        raise ValueError("Episode records must share evidence, action, and head dimensions.")

    batch_size = len(records)
    max_transitions = max(record.transition_length for record in records)
    max_states = max_transitions + 1
    evidence_dim = next(iter(evidence_dims))
    n_actions = next(iter(action_dims))
    n_heads = next(iter(head_counts))

    evidence = torch.zeros(batch_size, max_states, evidence_dim, dtype=torch.float32)
    valid_actions = torch.zeros(batch_size, max_states, n_actions, dtype=torch.bool)
    actions = torch.full((batch_size, max_transitions), -1, dtype=torch.long)
    rewards = torch.zeros(batch_size, max_transitions, dtype=torch.float32)
    discounts = torch.zeros(batch_size, max_transitions, dtype=torch.float32)
    dones = torch.zeros(batch_size, max_transitions, dtype=torch.bool)
    truncated = torch.zeros(batch_size, max_transitions, dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)
    bootstrap_mask = torch.zeros(batch_size, n_heads, dtype=torch.bool)

    for row, record in enumerate(records):
        transition_length = record.transition_length
        state_length = transition_length + 1
        evidence[row, :state_length] = record.evidence
        valid_actions[row, :state_length] = record.valid_actions
        actions[row, :transition_length] = record.actions
        rewards[row, :transition_length] = record.rewards
        discounts[row, :transition_length] = record.discounts
        dones[row, :transition_length] = record.dones
        truncated[row, :transition_length] = record.truncated
        lengths[row] = transition_length
        bootstrap_mask[row] = record.bootstrap.as_tensor()

    spec_sha256 = next(iter(spec_hashes))
    return SequenceTDBatch(
        evidence=EvidenceBatch(
            evidence=evidence,
            valid_actions=valid_actions,
            lengths=lengths + 1,
            spec_sha256=spec_sha256,
        ),
        actions=actions,
        rewards=rewards,
        discounts=discounts,
        dones=dones,
        truncated=truncated,
        lengths=lengths,
        bootstrap_mask=bootstrap_mask,
        episode_ids=tuple(record.episode_id for record in records),
    )


def valid_action_dueling(
    value: torch.Tensor,
    advantage: torch.Tensor,
    valid_actions: torch.Tensor,
    *,
    invalid_action_value: float = -torch.inf,
) -> torch.Tensor:
    """Combine dueling heads while centering only over valid actions."""

    if not math.isinf(float(invalid_action_value)) or float(invalid_action_value) >= 0.0:
        raise ValueError("invalid_action_value must be negative infinity.")
    if not isinstance(value, torch.Tensor) or not isinstance(advantage, torch.Tensor):
        raise TypeError("value and advantage must be tensors.")
    if not torch.is_floating_point(value) or not torch.is_floating_point(advantage):
        raise TypeError("value and advantage must use floating-point dtypes.")
    if value.device != advantage.device or value.dtype != advantage.dtype:
        raise ValueError("value and advantage must share device and dtype.")
    if value.shape != advantage.shape[:-1] + (1,):
        raise ValueError(
            "value must have the same leading dimensions as advantage and end in 1."
        )
    valid = _strict_bool_tensor(
        valid_actions,
        name="valid_actions",
        device=advantage.device,
    )
    while valid.ndim < advantage.ndim:
        valid = valid.unsqueeze(-2)
    try:
        valid = torch.broadcast_to(valid, advantage.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_actions shape {tuple(valid_actions.shape)} cannot broadcast to "
            f"advantage shape {tuple(advantage.shape)}."
        ) from exc
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(advantage).all()):
        raise ValueError("Dueling value and advantage logits must be finite.")
    valid_float = valid.to(dtype=advantage.dtype)
    valid_count = valid_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
    valid_mean = (advantage * valid_float).sum(dim=-1, keepdim=True) / valid_count
    q_values = value + advantage - valid_mean
    return q_values.masked_fill(~valid, float(invalid_action_value))


def ensemble_diversity_telemetry(
    q_values: torch.Tensor,
    valid_actions: torch.Tensor,
    *,
    prior_q_values: torch.Tensor | None = None,
    prior_scale: float = 0.0,
) -> dict[str, Any]:
    """Measure ensemble diversity on the valid normalized-advantage support.

    ``q_values`` has shape ``[..., K, A]`` and ``valid_actions`` has shape
    ``[..., A]``.  Invalid and padded actions never enter the calculation.  The
    effective rank is the covariance participation ratio across heads; the
    prior ratio compares the root-mean-square scaled prior contribution with
    the root-mean-square final normalized-advantage code.
    """

    if not isinstance(q_values, torch.Tensor) or not torch.is_floating_point(q_values):
        raise TypeError("q_values must be a floating-point tensor.")
    if q_values.ndim < 2:
        raise ValueError("q_values must have shape [..., K, A].")
    if q_values.shape[-2] <= 0 or q_values.shape[-1] <= 0:
        raise ValueError("q_values must contain at least one head and one action.")
    if not math.isfinite(float(prior_scale)) or float(prior_scale) < 0.0:
        raise ValueError("prior_scale must be finite and non-negative.")
    if float(prior_scale) > 0.0 and prior_q_values is None:
        raise ValueError(
            "A positive prior_scale requires the unscaled prior_q_values telemetry."
        )
    valid = _strict_bool_tensor(
        valid_actions,
        name="valid_actions",
        device=q_values.device,
    )
    expected_valid_shape = q_values.shape[:-2] + (q_values.shape[-1],)
    if valid.shape != expected_valid_shape:
        raise ValueError(
            "valid_actions must have shape [..., A] matching q_values; "
            f"expected {tuple(expected_valid_shape)}, got {tuple(valid.shape)}."
        )
    if not bool(valid.any()):
        raise ValueError("At least one valid action is required for telemetry.")
    expanded_valid = valid.unsqueeze(-2).expand_as(q_values)
    if not bool(torch.isfinite(q_values[expanded_valid]).all()):
        raise ValueError("Q values on the valid action support must be finite.")

    def _normalized_valid_vectors(values: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~expanded_valid, -torch.inf)
        best = masked.max(dim=-1, keepdim=True).values
        normalized = torch.where(expanded_valid, values - best, torch.zeros_like(values))
        head_first = normalized.movedim(-2, 0).reshape(values.shape[-2], -1)
        return head_first[:, valid.reshape(-1)]

    vectors = _normalized_valid_vectors(q_values)
    n_heads = int(vectors.shape[0])
    support_size = int(vectors.shape[1])
    centered = vectors - vectors.mean(dim=1, keepdim=True)
    covariance = centered @ centered.transpose(0, 1)
    covariance = covariance / float(max(support_size - 1, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    eigen_sum = eigenvalues.sum()
    eigen_square_sum = eigenvalues.square().sum()
    if float(eigen_square_sum.detach().cpu().item()) == 0.0:
        effective_rank = 0.0
    else:
        effective_rank = min(
            float(n_heads),
            max(
                0.0,
                float(
                    (eigen_sum.square() / eigen_square_sum).detach().cpu().item()
                ),
            ),
        )

    variances = covariance.diagonal().clamp_min(0.0)
    denominator = torch.sqrt(variances[:, None] * variances[None, :])
    correlation = torch.where(
        denominator > 0.0,
        covariance / denominator.clamp_min(torch.finfo(covariance.dtype).tiny),
        torch.zeros_like(covariance),
    ).clamp(min=-1.0, max=1.0)
    if n_heads > 1:
        off_diagonal = ~torch.eye(n_heads, dtype=torch.bool, device=q_values.device)
        head_correlation_mean = float(
            correlation[off_diagonal].mean().detach().cpu().item()
        )
        head_correlation_absolute_mean = float(
            correlation[off_diagonal].abs().mean().detach().cpu().item()
        )
    else:
        head_correlation_mean = None
        head_correlation_absolute_mean = None

    final_rms = float(vectors.square().mean().sqrt().detach().cpu().item())
    prior_scaled_rms = 0.0
    prior_contribution_ratio = 0.0
    if prior_q_values is not None:
        if not isinstance(prior_q_values, torch.Tensor):
            raise TypeError("prior_q_values must be a tensor when supplied.")
        if prior_q_values.shape != q_values.shape:
            raise ValueError("prior_q_values must have the same shape as q_values.")
        if prior_q_values.device != q_values.device or prior_q_values.dtype != q_values.dtype:
            raise ValueError("prior_q_values must share q_values device and dtype.")
        if not bool(torch.isfinite(prior_q_values[expanded_valid]).all()):
            raise ValueError("Prior Q values on the valid action support must be finite.")
        prior_vectors = _normalized_valid_vectors(prior_q_values)
        prior_scaled_rms = float(
            (float(prior_scale) * prior_vectors).square().mean().sqrt().detach().cpu().item()
        )
        if final_rms > 0.0:
            prior_contribution_ratio = float(prior_scaled_rms / final_rms)

    return {
        "definition": "valid_normalized_advantage_covariance_participation_ratio_v1",
        "n_heads": n_heads,
        "support_size": support_size,
        "effective_rank": effective_rank,
        "head_correlation_matrix": correlation.detach().cpu().tolist(),
        "head_correlation_mean": head_correlation_mean,
        "head_correlation_absolute_mean": head_correlation_absolute_mean,
        "prior_scale": float(prior_scale),
        "prior_scaled_rms": prior_scaled_rms,
        "final_normalized_advantage_rms": final_rms,
        "prior_contribution_ratio": prior_contribution_ratio,
    }


class RecurrentEnsembleQ(nn.Module):
    """Head-specific recurrent ensemble with an independent frozen prior."""

    def __init__(
        self,
        evidence_dim: int,
        n_actions: int,
        *,
        n_heads: int = 1,
        encoder_dim: int = 128,
        recurrent_dim: int = 128,
        prior_scale: float = 0.0,
        prior_encoder_dim: int | None = None,
        prior_recurrent_dim: int | None = None,
        prior_seed: int = 0,
        evidence_spec_sha256: str | None = None,
        invalid_action_value: float = -torch.inf,
    ):
        super().__init__()
        resolved_dimensions = {}
        for name, value in (
            ("evidence_dim", evidence_dim),
            ("n_actions", n_actions),
            ("n_heads", n_heads),
            ("encoder_dim", encoder_dim),
            ("recurrent_dim", recurrent_dim),
        ):
            resolved_dimensions[name] = _positive_int(value, name=name)
        if not math.isfinite(float(prior_scale)) or float(prior_scale) < 0.0:
            raise ValueError("prior_scale must be non-negative.")
        prior_seed = _nonnegative_int(prior_seed, name="prior_seed")
        if not math.isinf(float(invalid_action_value)) or float(invalid_action_value) >= 0.0:
            raise ValueError("invalid_action_value must be negative infinity.")
        _require_sha256(evidence_spec_sha256, name="evidence_spec_sha256")

        self.evidence_dim = resolved_dimensions["evidence_dim"]
        self.n_actions = resolved_dimensions["n_actions"]
        self.n_heads = resolved_dimensions["n_heads"]
        self.encoder_dim = resolved_dimensions["encoder_dim"]
        self.recurrent_dim = resolved_dimensions["recurrent_dim"]
        self.prior_scale = float(prior_scale)
        self.prior_seed = prior_seed
        self.evidence_spec_sha256 = evidence_spec_sha256
        self.invalid_action_value = float(invalid_action_value)

        self.shared_encoder = nn.Sequential(
            nn.Linear(self.evidence_dim, self.encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(self.encoder_dim),
        )
        self.head_grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.encoder_dim,
                    hidden_size=self.recurrent_dim,
                    batch_first=True,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.value_heads = nn.ModuleList(
            [nn.Linear(self.recurrent_dim, 1) for _ in range(self.n_heads)]
        )
        self.advantage_heads = nn.ModuleList(
            [nn.Linear(self.recurrent_dim, self.n_actions) for _ in range(self.n_heads)]
        )

        self.prior_encoder: nn.Module | None = None
        self.prior_grus = nn.ModuleList()
        self.prior_value_heads = nn.ModuleList()
        self.prior_advantage_heads = nn.ModuleList()
        self.prior_encoder_dim = _positive_int(
            (
                max(1, self.encoder_dim // 2)
                if prior_encoder_dim is None
                else prior_encoder_dim
            ),
            name="prior_encoder_dim",
        )
        self.prior_recurrent_dim = _positive_int(
            (
                max(1, self.recurrent_dim // 2)
                if prior_recurrent_dim is None
                else prior_recurrent_dim
            ),
            name="prior_recurrent_dim",
        )
        if self.prior_scale > 0.0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.prior_seed)
                self.prior_encoder = nn.Sequential(
                    nn.Linear(self.evidence_dim, self.prior_encoder_dim),
                    nn.Tanh(),
                )
                self.prior_grus = nn.ModuleList(
                    [
                        nn.GRU(
                            input_size=self.prior_encoder_dim,
                            hidden_size=self.prior_recurrent_dim,
                            batch_first=True,
                        )
                        for _ in range(self.n_heads)
                    ]
                )
                self.prior_value_heads = nn.ModuleList(
                    [nn.Linear(self.prior_recurrent_dim, 1) for _ in range(self.n_heads)]
                )
                self.prior_advantage_heads = nn.ModuleList(
                    [
                        nn.Linear(self.prior_recurrent_dim, self.n_actions)
                        for _ in range(self.n_heads)
                    ]
                )
            for module in self._prior_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()

    @property
    def representation_dim(self) -> int:
        return self.n_heads * self.recurrent_dim

    def train(self, mode: bool = True) -> "RecurrentEnsembleQ":
        super().train(mode)
        for module in self._prior_modules():
            module.eval()
        return self

    def forward(self, batch: EvidenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_sequence(batch)

    def forward_mean_sequence(self, batch: EvidenceBatch) -> torch.Tensor:
        q_values, _ = self.forward_sequence(batch)
        return q_values.mean(dim=2)

    def forward_sequence(
        self,
        batch: EvidenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q ``[B,S,K,A]`` and recurrent state ``Z[B,S,K*H]``.

        ``Z`` is exposed for diagnostics and baseline interfaces. It is not named
        the retained quotient; the primary Path C decision code is normalized
        advantage computed from the returned Q tensor.
        """

        self._validate_batch(batch)
        encoded = self.shared_encoder(batch.evidence)
        learned_states = []
        learned_q = []
        for head_idx, gru in enumerate(self.head_grus):
            states, _ = gru(encoded)
            learned_states.append(states)
            learned_q.append(
                valid_action_dueling(
                    self.value_heads[head_idx](states),
                    self.advantage_heads[head_idx](states),
                    batch.valid_actions,
                    invalid_action_value=self.invalid_action_value,
                )
            )
        q_values = torch.stack(learned_q, dim=2)
        if self.prior_scale > 0.0:
            prior_q = self.forward_prior_sequence(batch)
            valid = batch.valid_actions.unsqueeze(2)
            q_values = (q_values + self.prior_scale * prior_q).masked_fill(
                ~valid,
                self._mask_value(q_values.dtype),
            )

        state_stack = torch.stack(learned_states, dim=2)
        representation = state_stack.reshape(
            state_stack.shape[0],
            state_stack.shape[1],
            -1,
        )
        time_mask = batch.time_mask
        q_values = q_values.masked_fill(~time_mask[:, :, None, None], 0.0)
        representation = representation.masked_fill(~time_mask[:, :, None], 0.0)
        return q_values, representation

    def forward_prior_sequence(self, batch: EvidenceBatch) -> torch.Tensor:
        """Return the frozen prior contribution before multiplying by prior_scale."""

        self._validate_batch(batch)
        if self.prior_encoder is None:
            return batch.evidence.new_zeros(
                batch.evidence.shape[0],
                batch.evidence.shape[1],
                self.n_heads,
                self.n_actions,
            )
        with torch.no_grad():
            encoded = self.prior_encoder(batch.evidence)
            prior_q = []
            for head_idx, gru in enumerate(self.prior_grus):
                states, _ = gru(encoded)
                prior_q.append(
                    valid_action_dueling(
                        self.prior_value_heads[head_idx](states),
                        self.prior_advantage_heads[head_idx](states),
                        batch.valid_actions,
                        invalid_action_value=self.invalid_action_value,
                    )
                )
            values = torch.stack(prior_q, dim=2)
            return values.masked_fill(~batch.time_mask[:, :, None, None], 0.0)

    def prior_state_sha256(self) -> str | None:
        if self.prior_encoder is None:
            return None
        digest = hashlib.sha256()
        for prefix, module in (
            ("encoder", self.prior_encoder),
            ("grus", self.prior_grus),
            ("value_heads", self.prior_value_heads),
            ("advantage_heads", self.prior_advantage_heads),
        ):
            for name, tensor in sorted(module.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode("utf-8"))
                digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_same_fixed_prior(self, other: "RecurrentEnsembleQ") -> None:
        if not isinstance(other, RecurrentEnsembleQ):
            raise TypeError("Fixed-prior comparison requires RecurrentEnsembleQ models.")
        if self.prior_scale != other.prior_scale:
            raise ValueError("Online and target prior_scale values differ.")
        if self.prior_scale > 0.0 and self.prior_state_sha256() != other.prior_state_sha256():
            raise ValueError("Online and target networks do not use the same fixed prior.")

    def _validate_batch(self, batch: EvidenceBatch) -> None:
        if not isinstance(batch, EvidenceBatch):
            raise TypeError("forward_sequence expects an EvidenceBatch.")
        if batch.evidence.shape[-1] != self.evidence_dim:
            raise ValueError(
                f"Evidence dimension must be {self.evidence_dim}; "
                f"got {batch.evidence.shape[-1]}."
            )
        if batch.valid_actions.shape[-1] != self.n_actions:
            raise ValueError(
                f"Action dimension must be {self.n_actions}; "
                f"got {batch.valid_actions.shape[-1]}."
            )
        if (
            self.evidence_spec_sha256 is not None
            and batch.spec_sha256 != self.evidence_spec_sha256
        ):
            raise ValueError("EvidenceBatch does not match the model's frozen evidence spec.")

    def _prior_modules(self) -> tuple[nn.Module, ...]:
        if self.prior_encoder is None:
            return ()
        return (
            self.prior_encoder,
            self.prior_grus,
            self.prior_value_heads,
            self.prior_advantage_heads,
        )

    def _mask_value(self, dtype: torch.dtype) -> float:
        del dtype
        return self.invalid_action_value


@dataclass(frozen=True)
class SequenceTDLossOutput:
    loss: torch.Tensor
    per_head_loss: torch.Tensor
    head_support: torch.Tensor
    supported_heads: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor


def per_head_masked_mean(
    losses: torch.Tensor,
    transition_mask: torch.Tensor,
    bootstrap_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each head by its own active episode/time support."""

    if losses.ndim != 3:
        raise ValueError("losses must have shape [B,T,K].")
    transition_mask = torch.as_tensor(
        transition_mask,
        dtype=torch.bool,
        device=losses.device,
    )
    bootstrap_mask = torch.as_tensor(
        bootstrap_mask,
        dtype=torch.bool,
        device=losses.device,
    )
    if transition_mask.shape != losses.shape[:2]:
        raise ValueError("transition_mask must have shape [B,T].")
    if bootstrap_mask.shape != (losses.shape[0], losses.shape[2]):
        raise ValueError("bootstrap_mask must have shape [B,K].")
    mask = transition_mask.unsqueeze(-1) & bootstrap_mask.unsqueeze(1)
    support = mask.sum(dim=(0, 1))
    supported = support > 0
    totals = (losses * mask.to(dtype=losses.dtype)).sum(dim=(0, 1))
    per_head = torch.where(
        supported,
        totals / support.clamp(min=1).to(dtype=losses.dtype),
        torch.zeros_like(totals),
    )
    return per_head, support, supported


def sequence_td_core(
    q_online_all: torch.Tensor,
    q_target_all: torch.Tensor,
    *,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discounts: torch.Tensor,
    n_heads: int,
    td_loss: str = "huber",
    huber_delta: float = 1.0,
    double_q: bool = True,
    vmax: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared double-Q sequence TD math over aligned Q traces ``[B,S,K,A]``.

    This is the single home of the TD target/prediction arithmetic; both the
    option-level ``sequence_td_loss`` and the standard-path primitive loss
    delegate here so a correction cannot reach one training regime and silently
    miss the other. Returns ``(predictions, targets, elementwise_loss)``.
    """

    safe_actions = actions.clamp(min=0)
    gather_index = safe_actions[:, :, None, None].expand(-1, -1, int(n_heads), 1)
    predictions = q_online_all[:, :-1].gather(-1, gather_index).squeeze(-1)

    with torch.no_grad():
        q_target_next = q_target_all[:, 1:]
        if double_q:
            next_actions = q_online_all[:, 1:].argmax(dim=-1)
            next_values = q_target_next.gather(
                -1,
                next_actions.unsqueeze(-1),
            ).squeeze(-1)
        else:
            next_values = q_target_next.max(dim=-1).values
        next_values = torch.where(
            dones.unsqueeze(-1),
            torch.zeros_like(next_values),
            next_values,
        )
        targets = rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * next_values
        if vmax is not None:
            targets = targets.clamp(min=-float(vmax), max=float(vmax))

    if td_loss == "mse":
        elementwise = F.mse_loss(predictions, targets, reduction="none")
    elif td_loss == "huber":
        if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive when td_loss='huber'.")
        elementwise = F.smooth_l1_loss(
            predictions,
            targets,
            beta=float(huber_delta),
            reduction="none",
        )
    else:
        raise ValueError("td_loss must be one of {'huber', 'mse'}.")
    return predictions, targets, elementwise


def sequence_td_loss(
    q_net: RecurrentEnsembleQ,
    target_q_net: RecurrentEnsembleQ,
    batch: SequenceTDBatch,
    *,
    td_loss: str = "huber",
    huber_delta: float = 1.0,
    double_q: bool = True,
    vmax: float | None = None,
    fail_on_zero_support_head: bool = False,
    require_matching_prior: bool = True,
    return_details: bool = False,
) -> torch.Tensor | SequenceTDLossOutput:
    """Full-sequence double-Q TD loss with episode-level head bootstrap.

    ``discounts`` is consumed verbatim and must already contain the semi-Markov
    factor ``gamma ** option_duration``. Online and target networks each unroll
    the complete state sequence once; next-state values are obtained by a one-step
    shift on those aligned traces.
    """

    if not isinstance(batch, SequenceTDBatch):
        raise TypeError("batch must be a SequenceTDBatch.")
    if q_net.n_heads != target_q_net.n_heads or q_net.n_actions != target_q_net.n_actions:
        raise ValueError("Online and target ensemble dimensions must match.")
    if batch.n_heads != q_net.n_heads:
        raise ValueError(
            f"Batch bootstrap mask has {batch.n_heads} heads; model has {q_net.n_heads}."
        )
    if require_matching_prior:
        q_net.assert_same_fixed_prior(target_q_net)
    if vmax is not None and (
        not math.isfinite(float(vmax)) or float(vmax) <= 0.0
    ):
        raise ValueError("vmax must be positive when supplied.")

    q_online_all, _ = q_net.forward_sequence(batch.evidence)
    with torch.no_grad():
        q_target_all, _ = target_q_net.forward_sequence(batch.evidence)
    expected_q_shape = (
        batch.evidence.evidence.shape[0],
        batch.evidence.evidence.shape[1],
        q_net.n_heads,
        q_net.n_actions,
    )
    if q_online_all.shape != expected_q_shape or q_target_all.shape != expected_q_shape:
        raise ValueError(
            "forward_sequence must return Q[B,S,K,A] matching the batch dimensions."
        )

    predictions, targets, elementwise = sequence_td_core(
        q_online_all,
        q_target_all,
        actions=batch.actions,
        rewards=batch.rewards,
        dones=batch.dones,
        discounts=batch.discounts,
        n_heads=q_net.n_heads,
        td_loss=td_loss,
        huber_delta=huber_delta,
        double_q=double_q,
        vmax=vmax,
    )

    per_head, support, supported = per_head_masked_mean(
        elementwise,
        batch.transition_mask,
        batch.bootstrap_mask,
    )
    if not bool(supported.any()):
        raise ValueError("No ensemble head has TD support in this batch.")
    if fail_on_zero_support_head and not bool(supported.all()):
        missing = torch.nonzero(~supported, as_tuple=False).reshape(-1).tolist()
        raise ValueError(f"Ensemble head(s) have zero TD support: {missing}.")
    loss = per_head[supported].mean()
    output = SequenceTDLossOutput(
        loss=loss,
        per_head_loss=per_head,
        head_support=support,
        supported_heads=supported,
        predictions=predictions,
        targets=targets,
    )
    return output if return_details else output.loss
