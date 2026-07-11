from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence


AUDIT_BATTERY_SCHEMA_VERSION = "path_c_audit_battery_v1"
INVALID_OPTION_POLICY = "single_noop_primitive_then_advance"
SUPPORT_VIOLATION_TOKEN = "support_violation"

ProbeScope = Literal["core", "design_only"]
AuditRole = Literal["design", "calibration", "locked_audit"]


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProbeScriptV1:
    """One frozen open-loop option script.

    ``sampling_probability`` is scoped to either the method-independent core table
    or the design-only table. Invalid options do not trigger an adaptive replacement:
    the registered intervention executes one noop primitive, records a support
    violation, and advances to the next requested option.
    """

    script_id: str
    option_ids: tuple[int, ...]
    sampling_probability: float
    scope: ProbeScope = "core"
    invalid_option_policy: str = INVALID_OPTION_POLICY
    support_violation_token: str = SUPPORT_VIOLATION_TOKEN

    def __post_init__(self) -> None:
        if not str(self.script_id).strip():
            raise ValueError("ProbeScriptV1.script_id must be non-empty.")
        if self.scope not in {"core", "design_only"}:
            raise ValueError("ProbeScriptV1.scope must be 'core' or 'design_only'.")
        if not self.option_ids:
            raise ValueError("ProbeScriptV1.option_ids must be non-empty.")
        if any(isinstance(value, bool) or int(value) < 0 for value in self.option_ids):
            raise ValueError("Probe option ids must be non-negative integers.")
        probability = float(self.sampling_probability)
        if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
            raise ValueError("Probe sampling_probability must be in (0, 1].")
        if self.invalid_option_policy != INVALID_OPTION_POLICY:
            raise ValueError(
                "Probe invalid-option behavior is frozen as one noop primitive followed "
                "by advancing the open-loop script."
            )
        if self.support_violation_token != SUPPORT_VIOLATION_TOKEN:
            raise ValueError("Probe support-violation token is part of the frozen intervention.")

    @property
    def T_probe(self) -> int:
        """Number of option decisions in this probe script."""

        return len(self.option_ids)

    def to_payload(self) -> dict[str, Any]:
        return {
            "script_id": str(self.script_id),
            "option_ids": [int(value) for value in self.option_ids],
            "sampling_probability": float(self.sampling_probability),
            "scope": str(self.scope),
            "invalid_option_policy": str(self.invalid_option_policy),
            "support_violation_token": str(self.support_violation_token),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProbeScriptV1":
        expected = {
            "script_id",
            "option_ids",
            "sampling_probability",
            "scope",
            "invalid_option_policy",
            "support_violation_token",
        }
        _reject_unknown_keys(payload, expected, "probe script")
        missing = sorted(expected.difference(payload))
        if missing:
            raise ValueError("Probe script is missing field(s): " + ", ".join(missing))
        raw_options = payload["option_ids"]
        if not isinstance(raw_options, (list, tuple)):
            raise ValueError("Probe script option_ids must be a list.")
        return cls(
            script_id=str(payload["script_id"]),
            option_ids=tuple(int(value) for value in raw_options),
            sampling_probability=float(payload["sampling_probability"]),
            scope=str(payload["scope"]),  # type: ignore[arg-type]
            invalid_option_policy=str(payload["invalid_option_policy"]),
            support_violation_token=str(payload["support_violation_token"]),
        )


@dataclass(frozen=True)
class FrozenAuditBatteryV1:
    """Versioned probe registry frozen before any kernel readout.

    ``T_probe`` is the number of option decisions in a script. ``L_inner`` is the
    number of conditionally repeated continuations for one posterior outer draw.
    Keeping these names separate prevents the two budgets from being conflated.
    """

    battery_id: str
    battery_version: str
    T_probe: int
    L_inner: int
    core_scripts: tuple[ProbeScriptV1, ...]
    design_only_scripts: tuple[ProbeScriptV1, ...] = ()
    schema_version: str = AUDIT_BATTERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_BATTERY_SCHEMA_VERSION:
            raise ValueError(
                f"Audit battery schema_version must be {AUDIT_BATTERY_SCHEMA_VERSION!r}."
            )
        if not str(self.battery_id).strip() or not str(self.battery_version).strip():
            raise ValueError("Audit battery id and version must be non-empty.")
        if isinstance(self.T_probe, bool) or int(self.T_probe) <= 0:
            raise ValueError("Audit battery T_probe must be positive.")
        if isinstance(self.L_inner, bool) or int(self.L_inner) <= 0:
            raise ValueError("Audit battery L_inner must be positive.")
        if not self.core_scripts:
            raise ValueError("Audit battery must contain at least one core probe script.")
        all_scripts = self.core_scripts + self.design_only_scripts
        ids = [script.script_id for script in all_scripts]
        if len(ids) != len(set(ids)):
            raise ValueError("Audit battery script ids must be globally unique.")
        for script in self.core_scripts:
            if script.scope != "core":
                raise ValueError("core_scripts may contain only scope='core' scripts.")
            self._validate_length(script)
        for script in self.design_only_scripts:
            if script.scope != "design_only":
                raise ValueError(
                    "design_only_scripts may contain only scope='design_only' scripts."
                )
            self._validate_length(script)
        _validate_probability_table(self.core_scripts, "core")
        if self.design_only_scripts:
            _validate_probability_table(self.design_only_scripts, "design_only")

    def _validate_length(self, script: ProbeScriptV1) -> None:
        if script.T_probe != int(self.T_probe):
            raise ValueError(
                f"Probe script {script.script_id!r} has T_probe={script.T_probe}; "
                f"battery requires {self.T_probe}."
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._unsigned_payload())

    def scripts_for_role(self, role: AuditRole) -> tuple[ProbeScriptV1, ...]:
        """Return the only scripts admissible for a data role.

        Design-only scripts can influence selection on the design split. They are
        mechanically excluded from calibration and the locked audit.
        """

        if role == "design":
            return self.core_scripts + self.design_only_scripts
        if role in {"calibration", "locked_audit"}:
            return self.core_scripts
        raise ValueError(f"Unsupported audit role: {role!r}")

    def sampling_table(self, scope: ProbeScope) -> dict[str, float]:
        scripts = self.core_scripts if scope == "core" else self.design_only_scripts
        return {
            script.script_id: float(script.sampling_probability)
            for script in scripts
        }

    def to_manifest(self) -> dict[str, Any]:
        payload = self._unsigned_payload()
        payload["sha256"] = self.sha256
        return payload

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "battery_id": str(self.battery_id),
            "battery_version": str(self.battery_version),
            "T_probe": int(self.T_probe),
            "L_inner": int(self.L_inner),
            "core_scripts": [script.to_payload() for script in self.core_scripts],
            "design_only_scripts": [
                script.to_payload() for script in self.design_only_scripts
            ],
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "FrozenAuditBatteryV1":
        expected = {
            "schema_version",
            "battery_id",
            "battery_version",
            "T_probe",
            "L_inner",
            "core_scripts",
            "design_only_scripts",
            "sha256",
        }
        _reject_unknown_keys(payload, expected, "audit battery")
        missing = sorted(expected.difference(payload))
        if missing:
            raise ValueError("Audit battery is missing field(s): " + ", ".join(missing))
        core = _script_tuple(payload["core_scripts"], "core_scripts")
        design = _script_tuple(payload["design_only_scripts"], "design_only_scripts")
        battery = cls(
            schema_version=str(payload["schema_version"]),
            battery_id=str(payload["battery_id"]),
            battery_version=str(payload["battery_version"]),
            T_probe=int(payload["T_probe"]),
            L_inner=int(payload["L_inner"]),
            core_scripts=core,
            design_only_scripts=design,
        )
        if str(payload["sha256"]) != battery.sha256:
            raise ValueError("Audit battery SHA-256 does not match its canonical contents.")
        return battery


@dataclass(frozen=True)
class ProbeStepDecisionV1:
    script_id: str
    step_index: int
    requested_option_id: int
    executed_option_id: int | None
    fallback_primitive_action: int | None
    support_violation: bool
    support_violation_token: str | None
    intervention_sha256: str


def resolve_probe_step(
    script: ProbeScriptV1,
    step_index: int,
    valid_option_ids: Iterable[int],
    *,
    noop_primitive_action: int,
) -> ProbeStepDecisionV1:
    """Resolve one open-loop request without adapting the probe intervention."""

    if isinstance(step_index, bool) or not 0 <= int(step_index) < script.T_probe:
        raise IndexError("Probe step_index is outside the frozen script.")
    valid = {int(value) for value in valid_option_ids}
    requested = int(script.option_ids[int(step_index)])
    support_violation = requested not in valid
    executed_option = None if support_violation else requested
    fallback = int(noop_primitive_action) if support_violation else None
    token = script.support_violation_token if support_violation else None
    intervention_payload = {
        "script": script.to_payload(),
        "step_index": int(step_index),
        "requested_option_id": requested,
        "invalid_option_policy": script.invalid_option_policy,
        "fallback_primitive_action": fallback,
        "support_violation_token": token,
    }
    return ProbeStepDecisionV1(
        script_id=script.script_id,
        step_index=int(step_index),
        requested_option_id=requested,
        executed_option_id=executed_option,
        fallback_primitive_action=fallback,
        support_violation=support_violation,
        support_violation_token=token,
        intervention_sha256=canonical_sha256(intervention_payload),
    )


def _validate_probability_table(
    scripts: Sequence[ProbeScriptV1],
    name: str,
) -> None:
    total = math.fsum(float(script.sampling_probability) for script in scripts)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Audit battery {name} sampling probabilities must sum to one.")


def _script_tuple(value: Any, name: str) -> tuple[ProbeScriptV1, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Audit battery {name} must be a list.")
    scripts = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Audit battery {name}[{index}] must be a mapping.")
        scripts.append(ProbeScriptV1.from_payload(item))
    return tuple(scripts)


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    unknown = sorted(set(map(str, payload)).difference(expected))
    if unknown:
        raise ValueError(f"Unknown {name} field(s): " + ", ".join(unknown))
