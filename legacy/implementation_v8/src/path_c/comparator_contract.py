"""Run-independent scientific contract for the frozen history comparator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .continuation import RAW_REWARD_DEFINITION
from .experiment import OFFICIAL_SOURCE_COMMIT, PartnerManifest, RunConfig


def _hash_rows(rows: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparatorContract:
    layout: str
    observation_contract: str
    view_radius: int
    reward_definition: str
    gamma: float
    continuation_horizon: int
    probe_steps: int
    signature_distance_threshold: float
    task_match_epsilon: float
    official_source_commit: str
    reference_policy_set_hash: str
    fit_partner_parent_hash: str
    validation_partner_parent_hash: str

    def __post_init__(self) -> None:
        if not self.layout or not self.observation_contract:
            raise ValueError("Comparator layout and observation contract are required.")
        if self.reward_definition != RAW_REWARD_DEFINITION:
            raise ValueError("Comparator reward definition is not the DEPI estimand.")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("Comparator gamma must be in (0, 1].")
        if min(int(self.continuation_horizon), int(self.probe_steps)) <= 0:
            raise ValueError("Comparator horizon and probe length must be positive.")
        if float(self.signature_distance_threshold) <= 0.0:
            raise ValueError("Comparator signature threshold must be positive.")
        if float(self.task_match_epsilon) <= 0.0:
            raise ValueError("Comparator task-match epsilon must be positive.")
        if self.official_source_commit != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("Comparator Official source commit differs.")
        for digest in (
            self.reference_policy_set_hash,
            self.fit_partner_parent_hash,
            self.validation_partner_parent_hash,
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Comparator lineage hashes must be SHA-256 values.")

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ComparatorContract":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Comparator contract schema differs.")
        return cls(**dict(payload))


def comparator_contract_for_run(
    *,
    config: RunConfig,
    manifest: PartnerManifest,
    reference_policy_set_hash: str,
) -> ComparatorContract:
    """Project only comparator-relevant fields from a run configuration."""

    def parent_hash(role: str) -> str:
        runs = manifest.by_role(role)
        return _hash_rows(
            [
                f"{run.parent_training_run_id}:{run.checkpoint_sha256}"
                for run in runs
            ]
        )

    return ComparatorContract(
        layout=config.environment.layout,
        observation_contract=(
            "official_ego_local_observation_l1_l5:"
            f"view_radius={config.environment.agent_view_size}:"
            "task_planes_v1:instant_partner_v1"
        ),
        view_radius=int(config.environment.agent_view_size),
        reward_definition=RAW_REWARD_DEFINITION,
        gamma=float(config.ppo.gamma),
        continuation_horizon=int(config.anchors.continuation_horizon),
        probe_steps=int(config.anchors.probe_steps),
        signature_distance_threshold=float(
            config.anchors.signature_distance_threshold
        ),
        task_match_epsilon=float(
            getattr(config.anchors, "task_match_epsilon", 2.0)
        ),
        official_source_commit=OFFICIAL_SOURCE_COMMIT,
        reference_policy_set_hash=str(reference_policy_set_hash),
        fit_partner_parent_hash=parent_hash("comparator_fit"),
        validation_partner_parent_hash=parent_hash("comparator_validation"),
    )


def comparator_reference_policy_set_hash(
    *, reference_checkpoint_sha256: str, manifest: PartnerManifest
) -> str:
    checkpoint_hashes = sorted(
        {
            run.checkpoint_sha256
            for role in ("comparator_fit", "comparator_validation")
            for run in manifest.by_role(role)
        }
    )
    return _hash_rows([str(reference_checkpoint_sha256), *checkpoint_hashes])


__all__ = [
    "ComparatorContract",
    "comparator_contract_for_run",
    "comparator_reference_policy_set_hash",
]
