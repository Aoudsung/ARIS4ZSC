"""Published-protocol SP/XP evaluation for standard Path C checkpoints."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
import torch.nn as nn

from experiments.overcooked_v2.batched_rollout import BatchedEnvPool
from experiments.overcooked_v2.path_c_backbone_ppo import sample_actor_actions
from experiments.overcooked_v2.path_c_seed import (
    derive_ocv2_execution_seed,
    validate_unique_execution_seed_mapping,
)
from experiments.overcooked_v2.path_c_standard import (
    PrimitiveRecurrentEnsembleQ,
    StandardEnvConfig,
    choose_primitive_actions,
    load_standard_checkpoint,
    shared_team_reward,
    single_step_batch,
    validate_probe_config,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    decompose_raw_reward_events,
    inspect_standard_checkpoints,
    write_return_distribution,
)
from experiments.overcooked_v2.path_c_standard_training import derive_standard_seed
from experiments.overcooked_v2.path_c_adaptation import PathCAdaptationPolicy
from experiments.overcooked_v2.path_c_response_probe import (
    REGISTERED_LOCAL_RESPONSE_SPEC,
    initial_partner_belief,
    local_non_agent_change_from_default_observation,
    partner_visible_from_default_observation,
    response_tokens_from_observations,
    select_finite_prototype_two_action_surrogate,
    select_response_probe,
    update_partner_belief,
    validate_response_probe_config,
)


STANDARD_EVALUATION_SCHEMA_VERSION = "path_c_standard_evaluation_v1"


def validate_evaluation_probe_config(
    model: nn.Module,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind one probe schema to the loaded policy class before any rollout starts."""

    adaptation_model = isinstance(model, PathCAdaptationPolicy)
    response_schema = isinstance(payload, Mapping) and "controller" in payload
    if adaptation_model != response_schema:
        raise ValueError(
            "Adaptation checkpoints require the response-controller probe schema; "
            "legacy checkpoints require the advantage-disagreement schema."
        )
    normalized_payload = payload
    if adaptation_model and payload.get("controller") == "response_voi":
        normalized_payload = dict(payload)
        normalized_payload["controller"] = "generic_response_information"
    config = (
        validate_response_probe_config(normalized_payload)
        if adaptation_model
        else validate_probe_config(payload)
    )
    if adaptation_model and config["enabled"] != (config["controller"] != "off"):
        raise ValueError("probe.enabled must agree with whether the controller is off.")
    if adaptation_model and config["response_model_mode"] != model.response_model_mode:
        raise ValueError("Probe config and adaptation checkpoint use different response models.")
    if adaptation_model and int(config["response_ensemble_size"]) != int(
        model.response_ensemble.ensemble_size
    ):
        raise ValueError("Probe config and checkpoint disagree on partner hypotheses.")
    if adaptation_model and int(config["response_ensemble_size"]) != int(
        model.partner_value_ensemble.ensemble_size
    ):
        raise ValueError("Probe config and checkpoint disagree on partner-value heads.")
    return config


@dataclass(frozen=True, order=True)
class PolicyCheckpoint:
    seed: int
    path: Path


@dataclass(frozen=True, order=True)
class StandardPairing:
    policy_0_seed: int
    policy_1_seed: int

    @property
    def split(self) -> str:
        return "sp" if self.policy_0_seed == self.policy_1_seed else "xp"


@dataclass(frozen=True)
class StandardEpisodeReturn:
    schema_version: str
    layout: str
    split: str
    policy_0_seed: int
    policy_1_seed: int
    episode_index: int
    canonical_episode_seed: int
    raw_episode_return: float
    environment_steps: int
    policy_0_probe_count: int
    policy_1_probe_count: int
    correct_delivery_count: int = 0
    wrong_delivery_count: int = 0
    indicator_activation_count: int = 0
    ambiguous_reward_step_count: int = 0
    probe_cost_adjusted_return: float | None = None
    policy_0_input_contract: str | None = None
    policy_1_input_contract: str | None = None
    policy_0_final_partner_belief_entropy: float | None = None
    policy_1_final_partner_belief_entropy: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "path_c_standard_episode_return_v1",
            "path_c_standard_episode_return_v2",
            "path_c_standard_episode_return_v3",
        }:
            raise ValueError("Unsupported standard episode-row schema.")
        if self.split not in {"sp", "xp"}:
            raise ValueError("Standard episode row split must be SP or XP.")
        expected_split = (
            "sp" if self.policy_0_seed == self.policy_1_seed else "xp"
        )
        if self.split != expected_split:
            raise ValueError("Standard episode row split disagrees with its pairing.")
        if self.environment_steps != 400:
            raise ValueError("Standard Test Time episode rows must contain 400 steps.")
        if not math.isfinite(float(self.raw_episode_return)):
            raise ValueError("Standard episode return must be finite.")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative.")
        if self.policy_0_probe_count < 0 or self.policy_1_probe_count < 0:
            raise ValueError("Probe counts must be non-negative.")
        for value in (
            self.policy_0_final_partner_belief_entropy,
            self.policy_1_final_partner_belief_entropy,
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError("Partner belief entropy must be finite and non-negative.")
        for name in (
            "correct_delivery_count",
            "wrong_delivery_count",
            "indicator_activation_count",
            "ambiguous_reward_step_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.probe_cost_adjusted_return is not None and not math.isfinite(
            float(self.probe_cost_adjusted_return)
        ):
            raise ValueError("probe_cost_adjusted_return must be finite when present.")
        if self.schema_version in {
            "path_c_standard_episode_return_v2",
            "path_c_standard_episode_return_v3",
        } and (
            not self.policy_0_input_contract or not self.policy_1_input_contract
        ):
            raise ValueError("New episode rows must record both input contracts.")


def build_standard_pairings(policy_seeds: Sequence[int]) -> tuple[StandardPairing, ...]:
    seeds = tuple(int(item) for item in policy_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Policy seeds must be non-empty and unique.")
    self_play = [StandardPairing(seed, seed) for seed in seeds]
    cross_play = [
        StandardPairing(first, second)
        for first, second in itertools.permutations(seeds, 2)
    ]
    return tuple(self_play + cross_play)


def load_episode_rows(path: str | Path) -> list[StandardEpisodeReturn]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[StandardEpisodeReturn] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        try:
            rows.append(StandardEpisodeReturn(**payload))
        except Exception as exc:
            raise ValueError(
                f"Invalid standard episode row at line {line_number}."
            ) from exc
    return rows


def append_episode_rows(
    path: str | Path,
    rows: Iterable[StandardEpisodeReturn],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        handle.flush()


def _pairing_means(
    rows: Sequence[StandardEpisodeReturn],
    *,
    expected_episodes_per_pairing: int,
) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    episode_ids: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (row.policy_0_seed, row.policy_1_seed)
        grouped.setdefault(key, []).append(float(row.raw_episode_return))
        ids = episode_ids.setdefault(key, set())
        if row.episode_index in ids:
            raise ValueError(f"Pairing {key} repeats episode index {row.episode_index}.")
        ids.add(row.episode_index)
    means: dict[tuple[int, int], float] = {}
    for key, values in grouped.items():
        if len(values) != int(expected_episodes_per_pairing):
            raise ValueError(
                f"Pairing {key} has {len(values)} rows; expected "
                f"{expected_episodes_per_pairing}."
            )
        if episode_ids[key] != set(range(expected_episodes_per_pairing)):
            raise ValueError(f"Pairing {key} does not contain the complete episode index set.")
        means[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return means


def _adjusted_pairing_means(
    rows: Sequence[StandardEpisodeReturn],
    *,
    expected_episodes_per_pairing: int,
) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.policy_0_seed, row.policy_1_seed), []).append(
            float(
                row.raw_episode_return
                if row.probe_cost_adjusted_return is None
                else row.probe_cost_adjusted_return
            )
        )
    if any(len(values) != expected_episodes_per_pairing for values in grouped.values()):
        raise ValueError("Adjusted-return pairing rows are incomplete.")
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def two_way_cluster_bootstrap_ci(
    pairing_means: Mapping[tuple[int, int], float],
    policy_seeds: Sequence[int],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Node bootstrap using one sampled multiplicity in both policy positions."""

    seeds = tuple(int(item) for item in policy_seeds)
    if len(seeds) < 2:
        raise ValueError("XP bootstrap requires at least two policy seeds.")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1).")
    for first, second in itertools.permutations(seeds, 2):
        if (first, second) not in pairing_means:
            raise ValueError("XP matrix is incomplete for two-way bootstrap.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    seed_count = len(seeds)
    for replicate in range(int(replicates)):
        node_draw = rng.choice(seeds, size=seed_count, replace=True)
        values = [
            float(pairing_means[(int(first), int(second))])
            for first in node_draw
            for second in node_draw
            if int(first) != int(second)
        ]
        if not values:
            estimates[replicate] = np.nan
        else:
            estimates[replicate] = float(np.mean(values))
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < max(10, int(replicates) // 2):
        raise ValueError("Too few finite two-way bootstrap replicates.")
    tail = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def summarize_standard_rows(
    rows: Sequence[StandardEpisodeReturn],
    *,
    policy_seeds: Sequence[int],
    episodes_per_pairing: int,
    standard_deviation_ddof: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seeds = tuple(int(item) for item in policy_seeds)
    expected_pairings = set(build_standard_pairings(seeds))
    observed_pairings = {
        StandardPairing(row.policy_0_seed, row.policy_1_seed) for row in rows
    }
    if observed_pairings != expected_pairings:
        missing = sorted(expected_pairings - observed_pairings)
        unexpected = sorted(observed_pairings - expected_pairings)
        raise ValueError(
            f"Standard pairing matrix mismatch: missing={missing}, unexpected={unexpected}."
        )
    layouts = {row.layout for row in rows}
    if len(layouts) != 1:
        raise ValueError("One standard summary must contain exactly one layout.")
    means = _pairing_means(
        rows,
        expected_episodes_per_pairing=episodes_per_pairing,
    )
    sp_values = np.asarray([means[(seed, seed)] for seed in seeds], dtype=np.float64)
    xp_values = np.asarray(
        [means[(first, second)] for first, second in itertools.permutations(seeds, 2)],
        dtype=np.float64,
    )
    ddof = int(standard_deviation_ddof)
    if ddof < 0 or ddof >= min(sp_values.size, xp_values.size):
        raise ValueError("standard_deviation_ddof is invalid for the pairing counts.")
    ci_low, ci_high = two_way_cluster_bootstrap_ci(
        means,
        seeds,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    sp_mean = float(sp_values.mean())
    xp_mean = float(xp_values.mean())
    adjusted_means = _adjusted_pairing_means(
        rows,
        expected_episodes_per_pairing=episodes_per_pairing,
    )
    adjusted_sp = np.asarray(
        [adjusted_means[(seed, seed)] for seed in seeds],
        dtype=np.float64,
    )
    adjusted_xp = np.asarray(
        [
            adjusted_means[(first, second)]
            for first, second in itertools.permutations(seeds, 2)
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": "path_c_standard_summary_v1",
        "layout": next(iter(layouts)),
        "policy_seeds": list(seeds),
        "sp_pairing_count": int(sp_values.size),
        "xp_directed_pairing_count": int(xp_values.size),
        "episodes_per_pairing": int(episodes_per_pairing),
        "episode_environment_steps": 400,
        "raw_episode_row_count": len(rows),
        "sp_mean": sp_mean,
        "sp_std": float(np.std(sp_values, ddof=ddof)),
        "xp_mean": xp_mean,
        "xp_std": float(np.std(xp_values, ddof=ddof)),
        "sp_minus_xp_gap": sp_mean - xp_mean,
        "support_probe_cost_adjusted_sp_mean": float(adjusted_sp.mean()),
        "support_probe_cost_adjusted_xp_mean": float(adjusted_xp.mean()),
        "correct_delivery_count": int(
            sum(row.correct_delivery_count for row in rows)
        ),
        "wrong_delivery_count": int(
            sum(row.wrong_delivery_count for row in rows)
        ),
        "indicator_activation_count": int(
            sum(row.indicator_activation_count for row in rows)
        ),
        "ambiguous_reward_step_count": int(
            sum(row.ambiguous_reward_step_count for row in rows)
        ),
        "xp_two_way_cluster_bootstrap_ci": [ci_low, ci_high],
        "bootstrap_confidence": float(bootstrap_confidence),
        "bootstrap_replicates": int(bootstrap_replicates),
        "standard_deviation_ddof": ddof,
        "return_definition": "sum_of_raw_rewards_agent_0_over_400_steps",
    }


def summarize_standard_calibration_rows(
    rows: Sequence[StandardEpisodeReturn],
    *,
    policy_seed: int,
    episodes_per_pairing: int,
) -> dict[str, Any]:
    expected_pairing = StandardPairing(int(policy_seed), int(policy_seed))
    observed_pairings = {
        StandardPairing(row.policy_0_seed, row.policy_1_seed) for row in rows
    }
    if observed_pairings != {expected_pairing}:
        raise ValueError("Calibration rows must contain exactly one self-play pairing.")
    _pairing_means(rows, expected_episodes_per_pairing=episodes_per_pairing)
    values = np.asarray([row.raw_episode_return for row in rows], dtype=np.float64)
    if values.size != int(episodes_per_pairing) or not bool(np.isfinite(values).all()):
        raise ValueError("Calibration requires the complete finite return schedule.")
    policy_0_probes = np.asarray(
        [row.policy_0_probe_count for row in rows],
        dtype=np.float64,
    )
    policy_1_probes = np.asarray(
        [row.policy_1_probe_count for row in rows],
        dtype=np.float64,
    )
    adjusted_values = np.asarray(
        [
            row.raw_episode_return
            if row.probe_cost_adjusted_return is None
            else row.probe_cost_adjusted_return
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": "path_c_standard_calibration_summary_v1",
        "calibration_only": True,
        "scientific_readout_allowed": False,
        "layout": rows[0].layout,
        "policy_seed": int(policy_seed),
        "episodes": int(values.size),
        "episode_environment_steps": 400,
        "raw_episode_row_count": int(values.size),
        "raw_return_mean": float(values.mean()),
        "raw_return_std": float(values.std(ddof=0)),
        "raw_return_min": float(values.min()),
        "raw_return_max": float(values.max()),
        "raw_return_p10": float(np.quantile(values, 0.10)),
        "raw_return_median": float(np.quantile(values, 0.50)),
        "raw_return_p90": float(np.quantile(values, 0.90)),
        "negative_return_fraction": float(np.mean(values < 0.0)),
        "zero_return_fraction": float(np.mean(values == 0.0)),
        "positive_return_fraction": float(np.mean(values > 0.0)),
        "policy_0_probe_count_mean": float(policy_0_probes.mean()),
        "policy_1_probe_count_mean": float(policy_1_probes.mean()),
        "policy_0_probe_count_total": int(policy_0_probes.sum()),
        "policy_1_probe_count_total": int(policy_1_probes.sum()),
        "support_probe_cost_adjusted_return_mean": float(
            adjusted_values.mean()
        ),
        "correct_delivery_count": int(
            sum(row.correct_delivery_count for row in rows)
        ),
        "wrong_delivery_count": int(
            sum(row.wrong_delivery_count for row in rows)
        ),
        "indicator_activation_count": int(
            sum(row.indicator_activation_count for row in rows)
        ),
        "ambiguous_reward_step_count": int(
            sum(row.ambiguous_reward_step_count for row in rows)
        ),
        "return_definition": "sum_of_raw_rewards_agent_0_over_400_steps",
        "interpretation_boundary": (
            "Single-checkpoint self-play calibration; not a multi-seed scientific result."
        ),
    }


@dataclass(frozen=True)
class StandardEvaluationSpec:
    run_kind: str
    scientific_readout_allowed: bool
    evaluation_seed: int
    episodes_per_pairing: int
    evaluation_batch_size: int
    expected_policy_count: int
    standard_deviation_ddof: int
    bootstrap_replicates: int
    bootstrap_confidence: float
    probe_cost_per_use: float
    diagnostic_probe_mode: str
    output_dir: Path
    checkpoint_paths: tuple[Path, ...]
    diagnostic_checkpoint_paths: tuple[Path, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StandardEvaluationSpec":
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("Standard evaluation config requires an evaluation mapping.")
        run_kind = str(payload.get("run_kind", "smoke"))
        if run_kind not in {
            "smoke",
            "formal",
            "calibration",
            "admission",
            "diagnostic",
        }:
            raise ValueError("Unsupported standard evaluation run_kind.")
        scientific = bool(payload.get("scientific_readout_allowed", False))
        if run_kind == "smoke" and scientific:
            raise ValueError("Smoke evaluation cannot permit scientific readout.")
        if run_kind == "calibration" and scientific:
            raise ValueError("Calibration cannot permit scientific readout.")
        if run_kind in {"admission", "diagnostic"} and scientific:
            raise ValueError("Admission and diagnostic evaluation are non-scientific.")
        episodes = int(evaluation["episodes_per_pairing"])
        expected = int(evaluation["expected_policy_count"])
        if episodes <= 0 or expected <= 0:
            raise ValueError("Evaluation requires positive episode and policy counts.")
        if run_kind != "calibration" and expected <= 1:
            raise ValueError("SP/XP evaluation requires at least two policies.")
        if run_kind == "formal" and (episodes != 500 or expected != 10):
            raise ValueError("Formal evaluation requires 10 policies and 500 episodes per pairing.")
        if run_kind == "calibration" and (episodes != 500 or expected != 1):
            raise ValueError(
                "Calibration requires one policy and 500 self-play episodes."
            )
        paths = tuple(Path(item) for item in payload["checkpoint_paths"])
        if len(paths) != expected or len(set(paths)) != len(paths):
            raise ValueError("checkpoint_paths must match expected_policy_count without repeats.")
        diagnostic_paths = tuple(
            Path(item) for item in payload.get("diagnostic_checkpoint_paths", ())
        )
        if run_kind == "calibration" and (
            len(diagnostic_paths) != 5 or len(set(diagnostic_paths)) != 5
        ):
            raise ValueError(
                "Calibration requires four partner snapshots and one final diagnostic checkpoint."
            )
        if run_kind != "calibration" and diagnostic_paths:
            raise ValueError(
                "diagnostic_checkpoint_paths are calibration-only."
            )
        confidence = float(evaluation.get("bootstrap_confidence", 0.95))
        if not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap_confidence must lie in (0,1).")
        evaluation_batch_size = int(evaluation["evaluation_batch_size"])
        bootstrap_replicates = int(evaluation.get("bootstrap_replicates", 2000))
        if evaluation_batch_size <= 0 or bootstrap_replicates <= 0:
            raise ValueError(
                "evaluation_batch_size and bootstrap_replicates must be positive."
            )
        if run_kind == "formal" and bootstrap_replicates != 9_999:
            raise ValueError(
                "Formal proposal evaluation requires 9,999 node-bootstrap replicates."
            )
        if episodes % evaluation_batch_size:
            raise ValueError(
                "evaluation_batch_size must divide episodes_per_pairing so every "
                "chunk keeps one fixed XLA batch shape."
            )
        probe_cost = float(evaluation.get("probe_cost_per_use", 0.0))
        if not math.isfinite(probe_cost) or probe_cost < 0.0:
            raise ValueError("evaluation.probe_cost_per_use must be non-negative.")
        diagnostic_probe_mode = str(
            evaluation.get("diagnostic_probe_mode", "bilateral")
        )
        if diagnostic_probe_mode not in {"bilateral", "slot0_only"}:
            raise ValueError(
                "evaluation.diagnostic_probe_mode must be bilateral or slot0_only."
            )
        if diagnostic_probe_mode == "slot0_only" and run_kind != "diagnostic":
            raise ValueError("Single-sided probing is diagnostic-only.")
        return cls(
            run_kind=run_kind,
            scientific_readout_allowed=scientific,
            evaluation_seed=int(payload["evaluation_seed"]),
            episodes_per_pairing=episodes,
            evaluation_batch_size=evaluation_batch_size,
            expected_policy_count=expected,
            standard_deviation_ddof=int(
                evaluation.get("standard_deviation_ddof", 0)
            ),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_confidence=confidence,
            probe_cost_per_use=probe_cost,
            diagnostic_probe_mode=diagnostic_probe_mode,
            output_dir=Path(payload["output_dir"]),
            checkpoint_paths=paths,
            diagnostic_checkpoint_paths=diagnostic_paths,
        )


def load_standard_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Standard evaluation config must contain a mapping.")
    config = dict(payload)
    if config.get("schema_version") != STANDARD_EVALUATION_SCHEMA_VERSION:
        raise ValueError("Unsupported standard evaluation config schema.")
    env_path = Path(config["environment_config"])
    if not env_path.is_absolute():
        env_path = (config_path.parent / env_path).resolve()
    config["environment"] = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    resolved_checkpoints = []
    for raw_path in config["checkpoint_paths"]:
        checkpoint = Path(raw_path)
        if not checkpoint.is_absolute():
            checkpoint = (config_path.parent / checkpoint).resolve()
        resolved_checkpoints.append(str(checkpoint))
    config["checkpoint_paths"] = resolved_checkpoints
    resolved_diagnostic_checkpoints = []
    for raw_path in config.get("diagnostic_checkpoint_paths", ()):
        checkpoint = Path(raw_path)
        if not checkpoint.is_absolute():
            checkpoint = (config_path.parent / checkpoint).resolve()
        resolved_diagnostic_checkpoints.append(str(checkpoint))
    config["diagnostic_checkpoint_paths"] = resolved_diagnostic_checkpoints
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        config["output_dir"] = str((config_path.parent / output_dir).resolve())
    return config


class StandardPairingEvaluator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = StandardEvaluationSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.device = torch.device(str(config.get("device", "cuda")))
        probe_payload = config.get("probe")
        self.output_dir = self.spec.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.output_dir / "episode_returns.jsonl"
        # One adapter and one vmapped env pool per batch size for the whole
        # evaluation; rebuilding them per chunk would recompile the XLA step.
        self._adapter = self.env_config.make_adapter()
        self._pools: dict[int, BatchedEnvPool] = {}
        self.checkpoints: list[PolicyCheckpoint] = []
        self.models: dict[int, nn.Module] = {}
        self.checkpoint_metadata: dict[int, dict[str, Any]] = {}
        for path in self.spec.checkpoint_paths:
            model, metadata = load_standard_checkpoint(path, device=self.device)
            seed = int(metadata["seed"])
            if seed in self.models:
                raise ValueError(f"Two checkpoints use training seed {seed}.")
            expected_phase = "path_c_final"
            if metadata.get("phase") != expected_phase:
                raise ValueError(
                    f"{self.spec.run_kind} evaluation requires {expected_phase} checkpoints."
                )
            if metadata.get("layout") != self.env_config.layout:
                raise ValueError("Checkpoint layout differs from evaluation layout.")
            if metadata.get("environment") != self.env_config.to_mapping():
                raise ValueError(
                    "Checkpoint environment semantics differ from evaluation semantics."
                )
            if self.spec.run_kind in {"formal", "calibration"}:
                if metadata.get("run_kind") != "formal" or not bool(
                    metadata.get("scientific_readout_allowed", False)
                ):
                    raise ValueError(
                        "Formal evaluation and calibration require a completed formal checkpoint."
                    )
                expected_steps = (
                    10_000_000
                    if isinstance(model, PathCAdaptationPolicy)
                    else 30_000_000
                )
                if int(metadata.get("environment_steps", -1)) != expected_steps:
                    raise ValueError(
                        "Formal evaluation requires the model-specific completed budget."
                    )
            if self.spec.run_kind == "formal":
                if not isinstance(model, PathCAdaptationPolicy):
                    raise ValueError(
                        "Historical advantage-disagreement checkpoints cannot serve "
                        "as the proposal's formal main method."
                    )
                if model.partner_action_channel:
                    raise ValueError(
                        "Formal proposal evaluation requires official local observations."
                    )
                if metadata.get("response_vocabulary_sha256") != (
                    REGISTERED_LOCAL_RESPONSE_SPEC.sha256
                ):
                    raise ValueError(
                        "Formal proposal evaluation requires the registered response vocabulary."
                    )
                formal_condition = str(metadata.get("condition_id"))
                expected_controller = {
                    "decision_focused": "registered_response_sequential_branch_v1",
                    "random_safe_probe": "registered_random_safe_probe_v1",
                    "no_probe": "off",
                    "generic_response_information": "generic_response_information",
                }.get(formal_condition)
                if expected_controller is None or metadata.get("controller") != expected_controller:
                    raise ValueError(
                        "Formal proposal evaluation requires a registered condition "
                        "whose controller matches its checkpoint."
                    )
                if expected_controller in {
                    "registered_response_sequential_branch_v1",
                    "registered_random_safe_probe_v1",
                }:
                    raise NotImplementedError(
                        "Formal evaluation cannot substitute a diagnostic controller for "
                        "the registered sequential or matched random-safe controller."
                    )
                if expected_controller == "generic_response_information":
                    calibration_binding = metadata.get(
                        "probe_calibration_binding"
                    )
                    required_calibration_hashes = {
                        "summary_sha256",
                        "rows_sha256",
                        "calibration_contract_sha256",
                        "calibration_implementation_sha256",
                    }
                    if not isinstance(calibration_binding, Mapping) or any(
                        not isinstance(calibration_binding.get(field), str)
                        or len(calibration_binding[field]) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in calibration_binding[field]
                        )
                        for field in required_calibration_hashes
                    ):
                        raise ValueError(
                            "Formal response-information evaluation requires a "
                            "content-addressed, recomputed calibration ledger."
                        )
            self.checkpoints.append(PolicyCheckpoint(seed=seed, path=path))
            self.models[seed] = model
            self.checkpoint_metadata[seed] = dict(metadata)
        self.checkpoints.sort()
        architectures = {
            json.dumps(model.architecture_manifest(), sort_keys=True)
            for model in self.models.values()
        }
        if len(architectures) != 1:
            raise ValueError("All policies in one SP/XP matrix must share an architecture.")
        first_model = next(iter(self.models.values()))
        self.probe_configs: dict[int, dict[str, Any]] = {}
        for seed, model in self.models.items():
            model_probe_payload = probe_payload
            if (
                isinstance(model, PathCAdaptationPolicy)
                and model.response_model_mode == "partner_conditioned"
            ):
                metadata = self.checkpoint_metadata[seed]
                controller = str(metadata.get("controller"))
                threshold_field = (
                    "surrogate_score_threshold"
                    if controller == "finite_prototype_two_action_surrogate"
                    else "response_disagreement_threshold"
                )
                if threshold_field not in metadata:
                    raise ValueError(
                        "Adaptation checkpoints must bind their calibrated threshold."
                    )
                bound_threshold = float(metadata[threshold_field])
                if not isinstance(probe_payload, Mapping):
                    raise ValueError("Adaptation evaluation requires a probe mapping.")
                configured_controller = str(probe_payload.get("controller"))
                if configured_controller == "response_voi":
                    configured_controller = "generic_response_information"
                if configured_controller != controller:
                    raise ValueError(
                        "Evaluation controller differs from the trained checkpoint."
                    )
                if probe_payload.get(
                    "response_route", "use_registered_response"
                ) != metadata.get("response_route", "use_registered_response"):
                    raise ValueError(
                        "Evaluation response route differs from the trained checkpoint."
                    )
                configured_threshold = probe_payload.get(threshold_field)
                if configured_threshold is not None and not math.isclose(
                    float(configured_threshold),
                    bound_threshold,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "Evaluation probe threshold differs from a trained checkpoint."
                    )
                model_probe_payload = dict(probe_payload)
                model_probe_payload[threshold_field] = bound_threshold
            config_for_model = validate_evaluation_probe_config(
                model, model_probe_payload
            )
            if config_for_model["enabled"] and model.n_heads <= 1:
                raise ValueError("Enabled Path C probing requires more than one ensemble head.")
            self.probe_configs[seed] = config_for_model
        self.probe_config = self.probe_configs[self.checkpoints[0].seed]

    def run(self) -> dict[str, Any]:
        policy_seeds = [item.seed for item in self.checkpoints]
        pairings = build_standard_pairings(policy_seeds)
        existing_rows = load_episode_rows(self.rows_path)
        allowed_seeds = set(policy_seeds)
        for row in existing_rows:
            if row.layout != self.env_config.layout:
                raise ValueError("Existing episode rows belong to a different layout.")
            if row.policy_0_seed not in allowed_seeds or row.policy_1_seed not in allowed_seeds:
                raise ValueError("Existing episode rows belong to different checkpoints.")
            if row.episode_index >= self.spec.episodes_per_pairing:
                raise ValueError("Existing episode rows exceed the configured schedule.")
        existing_keys = {
            (row.policy_0_seed, row.policy_1_seed, row.episode_index)
            for row in existing_rows
        }
        for pairing in pairings:
            # Reject 64->32-bit execution-seed collisions across the pairing's
            # complete episode schedule: a collision would record two identical
            # rollouts as independent episodes and silently deflate variance.
            validate_unique_execution_seed_mapping(
                (
                    self._canonical_episode_seed(pairing, episode_index)
                    for episode_index in range(self.spec.episodes_per_pairing)
                ),
                name=(
                    "standard evaluation episode seeds for pairing "
                    f"({pairing.policy_0_seed},{pairing.policy_1_seed})"
                ),
            )
            missing = [
                episode_index
                for episode_index in range(self.spec.episodes_per_pairing)
                if (
                    pairing.policy_0_seed,
                    pairing.policy_1_seed,
                    episode_index,
                )
                not in existing_keys
            ]
            for offset in range(0, len(missing), self.spec.evaluation_batch_size):
                indices = missing[offset : offset + self.spec.evaluation_batch_size]
                rows = self._evaluate_chunk(pairing, indices)
                append_episode_rows(self.rows_path, rows)
                for row in rows:
                    existing_keys.add(
                        (row.policy_0_seed, row.policy_1_seed, row.episode_index)
                    )
        rows = load_episode_rows(self.rows_path)
        if self.spec.run_kind == "calibration":
            summary = summarize_standard_calibration_rows(
                rows,
                policy_seed=policy_seeds[0],
                episodes_per_pairing=self.spec.episodes_per_pairing,
            )
            health = inspect_standard_checkpoints(
                self.spec.diagnostic_checkpoint_paths,
                output_dir=self.output_dir,
                device=self.device,
            )
            distribution_path = self.output_dir / "return_distribution.png"
            write_return_distribution(
                [row.raw_episode_return for row in rows],
                distribution_path,
            )
            summary.update(
                {
                    "checkpoint_health_path": str(
                        self.output_dir / "checkpoint_health.json"
                    ),
                    "checkpoint_parameter_health_plot": str(
                        self.output_dir / "checkpoint_parameter_health.png"
                    ),
                    "return_distribution_plot": str(distribution_path),
                    "fixed_prior_consistent_across_checkpoints": bool(
                        health["fixed_prior_consistent"]
                    ),
                }
            )
        else:
            summary = summarize_standard_rows(
                rows,
                policy_seeds=policy_seeds,
                episodes_per_pairing=self.spec.episodes_per_pairing,
                standard_deviation_ddof=self.spec.standard_deviation_ddof,
                bootstrap_replicates=self.spec.bootstrap_replicates,
                bootstrap_confidence=self.spec.bootstrap_confidence,
                bootstrap_seed=self.spec.evaluation_seed,
            )
        summary.update(
            {
                "run_kind": self.spec.run_kind,
                "scientific_readout_allowed": self.spec.scientific_readout_allowed,
                "checkpoint_paths": [str(item.path) for item in self.checkpoints],
                "evaluation_environment_steps": len(rows) * self.env_config.max_steps,
                "probe_controller": self.probe_config.get(
                    "controller", "advantage_disagreement"
                ),
                "probe_budget_per_episode": self.probe_config.get(
                    "probe_budget_per_episode"
                ),
                "probe_window_environment_steps": self.probe_config.get(
                    "probe_window_environment_steps"
                ),
                "response_disagreement_threshold_by_seed": {
                    str(seed): config.get("response_disagreement_threshold")
                    for seed, config in self.probe_configs.items()
                    if "response_disagreement_threshold" in config
                },
                "probe_cost_per_use": self.spec.probe_cost_per_use,
            }
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def _canonical_episode_seed(
        self,
        pairing: StandardPairing,
        episode_index: int,
    ) -> int:
        return derive_standard_seed(
            self.spec.evaluation_seed,
            self.env_config.layout,
            pairing.policy_0_seed,
            pairing.policy_1_seed,
            int(episode_index),
        )

    def _pool_for_batch_size(self, batch_size: int) -> BatchedEnvPool:
        pool = self._pools.get(batch_size)
        if pool is None:
            pool = BatchedEnvPool(self._adapter.env, batch_size)
            self._pools[batch_size] = pool
        return pool

    def _evaluate_chunk(
        self,
        pairing: StandardPairing,
        episode_indices: Sequence[int],
    ) -> list[StandardEpisodeReturn]:
        if not episode_indices:
            return []
        model_0 = self.models[pairing.policy_0_seed]
        model_1 = self.models[pairing.policy_1_seed]
        pairing_probe_configs = {
            0: self.probe_configs[pairing.policy_0_seed],
            1: self.probe_configs[pairing.policy_1_seed],
        }
        model_0.eval()
        model_1.eval()
        batch_size = len(episode_indices)
        pool = self._pool_for_batch_size(batch_size)
        canonical_seeds = [
            self._canonical_episode_seed(pairing, episode_index)
            for episode_index in episode_indices
        ]
        execution_seeds = np.asarray(
            [derive_ocv2_execution_seed(seed) for seed in canonical_seeds],
            dtype=np.uint32,
        )
        pool.reset(execution_seeds)
        observations = pool.snapshot_obs()
        previous_actions = {
            slot: np.full(batch_size, model_0.n_actions, dtype=np.int64)
            for slot in (0, 1)
        }
        previous_rewards = {
            slot: np.zeros(batch_size, dtype=np.float32) for slot in (0, 1)
        }
        input_contracts = {
            slot: str(models.input_contract)
            if isinstance(models := (model_0 if slot == 0 else model_1), PathCAdaptationPolicy)
            else str(models.architecture_manifest()["input_contract"])
            for slot in (0, 1)
        }
        episode_starts = np.ones(batch_size, dtype=bool)
        states = {
            0: model_0.initial_state(batch_size, device=self.device),
            1: model_1.initial_state(batch_size, device=self.device),
        }
        partner_beliefs: dict[int, torch.Tensor | None] = {}
        pending_response_probabilities: dict[int, torch.Tensor | None] = {}
        pending_continuation_values: dict[int, torch.Tensor | None] = {}
        pending_response_is_probe: dict[int, np.ndarray | None] = {}
        previous_local_observations: dict[int, np.ndarray | None] = {}
        for slot, model in ((0, model_0), (1, model_1)):
            partner_beliefs[slot] = (
                initial_partner_belief(
                    batch_size,
                    model.response_ensemble.ensemble_size,
                    device=self.device,
                )
                if isinstance(model, PathCAdaptationPolicy)
                and model.response_model_mode == "partner_conditioned"
                else None
            )
            pending_response_probabilities[slot] = None
            pending_continuation_values[slot] = None
            pending_response_is_probe[slot] = None
            previous_local_observations[slot] = None
        totals = np.zeros(batch_size, dtype=np.float64)
        probes = {
            slot: np.zeros(batch_size, dtype=np.int64) for slot in (0, 1)
        }
        correct_deliveries = np.zeros(batch_size, dtype=np.int64)
        wrong_deliveries = np.zeros(batch_size, dtype=np.int64)
        indicator_activations = np.zeros(batch_size, dtype=np.int64)
        ambiguous_reward_steps = np.zeros(batch_size, dtype=np.int64)
        rngs = {
            slot: np.random.default_rng(
                derive_ocv2_execution_seed(
                    derive_standard_seed(
                        self.spec.evaluation_seed,
                        self.env_config.layout,
                        pairing.policy_0_seed,
                        pairing.policy_1_seed,
                        "policy_rng",
                        slot,
                    )
                )
            )
            for slot in (0, 1)
        }
        for episode_step in range(self.env_config.max_steps):
            actions: dict[int, np.ndarray] = {}
            current_response_probabilities: dict[int, torch.Tensor | None] = {
                0: None,
                1: None,
            }
            current_continuation_values: dict[int, torch.Tensor | None] = {
                0: None,
                1: None,
            }
            current_probe_flags: dict[int, np.ndarray | None] = {0: None, 1: None}
            current_partner_visible: dict[int, np.ndarray] = {}
            for slot, model in ((0, model_0), (1, model_1)):
                if isinstance(model, PathCAdaptationPolicy):
                    current_partner_visible[slot] = (
                        partner_visible_from_default_observation(
                            observations[f"agent_{slot}"],
                            indicate_successful_delivery=(
                                self.env_config.indicate_successful_delivery
                            ),
                        )
                    )
                    belief = partner_beliefs[slot]
                    pending = pending_response_probabilities[slot]
                    pending_probe = pending_response_is_probe[slot]
                    previous_local = previous_local_observations[slot]
                    if (
                        model.response_model_mode == "partner_conditioned"
                        and not model.partner_action_channel
                        and belief is not None
                        and pending is not None
                    ):
                        if pending_probe is None or previous_local is None:
                            raise RuntimeError(
                                "A pending local response lacks its source history."
                            )
                        local_change = local_non_agent_change_from_default_observation(
                            previous_local[:, None],
                            observations[f"agent_{slot}"][:, None],
                            indicate_successful_delivery=(
                                self.env_config.indicate_successful_delivery
                            ),
                        )[:, 0]
                        response_tokens = response_tokens_from_observations(
                            partner_actions=np.zeros(batch_size, dtype=np.int64),
                            partner_visible=current_partner_visible[slot],
                            local_non_agent_change=local_change,
                            partner_action_channel=False,
                        )
                        observed = (
                            ~pending_probe
                            if pairing_probe_configs[slot]["response_route"]
                            == "mask_current_probe_response"
                            else np.ones(batch_size, dtype=bool)
                        )
                        partner_beliefs[slot] = update_partner_belief(
                            belief,
                            pending,
                            torch.as_tensor(
                                response_tokens,
                                dtype=torch.long,
                                device=self.device,
                            ),
                            probability_floor=float(
                                pairing_probe_configs[slot][
                                    "belief_probability_floor"
                                ]
                            ),
                            observed=torch.as_tensor(observed, device=self.device),
                        )
                batch = single_step_batch(
                    observations[f"agent_{slot}"],
                    previous_actions[slot],
                    previous_rewards[slot],
                    episode_starts,
                    n_actions=model.n_actions,
                    device=self.device,
                )
                if isinstance(model, PathCAdaptationPolicy):
                    partner_previous = torch.as_tensor(
                        previous_actions[1 - slot],
                        dtype=torch.long,
                        device=self.device,
                    ).reshape(batch_size, 1)
                    with torch.no_grad():
                        q_values, features, states[slot] = model.forward_step(
                            batch,
                            states[slot],
                            partner_previous_actions=(
                                partner_previous if model.partner_action_channel else None
                            ),
                        )
                        actor_logits = model.actor_logits(features)
                else:
                    with torch.no_grad():
                        q_values, _, states[slot] = model.forward_step(batch, states[slot])
                slot_probe_config = pairing_probe_configs[slot]
                slot_probe_enabled = slot_probe_config["enabled"] and not (
                    self.spec.diagnostic_probe_mode == "slot0_only" and slot == 1
                )
                allowed = (
                    (probes[slot] < slot_probe_config["probe_budget_per_episode"])
                    & (episode_step < slot_probe_config["probe_window_environment_steps"])
                ) if slot_probe_enabled else np.zeros(batch_size, dtype=bool)
                if isinstance(model, PathCAdaptationPolicy):
                    controller = slot_probe_config["controller"]
                    greedy_actions = actor_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                    selected_actions = sample_actor_actions(
                        actor_logits,
                        rngs[slot].random(batch_size),
                    )
                    base_actions = torch.as_tensor(
                        selected_actions, dtype=torch.long, device=self.device
                    )
                    pending_values = pending_continuation_values[slot]
                    if (
                        controller == "finite_prototype_two_action_surrogate"
                        and pending_values is not None
                    ):
                        belief = partner_beliefs[slot]
                        if belief is None:
                            raise ValueError(
                                "Decision-focused evaluation requires a partner belief."
                            )
                        belief_values = torch.einsum(
                            "bk,bka->ba", belief, pending_values
                        )
                        base_actions = belief_values.argmax(dim=-1)
                        selected_actions = (
                            base_actions.detach().cpu().numpy().astype(np.int64)
                        )
                    selected_probes = np.zeros(batch_size, dtype=bool)
                    response_probabilities = None
                    continuation_values = None
                    if model.response_model_mode == "partner_conditioned":
                        with torch.no_grad():
                            response_probabilities = (
                                model.response_ensemble.probabilities_for_candidates(
                                    features.detach(),
                                    torch.arange(model.n_actions, device=self.device),
                                )
                            )
                            continuation_values = (
                                model.partner_value_ensemble.values_for_candidates(
                                    features.detach(),
                                    torch.arange(model.n_actions, device=self.device),
                                )
                            )
                    if (
                        controller == "finite_prototype_two_action_surrogate"
                        and bool(allowed.any())
                    ):
                        if (
                            response_probabilities is None
                            or continuation_values is None
                            or partner_beliefs[slot] is None
                        ):
                            raise ValueError(
                                "The finite-prototype surrogate lacks a required model."
                            )
                        decision = select_finite_prototype_two_action_surrogate(
                            actor_logits=actor_logits,
                            candidate_probabilities=response_probabilities,
                            continuation_values=continuation_values,
                            partner_belief=partner_beliefs[slot],
                            valid_actions=batch.valid_actions[:, 0],
                            surrogate_score_threshold=slot_probe_config[
                                "surrogate_score_threshold"
                            ],
                            max_probe_task_cost=slot_probe_config[
                                "max_probe_task_cost"
                            ],
                            probe_allowed=allowed,
                        )
                        selected_actions = decision.actions.copy()
                        selected_probes = decision.is_probe
                    elif (
                        controller == "generic_response_information"
                        and bool(allowed.any())
                    ):
                        if response_probabilities is None:
                            raise ValueError(
                                "The response-information baseline lacks response models."
                            )
                        decision = select_response_probe(
                            actor_logits=actor_logits,
                            critic_q_values=q_values,
                            candidate_probabilities=response_probabilities,
                            valid_actions=batch.valid_actions[:, 0],
                            response_disagreement_threshold=slot_probe_config[
                                "response_disagreement_threshold"
                            ],
                            max_probe_regret=slot_probe_config["max_probe_regret"],
                            probe_allowed=allowed,
                            partner_weights=partner_beliefs[slot],
                            baseline_actions=base_actions,
                        )
                        selected_actions[decision.is_probe] = (
                            decision.actions[decision.is_probe]
                        )
                        selected_probes = decision.is_probe
                    elif controller == "advantage_disagreement" and bool(allowed.any()):
                        decision = choose_primitive_actions(
                            q_values,
                            batch.valid_actions[:, 0],
                            rng=rngs[slot],
                            probe_enabled=True,
                            disagreement_threshold=slot_probe_config[
                                "advantage_disagreement_threshold"
                            ],
                            max_probe_regret=slot_probe_config["max_probe_regret"],
                            probe_allowed=allowed,
                        )
                        selected_probes = decision.is_probe & (
                            decision.actions != selected_actions
                        )
                        selected_actions[selected_probes] = decision.actions[
                            selected_probes
                        ]
                    elif controller == "random" and bool(allowed.any()):
                        candidates = rngs[slot].integers(0, model.n_actions, size=batch_size)
                        mean_q = q_values.mean(dim=1).cpu().numpy()
                        rows = np.arange(batch_size)
                        regrets = mean_q[rows, greedy_actions] - mean_q[rows, candidates]
                        selected_probes = (
                            allowed
                            & (candidates != greedy_actions)
                            & (candidates != selected_actions)
                            & (regrets <= slot_probe_config["max_probe_regret"])
                        )
                        selected_actions[selected_probes] = candidates[selected_probes]
                    elif controller != "off" and slot_probe_enabled:
                        raise ValueError(f"Unsupported adaptation probe controller {controller!r}.")
                    actions[slot] = selected_actions
                    probes[slot] += selected_probes.astype(np.int64)
                    if model.response_model_mode == "partner_conditioned":
                        if response_probabilities is None or continuation_values is None:
                            raise RuntimeError(
                                "Partner-conditioned evaluation predictions are missing."
                            )
                        executed = torch.as_tensor(
                            selected_actions, dtype=torch.long, device=self.device
                        )
                        row_index = torch.arange(batch_size, device=self.device)
                        current_response_probabilities[slot] = (
                            response_probabilities[row_index, executed]
                        )
                        current_continuation_values[slot] = (
                            continuation_values[row_index, executed]
                        )
                        current_probe_flags[slot] = selected_probes.copy()
                else:
                    decision = choose_primitive_actions(
                        q_values,
                        batch.valid_actions[:, 0],
                        rng=rngs[slot],
                        epsilon=0.0,
                        probe_enabled=slot_probe_enabled,
                        disagreement_threshold=slot_probe_config["disagreement_threshold"],
                        max_probe_regret=slot_probe_config["max_probe_regret"],
                        probe_allowed=allowed if slot_probe_enabled else None,
                        disagreement_stat=slot_probe_config["disagreement_stat"],
                    )
                    actions[slot] = decision.actions
                    probes[slot] += decision.is_probe.astype(np.int64)
            for slot, model in ((0, model_0), (1, model_1)):
                belief = partner_beliefs[slot]
                pending = pending_response_probabilities[slot]
                if (
                    isinstance(model, PathCAdaptationPolicy)
                    and model.response_model_mode == "partner_conditioned"
                    and model.partner_action_channel
                    and belief is not None
                    and pending is not None
                ):
                    pending_probe = pending_response_is_probe[slot]
                    if pending_probe is None:
                        raise RuntimeError("A pending action response lacks its source action.")
                    response_tokens = response_tokens_from_observations(
                        partner_actions=actions[1 - slot],
                        partner_visible=current_partner_visible[slot],
                        local_non_agent_change=np.zeros(batch_size, dtype=bool),
                        partner_action_channel=model.partner_action_channel,
                    )
                    partner_beliefs[slot] = update_partner_belief(
                        belief,
                        pending,
                        torch.as_tensor(
                            response_tokens,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        probability_floor=float(
                            pairing_probe_configs[slot]["belief_probability_floor"]
                        ),
                        observed=torch.as_tensor(
                            ~pending_probe
                            if pairing_probe_configs[slot]["response_route"]
                            == "mask_current_probe_response"
                            else np.ones(batch_size, dtype=bool),
                            device=self.device,
                        ),
                    )
                pending_response_probabilities[slot] = (
                    current_response_probabilities[slot]
                )
                pending_continuation_values[slot] = current_continuation_values[slot]
                pending_response_is_probe[slot] = current_probe_flags[slot]
            next_observations, _, rewards, dones, _ = pool.step_joint(
                actions[0],
                actions[1],
            )
            reward = shared_team_reward(rewards).astype(np.float64)
            totals += reward
            for row, row_reward in enumerate(reward):
                events = decompose_raw_reward_events(
                    np.asarray([row_reward], dtype=np.float64)
                )
                correct_deliveries[row] += events.correct_delivery_count
                wrong_deliveries[row] += events.wrong_delivery_count
                indicator_activations[row] += events.indicator_activation_count
                ambiguous_reward_steps[row] += events.ambiguous_step_count
            episode_starts = np.asarray(dones["__all__"], dtype=bool)
            for slot in (0, 1):
                previous_local_observations[slot] = observations[
                    f"agent_{slot}"
                ].copy()
                previous_actions[slot] = actions[slot]
                previous_rewards[slot] = reward.astype(np.float32)
            observations = {
                key: np.asarray(value) for key, value in next_observations.items()
            }
        if not bool(episode_starts.all()):
            raise ValueError("Standard evaluator did not reach the 400-step boundary.")
        final_belief_entropy: dict[int, np.ndarray | None] = {}
        for slot in (0, 1):
            belief = partner_beliefs[slot]
            if belief is None:
                final_belief_entropy[slot] = None
            else:
                entropy = -(
                    belief * belief.clamp_min(torch.finfo(belief.dtype).tiny).log()
                ).sum(dim=-1)
                final_belief_entropy[slot] = entropy.detach().cpu().numpy()
        return [
            StandardEpisodeReturn(
                schema_version="path_c_standard_episode_return_v3",
                layout=self.env_config.layout,
                split=pairing.split,
                policy_0_seed=pairing.policy_0_seed,
                policy_1_seed=pairing.policy_1_seed,
                episode_index=int(episode_index),
                canonical_episode_seed=int(canonical_seed),
                raw_episode_return=float(total),
                environment_steps=self.env_config.max_steps,
                policy_0_probe_count=int(probes[0][row]),
                policy_1_probe_count=int(probes[1][row]),
                correct_delivery_count=int(correct_deliveries[row]),
                wrong_delivery_count=int(wrong_deliveries[row]),
                indicator_activation_count=int(indicator_activations[row]),
                ambiguous_reward_step_count=int(ambiguous_reward_steps[row]),
                probe_cost_adjusted_return=float(
                    total
                    - self.spec.probe_cost_per_use
                    * (int(probes[0][row]) + int(probes[1][row]))
                ),
                policy_0_input_contract=input_contracts[0],
                policy_1_input_contract=input_contracts[1],
                policy_0_final_partner_belief_entropy=(
                    None
                    if final_belief_entropy[0] is None
                    else float(final_belief_entropy[0][row])
                ),
                policy_1_final_partner_belief_entropy=(
                    None
                    if final_belief_entropy[1] is None
                    else float(final_belief_entropy[1][row])
                ),
            )
            for row, (episode_index, canonical_seed, total) in enumerate(
                zip(episode_indices, canonical_seeds, totals, strict=True)
            )
        ]


def run_standard_evaluation(config_path: str | Path) -> dict[str, Any]:
    config = load_standard_evaluation_config(config_path)
    return StandardPairingEvaluator(config).run()
