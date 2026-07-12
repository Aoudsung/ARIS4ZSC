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

from experiments.overcooked_v2.batched_rollout import BatchedEnvPool
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
    inspect_standard_checkpoints,
    write_return_distribution,
)
from experiments.overcooked_v2.path_c_standard_training import derive_standard_seed


STANDARD_EVALUATION_SCHEMA_VERSION = "path_c_standard_evaluation_v1"


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

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_standard_episode_return_v1":
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


def two_way_cluster_bootstrap_ci(
    pairing_means: Mapping[tuple[int, int], float],
    policy_seeds: Sequence[int],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Pigeonhole bootstrap over row and column training-seed clusters."""

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
        row_draw = rng.choice(seeds, size=seed_count, replace=True)
        column_draw = rng.choice(seeds, size=seed_count, replace=True)
        values = [
            float(pairing_means[(int(first), int(second))])
            for first in row_draw
            for second in column_draw
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
    output_dir: Path
    checkpoint_paths: tuple[Path, ...]
    diagnostic_checkpoint_paths: tuple[Path, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StandardEvaluationSpec":
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("Standard evaluation config requires an evaluation mapping.")
        run_kind = str(payload.get("run_kind", "smoke"))
        if run_kind not in {"smoke", "formal", "calibration"}:
            raise ValueError("run_kind must be 'smoke', 'formal', or 'calibration'.")
        scientific = bool(payload.get("scientific_readout_allowed", False))
        if run_kind == "smoke" and scientific:
            raise ValueError("Smoke evaluation cannot permit scientific readout.")
        if run_kind == "calibration" and scientific:
            raise ValueError("Calibration cannot permit scientific readout.")
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
        if episodes % evaluation_batch_size:
            raise ValueError(
                "evaluation_batch_size must divide episodes_per_pairing so every "
                "chunk keeps one fixed XLA batch shape."
            )
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
        self.probe_config = validate_probe_config(config.get("probe"))
        self.output_dir = self.spec.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.output_dir / "episode_returns.jsonl"
        # One adapter and one vmapped env pool per batch size for the whole
        # evaluation; rebuilding them per chunk would recompile the XLA step.
        self._adapter = self.env_config.make_adapter()
        self._pools: dict[int, BatchedEnvPool] = {}
        self.checkpoints: list[PolicyCheckpoint] = []
        self.models: dict[int, PrimitiveRecurrentEnsembleQ] = {}
        for path in self.spec.checkpoint_paths:
            model, metadata = load_standard_checkpoint(path, device=self.device)
            seed = int(metadata["seed"])
            if seed in self.models:
                raise ValueError(f"Two checkpoints use training seed {seed}.")
            if metadata.get("phase") != "path_c_final":
                raise ValueError("Standard evaluation only accepts final Path C checkpoints.")
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
                if int(metadata.get("environment_steps", -1)) != 30_000_000:
                    raise ValueError(
                        "Formal evaluation and calibration require 30,000,000-step checkpoints."
                    )
            self.checkpoints.append(PolicyCheckpoint(seed=seed, path=path))
            self.models[seed] = model
        self.checkpoints.sort()
        architectures = {
            json.dumps(model.architecture_manifest(), sort_keys=True)
            for model in self.models.values()
        }
        if len(architectures) != 1:
            raise ValueError("All policies in one SP/XP matrix must share an architecture.")
        if self.probe_config["enabled"] and next(
            iter(self.models.values())
        ).n_heads <= 1:
            raise ValueError("Enabled Path C probing requires more than one ensemble head.")

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
        episode_starts = np.ones(batch_size, dtype=bool)
        states = {
            0: model_0.initial_state(batch_size, device=self.device),
            1: model_1.initial_state(batch_size, device=self.device),
        }
        totals = np.zeros(batch_size, dtype=np.float64)
        probes = {
            slot: np.zeros(batch_size, dtype=np.int64) for slot in (0, 1)
        }
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
        for _ in range(self.env_config.max_steps):
            actions: dict[int, np.ndarray] = {}
            for slot, model in ((0, model_0), (1, model_1)):
                batch = single_step_batch(
                    observations[f"agent_{slot}"],
                    previous_actions[slot],
                    previous_rewards[slot],
                    episode_starts,
                    n_actions=model.n_actions,
                    device=self.device,
                )
                with torch.no_grad():
                    q_values, _, states[slot] = model.forward_step(batch, states[slot])
                decision = choose_primitive_actions(
                    q_values,
                    batch.valid_actions[:, 0],
                    rng=rngs[slot],
                    epsilon=0.0,
                    probe_enabled=self.probe_config["enabled"],
                    disagreement_threshold=self.probe_config["disagreement_threshold"],
                    return_floor=self.probe_config["return_floor"],
                    disagreement_stat=self.probe_config["disagreement_stat"],
                )
                actions[slot] = decision.actions
                probes[slot] += decision.is_probe.astype(np.int64)
            next_observations, _, rewards, dones, _ = pool.step_joint(
                actions[0],
                actions[1],
            )
            reward = shared_team_reward(rewards).astype(np.float64)
            totals += reward
            episode_starts = np.asarray(dones["__all__"], dtype=bool)
            for slot in (0, 1):
                previous_actions[slot] = actions[slot]
                previous_rewards[slot] = reward.astype(np.float32)
            observations = {
                key: np.asarray(value) for key, value in next_observations.items()
            }
        if not bool(episode_starts.all()):
            raise ValueError("Standard evaluator did not reach the 400-step boundary.")
        return [
            StandardEpisodeReturn(
                schema_version="path_c_standard_episode_return_v1",
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
            )
            for row, (episode_index, canonical_seed, total) in enumerate(
                zip(episode_indices, canonical_seeds, totals, strict=True)
            )
        ]


def run_standard_evaluation(config_path: str | Path) -> dict[str, Any]:
    config = load_standard_evaluation_config(config_path)
    return StandardPairingEvaluator(config).run()
