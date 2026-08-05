"""Aggregate development runs for the registered p_stay sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import yaml

from src.path_c.protocol_diagnostics import (
    PROTOCOL_STAY_SENSITIVITY,
    analytic_posterior_memory_curves,
    empirical_protocol_memory_metrics,
)
from src.path_c.storage import ensure_run_identity, runtime_provenance, sha256_path, write_json


def collect_protocol_sensitivity_sequences(args: argparse.Namespace) -> None:
    """Replay a shared legal-history panel through one frozen development model."""

    import jax.numpy as jnp

    from experiments.overcooked_v2.comparator_app import load_component_diagnostic_panel
    from experiments.overcooked_v2.deployment import load_deployment
    from src.path_c.experiment import load_config
    from src.path_c.model import initial_policy_state
    from src.path_c.response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

    config = load_config(args.config, run_kind="development")
    if float(config.model.protocol_stay_probability) not in PROTOCOL_STAY_SENSITIVITY:
        raise ValueError("Protocol sensitivity config has an unregistered p_stay.")
    comparator_path = Path(args.panel_comparator).resolve()
    panel = load_component_diagnostic_panel(comparator_path)
    deployment = load_deployment(
        Path(args.training_run).resolve() / "final_deployment", config
    )
    history_observations = jnp.asarray(
        panel["history_observations"], dtype=jnp.float32
    )
    current = jnp.asarray(panel["current_observations"], dtype=jnp.float32)
    actions = jnp.asarray(panel["history_actions"], dtype=jnp.int32)
    observations = jnp.concatenate((history_observations, current[:, None]), axis=1)
    sequence = jnp.swapaxes(observations, 0, 1)
    count, history_steps = actions.shape
    previous_actions = jnp.concatenate(
        (jnp.zeros((count, 1), dtype=jnp.int32), actions), axis=1
    )
    starts = jnp.zeros((count, history_steps + 1), dtype=jnp.bool_).at[:, 0].set(True)
    state = initial_policy_state(
        batch_size=count,
        observation_shape=tuple(deployment.model.observation_shape),
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        capability_dim=config.model.capability_dim,
        component_embedding_dim=config.model.component_embedding_dim,
        protocol_components=config.model.protocol_components,
    )
    unused_state, context = deployment.model.apply(
        {"params": deployment.params},
        state,
        sequence,
        jnp.swapaxes(previous_actions, 0, 1),
        jnp.swapaxes(starts, 0, 1),
        method=deployment.model.context_sequence,
    )
    del unused_state
    probabilities = np.asarray(
        jnp.swapaxes(context.protocol_probabilities, 0, 1), dtype=np.float64
    )
    planes = official_partner_observation_planes(sequence.shape[-1])
    previous = jnp.concatenate((sequence[:1], sequence[:-1]), axis=0)
    targets = extract_partner_response_targets(previous, sequence, planes=planes)
    evidence_valid = np.asarray(
        jnp.swapaxes(targets.visibility > 0.5, 0, 1), dtype=bool
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "version": 1,
            "artifact_type": "depi_protocol_sensitivity_sequences",
            "development_only": True,
            "layout": config.environment.layout,
            "protocol_stay_probability": float(
                config.model.protocol_stay_probability
            ),
            "panel_sha256": panel["source"]["sha256"],
            "training_run": {
                "path": str(Path(args.training_run).resolve()),
                "sha256": sha256_path(Path(args.training_run).resolve()),
            },
            "panel_comparator": {
                "path": str(comparator_path),
                "sha256": sha256_path(comparator_path),
            },
            "sequences": [
                {
                    "panel_id": str(panel["panel_ids"][index]),
                    "posterior": probabilities[index].tolist(),
                    "evidence_valid": evidence_valid[index].tolist(),
                }
                for index in range(count)
            ],
        },
    )


def summarize_protocol_sensitivity(args: argparse.Namespace) -> None:
    by_stay: dict[float, tuple[Path, Mapping[str, Any]]] = {}
    for raw in args.input:
        stay_text, path_text = raw.split("=", 1)
        stay = float(stay_text)
        path = Path(path_text).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if stay in by_stay or stay not in PROTOCOL_STAY_SENSITIVITY:
            raise ValueError("Protocol sensitivity inputs must cover registered p_stay once.")
        if (
            not isinstance(payload, Mapping)
            or payload.get("artifact_type") != "depi_protocol_sensitivity_sequences"
            or payload.get("development_only") is not True
            or float(payload.get("protocol_stay_probability", -1.0)) != stay
            or not isinstance(payload.get("sequences"), list)
        ):
            raise ValueError("Protocol sensitivity sequence artifact differs.")
        by_stay[stay] = (path, payload)
    if set(by_stay) != set(PROTOCOL_STAY_SENSITIVITY):
        raise ValueError("Protocol sensitivity requires p_stay={0.90,0.97,0.99}.")
    layouts = {str(payload["layout"]) for _, payload in by_stay.values()}
    panels = {str(payload["panel_sha256"]) for _, payload in by_stay.values()}
    if len(layouts) != 1 or len(panels) != 1:
        raise ValueError("Protocol sensitivity runs must share layout and state panel.")
    empirical = {
        f"{stay:.2f}": empirical_protocol_memory_metrics(payload["sequences"])
        for stay, (_, payload) in sorted(by_stay.items())
    }
    sources = {
        f"{stay:.2f}": {"path": str(path), "sha256": sha256_path(path)}
        for stay, (path, _) in sorted(by_stay.items())
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "summarize-protocol-sensitivity",
            "repository_runtime": runtime_provenance(),
            "sources": sources,
            "development_only": True,
        },
    )
    write_json(
        output / "protocol_sensitivity.json",
        {
            "version": 1,
            "artifact_type": "depi_protocol_stay_sensitivity",
            "development_only": True,
            "layout": next(iter(layouts)),
            "shared_panel_sha256": next(iter(panels)),
            "registered_stay_probabilities": list(PROTOCOL_STAY_SENSITIVITY),
            "analytic_memory_curves": analytic_posterior_memory_curves(),
            "empirical": empirical,
            "sources": sources,
        },
    )


def run_protocol_sensitivity_matrix(args: argparse.Namespace) -> None:
    """Train and replay the full development p_stay sensitivity on one layout."""

    from experiments.overcooked_v2.training_app import run_training
    from src.path_c.experiment import CONFIG_VERSION

    source_config = Path(args.config).resolve()
    base = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in args.seed_index)
    if tuple(sorted(seeds)) != tuple(range(10)):
        raise ValueError("Protocol sensitivity requires development seeds 0..9.")
    output = Path(args.output).resolve()
    input_descriptors = []
    for stay in PROTOCOL_STAY_SENSITIVITY:
        payload = dict(base)
        payload["version"] = CONFIG_VERSION
        payload["method_variant"] = "b2"
        payload["model"] = dict(payload["model"])
        payload["model"]["protocol_stay_probability"] = float(stay)
        config_path = output / "configs" / f"pstay-{stay:.2f}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        combined_sequences = []
        panel_sha = None
        layout = None
        for seed in seeds:
            run = output / "runs" / f"pstay-{stay:.2f}" / f"seed-{seed}"
            run_training(
                SimpleNamespace(
                    config=str(config_path),
                    partner_manifest=str(Path(args.partner_manifest).resolve()),
                    ego_run_id=f"depi-pstay-{stay:.2f}-seed-{seed}",
                    seed_index=seed,
                    run_kind="development",
                    output=str(run),
                    pair_comparator=str(Path(args.pair_comparator).resolve()),
                    comparator_reference_ego_checkpoint=str(
                        Path(args.comparator_reference_ego_checkpoint).resolve()
                    ),
                    resume=bool(args.resume),
                    skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                    _execution_scope="protocol-sensitivity-matrix",
                )
            )
            sequence_path = (
                output
                / "sequences"
                / f"pstay-{stay:.2f}"
                / f"seed-{seed}.json"
            )
            collect_protocol_sensitivity_sequences(
                SimpleNamespace(
                    config=str(config_path),
                    training_run=str(run),
                    panel_comparator=str(Path(args.pair_comparator).resolve()),
                    output=str(sequence_path),
                )
            )
            sequence_payload = json.loads(sequence_path.read_text(encoding="utf-8"))
            panel_sha = (
                sequence_payload["panel_sha256"]
                if panel_sha is None
                else panel_sha
            )
            layout = sequence_payload["layout"] if layout is None else layout
            if (
                panel_sha != sequence_payload["panel_sha256"]
                or layout != sequence_payload["layout"]
            ):
                raise RuntimeError("Protocol sensitivity seed panels differ.")
            combined_sequences.extend(
                {
                    **row,
                    "panel_id": f"seed-{seed}:{row['panel_id']}",
                }
                for row in sequence_payload["sequences"]
            )
        combined_path = output / "sequences" / f"pstay-{stay:.2f}.json"
        write_json(
            combined_path,
            {
                "version": 1,
                "artifact_type": "depi_protocol_sensitivity_sequences",
                "development_only": True,
                "layout": layout,
                "protocol_stay_probability": float(stay),
                "panel_sha256": panel_sha,
                "sequences": combined_sequences,
            },
        )
        input_descriptors.append(f"{stay:.2f}={combined_path}")
    summarize_protocol_sensitivity(
        SimpleNamespace(input=input_descriptors, output=str(output / "summary"))
    )


__all__ = [
    "collect_protocol_sensitivity_sequences",
    "run_protocol_sensitivity_matrix",
    "summarize_protocol_sensitivity",
]
