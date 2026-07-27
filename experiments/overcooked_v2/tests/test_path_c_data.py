from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.path_c.experiment import (
    METHOD_VERSION,
    Population,
    PopulationEntry,
    load_config,
    load_population,
    write_population,
)
from src.path_c.storage import (
    CompleteConsoleLog,
    ensure_run_identity,
    evaluation_identity,
    read_array_chunks,
    read_run_identity,
    training_identity,
    write_array_chunks,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[3]
SIMPLE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_simple.yaml"
WIDE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_wide.yaml"


def test_run_kind_selects_registered_budget() -> None:
    development = load_config(SIMPLE_CONFIG, run_kind="development")
    formal = load_config(SIMPLE_CONFIG, run_kind="formal")
    assert development.environment.num_envs == 32
    assert development.training.environment_steps == 1_228_800
    assert development.training.minibatches_per_epoch == 8
    assert formal.environment.num_envs == 250
    assert formal.training.environment_steps == 11_000_000
    assert formal.training.minibatches_per_epoch == 50


def test_layout_configs_differ_only_by_layout() -> None:
    simple = load_config(SIMPLE_CONFIG, run_kind="development")
    wide = load_config(WIDE_CONFIG, run_kind="development")
    assert simple.environment.layout == "test_time_simple"
    assert wide.environment.layout == "test_time_wide"
    assert replace(simple, environment=wide.environment) == wide


def test_unknown_configuration_field_fails(tmp_path: Path) -> None:
    import yaml

    payload = yaml.safe_load(SIMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["silent_limit"] = 10
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path, run_kind="development")


def test_run_identity_is_single_literal_source(tmp_path: Path) -> None:
    config = load_config(SIMPLE_CONFIG, run_kind="development")
    expected = training_identity(
        config=config,
        seed=100,
        outer_unit_id=0,
        reference_checkpoint=tmp_path / "reference",
        partner_checkpoints=(tmp_path / "partner-a", tmp_path / "partner-b"),
    )
    ensure_run_identity(tmp_path / "run", expected)
    assert read_run_identity(tmp_path / "run") == expected

    changed = dict(expected)
    changed["outer_unit_id"] = 1
    with pytest.raises(RuntimeError, match="different experiment"):
        ensure_run_identity(tmp_path / "run", changed)


def test_evaluation_identity_binds_population_and_seed(tmp_path: Path) -> None:
    config = load_config(SIMPLE_CONFIG, run_kind="development")
    population = Population(
        name="development",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(index, tmp_path / f"run-{index}") for index in range(10)
        ),
    )
    first = evaluation_identity(
        config=config, seed=17, population=population.to_mapping()
    )
    second = evaluation_identity(
        config=config, seed=18, population=population.to_mapping()
    )
    assert first["method"] == METHOD_VERSION
    assert first != second


def test_population_records_training_run_directories_only(tmp_path: Path) -> None:
    population = Population(
        name="posterior-use",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(index, tmp_path / f"run-{index}") for index in range(10)
        ),
    )
    path = write_population(tmp_path / "population.json", population)
    assert load_population(path) == population
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["policies"][0]) == {"outer_unit_id", "run_directory"}


def test_jsonl_and_array_chunks_are_lossless(tmp_path: Path) -> None:
    long_text = "响应🙂\n制表符\t引号\"" * 100
    rows = ({"index": index, "text": long_text + str(index)} for index in range(257))
    path = write_jsonl(tmp_path / "records.jsonl", rows)
    restored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(restored) == 257
    assert restored[-1]["text"] == long_text + "256"

    values = np.arange(2_317 * 7, dtype=np.float32).reshape(2_317, 7)
    chunks = write_array_chunks(
        tmp_path / "arrays", name="trajectory", values=values, rows_per_chunk=97
    )
    np.testing.assert_array_equal(read_array_chunks(chunks), values)


def test_console_log_keeps_exception_trace(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="完整异常消息"):
        with CompleteConsoleLog(tmp_path / "logs"):
            raise RuntimeError("完整异常消息")
    stderr = (tmp_path / "logs/stderr.log").read_text(encoding="utf-8")
    assert "Traceback" in stderr
    assert "RuntimeError: 完整异常消息" in stderr
