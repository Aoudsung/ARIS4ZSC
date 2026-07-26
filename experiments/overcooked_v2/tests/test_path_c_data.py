from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.path_c.storage import (
    CompleteConsoleLog,
    load_config,
    orbax_manager,
    read_array_chunks,
    read_parquet,
    restore_latest_checkpoint,
    save_checkpoint,
    write_array_chunks,
    write_jsonl,
    write_parquet,
)


ROOT = Path(__file__).resolve().parents[3]
SIMPLE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_simple.yaml"
WIDE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_wide.yaml"


def test_layout_configs_have_one_shape_and_no_unused_aliases() -> None:
    simple = load_config(SIMPLE_CONFIG)
    wide = load_config(WIDE_CONFIG)
    assert simple.environment.layout == "test_time_simple"
    assert wide.environment.layout == "test_time_wide"
    assert replace(simple, environment=wide.environment) == wide


def test_unknown_configuration_field_fails_instead_of_being_ignored(
    tmp_path: Path,
) -> None:
    import yaml

    payload = yaml.safe_load(SIMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["silent_limit"] = 10
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_long_jsonl_and_special_characters_round_trip_without_truncation(
    tmp_path: Path,
) -> None:
    long_text = "响应🙂\n制表符\t引号\"" * 1_000
    rows = (
        {"index": index, "text": long_text + str(index)}
        for index in range(257)
    )
    path = write_jsonl(tmp_path / "records.jsonl", rows)
    restored = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(restored) == 257
    assert restored[0]["text"] == long_text + "0"
    assert restored[-1]["text"] == long_text + "256"


def test_console_log_keeps_the_complete_exception_trace(tmp_path: Path) -> None:
    log_directory = tmp_path / "logs"
    with pytest.raises(RuntimeError, match="完整异常消息"):
        with CompleteConsoleLog(log_directory):
            raise RuntimeError("完整异常消息")
    stderr = (log_directory / "stderr.log").read_text(encoding="utf-8")
    assert "Traceback" in stderr
    assert "RuntimeError: 完整异常消息" in stderr


def test_parquet_preserves_every_row_and_nested_values(tmp_path: Path) -> None:
    rows = [
        {
            "episode": index,
            "return": float(index) - 750.5,
            "responses": [index % 16, (index + 1) % 16],
            "label": f"回合-{index}",
        }
        for index in range(1_501)
    ]
    path = write_parquet(tmp_path / "episodes.parquet", rows)
    assert read_parquet(path) == rows


def test_lossless_array_chunks_restore_full_array(tmp_path: Path) -> None:
    values = np.arange(23_017 * 7, dtype=np.float32).reshape(23_017, 7)
    paths = write_array_chunks(
        tmp_path / "arrays", name="trajectory", values=values, rows_per_chunk=997
    )
    assert len(paths) > 20
    np.testing.assert_array_equal(read_array_chunks(paths), values)


def test_orbax_resume_uses_latest_complete_step_and_preserves_counts(
    tmp_path: Path,
) -> None:
    manager = orbax_manager(tmp_path / "checkpoints")
    save_checkpoint(
        manager,
        step=400,
        state={"effective_environment_steps": 400, "completed_episodes": 1},
    )
    save_checkpoint(
        manager,
        step=800,
        state={"effective_environment_steps": 800, "completed_episodes": 2},
    )
    restored = restore_latest_checkpoint(manager)
    assert restored is not None
    step, state = restored
    assert step == 800
    assert state == {"effective_environment_steps": 800, "completed_episodes": 2}
