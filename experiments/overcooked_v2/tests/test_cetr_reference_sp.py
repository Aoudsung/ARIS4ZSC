from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_reference_sp_maps_engineering_seed_minus_one_to_zero(tmp_path: Path, monkeypatch) -> None:
    import experiments.overcooked_v2.reference_sp_app as reference_sp_app
    from src.cetr_zsc.storage import read_json

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    keys = np.arange(20, dtype=np.uint32).reshape(10, 2)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        reference_sp_app,
        "population_sp_keys",
        lambda: keys,
    )
    monkeypatch.setattr(
        reference_sp_app,
        "restore_official_checkpoint",
        lambda path: (None, None),
    )
    monkeypatch.setattr(reference_sp_app, "official_policy", lambda params, config: object())
    monkeypatch.setattr(
        reference_sp_app,
        "VectorEnvironment",
        SimpleNamespace(create=lambda config: SimpleNamespace(environment=object())),
    )

    def fake_reference_returns(*, policy, environment, root_key, episodes):
        del policy, environment
        observed["root_key"] = root_key
        observed["episodes"] = episodes
        return np.asarray([3.0, 5.0], dtype=np.float64)

    monkeypatch.setattr(reference_sp_app, "_reference_returns", fake_reference_returns)
    args = SimpleNamespace(
        config="experiments/overcooked_v2/configs/cetr_simple_mechanical.yaml",
        run_kind="mechanical",
        sp_checkpoint=str(checkpoint),
        seed_index=-1,
        output=str(tmp_path / "reference.json"),
    )
    reference_sp_app.measure_reference_sp(args)

    payload = read_json(tmp_path / "reference.json")
    assert payload["seed_index"] == 0
    np.testing.assert_array_equal(np.asarray(observed["root_key"]), keys[0])
    assert observed["episodes"] == 500
    assert payload["tau_sp"] == 4.0


def test_reference_sp_reuses_population_diagonal_key_schedule() -> None:
    import experiments.overcooked_v2.evaluation_app as evaluation_app
    import experiments.overcooked_v2.reference_sp_app as reference_sp_app

    assert reference_sp_app.population_sp_keys is evaluation_app.population_sp_keys


def test_formal_reference_sp_requires_cuda_platform(tmp_path: Path, monkeypatch) -> None:
    import experiments.overcooked_v2.reference_sp_app as reference_sp_app

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        reference_sp_app,
        "load_config",
        lambda path, run_kind: SimpleNamespace(run_kind="formal"),
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    args = SimpleNamespace(
        config="unused.yaml",
        run_kind="formal",
        sp_checkpoint=str(checkpoint),
        seed_index=0,
        output=str(tmp_path / "reference.json"),
    )
    try:
        reference_sp_app.measure_reference_sp(args)
    except RuntimeError as error:
        assert "JAX_PLATFORMS=cuda" in str(error)
    else:
        raise AssertionError("formal reference-SP measurement accepted a non-CUDA platform")


def test_formal_reference_sp_requires_one_visible_gpu(tmp_path: Path, monkeypatch) -> None:
    import jax

    import experiments.overcooked_v2.reference_sp_app as reference_sp_app

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        reference_sp_app,
        "load_config",
        lambda path, run_kind: SimpleNamespace(run_kind="formal"),
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setattr(jax, "devices", lambda: [])
    args = SimpleNamespace(
        config="unused.yaml",
        run_kind="formal",
        sp_checkpoint=str(checkpoint),
        seed_index=0,
        output=str(tmp_path / "reference.json"),
    )
    try:
        reference_sp_app.measure_reference_sp(args)
    except RuntimeError as error:
        assert "exactly one visible CUDA GPU" in str(error)
    else:
        raise AssertionError("formal reference-SP measurement accepted zero visible GPUs")


def test_formal_reference_sp_rejects_registered_peak_memory_limit(
    tmp_path: Path, monkeypatch
) -> None:
    import jax

    import experiments.overcooked_v2.reference_sp_app as reference_sp_app
    from src.cetr_zsc.config import FORMAL_PEAK_MEMORY_LIMIT_BYTES

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    config = SimpleNamespace(
        run_kind="formal",
        environment=SimpleNamespace(layout="test_time_simple"),
        evaluation=SimpleNamespace(episodes_per_pairing=1),
    )
    monkeypatch.setattr(reference_sp_app, "load_config", lambda path, run_kind: config)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(reference_sp_app, "validate_official_runtime", lambda: None)
    monkeypatch.setattr(reference_sp_app, "restore_official_checkpoint", lambda path: (None, None))
    monkeypatch.setattr(reference_sp_app, "official_policy", lambda params, config: object())
    monkeypatch.setattr(
        reference_sp_app,
        "VectorEnvironment",
        SimpleNamespace(create=lambda config: SimpleNamespace(environment=object())),
    )
    monkeypatch.setattr(reference_sp_app, "population_sp_keys", lambda: [0])
    monkeypatch.setattr(
        reference_sp_app,
        "_reference_returns",
        lambda **kwargs: np.asarray([1.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        reference_sp_app,
        "peak_device_memory_bytes",
        lambda: FORMAL_PEAK_MEMORY_LIMIT_BYTES,
    )
    args = SimpleNamespace(
        config="unused.yaml",
        run_kind="formal",
        sp_checkpoint=str(checkpoint),
        seed_index=0,
        output=str(tmp_path / "reference.json"),
    )
    with pytest.raises(RuntimeError, match="peak device memory"):
        reference_sp_app.measure_reference_sp(args)
