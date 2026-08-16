from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _manifest(path: Path) -> Path:
    rows = []
    index = 0
    for mechanism in ("sp", "sa", "op", "fcp"):
        for parent_index in range(4):
            rows.append(
                {
                    "run_id": f"partner-{index}",
                    "role": "confirmatory",
                    "checkpoint": "unused-checkpoint",
                    "parent_training_run_id": f"{mechanism}-parent-{parent_index}",
                    "generation_mechanism": mechanism,
                    "checkpoint_stage": 1.0,
                    "hyperparameter_family": "test",
                    "seed": index,
                    "seed_index": index,
                    "jax_prng_key": [0, index],
                    "owner_seed_index": None,
                    "co_training_group_id": None,
                    "partner_type_id": None,
                }
            )
            index += 1
    payload = {"version": 2, "layout": "test_time_simple", "runs": rows}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evaluation(
    root: Path,
    *,
    method: str,
    prefix: str,
    value: float,
    partner_manifest: Path,
) -> Path:
    directory = root / method
    directory.mkdir()
    raw = directory / "episode_returns.jsonl"
    with raw.open("w", encoding="utf-8") as handle:
        for ego_index in range(10):
            for partner_index in range(16):
                handle.write(
                    json.dumps(
                        {
                            "ego_run_index": ego_index,
                            "ego_run_id": f"{prefix}-run-{ego_index}",
                            "partner_run_id": f"partner-{partner_index}",
                            "raw_return": value,
                        }
                    )
                    + "\n"
                )
    (directory / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "artifact_type": "cetr_raw_evaluation",
                "evaluation_mode": "common_partner",
                "method": method,
                "layout": "test_time_simple",
                "raw": {"path": str(raw.resolve())},
                "partner_manifest": {"path": str(partner_manifest.resolve())},
            }
        ),
        encoding="utf-8",
    )
    return directory


def _claim_inputs(tmp_path: Path, *, external_delta: float, sp_delta: float):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(tmp_path / "partners.json")
    cetr = _evaluation(
        tmp_path,
        method="cetr-zsc",
        prefix="cetr",
        value=external_delta,
        partner_manifest=manifest,
    )
    fcp = _evaluation(
        tmp_path,
        method="fcp",
        prefix="fcp",
        value=0.0,
        partner_manifest=manifest,
    )
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps({"sp_diagonal": [1.0 + sp_delta] * 10}), encoding="utf-8"
    )
    references = []
    for seed in range(10):
        reference = tmp_path / f"reference-{seed}.json"
        reference.write_text(
            json.dumps(
                {
                    "artifact_type": "cetr_reference_sp",
                    "version": 2,
                    "layout": "test_time_simple",
                    "seed_index": seed,
                    "tau_sp": 1.0,
                    "episodes_per_pairing": 500,
                    "evaluation_root_seed": 0,
                    "source_checkpoint": str((tmp_path / f"ckpt-{seed}").resolve()),
                }
            ),
            encoding="utf-8",
        )
        references.append(str(reference))
    return SimpleNamespace(
        cetr_evaluation=[str(cetr)],
        baseline_evaluation=[str(fcp)],
        cetr_population=[str(population)],
        reference_sp=references,
        output=str(tmp_path / "claim"),
    )


def test_run_index_alignment_accepts_different_ego_run_ids_and_decides_go(tmp_path, monkeypatch):
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 64)
    args = _claim_inputs(tmp_path, external_delta=10.0, sp_delta=1.0)
    claim_app.build_claim(args)
    report = json.loads((tmp_path / "claim" / "claim.json").read_text())
    assert report["decision"]["status"] == "GO"
    assert all(row["difference"] == 1.0 for row in report["sp_unit_table"])
    assert report["panel"]["ego_run_indexes"] == list(range(10))


def test_claim_decides_no_go_for_clear_external_and_sp_failures(tmp_path, monkeypatch):
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 64)
    args = _claim_inputs(tmp_path, external_delta=0.0, sp_delta=-1.0)
    # Make FCP exceed CETR on every external unit.
    baseline_raw = Path(args.baseline_evaluation[0]) / "episode_returns.jsonl"
    rows = [json.loads(line) for line in baseline_raw.read_text().splitlines()]
    baseline_raw.write_text(
        "".join(json.dumps({**row, "raw_return": 10.0}) + "\n" for row in rows),
        encoding="utf-8",
    )
    claim_app.build_claim(args)
    report = json.loads((tmp_path / "claim" / "claim.json").read_text())
    assert report["decision"]["status"] == "NO-GO"


def test_fcp_method_validation_rejects_non_fcp_sets(tmp_path, monkeypatch):
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
    args = _claim_inputs(tmp_path, external_delta=1.0, sp_delta=1.0)
    baseline = Path(args.baseline_evaluation[0]) / "evaluation_summary.json"
    payload = json.loads(baseline.read_text())
    payload["method"] = "op"
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the FCP"):
        claim_app.build_claim(args)

    args = _claim_inputs(tmp_path / "mixed", external_delta=1.0, sp_delta=1.0)
    extra = _evaluation(
        tmp_path / "mixed",
        method="op",
        prefix="op",
        value=0.0,
        partner_manifest=Path(args.baseline_evaluation[0]).parent / "../partners.json",
    )
    args.baseline_evaluation.append(str(extra))
    with pytest.raises(ValueError, match="exactly the FCP"):
        claim_app.build_claim(args)


def test_crossed_bootstrap_resamples_both_axes(monkeypatch):
    import experiments.overcooked_v2.claim_app as claim_app

    calls: list[np.ndarray] = []

    class FixedRng:
        def integers(self, low, high, size):
            del low
            values = np.full(size, len(calls) // 2 % high, dtype=np.int64)
            calls.append(values.copy())
            return values

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 4)
    monkeypatch.setattr(claim_app.np.random, "default_rng", lambda seed: FixedRng())
    result = claim_app._bootstrap_external(
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 0.0, 0.0]]),
        np.zeros((3, 4), dtype=np.float64),
        statistic="mean",
        seed=7,
    )
    ego_draws = calls[0::2]
    parent_draws = calls[1::2]
    assert len(ego_draws) == len(parent_draws) == 4
    assert not np.array_equal(ego_draws[0], ego_draws[1])
    assert not np.array_equal(parent_draws[0], parent_draws[1])
    assert all(draw.shape == (4,) for draw in parent_draws)
    assert result["interval_95"][0] < result["interval_95"][1]


def test_sp_pairing_requires_all_seeds_and_reports_hand_calculated_differences(tmp_path, monkeypatch):
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
    args = _claim_inputs(tmp_path, external_delta=10.0, sp_delta=1.0)
    args.reference_sp = args.reference_sp[:-1]
    with pytest.raises(ValueError, match="ten reference artifacts"):
        claim_app.build_claim(args)


def test_parent_level_cvar50_matches_hand_calculated_column_tail() -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    values = np.arange(16, dtype=np.float64)[None, :] + np.arange(10, dtype=np.float64)[:, None]
    # Parent means are 4.5 through 19.5; the lower eight average to 8.0.
    assert claim_app._cvar50(values) == 8.0


def test_gate_combinations_fail_independently(tmp_path, monkeypatch) -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    cases = (
        ("mean", -1.0, 1.0, "INCONCLUSIVE"),
        ("cvar50", 1.0, -1.0, "INCONCLUSIVE"),
        ("sp", 1.0, 1.0, "NO-GO"),
    )
    for name, mean_lcb, cvar_lcb, expected_status in cases:
        root = tmp_path / name
        args = _claim_inputs(root, external_delta=10.0, sp_delta=-1.0 if name == "sp" else 1.0)

        def fake_bootstrap(left, right, *, statistic, seed=0):
            del left, right, seed
            lcb = mean_lcb if statistic == "mean" else cvar_lcb
            return {
                "estimate": lcb,
                "interval_95": [lcb, lcb],
                "lcb95": lcb,
                "bootstrap_replicates": 8,
                "bootstrap_unit": "ego_run_x_parent_lineage",
            }

        monkeypatch.setattr(claim_app, "_bootstrap_external", fake_bootstrap)
        monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
        claim_app.build_claim(args)
        report = json.loads((root / "claim" / "claim.json").read_text())
        assert report["decision"]["status"] == expected_status


def test_sp_pairing_rejects_duplicate_and_wrong_seed_sets(tmp_path, monkeypatch) -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
    duplicate = _claim_inputs(tmp_path / "duplicate", external_delta=10.0, sp_delta=1.0)
    payload = json.loads(Path(duplicate.reference_sp[1]).read_text())
    payload["seed_index"] = 0
    Path(duplicate.reference_sp[1]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        claim_app.build_claim(duplicate)

    wrong = _claim_inputs(tmp_path / "wrong", external_delta=10.0, sp_delta=1.0)
    payload = json.loads(Path(wrong.reference_sp[-1]).read_text())
    payload["seed_index"] = 10
    Path(wrong.reference_sp[-1]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one reference artifact"):
        claim_app.build_claim(wrong)


def test_cetr_slot_rejects_a_non_cetr_method_summary(tmp_path, monkeypatch) -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
    args = _claim_inputs(tmp_path, external_delta=10.0, sp_delta=1.0)
    summary = Path(args.cetr_evaluation[0]) / "evaluation_summary.json"
    payload = json.loads(summary.read_text())
    payload["method"] = "fcp"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cetr-zsc"):
        claim_app.build_claim(args)


def test_claim_matrix_is_fail_closed_at_registered_scale() -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    small = {
        (ego, f"parent-{parent}", "sp"): 0.0
        for ego in range(2)
        for parent in range(2)
    }
    with pytest.raises(ValueError, match="registered ego"):
        claim_app._matrix(small)

    undersized = {
        (ego, f"parent-{parent}", ("sp", "sa", "op", "fcp")[parent // 4]): 0.0
        for ego in range(10)
        for parent in range(15)
    }
    with pytest.raises(ValueError, match="parent count"):
        claim_app._matrix(undersized)

    valid = {
        (ego, f"parent-{parent}", ("sp", "sa", "op", "fcp")[parent // 4]): float(parent)
        for ego in range(10)
        for parent in range(16)
    }
    values, egos, parents = claim_app._matrix(valid)
    assert values.shape == (10, 16)
    assert egos == tuple(range(10))
    assert len(parents) == 16


def test_sp_pairing_uses_each_seed_tau_individually(tmp_path, monkeypatch) -> None:
    import experiments.overcooked_v2.claim_app as claim_app

    monkeypatch.setattr(claim_app, "OFFICIAL_BOOTSTRAP_REPLICATES", 8)
    args = _claim_inputs(tmp_path, external_delta=10.0, sp_delta=1.0)
    population = Path(args.cetr_population[0])
    population.write_text(
        json.dumps({"sp_diagonal": [20.0 + seed for seed in range(10)]}),
        encoding="utf-8",
    )
    for seed, reference_path in enumerate(args.reference_sp):
        payload = json.loads(Path(reference_path).read_text())
        payload["tau_sp"] = 10.0 + seed
        Path(reference_path).write_text(json.dumps(payload), encoding="utf-8")

    claim_app.build_claim(args)
    report = json.loads((tmp_path / "claim" / "claim.json").read_text())
    assert [row["tau_sp"] for row in report["sp_unit_table"]] == [10.0 + seed for seed in range(10)]
    assert [row["difference"] for row in report["sp_unit_table"]] == [10.0] * 10
