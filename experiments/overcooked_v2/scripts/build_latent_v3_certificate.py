"""Aggregate Link-A latent_v3 Phenomenon-Existence Certificate artifacts."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _get(dct: dict[str, Any], path: list[str], default=None):
    cur: Any = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _scan_chunks(chunks: str | Path, golden_review: str | None) -> dict[str, Any]:
    metas = []
    for path in sorted(glob.glob(str(Path(chunks) / "*.meta.json"))):
        metas.append(_load(path))
    policies = sorted(set(str(meta.get("evidence_policy")) for meta in metas))
    oracle_source_count = int(sum(int(meta.get("oracle_source_count", 0)) for meta in metas))
    families = sorted(set(
        str(_get(meta, ["latent_partner_spec", "mode", "family"], "unknown"))
        for meta in metas
        if _get(meta, ["latent_partner_spec", "mode", "family"]) is not None
    ))
    golden_files = sorted(glob.glob(str(Path(chunks) / "*.golden.txt")))
    review = _load(golden_review) if golden_review else {}
    family_review = review.get("families", {}) if isinstance(review, dict) else {}
    required_families = set(families)
    reviewed_pass = {
        family
        for family, verdict in family_review.items()
        if str(verdict).upper() == "PASS"
    }
    return {
        "n_chunks": len(metas),
        "n_golden_files": len(golden_files),
        "evidence_policies": policies,
        "oracle_source_count": oracle_source_count,
        "families": families,
        "golden_review_path": str(golden_review) if golden_review else None,
        "golden_reviewed_pass_families": sorted(reviewed_pass),
        "golden_review_pass": bool(required_families and required_families <= reviewed_pass),
        "PASS": bool(
            metas
            and oracle_source_count == 0
            and policies == ["behavior_inferred_v1"]
            and required_families
            and required_families <= reviewed_pass
        ),
    }


def _criterion(name: str, value: float, threshold: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "value": value, "threshold": threshold, "pass": bool(passed)}


def run(args: argparse.Namespace) -> None:
    readout = _load(args.d1_readout)
    tune = _load(args.d1_tune)
    mode_cert = _load(args.mode_cert)
    voi = _load(args.voi)
    wiring = _scan_chunks(args.chunks, args.golden_review)

    blind = readout["splits"]["blind_terminal_main"]
    c1_value = float(blind["auc_ps_nohist"])
    c2_gain = float(blind["gain"])
    c2_ci_lo = float(_get(blind, ["gain_ci", "lo"], float("nan")))
    indist = tune["k_blocks"]["k5"]["indist_val_main_ensemble"]
    c3_value = float(indist["auc_ps_idoracle"]) - float(indist["auc_ps_nohist"])
    c4_full = float(mode_cert["variants"]["history_full"]["blind_balanced_acc_median"])
    c4_state = float(mode_cert["variants"]["state_only"]["blind_balanced_acc_median"])
    c5_value = float(voi["criteria"]["min_relative_lift"])

    criteria = {
        "C-1": _criterion("unsaturation", c1_value, "NOHIST blind AUC <= 0.80", c1_value <= 0.80),
        "C-2": {
            "name": "signal exists and needs history",
            "gain": c2_gain,
            "gain_ci_lo": c2_ci_lo,
            "threshold": "G_blind >= 0.10 and gain_ci.lo > 0",
            "pass": bool(c2_gain >= 0.10 and c2_ci_lo > 0.0),
        },
        "C-3": _criterion(
            "identity beyond state",
            c3_value,
            "IDORACLE - NOHIST indist >= 0.05",
            c3_value >= 0.05,
        ),
        "C-4": {
            "name": "identifiability",
            "history_full_balanced_acc": c4_full,
            "state_only_balanced_acc": c4_state,
            "threshold": "history_full >= 0.75 and state_only <= 0.60",
            "pass": bool(mode_cert["criteria"]["PASS"]),
        },
        "C-5": _criterion(
            "value of information",
            c5_value,
            "mode_oracle relative lift >= 0.15 vs best blind scripted ego",
            bool(voi["criteria"]["PASS"]),
        ),
        "C-6": {
            "name": "wiring",
            "summary": wiring,
            "threshold": "oracle_source_count=0, evidence_policy=behavior_inferred_v1, golden PASS per family",
            "pass": bool(wiring["PASS"]),
        },
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    cert = {
        "certificate": "latent_v3 Phenomenon-Existence Certificate",
        "status": "PASS" if passed else "FAIL",
        "commit": str(args.commit),
        "artifacts": {
            "d1_readout": str(args.d1_readout),
            "d1_tune": str(args.d1_tune),
            "mode_cert": str(args.mode_cert),
            "voi": str(args.voi),
            "chunks": str(args.chunks),
            "golden_review": str(args.golden_review) if args.golden_review else None,
        },
        "commands": [
            "EPS_SCALE=2 bash experiments/overcooked_v2/scripts/diag_link_a_generate_all.sh <anchor_ckpt> <chunks_dir> <jobs>",
            "python experiments/overcooked_v2/scripts/diag_d1_train.py --chunks <chunks_dir> --out <d1_out> --stage gate",
            "python experiments/overcooked_v2/scripts/diag_d1_train.py --chunks <chunks_dir> --out <d1_out> --stage tune",
            "python experiments/overcooked_v2/scripts/diag_d1_train.py --chunks <chunks_dir> --out <d1_out> --stage readout",
            "python experiments/overcooked_v2/scripts/diag_link_a_mode_cert.py --chunks <chunks_dir> --out <mode_cert>",
            "python experiments/overcooked_v2/scripts/latent_v3_voi_probe.py --checkpoint <anchor_ckpt> --out <voi>",
        ],
        "criteria": criteria,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(cert, indent=1))
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# latent_v3 Phenomenon-Existence Certificate",
        "",
        f"Status: **{cert['status']}**",
        f"Commit: `{cert['commit']}`",
        "",
        "| Check | Value | Threshold | Pass |",
        "|---|---:|---|---|",
    ]
    for key, item in criteria.items():
        if key == "C-2":
            value = f"gain={item['gain']:.6g}, ci_lo={item['gain_ci_lo']:.6g}"
        elif key == "C-4":
            value = (
                f"full={item['history_full_balanced_acc']:.6g}, "
                f"state={item['state_only_balanced_acc']:.6g}"
            )
        elif key == "C-6":
            value = (
                f"oracle={wiring['oracle_source_count']}, "
                f"golden_families={len(wiring['golden_reviewed_pass_families'])}"
            )
        else:
            value = f"{float(item['value']):.6g}"
        lines.append(
            f"| {key} {item['name']} | {value} | {item['threshold']} | {item['pass']} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        f"- D1 readout: `{args.d1_readout}`",
        f"- D1 tune: `{args.d1_tune}`",
        f"- Mode cert: `{args.mode_cert}`",
        f"- VOI: `{args.voi}`",
        f"- Chunks: `{args.chunks}`",
    ])
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": cert["status"], "criteria": {k: v["pass"] for k, v in criteria.items()}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-readout", required=True)
    ap.add_argument("--d1-tune", required=True)
    ap.add_argument("--mode-cert", required=True)
    ap.add_argument("--voi", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--golden-review", default=None)
    ap.add_argument("--commit", default="UNKNOWN")
    ap.add_argument("--out-json", default="artifacts/latent_v3_phenomenon_certificate.json")
    ap.add_argument("--out-md", default="artifacts/latent_v3_phenomenon_certificate.md")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

