"""S2 controlled phase-diagram task (Bernoulli evidence routing).

Implements the registered specification
``docs/research/S2_PHASE_TASK_SPEC.md``.  Pure exact enumeration with no
Monte Carlo and no learned components: every quantity is a closed-form or
exact-sum value of the finite task family.  Any registered check failure is
reported fail-closed and the process exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REGISTERED_GRID = {
    "delta": (2.0, 4.0, 8.0),
    "kappa": (0.05, 0.1, 0.2, 0.35, 0.5),
    "n_e": (8, 16, 32),
    "c": (0.0, 0.05),
    "lock": (False, True),
}
DIRECTED_WINDOW_POINT = {
    "delta": 2.0,
    "kappa": 0.05,
    "n_e": 32,
    "c": 0.05,
    "n_lock": 32,
    "t_prefix": 64,
}
CHECK_TOLERANCE = 1e-9


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """Exact Binomial(n, p) pmf in log space (underflow-safe for large n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly inside (0, 1)")
    if n == 0:
        return np.ones(1, dtype=np.float64)
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    log_n_fact = math.lgamma(n + 1)
    log_pmf = np.array(
        [
            log_n_fact
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
            for k in range(n + 1)
        ],
        dtype=np.float64,
    )
    pmf = np.exp(log_pmf - log_pmf.max())
    pmf /= pmf.sum()
    return pmf


def binomial_tv(n: int, p0: float, p1: float) -> float:
    """Exact total variation between Binomial(n, p0) and Binomial(n, p1)."""
    if n == 0:
        return 0.0
    return 0.5 * float(
        np.abs(binomial_pmf(n, p0) - binomial_pmf(n, p1)).sum()
    )


_TV_SEQUENCE_CACHE: dict[tuple[float, int], np.ndarray] = {}


def tv_sequence(n_max: int, kappa: float) -> np.ndarray:
    key = (kappa, n_max)
    cached = _TV_SEQUENCE_CACHE.get(key)
    if cached is not None:
        return cached
    p0, p1 = evidence_probabilities(kappa)
    sequence = np.array([binomial_tv(n, p0, p1) for n in range(n_max + 1)])
    _TV_SEQUENCE_CACHE[key] = sequence
    return sequence


def evidence_probabilities(kappa: float) -> tuple[float, float]:
    if not 0.0 < kappa <= 0.5:
        raise ValueError("kappa must be in (0, 0.5]")
    return (1.0 - kappa) / 2.0, (1.0 + kappa) / 2.0


def t3_lower_bound(delta: float, epsilon: float, kappa: float) -> int:
    return math.ceil(math.log(delta / (4.0 * epsilon)) / (4.0 * kappa * kappa))


def t3_upper_bound(delta: float, epsilon: float, kappa: float) -> int:
    return math.ceil(2.0 * math.log(delta / epsilon) / (kappa * kappa))


def evaluate_point(
    delta: float,
    kappa: float,
    n_e: int,
    c: float,
    n_lock: int,
    t_prefix: int,
    epsilon: float,
) -> dict:
    """Exact six-control values at the registered commit time n_e.

    All controls are evaluated committing at the same time n_e so every
    object pays the identical waiting cost; the T2 identity then holds at
    fixed commit time.  ``v_hz_free`` reports the label-informed value that
    commits immediately (the J_HZ reference used in window accounting).
    """
    if n_lock > n_e:
        raise ValueError("n_lock cannot exceed n_e")
    p0, p1 = evidence_probabilities(kappa)
    n_scan = max(n_e, t3_upper_bound(delta, epsilon, kappa))
    tvs = tv_sequence(n_scan, kappa)

    wait = c * (t_prefix + n_e)
    v_fix = delta / 2.0 - wait
    v_state = delta * (1.0 + kappa) / 2.0 - wait
    v_hist = delta * (1.0 + float(tvs[n_e])) / 2.0 - wait
    v_hz = delta - wait
    v_hz_free = delta - c * t_prefix
    v_full = v_hz

    regrets = delta * (1.0 - tvs) / 2.0
    n_emp = int(np.argmax(regrets <= epsilon)) if (regrets <= epsilon).any() else -1

    point = {
        "delta": delta,
        "kappa": kappa,
        "n_e": n_e,
        "c": c,
        "n_lock": n_lock,
        "t_prefix": t_prefix,
        "epsilon": epsilon,
        "v_fix": v_fix,
        "v_state": v_state,
        "v_hist": v_hist,
        "v_hz": v_hz,
        "v_hz_free": v_hz_free,
        "v_full": v_full,
        "tv_at_n_e": float(tvs[n_e]),
        "n_emp": n_emp,
        "n_lower": t3_lower_bound(delta, epsilon, kappa),
        "n_upper": t3_upper_bound(delta, epsilon, kappa),
    }
    point["checks"] = run_checks(point, tvs)
    return point


def run_checks(point: dict, tvs: np.ndarray) -> dict:
    tol = CHECK_TOLERANCE
    delta = point["delta"]
    tv_n = point["tv_at_n_e"]

    chain = [
        point["v_fix"],
        point["v_state"],
        point["v_hist"],
        point["v_hz"],
        point["v_full"],
    ]
    chain_ok = all(right >= left - tol for left, right in zip(chain, chain[1:]))

    t2_gap_hz = abs(point["v_hz"] - point["v_hist"] - delta * (1.0 - tv_n) / 2.0)
    t2_gap_fix = abs(
        (point["v_hist"] - point["v_fix"]) - delta * tv_n / 2.0
    )
    t2_ok = t2_gap_hz <= tol and t2_gap_fix <= tol

    if point["n_emp"] < 0:
        t3_ok, t3_detail = False, "n_emp not found within scan range"
    else:
        t3_ok = point["n_lower"] <= point["n_emp"] <= point["n_upper"]
        t3_detail = (
            f"n_lower={point['n_lower']} n_emp={point['n_emp']} "
            f"n_upper={point['n_upper']}"
        )

    return {
        "t1_chain_ok": bool(chain_ok),
        "t2_ok": bool(t2_ok),
        "t2_residual_hz": t2_gap_hz,
        "t2_residual_fix": t2_gap_fix,
        "t3_ok": bool(t3_ok),
        "t3_detail": t3_detail,
    }


def directed_window_point() -> dict:
    """Check 4: late evidence with waiting cost kills the end-to-end gain."""
    cfg = DIRECTED_WINDOW_POINT
    delta, kappa = cfg["delta"], cfg["kappa"]
    epsilon = delta / 8.0
    n_scan = max(cfg["n_e"], t3_upper_bound(delta, epsilon, kappa))
    tvs = tv_sequence(n_scan, kappa)

    j_fix0 = delta / 2.0
    j_sfix = -cfg["c"] * cfg["t_prefix"] + delta / 2.0
    j_hz = delta - cfg["c"] * cfg["t_prefix"]
    routed = [
        -cfg["c"] * (cfg["t_prefix"] + n) + delta * (1.0 + float(tvs[n])) / 2.0
        for n in range(cfg["n_e"] + 1)
    ]
    j_route = max(routed)

    return {
        **cfg,
        "epsilon": epsilon,
        "j_fix0": j_fix0,
        "j_sfix": j_sfix,
        "j_hz": j_hz,
        "j_route": j_route,
        "oracle_gain": j_hz - j_sfix,
        "end_to_end_gain": j_route - j_fix0,
        "checks": {
            "window_oracle_positive": bool(j_hz - j_sfix > 0.0),
            "window_end_to_end_nonpositive": bool(j_route - j_fix0 <= 0.0),
        },
    }


def build_grid() -> list[dict]:
    grid = []
    for delta in REGISTERED_GRID["delta"]:
        for kappa in REGISTERED_GRID["kappa"]:
            for n_e in REGISTERED_GRID["n_e"]:
                for c in REGISTERED_GRID["c"]:
                    for lock in REGISTERED_GRID["lock"]:
                        n_lock = n_e // 2 if lock else n_e
                        grid.append(
                            {
                                "delta": delta,
                                "kappa": kappa,
                                "n_e": n_e,
                                "c": c,
                                "n_lock": n_lock,
                                "t_prefix": 0,
                            }
                        )
    return grid


def figure1_slice(points: list[dict]) -> list[dict]:
    """Recoverable value on the zero-wait-cost slice for Figure 1."""
    return [
        {
            "delta": p["delta"],
            "kappa": p["kappa"],
            "n_e": p["n_e"],
            "recoverable": p["v_hist"] - p["v_fix"],
        }
        for p in points
        if p["c"] == 0.0 and not _is_locked(p)
    ]


def _is_locked(point: dict) -> bool:
    return point["n_lock"] < point["n_e"]


def run_scan(quick: bool) -> dict:
    grid = build_grid()
    if quick:
        grid = grid[:4]
    points = [
        evaluate_point(epsilon=spec["delta"] / 8.0, **spec) for spec in grid
    ]
    window = directed_window_point()

    failures = []
    for point in points:
        checks = point["checks"]
        for key in ("t1_chain_ok", "t2_ok", "t3_ok"):
            if not checks[key]:
                failures.append({"point": point, "failed": key})
    window_checks = window["checks"]
    for key, passed in window_checks.items():
        if not passed:
            failures.append({"point": window, "failed": key})

    return {
        "points": points,
        "directed_window": window,
        "figure1": figure1_slice(points),
        "summary": {
            "n_points": len(points),
            "t1_pass": sum(p["checks"]["t1_chain_ok"] for p in points),
            "t2_pass": sum(p["checks"]["t2_ok"] for p in points),
            "t3_pass": sum(p["checks"]["t3_ok"] for p in points),
            "window_pass": all(window_checks.values()),
            "max_t2_residual": max(
                max(p["checks"]["t2_residual_hz"], p["checks"]["t2_residual_fix"])
                for p in points
            ),
            "failures": len(failures),
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S2 controlled phase-diagram exact enumeration."
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first four grid points (smoke mode).",
    )
    args = parser.parse_args()

    result = run_scan(quick=args.quick)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "points.json").write_text(
        json.dumps(result["points"], indent=2), encoding="utf-8"
    )
    (output / "figure1.json").write_text(
        json.dumps(result["figure1"], indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "summary": result["summary"],
                "directed_window": result["directed_window"],
                "failures": result["failures"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        raise SystemExit(
            f"fail-closed: {len(result['failures'])} registered check failures"
        )


if __name__ == "__main__":
    main()
