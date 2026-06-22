"""
Symbolic toy pilot for the first experiment-bridge milestone.

This runner targets the plan's first toy-game priority before any OvercookedV2
work: compare G-TVOI, MI probing, passive inference, random task-valid options,
and an oracle diagnostic policy on known ground-truth interaction factors.

It deliberately evaluates against the toy environment's ground-truth convention
assignment. No method is scored against another model's output.
"""

import argparse
import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import bootstrap_ci, expected_calibration_error, pareto_frontier
from common.utils import save_results, set_seed
from toy_factor_game.env import (
    DELIVER_LEFT,
    DELIVER_RIGHT,
    FACTOR_MODES,
    NUM_FACTORS,
    RESOURCE_A,
    RESOURCE_B,
    Action,
    ConventionAssignment,
    ToyFactorGameEnv,
)
from toy_factor_game.options import OPTION_NAMES, OptionID, get_option_action


METHODS = ("gtvoi", "mi", "passive", "random", "oracle")
DIAGNOSTIC_OPTIONS = {
    OptionID.CROSS_CORRIDOR,
    OptionID.WAIT_AT_BOTTLENECK,
    OptionID.GOTO_RESOURCE_A,
    OptionID.DELIVER_LEFT,
    OptionID.DELIVER_RIGHT,
}


ConventionKey = tuple[int, ...]


def all_convention_keys() -> list[ConventionKey]:
    keys = []
    for f0 in range(FACTOR_MODES[0]):
        for f1 in range(FACTOR_MODES[1]):
            for f2 in range(FACTOR_MODES[2]):
                keys.append((f0, f1, f2))
    return keys


def key_to_assignment(key: ConventionKey) -> ConventionAssignment:
    return ConventionAssignment(modes={f: key[f] for f in range(NUM_FACTORS)})


def convention_label(key: ConventionKey) -> str:
    return "-".join(str(v) for v in key)


def entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


@dataclass
class BeliefState:
    """Joint posterior over the toy game's three ground-truth factor modes."""

    weights: dict[ConventionKey, float]

    @classmethod
    def uniform(cls) -> "BeliefState":
        keys = all_convention_keys()
        p = 1.0 / len(keys)
        return cls({key: p for key in keys})

    @classmethod
    def point_mass(cls, key: ConventionKey) -> "BeliefState":
        return cls({candidate: float(candidate == key) for candidate in all_convention_keys()})

    def normalized(self) -> "BeliefState":
        total = sum(self.weights.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Belief normalization failed: posterior mass is zero or non-finite")
        return BeliefState({key: value / total for key, value in self.weights.items()})

    def marginals(self) -> list[np.ndarray]:
        out = [np.zeros(FACTOR_MODES[f], dtype=np.float64) for f in range(NUM_FACTORS)]
        for key, prob in self.weights.items():
            for factor_idx, mode in enumerate(key):
                out[factor_idx][mode] += prob
        return out

    def entropy_vector(self) -> np.ndarray:
        return np.array([entropy(m) for m in self.marginals()], dtype=np.float64)

    def mode_predictions(self) -> list[int]:
        return [int(np.argmax(m)) for m in self.marginals()]

    def confidence_for_truth(self, truth: ConventionKey) -> list[float]:
        marginals = self.marginals()
        return [float(marginals[f][truth[f]]) for f in range(NUM_FACTORS)]

    def update_from_partner_action(
        self,
        env_before: ToyFactorGameEnv,
        observed_partner_action: int,
        likelihood_error: float,
    ) -> "BeliefState":
        if not 0.0 <= likelihood_error < 1.0:
            raise ValueError("--likelihood_error must be in [0, 1)")

        miss_prob = likelihood_error / max(1, len(Action) - 1)
        updated = {}
        for key, prior in self.weights.items():
            predicted = int(predict_partner_action(env_before, key))
            likelihood = 1.0 - likelihood_error if predicted == observed_partner_action else miss_prob
            updated[key] = prior * likelihood
        return BeliefState(updated).normalized()


def predict_partner_action(env_before: ToyFactorGameEnv, convention: ConventionKey) -> Action:
    candidate = copy.deepcopy(env_before)
    candidate.partner_convention = key_to_assignment(convention)
    return candidate._partner_action()


def valid_options(env: ToyFactorGameEnv) -> list[OptionID]:
    options = [OptionID.NOOP, OptionID.CROSS_CORRIDOR, OptionID.WAIT_AT_BOTTLENECK]
    if env.ego_carrying:
        options.extend([OptionID.DELIVER_LEFT, OptionID.DELIVER_RIGHT])
        if tuple(env.ego_pos) in (DELIVER_LEFT, DELIVER_RIGHT):
            options.append(OptionID.DROP)
    else:
        if env.resource_a_available:
            options.append(OptionID.GOTO_RESOURCE_A)
        if env.resource_b_available:
            options.append(OptionID.GOTO_RESOURCE_B)
        if tuple(env.ego_pos) in (RESOURCE_A, RESOURCE_B):
            options.append(OptionID.PICKUP)
    return sorted(set(options), key=int)


def option_target(option: OptionID):
    if option == OptionID.GOTO_RESOURCE_A:
        return RESOURCE_A
    if option == OptionID.GOTO_RESOURCE_B:
        return RESOURCE_B
    if option == OptionID.DELIVER_LEFT:
        return DELIVER_LEFT
    if option == OptionID.DELIVER_RIGHT:
        return DELIVER_RIGHT
    if option == OptionID.CROSS_CORRIDOR:
        return (3, 3)
    if option == OptionID.WAIT_AT_BOTTLENECK:
        return (3, 2)
    return None


def manhattan(a: Iterable[int], b: Iterable[int]) -> int:
    ar, ac = a
    br, bc = b
    return abs(ar - br) + abs(ac - bc)


def option_cost(option: OptionID) -> float:
    if option in (OptionID.NOOP, OptionID.WAIT_AT_BOTTLENECK):
        return 1.5
    if option == OptionID.CROSS_CORRIDOR:
        return 1.2
    return 1.0


def greedy_task_option(env: ToyFactorGameEnv, belief: BeliefState) -> OptionID:
    marginals = belief.marginals()

    if env.ego_carrying:
        if tuple(env.ego_pos) in (DELIVER_LEFT, DELIVER_RIGHT):
            return OptionID.DROP
        return OptionID.DELIVER_LEFT if marginals[2][0] >= marginals[2][1] else OptionID.DELIVER_RIGHT

    if tuple(env.ego_pos) in (RESOURCE_A, RESOURCE_B):
        return OptionID.PICKUP

    if env.resource_a_available and env.resource_b_available:
        return OptionID.GOTO_RESOURCE_A if marginals[1][0] >= marginals[1][1] else OptionID.GOTO_RESOURCE_B
    if env.resource_a_available:
        return OptionID.GOTO_RESOURCE_A
    if env.resource_b_available:
        return OptionID.GOTO_RESOURCE_B
    return OptionID.NOOP


def progress_bonus(env: ToyFactorGameEnv, option: OptionID, belief: BeliefState) -> float:
    target = option_target(option)
    bonus = 0.0
    if target is not None:
        bonus -= 0.05 * manhattan(env.ego_pos, target)
    if option == greedy_task_option(env, belief):
        bonus += 0.5
    if option in (OptionID.PICKUP, OptionID.DROP):
        bonus += 0.3
    return bonus


def expected_step_reward(env: ToyFactorGameEnv, option: OptionID, belief: BeliefState) -> float:
    action = get_option_action(option, env.ego_pos, env.ego_carrying)
    reward = 0.0
    for key, prob in belief.weights.items():
        branch = copy.deepcopy(env)
        branch.partner_convention = key_to_assignment(key)
        _obs, step_reward, _done, _info = branch.step(action)
        reward += prob * step_reward
    return float(reward)


def expected_entropy_after_option(
    env: ToyFactorGameEnv,
    option: OptionID,
    belief: BeliefState,
    likelihood_error: float,
) -> np.ndarray:
    action = get_option_action(option, env.ego_pos, env.ego_carrying)
    obs_prob: dict[int, float] = {}

    for key, prob in belief.weights.items():
        branch = copy.deepcopy(env)
        branch.partner_convention = key_to_assignment(key)
        _obs, _reward, _done, info = branch.step(action)
        observed = int(info["partner_action"])
        obs_prob[observed] = obs_prob.get(observed, 0.0) + prob

    expected = np.zeros(NUM_FACTORS, dtype=np.float64)
    for observed, prob in obs_prob.items():
        posterior = belief.update_from_partner_action(env, observed, likelihood_error)
        expected += prob * posterior.entropy_vector()
    return expected


def value_weights(env: ToyFactorGameEnv, option: OptionID) -> np.ndarray:
    near_bottleneck = (
        env.ego_pos[0] == 3
        or env.partner_pos[0] == 3
        or option in (OptionID.CROSS_CORRIDOR, OptionID.WAIT_AT_BOTTLENECK)
    )
    resource_active = (not env.ego_carrying) and (env.resource_a_available or env.resource_b_available)
    delivery_active = env.ego_carrying
    return np.array(
        [
            1.0 if near_bottleneck else 0.25,
            1.0 if resource_active else 0.15,
            1.0 if delivery_active else 0.15,
        ],
        dtype=np.float64,
    )


def score_option(
    env: ToyFactorGameEnv,
    option: OptionID,
    belief: BeliefState,
    method: str,
    alpha: float,
    beta: float,
    likelihood_error: float,
) -> tuple[float, dict[str, float]]:
    task_value = expected_step_reward(env, option, belief) + progress_bonus(env, option, belief)
    before_entropy = belief.entropy_vector()
    after_entropy = expected_entropy_after_option(env, option, belief, likelihood_error)
    info_gain = np.maximum(before_entropy - after_entropy, 0.0)

    if method == "mi":
        active_gain = float(info_gain.sum())
    elif method in ("gtvoi", "oracle"):
        active_gain = float(np.dot(value_weights(env, option), info_gain))
    else:
        active_gain = 0.0

    cost = option_cost(option)
    score = task_value + alpha * active_gain - beta * cost
    diagnostics = {
        "task_value": float(task_value),
        "active_gain": active_gain,
        "cost": cost,
        "entropy_gain_total": float(info_gain.sum()),
        "entropy_gain_value_weighted": float(np.dot(value_weights(env, option), info_gain)),
    }
    return float(score), diagnostics


def choose_option(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    method: str,
    rng: np.random.RandomState,
    alpha: float,
    beta: float,
    likelihood_error: float,
    truth: ConventionKey,
) -> tuple[OptionID, dict[str, float]]:
    options = valid_options(env)

    if method == "random":
        option = options[int(rng.randint(0, len(options)))]
        return option, {
            "task_value": 0.0,
            "active_gain": 0.0,
            "cost": option_cost(option),
            "entropy_gain_total": 0.0,
            "entropy_gain_value_weighted": 0.0,
        }

    if method == "oracle":
        true_belief = BeliefState.point_mass(truth)
        scored = []
        for option in options:
            task_value = expected_step_reward(env, option, true_belief) + progress_bonus(
                env, option, true_belief
            )
            before_entropy = belief.entropy_vector()
            after_entropy = expected_entropy_after_option(env, option, belief, likelihood_error)
            info_gain = np.maximum(before_entropy - after_entropy, 0.0)
            active_gain = float(np.dot(value_weights(env, option), info_gain))
            cost = option_cost(option)
            score = task_value + alpha * active_gain - beta * cost
            scored.append(
                (
                    (
                        score,
                        {
                            "task_value": float(task_value),
                            "active_gain": active_gain,
                            "cost": cost,
                            "entropy_gain_total": float(info_gain.sum()),
                            "entropy_gain_value_weighted": active_gain,
                        },
                    ),
                    option,
                )
            )
    else:
        scored = [
            (
                score_option(env, option, belief, method, alpha, beta, likelihood_error),
                option,
            )
            for option in options
        ]
    (score, diagnostics), option = max(scored, key=lambda item: (item[0][0], -int(item[1])))
    diagnostics["score"] = score
    return option, diagnostics


def is_aligned(belief: BeliefState, truth: ConventionKey, weights: np.ndarray, threshold: float) -> bool:
    predictions = belief.mode_predictions()
    for factor_idx, weight in enumerate(weights):
        if weight >= threshold and predictions[factor_idx] != truth[factor_idx]:
            return False
    return True


def run_episode(
    method: str,
    convention: ConventionKey,
    seed: int,
    max_steps: int,
    alpha: float,
    beta: float,
    likelihood_error: float,
    alignment_weight_threshold: float,
) -> dict[str, float | int | str]:
    rng = np.random.RandomState(seed)
    env = ToyFactorGameEnv(partner_convention=key_to_assignment(convention), max_steps=max_steps, seed=seed)
    belief = BeliefState.uniform()

    total_reward = 0.0
    early_reward = 0.0
    probe_cost = 0.0
    probe_count = 0
    collisions = 0
    alignment_time = max_steps + 1
    total_option_cost = 0.0
    selected_options: list[str] = []
    active_gains = []

    for step_idx in range(max_steps):
        env_before = copy.deepcopy(env)
        option, diagnostics = choose_option(
            env,
            belief,
            method,
            rng,
            alpha,
            beta,
            likelihood_error,
            convention,
        )
        action = get_option_action(option, env.ego_pos, env.ego_carrying)
        _obs, reward, done, info = env.step(action)

        belief = belief.update_from_partner_action(env_before, int(info["partner_action"]), likelihood_error)

        selected_options.append(OPTION_NAMES[option])
        total_reward += reward
        if step_idx < max(1, max_steps // 5):
            early_reward += reward
        collisions += int(info["collision"])
        total_option_cost += diagnostics["cost"]
        active_gains.append(diagnostics["active_gain"])

        is_probe = option in DIAGNOSTIC_OPTIONS and (
            method == "random" or diagnostics["active_gain"] > 1e-6
        )
        if is_probe:
            probe_count += 1
            probe_cost += diagnostics["cost"]

        weights = value_weights(env, option)
        if alignment_time == max_steps + 1 and is_aligned(
            belief, convention, weights, alignment_weight_threshold
        ):
            alignment_time = step_idx + 1

        if done:
            break

    final_entropies = belief.entropy_vector()
    final_predictions = belief.mode_predictions()
    truth_confidences = belief.confidence_for_truth(convention)
    final_factor_correct = [int(final_predictions[f] == convention[f]) for f in range(NUM_FACTORS)]

    row = {
        "method": method,
        "convention": convention_label(convention),
        "seed": seed,
        "episode_reward": float(total_reward),
        "early_reward": float(early_reward),
        "probe_cost": float(probe_cost),
        "probe_count": int(probe_count),
        "total_option_cost": float(total_option_cost),
        "collisions": int(collisions),
        "time_to_alignment": int(alignment_time),
        "final_entropy_total": float(final_entropies.sum()),
        "final_entropy_critical": float(final_entropies[:NUM_FACTORS].sum()),
        "final_factor_accuracy": float(np.mean(final_factor_correct)),
        "final_factor_confidence": float(np.mean(truth_confidences)),
        "mean_active_gain": float(np.mean(active_gains)) if active_gains else 0.0,
        "selected_options": ",".join(selected_options),
    }
    for factor_idx in range(NUM_FACTORS):
        row[f"factor_{factor_idx}_correct"] = int(final_factor_correct[factor_idx])
        row[f"factor_{factor_idx}_confidence"] = float(truth_confidences[factor_idx])
    return row


def summarize_method(rows: list[dict[str, float | int | str]]) -> dict[str, float | list[float]]:
    rewards = np.array([float(r["episode_reward"]) for r in rows], dtype=np.float64)
    probe_costs = np.array([float(r["probe_cost"]) for r in rows], dtype=np.float64)
    regrets = np.array([float(r["regret_to_oracle"]) for r in rows], dtype=np.float64)
    early_regrets = np.array([float(r["early_regret_to_oracle"]) for r in rows], dtype=np.float64)
    alignments = np.array([float(r["time_to_alignment"]) for r in rows], dtype=np.float64)
    factor_confidences = np.array(
        [
            float(row[f"factor_{factor_idx}_confidence"])
            for row in rows
            for factor_idx in range(NUM_FACTORS)
        ],
        dtype=np.float64,
    )
    factor_correct = np.array(
        [
            float(row[f"factor_{factor_idx}_correct"])
            for row in rows
            for factor_idx in range(NUM_FACTORS)
        ],
        dtype=np.float64,
    )
    episode_correct = np.array([float(r["final_factor_accuracy"]) for r in rows], dtype=np.float64)

    reward_ci = bootstrap_ci(rewards)
    regret_ci = bootstrap_ci(regrets)
    return {
        "n": len(rows),
        "episode_reward_mean": float(rewards.mean()),
        "episode_reward_ci": list(reward_ci),
        "probe_cost_mean": float(probe_costs.mean()),
        "regret_to_oracle_mean": float(regrets.mean()),
        "regret_to_oracle_ci": list(regret_ci),
        "early_regret_to_oracle_mean": float(early_regrets.mean()),
        "time_to_alignment_mean": float(alignments.mean()),
        "final_factor_accuracy_mean": float(episode_correct.mean()),
        "final_ece": expected_calibration_error(factor_confidences, factor_correct),
    }


def add_oracle_regrets(rows: list[dict[str, float | int | str]]) -> None:
    oracle_by_case = {
        (row["seed"], row["convention"]): row
        for row in rows
        if row["method"] == "oracle"
    }
    for row in rows:
        oracle = oracle_by_case[(row["seed"], row["convention"])]
        row["regret_to_oracle"] = float(oracle["episode_reward"]) - float(row["episode_reward"])
        row["early_regret_to_oracle"] = float(oracle["early_reward"]) - float(row["early_reward"])


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "convention",
        "seed",
        "episode_reward",
        "early_reward",
        "regret_to_oracle",
        "early_regret_to_oracle",
        "probe_cost",
        "probe_count",
        "total_option_cost",
        "collisions",
        "time_to_alignment",
        "final_entropy_total",
        "final_entropy_critical",
        "final_factor_accuracy",
        "final_factor_confidence",
        "factor_0_correct",
        "factor_0_confidence",
        "factor_1_correct",
        "factor_1_confidence",
        "factor_2_correct",
        "factor_2_confidence",
        "mean_active_gain",
        "selected_options",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_method_list(raw: str) -> list[str]:
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}; expected subset of {METHODS}")
    if "oracle" not in methods:
        methods.append("oracle")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes_per_convention", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_conventions", type=int, default=0,
                        help="Use the first N convention assignments; 0 means all")
    parser.add_argument("--methods", type=str, default="gtvoi,mi,passive,random,oracle")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--likelihood_error", type=float, default=0.05)
    parser.add_argument("--alignment_weight_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/toy_symbolic")
    parser.add_argument("--progress_every", type=int, default=0,
                        help="Print progress every N completed method episodes; 0 disables")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    if args.episodes_per_convention <= 0:
        raise ValueError("--episodes_per_convention must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.max_conventions < 0:
        raise ValueError("--max_conventions must be non-negative")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be non-negative")
    set_seed(min(seeds) if seeds else 0)

    rows: list[dict[str, float | int | str]] = []
    conventions = all_convention_keys()
    if args.max_conventions > 0:
        conventions = conventions[:args.max_conventions]
    total_runs = len(seeds) * len(conventions) * args.episodes_per_convention * len(methods)
    completed_runs = 0
    for seed in seeds:
        for convention in conventions:
            for repeat_idx in range(args.episodes_per_convention):
                episode_seed = seed * 1000 + repeat_idx
                for method in methods:
                    rows.append(
                        run_episode(
                            method=method,
                            convention=convention,
                            seed=episode_seed,
                            max_steps=args.max_steps,
                            alpha=args.alpha,
                            beta=args.beta,
                            likelihood_error=args.likelihood_error,
                            alignment_weight_threshold=args.alignment_weight_threshold,
                        )
                    )
                    completed_runs += 1
                    if args.progress_every and completed_runs % args.progress_every == 0:
                        print(
                            f"Completed {completed_runs}/{total_runs} method episodes",
                            flush=True,
                        )

    add_oracle_regrets(rows)
    summary = {
        method: summarize_method([row for row in rows if row["method"] == method])
        for method in methods
    }

    costs = np.array([summary[method]["probe_cost_mean"] for method in methods], dtype=np.float64)
    rewards = np.array([summary[method]["episode_reward_mean"] for method in methods], dtype=np.float64)
    frontier_idx = pareto_frontier(costs, rewards)
    frontier_methods = [methods[int(idx)] for idx in frontier_idx]

    output_dir = Path(args.output_dir)
    output = {
        "config": vars(args),
        "methods": methods,
        "n_conventions": len(conventions),
        "n_rows": len(rows),
        "summary": summary,
        "reward_probe_cost_pareto_methods": frontier_methods,
        "ground_truth_note": (
            "Factor accuracy, confidence, ECE, and regret use the toy environment's "
            "ConventionAssignment ground truth. No metric compares against another model output."
        ),
    }
    save_results(output, str(output_dir / "summary.json"))
    write_csv(rows, output_dir / "episodes.csv")

    print(f"Saved symbolic toy pilot summary to {output_dir / 'summary.json'}")
    print(f"Saved per-episode rows to {output_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
