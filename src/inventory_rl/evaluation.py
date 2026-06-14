"""Held-out, paired evaluation with bootstrap confidence intervals.

This is the module the "evaluation against baseline" grade rests on, so the protocol is
deliberately strict:

* **Held-out demand.** Every policy is scored on demand episodes drawn from a seed stream
  that is disjoint from the training stream (``seed_sets_disjoint`` proves it). Only these
  episodes are called "out-of-sample".
* **Paired design.** For each evaluation episode, every policy faces the *identical* demand
  realization, so differences reflect the policy, not luck. Comparisons are therefore paired,
  and confidence intervals come from a *paired* bootstrap that resamples episode indices.
* **True objective.** Returns are always computed under the balanced reward, even for an
  agent trained on the naive proxy -- you optimize the proxy, but you are graded on the
  business.
* **Multi-metric.** A single mean return hides reward hacking, so every run also reports
  service level, stockout frequency, average on-hand, and the holding/penalty/purchase cost
  breakdown.

The module is pure measurement: it imports no trainer and writes no plots, so it can be unit
-tested and reused by both the CLI and the evals harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy import stats

from inventory_rl.config import EnvParams
from inventory_rl.env import InventoryEnv
from inventory_rl.rewards import reward_balanced


class Policy(Protocol):
    """Anything that maps an observation to a discrete action index."""

    name: str

    def __call__(self, obs: np.ndarray) -> int: ...


# The per-episode metrics produced by `rollout`. Declared explicitly (rather than discovered
# from a throwaway rollout) so the metric contract is documented; kept in sync with `rollout`
# by `test_metric_keys_match_rollout`.
METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "service_level",
    "stockout_freq",
    "avg_on_hand",
    "avg_order",
    "revenue",
    "purchase_cost",
    "holding_cost",
    "penalty_cost",
    "approval_flags",
    "capacity_violations",
)


# Demand generation
def generate_demand_matrix(
    demand_process: Any, n_episodes: int, seed_base: int, horizon: int
) -> np.ndarray:
    """Pre-generate an ``(n_episodes, horizon)`` demand matrix from one seed stream.

    Each row uses its own ``default_rng(seed_base + i)`` so the matrix is reproducible and the
    seed stream is explicit and auditable.
    """
    rows = [
        demand_process.sample_episode(horizon, np.random.default_rng(seed_base + i))
        for i in range(n_episodes)
    ]
    return np.asarray(rows, dtype=np.int64)


def seed_sets_disjoint(train_base: int, n_train: int, eval_base: int, n_eval: int) -> bool:
    """True iff the training and evaluation seed ranges do not overlap (no demand leakage)."""
    train = range(train_base, train_base + n_train)
    held_out = range(eval_base, eval_base + n_eval)
    return train.stop <= held_out.start or held_out.stop <= train.start


# Single-episode rollout
@dataclass
class EpisodeResult:
    """One episode's true-objective return, summary metrics, and per-step trajectory."""

    total_return: float
    metrics: dict[str, float]
    trajectory: dict[str, list[int]]


def rollout(
    policy: Policy,
    demand_sequence: np.ndarray,
    env_params: EnvParams,
    lead_time: int | None = None,
) -> EpisodeResult:
    """Run one episode of ``policy`` against a fixed demand sequence under the balanced reward.

    The evaluation environment always uses the balanced reward, so a policy trained on the
    naive proxy is still measured on the true economic objective.
    """
    env = InventoryEnv(env_params, reward_fn=reward_balanced)
    options: dict[str, Any] = {"demand_sequence": demand_sequence}
    if lead_time is not None:
        options["lead_time"] = lead_time
    obs, _ = env.reset(seed=0, options=options)

    on_hand, order, demand, sales, unmet, ip = [], [], [], [], [], []
    approval_flags = capacity_violations = 0
    total_return = 0.0
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_return += reward
        on_hand.append(info["end_on_hand"])
        order.append(info["effective_order"])
        demand.append(info["demand"])
        sales.append(info["sales"])
        unmet.append(info["unmet"])
        ip.append(info["inventory_position"])
        approval_flags += int(info["approval_flagged"])
        capacity_violations += int(info["capacity_violation"])

    demand_arr = np.asarray(demand)
    sales_arr = np.asarray(sales)
    unmet_arr = np.asarray(unmet)
    on_hand_arr = np.asarray(on_hand)
    order_arr = np.asarray(order)
    total_demand = int(demand_arr.sum())

    metrics = {
        "total_return": total_return,
        "service_level": float(sales_arr.sum() / total_demand) if total_demand > 0 else 1.0,
        "stockout_freq": float((unmet_arr > 0).mean()),
        "avg_on_hand": float(on_hand_arr.mean()),
        "avg_order": float(order_arr.mean()),
        "revenue": float(env_params.price * sales_arr.sum()),
        "purchase_cost": float(env_params.unit_cost * order_arr.sum()),
        "holding_cost": float(env_params.holding_cost * on_hand_arr.sum()),
        "penalty_cost": float(env_params.stockout_penalty * unmet_arr.sum()),
        "approval_flags": float(approval_flags),
        "capacity_violations": float(capacity_violations),
    }
    trajectory = {
        "on_hand": on_hand,
        "order": order,
        "demand": demand,
        "sales": sales,
        "unmet": unmet,
        "inventory_position": ip,
    }
    return EpisodeResult(total_return=total_return, metrics=metrics, trajectory=trajectory)


# Paired multi-policy evaluation
@dataclass
class PolicyEvalResult:
    """Per-episode returns and metrics for one policy over the held-out set."""

    name: str
    returns: np.ndarray
    metrics: dict[str, np.ndarray]
    sample_trajectory: dict[str, list[int]]


def evaluate_policies(
    policies: list[Policy],
    demand_matrix: np.ndarray,
    env_params: EnvParams,
    lead_time: int | None = None,
) -> dict[str, PolicyEvalResult]:
    """Score every policy on every (shared) demand episode -- the paired evaluation core."""
    n_episodes = demand_matrix.shape[0]
    metric_keys = METRIC_KEYS
    results: dict[str, PolicyEvalResult] = {}
    for policy in policies:
        returns = np.zeros(n_episodes)
        metrics = {k: np.zeros(n_episodes) for k in metric_keys}
        sample_traj: dict[str, list[int]] = {}
        for i in range(n_episodes):
            ep = rollout(policy, demand_matrix[i], env_params, lead_time)
            returns[i] = ep.total_return
            for k in metric_keys:
                metrics[k][i] = ep.metrics[k]
            if i == 0:
                sample_traj = ep.trajectory
        results[policy.name] = PolicyEvalResult(policy.name, returns, metrics, sample_traj)
    return results


# Statistics
def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int, ci_level: float, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap the mean of ``values`` (resampling indices); return ``(mean, low, high)``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    low, high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(low), float(high)


@dataclass
class PairedComparison:
    """Paired comparison of a policy's returns against a baseline's, on the same episodes."""

    policy: str
    baseline: str
    mean_difference: float
    ci_low: float
    ci_high: float
    median_difference: float
    wilcoxon_p: float
    ttest_p: float
    n_episodes: int


def compare_paired(
    policy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    policy_name: str,
    baseline_name: str,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two policies on identical episodes: paired-bootstrap CI + Wilcoxon + paired t.

    Wilcoxon signed-rank is the primary test (it does not assume normal differences); the
    paired t-test is reported as a secondary cross-check. Both run on the per-episode
    differences, preserving the pairing.
    """
    diff = policy_returns - baseline_returns
    mean_diff, low, high = bootstrap_mean_ci(diff, n_boot, ci_level, seed)
    if np.allclose(diff, 0.0):
        # Identical policies: no difference to detect. scipy would return NaN here.
        wilcoxon_p = 1.0
        ttest_p = 1.0
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        ttest_p = float(stats.ttest_rel(policy_returns, baseline_returns).pvalue)
    if np.isnan(wilcoxon_p):
        wilcoxon_p = 1.0
    return PairedComparison(
        policy=policy_name,
        baseline=baseline_name,
        mean_difference=mean_diff,
        ci_low=low,
        ci_high=high,
        median_difference=float(np.median(diff)),
        wilcoxon_p=wilcoxon_p,
        ttest_p=ttest_p,
        n_episodes=len(diff),
    )


def summarize_metric(values: np.ndarray, n_boot: int, ci_level: float) -> dict[str, float]:
    """Mean and bootstrap CI for one metric, plus distribution tails across the held-out set.

    The course's offline-evaluation checklist asks for the *reward distribution* and *tail
    risks*, not just the average (p.150). So every metric also reports its 5th/95th percentiles
    and ``cvar05`` -- the mean of the worst 5% of episodes (expected shortfall). For the return,
    ``p05``/``cvar05`` is the downside a risk-averse operator actually cares about.
    """
    mean, low, high = bootstrap_mean_ci(values, n_boot, ci_level)
    p05 = float(np.quantile(values, 0.05))
    worst_5pct = values[values <= p05]
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "std": float(values.std()),
        "p05": p05,
        "p95": float(np.quantile(values, 0.95)),
        "cvar05": float(worst_5pct.mean()) if worst_5pct.size else p05,
    }


# Report assembly
@dataclass
class EvalReport:
    """The structured, committed record of an evaluation run (written to JSON)."""

    meta: dict[str, Any]
    policy_summaries: dict[str, dict[str, dict[str, float]]]
    comparisons: list[dict[str, Any]] = field(default_factory=list)


def build_report(
    results: dict[str, PolicyEvalResult],
    meta: dict[str, Any],
    baseline_name: str,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
) -> EvalReport:
    """Summarize every policy and compare each against the named baseline."""
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for name, res in results.items():
        summaries[name] = {"return": summarize_metric(res.returns, n_boot, ci_level)}
        for metric, values in res.metrics.items():
            summaries[name][metric] = summarize_metric(values, n_boot, ci_level)

    comparisons: list[dict[str, Any]] = []
    if baseline_name in results:
        base_returns = results[baseline_name].returns
        for name, res in results.items():
            if name == baseline_name:
                continue
            cmp = compare_paired(res.returns, base_returns, name, baseline_name, n_boot, ci_level)
            comparisons.append(asdict(cmp))
    return EvalReport(meta=meta, policy_summaries=summaries, comparisons=comparisons)


def write_report(report: EvalReport, path: str | Path) -> None:
    """Write the evaluation report as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")


def write_results_csv(
    results_by_regime: dict[str, dict[str, PolicyEvalResult]], path: str | Path
) -> None:
    """Write per-episode returns and key metrics in long format for plotting/inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["regime,policy,episode,total_return,service_level,avg_on_hand,holding_cost,penalty_cost"]
    for regime, results in results_by_regime.items():
        for name, res in results.items():
            for i in range(len(res.returns)):
                lines.append(
                    f"{regime},{name},{i},{res.returns[i]:.4f},"
                    f"{res.metrics['service_level'][i]:.4f},{res.metrics['avg_on_hand'][i]:.4f},"
                    f"{res.metrics['holding_cost'][i]:.4f},{res.metrics['penalty_cost'][i]:.4f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
