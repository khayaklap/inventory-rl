"""Tests for the held-out, paired evaluation protocol."""

from __future__ import annotations

import numpy as np

from inventory_rl.baselines import BaseStockPolicy, RandomPolicy, stationary_base_stock_level
from inventory_rl.config import EnvParams, EvalParams
from inventory_rl.demand import StationaryPoisson
from inventory_rl.evaluation import (
    bootstrap_mean_ci,
    build_report,
    compare_paired,
    evaluate_policies,
    generate_demand_matrix,
    rollout,
    seed_sets_disjoint,
)


def test_training_and_eval_seeds_are_disjoint() -> None:
    """The configured seed streams cannot leak training demand into the held-out set."""
    ev = EvalParams()
    n_training = 30_000  # QLearnParams.n_episodes
    assert seed_sets_disjoint(ev.train_seed_base, n_training, ev.eval_seed_base, ev.n_eval_episodes)
    assert seed_sets_disjoint(ev.train_seed_base, n_training, ev.val_seed_base, ev.n_eval_episodes)
    # A deliberately overlapping pair is correctly flagged.
    assert not seed_sets_disjoint(1_000, 30_000, 9_000, 100)


def test_demand_matrix_is_reproducible_and_shaped() -> None:
    """The demand matrix is deterministic given its seed base and has the right shape."""
    proc = StationaryPoisson(10.0)
    a = generate_demand_matrix(proc, 16, seed_base=3_000_000, horizon=52)
    b = generate_demand_matrix(proc, 16, seed_base=3_000_000, horizon=52)
    assert a.shape == (16, 52)
    assert np.array_equal(a, b)


def test_compare_paired_difference_matches_mean_of_differences() -> None:
    """The reported mean difference equals the mean of per-episode paired differences."""
    rng = np.random.default_rng(0)
    a = rng.normal(100, 10, size=200)
    b = rng.normal(90, 10, size=200)
    cmp = compare_paired(a, b, "a", "b", n_boot=2_000)
    assert np.isclose(cmp.mean_difference, np.mean(a - b))
    assert np.isclose(cmp.median_difference, np.median(a - b))
    assert cmp.ci_low <= cmp.mean_difference <= cmp.ci_high
    assert cmp.n_episodes == 200


def test_compare_paired_identical_returns_has_zero_difference() -> None:
    """Comparing a policy with itself yields a zero difference and no significant gap."""
    x = np.linspace(0, 100, 50)
    cmp = compare_paired(x, x.copy(), "x", "x", n_boot=1_000)
    assert cmp.mean_difference == 0.0
    assert cmp.wilcoxon_p == 1.0  # all-zero differences handled gracefully


def test_bootstrap_ci_brackets_the_mean() -> None:
    """The bootstrap CI contains the sample mean and is ordered low <= mean <= high."""
    values = np.array([10.0, 12.0, 11.0, 9.0, 10.5, 11.5, 10.0, 9.5])
    mean, low, high = bootstrap_mean_ci(values, n_boot=5_000, ci_level=0.95)
    assert np.isclose(mean, values.mean())
    assert low <= mean <= high


def test_summary_reports_distribution_tails() -> None:
    """Each metric summary carries the distribution tails (p05/p95/cvar05), not just the mean."""
    from inventory_rl.evaluation import summarize_metric

    values = np.arange(0.0, 100.0)  # 0..99
    summary = summarize_metric(values, n_boot=1_000, ci_level=0.95)
    assert {"p05", "p95", "cvar05"} <= set(summary)
    assert summary["p05"] < summary["mean"] < summary["p95"]
    assert summary["cvar05"] <= summary["p05"]  # expected shortfall is no better than the 5th pct


def test_rollout_return_equals_cost_decomposition() -> None:
    """Total return reconciles with revenue minus costs plus terminal salvage (no leakage)."""
    params = EnvParams()  # order_cost K = 0
    policy = BaseStockPolicy(stationary_base_stock_level(params, 10.0), params)
    demand = StationaryPoisson(10.0).sample_episode(params.horizon, np.random.default_rng(3_000_000))
    result = rollout(policy, demand, params)
    m = result.metrics
    salvage = params.salvage * result.trajectory["on_hand"][-1]
    reconstructed = m["revenue"] - m["purchase_cost"] - m["holding_cost"] - m["penalty_cost"] + salvage
    assert np.isclose(result.total_return, reconstructed)


def test_metric_keys_match_rollout() -> None:
    """The documented METRIC_KEYS constant stays in sync with what rollout actually produces."""
    from inventory_rl.evaluation import METRIC_KEYS

    params = EnvParams()
    policy = BaseStockPolicy(stationary_base_stock_level(params, 10.0), params)
    demand = StationaryPoisson(10.0).sample_episode(params.horizon, np.random.default_rng(3_000_000))
    produced = set(rollout(policy, demand, params).metrics)
    assert produced == set(METRIC_KEYS)


def test_base_stock_beats_random_on_held_out_set() -> None:
    """On the held-out set, the newsvendor policy earns more than random ordering."""
    params = EnvParams()
    matrix = generate_demand_matrix(StationaryPoisson(10.0), 30, seed_base=3_000_000, horizon=52)
    policies = [
        RandomPolicy(len(params.action_menu), seed=0),
        BaseStockPolicy(stationary_base_stock_level(params, 10.0), params),
    ]
    results = evaluate_policies(policies, matrix, params)
    assert results["base_stock"].returns.mean() > results["random"].returns.mean()
    # No policy may ever violate the hard capacity constraint.
    for res in results.values():
        assert res.metrics["capacity_violations"].sum() == 0


def test_build_report_has_summaries_and_comparisons() -> None:
    """The report carries per-policy summaries and pairwise comparisons to the baseline."""
    params = EnvParams()
    matrix = generate_demand_matrix(StationaryPoisson(10.0), 20, seed_base=3_000_000, horizon=52)
    policies = [
        RandomPolicy(len(params.action_menu), seed=0),
        BaseStockPolicy(stationary_base_stock_level(params, 10.0), params),
    ]
    results = evaluate_policies(policies, matrix, params)
    report = build_report(results, meta={"regime": "stationary"}, baseline_name="base_stock", n_boot=1_000)
    assert "base_stock" in report.policy_summaries
    assert "return" in report.policy_summaries["random"]
    assert any(c["policy"] == "random" and c["baseline"] == "base_stock" for c in report.comparisons)
