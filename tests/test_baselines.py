"""Tests for the baseline policies and the newsvendor level computation."""

from __future__ import annotations

import numpy as np
from scipy import stats

from inventory_rl.baselines import (
    BaseStockPolicy,
    ConstantOrderPolicy,
    RandomPolicy,
    SeasonalBaseStockPolicy,
    SSPolicy,
    base_stock_level,
    seasonal_base_stock_levels,
    ss_policy_levels,
    stationary_base_stock_level,
)
from inventory_rl.config import DemandParams, EnvParams


def test_base_stock_level_is_smallest_integer_meeting_critical_ratio() -> None:
    """S is the smallest integer with Poisson CDF over the protection interval >= critical ratio."""
    params = EnvParams()
    pi_mean = 10.0 * params.protection_interval  # 20
    s = base_stock_level(params, pi_mean)
    cr = params.critical_ratio
    assert stats.poisson.cdf(s, pi_mean) >= cr
    assert stats.poisson.cdf(s - 1, pi_mean) < cr


def test_stationary_base_stock_in_expected_range() -> None:
    """With Cu=11, Co=1, lambda=10, L=1, the order-up-to level is ~26-27."""
    s = stationary_base_stock_level(EnvParams(), 10.0)
    assert 24 <= s <= 28


def test_underage_cost_includes_lost_margin() -> None:
    """The newsvendor underage cost is lost margin plus penalty, not the penalty alone."""
    params = EnvParams()
    assert params.underage_cost == (params.price - params.unit_cost) + params.stockout_penalty
    assert params.critical_ratio == params.underage_cost / (params.underage_cost + params.overage_cost)


def test_base_stock_orders_to_close_the_gap_and_stops_at_level() -> None:
    """Base-stock orders to lift inventory position toward S and orders nothing once at/above S."""
    params = EnvParams()
    policy = BaseStockPolicy(level=26, params=params)
    # Inventory position 6 (on_hand) + 0 (pipeline): a large gap -> a positive order.
    low = np.array([6, 0, 0], dtype=np.float32)
    assert params.action_menu[policy(low)] > 0
    # Inventory position already at the level -> no order.
    at_level = np.array([26, 0, 0], dtype=np.float32)
    assert params.action_menu[policy(at_level)] == 0
    # Above the level -> still no order.
    above = np.array([40, 0, 0], dtype=np.float32)
    assert params.action_menu[policy(above)] == 0


def test_base_stock_respects_order_cap() -> None:
    """The base-stock order never exceeds the per-order cap, even with a huge gap."""
    params = EnvParams()
    policy = BaseStockPolicy(level=200, params=params)  # absurd level
    order = params.action_menu[policy(np.array([0, 0, 0], dtype=np.float32))]
    assert order <= params.q_max


def test_seasonal_levels_vary_with_season() -> None:
    """Seasonal order-up-to levels are higher in peak weeks than in trough weeks."""
    levels = seasonal_base_stock_levels(EnvParams(), DemandParams())
    assert levels[13] > levels[0]  # quarter-year peak above the mean week
    assert levels[39] < levels[0]  # three-quarter trough below it
    assert levels.max() > levels.min()


def test_seasonal_policy_uses_week_index() -> None:
    """The seasonal policy reads the week from the observation to pick its level."""
    params = EnvParams()
    levels = seasonal_base_stock_levels(params, DemandParams())
    policy = SeasonalBaseStockPolicy(levels, params)
    peak_obs = np.array([5, 0, 13], dtype=np.float32)
    trough_obs = np.array([5, 0, 39], dtype=np.float32)
    # The same inventory position orders more in a peak week than in a trough week.
    assert params.action_menu[policy(peak_obs)] >= params.action_menu[policy(trough_obs)]


def test_ss_policy_orders_only_below_reorder_point() -> None:
    """An (s, S) policy orders up to S below the reorder point and nothing at/above it."""
    params = EnvParams(order_cost=20.0)
    s, big_s = ss_policy_levels(params, 10.0)
    assert s == 15 and big_s == 35
    policy = SSPolicy(s, big_s, params)
    assert params.action_menu[policy(np.array([5, 0, 0], dtype=np.float32))] > 0  # below s
    assert params.action_menu[policy(np.array([20, 0, 0], dtype=np.float32))] == 0  # above s


def test_constant_order_policy_always_returns_its_index() -> None:
    """The constant-order policy ignores state and returns its fixed action (safety-ablation actor)."""
    params = EnvParams()
    policy = ConstantOrderPolicy(len(params.action_menu) - 1, params)
    assert policy(np.array([0, 0, 0], dtype=np.float32)) == len(params.action_menu) - 1
    assert policy(np.array([40, 10, 30], dtype=np.float32)) == len(params.action_menu) - 1


def test_random_policy_is_bounded_and_reproducible() -> None:
    """Random ordering stays within the action set and repeats under a fixed seed."""
    n = len(EnvParams().action_menu)
    a = [RandomPolicy(n, seed=3)(np.zeros(3, dtype=np.float32)) for _ in range(50)]
    b = [RandomPolicy(n, seed=3)(np.zeros(3, dtype=np.float32)) for _ in range(50)]
    assert a == b
    assert all(0 <= x < n for x in a)
