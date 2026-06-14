"""Baseline ordering policies, from a naive floor to a near-optimal analytical anchor.

The course's "baseline ladder" says to climb only as far as a problem demands and to justify
RL against the strongest simple baseline -- not a strawman. So this module provides:

* ``RandomPolicy`` -- the floor; any useful agent must beat it.
* ``BaseStockPolicy`` -- the real competitor: a newsvendor order-up-to level ``S`` derived in
  closed form. For stationary demand this is near-optimal, so a learned agent that merely
  *ties* it is the honest result (RL is not needed there).
* ``SeasonalBaseStockPolicy`` -- a time-indexed ``S_t``; the fair competitor under seasonal
  demand, so the seasonal "RL wins" story is not measured against a deliberately handicapped
  static rule.
* ``SSPolicy`` -- an ``(s, S)`` policy for the fixed-order-cost variant.

The newsvendor level uses the corrected underage cost ``Cu = (price - cost) + penalty`` (lost
margin *plus* the explicit penalty), critical ratio ``Cu / (Cu + Co)``, and the
protection-interval demand over ``L + 1`` periods. Each policy maps its desired order to the
nearest quantity on the shared discrete action menu, so it competes on the agent's action set.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy import stats

from inventory_rl.config import DemandParams, EnvParams
from inventory_rl.demand import SeasonalPoisson
from inventory_rl.env import OBS_ON_HAND, OBS_PIPELINE, OBS_WEEK, order_to_action_index


class Policy(Protocol):
    """A deterministic-or-seeded mapping from an observation to a discrete action index."""

    name: str

    def __call__(self, obs: np.ndarray) -> int: ...


def base_stock_level(params: EnvParams, protection_interval_mean: float) -> int:
    """Newsvendor order-up-to level for a given protection-interval demand mean.

    ``S`` is the smallest integer with ``P(D_PI <= S) >= critical_ratio``, where ``D_PI`` is
    Poisson over the protection interval. We use the exact discrete quantile (``ppf``) rather
    than a normal approximation, because Poisson demand is skewed and the normal approximation
    can be off by a full unit at this service level.
    """
    cr = params.critical_ratio
    level = int(stats.poisson.ppf(cr, protection_interval_mean))
    # ppf already returns the smallest integer with cdf >= cr, but guard the floating edge.
    if stats.poisson.cdf(level, protection_interval_mean) < cr:
        level += 1
    return level


def stationary_base_stock_level(params: EnvParams, demand_mean: float) -> int:
    """Base-stock level for stationary demand: newsvendor over ``protection_interval`` periods."""
    return base_stock_level(params, demand_mean * params.protection_interval)


def seasonal_base_stock_levels(params: EnvParams, demand: DemandParams) -> np.ndarray:
    """One order-up-to level per week, from the protection-interval demand starting that week.

    For week ``t`` the protection-interval mean is the sum of the seasonal rate over the next
    ``L + 1`` weeks. This is the fair seasonal competitor: it adapts ``S_t`` to the season the
    way a sophisticated planner would, so the learned agent must match an adaptive baseline,
    not a handicapped static one.
    """
    proc = SeasonalPoisson(demand.mean, demand.amplitude, demand.period, demand.min_rate)
    pi = params.protection_interval
    levels = np.zeros(demand.period, dtype=np.int64)
    for t in range(demand.period):
        pi_mean = sum(proc.rate((t + k) % demand.period) for k in range(pi))
        levels[t] = base_stock_level(params, pi_mean)
    return levels


def ss_policy_levels(params: EnvParams, demand_mean: float) -> tuple[int, int]:
    """Reorder point ``s`` and order-up-to ``S`` for the fixed-order-cost ``(s, S)`` variant.

    ``s`` is the newsvendor level over the lead-time-only demand; ``S = s + EOQ`` with the
    classic economic order quantity. This is a documented heuristic (the exact computation is
    Zheng-Federgruen); it is only meaningful when ``order_cost > 0``.
    """
    s = base_stock_level(params, demand_mean * params.lead_time)
    eoq = math.sqrt(2.0 * params.order_cost * demand_mean / params.holding_cost) if params.order_cost > 0 else 0.0
    big_s = s + int(round(eoq))
    return s, big_s


class RandomPolicy:
    """Uniform-random ordering -- the lower-bound baseline."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self.name = "random"

    def __call__(self, obs: np.ndarray) -> int:  # noqa: ARG002 - ignores state by design
        return int(self._rng.integers(self.n_actions))


class ConstantOrderPolicy:
    """Always request the same order quantity -- a worst-case actor for the safety ablation."""

    def __init__(self, action_index: int, params: EnvParams, name: str = "constant_order") -> None:
        self.action_index = action_index
        self.params = params
        self.name = name

    def __call__(self, obs: np.ndarray) -> int:  # noqa: ARG002 - ignores state by design
        return self.action_index


class BaseStockPolicy:
    """Order up to a fixed level ``S`` on inventory position (the newsvendor anchor)."""

    def __init__(self, level: int, params: EnvParams, name: str = "base_stock") -> None:
        self.level = level
        self.params = params
        self.name = name

    def __call__(self, obs: np.ndarray) -> int:
        ip = int(obs[OBS_ON_HAND] + obs[OBS_PIPELINE])
        order = min(max(self.level - ip, 0), self.params.q_max)
        return order_to_action_index(order, self.params.action_menu)


class SeasonalBaseStockPolicy:
    """Order up to a week-indexed level ``S_t`` (the adaptive seasonal competitor)."""

    def __init__(self, levels: np.ndarray, params: EnvParams, name: str = "seasonal_base_stock") -> None:
        self.levels = levels
        self.params = params
        self.name = name

    def __call__(self, obs: np.ndarray) -> int:
        week = int(obs[OBS_WEEK]) % len(self.levels)
        ip = int(obs[OBS_ON_HAND] + obs[OBS_PIPELINE])
        order = min(max(int(self.levels[week]) - ip, 0), self.params.q_max)
        return order_to_action_index(order, self.params.action_menu)


class SSPolicy:
    """``(s, S)`` policy: when inventory position drops below ``s``, order up to ``S``."""

    def __init__(self, reorder_point: int, level: int, params: EnvParams, name: str = "sS") -> None:
        self.reorder_point = reorder_point
        self.level = level
        self.params = params
        self.name = name

    def __call__(self, obs: np.ndarray) -> int:
        ip = int(obs[OBS_ON_HAND] + obs[OBS_PIPELINE])
        order = min(max(self.level - ip, 0), self.params.q_max) if ip < self.reorder_point else 0
        return order_to_action_index(order, self.params.action_menu)
