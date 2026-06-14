"""Tests for the reward functions, including the reward-hacking separation property."""

from __future__ import annotations

import math

from inventory_rl.config import EnvParams
from inventory_rl.rewards import StepResult, reward_balanced, reward_naive


def test_naive_and_balanced_rank_policies_differently() -> None:
    """The proxy must be gameable: a policy the naive reward prefers, the balanced reward rejects.

    This is the inventory analog of the course deck's reward_bad / reward_better example: an
    over-ordering step scores best under the service-only proxy (zero stockouts) but worst
    under the true objective (it paid for and is holding a wall of stock).
    """
    params = EnvParams()
    over_order = StepResult(sales=10, end_on_hand=45, unmet=0, order_qty=30, is_terminal=False)
    balanced = StepResult(sales=10, end_on_hand=8, unmet=2, order_qty=10, is_terminal=False)

    # Naive reward ranks the over-ordering step at least as high (it has no stockouts).
    assert reward_naive(over_order, params) >= reward_naive(balanced, params)
    # Balanced reward strictly prefers the lean policy: the ranking flips.
    assert reward_balanced(balanced, params) > reward_balanced(over_order, params)


def test_naive_reward_is_zero_iff_no_stockout() -> None:
    """The naive proxy is maximized (=0) exactly when demand is fully served."""
    params = EnvParams()
    served = StepResult(sales=10, end_on_hand=5, unmet=0, order_qty=10, is_terminal=False)
    short = StepResult(sales=8, end_on_hand=0, unmet=3, order_qty=0, is_terminal=False)
    assert reward_naive(served, params) == 0.0
    assert reward_naive(short, params) == -params.stockout_penalty * 3


def test_balanced_reward_penalizes_holding_and_stockouts_monotonically() -> None:
    """Each cost term moves the balanced reward in the documented direction."""
    params = EnvParams()
    base = StepResult(sales=10, end_on_hand=5, unmet=0, order_qty=10, is_terminal=False)
    more_holding = StepResult(sales=10, end_on_hand=15, unmet=0, order_qty=10, is_terminal=False)
    more_unmet = StepResult(sales=10, end_on_hand=5, unmet=4, order_qty=10, is_terminal=False)

    assert reward_balanced(more_holding, params) < reward_balanced(base, params)
    assert reward_balanced(more_unmet, params) < reward_balanced(base, params)


def test_balanced_reward_matches_closed_form() -> None:
    """The balanced reward equals its documented cost-at-order expression."""
    params = EnvParams()
    step = StepResult(sales=12, end_on_hand=7, unmet=1, order_qty=15, is_terminal=False)
    expected = (
        params.price * 12
        - params.unit_cost * 15
        - params.holding_cost * 7
        - params.stockout_penalty * 1
        - params.order_cost * 1
    )
    assert math.isclose(reward_balanced(step, params), expected)


def test_terminal_salvage_only_on_last_period() -> None:
    """Leftover stock is salvaged once, at the terminal step, and nowhere else."""
    params = EnvParams()
    interior = StepResult(sales=5, end_on_hand=10, unmet=0, order_qty=5, is_terminal=False)
    terminal = StepResult(sales=5, end_on_hand=10, unmet=0, order_qty=5, is_terminal=True)
    assert reward_balanced(terminal, params) - reward_balanced(interior, params) == params.salvage * 10


def test_rewards_finite_at_boundaries() -> None:
    """Rewards stay finite at the boundary states (empty and full warehouse)."""
    params = EnvParams()
    empty = StepResult(sales=0, end_on_hand=0, unmet=15, order_qty=0, is_terminal=False)
    full = StepResult(sales=10, end_on_hand=params.i_max, unmet=0, order_qty=30, is_terminal=True)
    assert math.isfinite(reward_balanced(empty, params))
    assert math.isfinite(reward_balanced(full, params))
    assert math.isfinite(reward_naive(empty, params))
