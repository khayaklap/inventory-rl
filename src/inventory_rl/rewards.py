"""Reward functions: one balanced objective, one naive proxy that is deliberately gameable.

This module is the heart of the reward-hacking demonstration. ``reward_balanced`` encodes
the real business objective (profit net of holding, stockouts, ordering, and end-of-horizon
salvage). ``reward_naive`` encodes a tempting but wrong proxy -- "just don't run out of
stock" -- which an optimizer will satisfy by ordering to the warehouse cap, scoring a great
service level while quietly destroying profit. Training an agent on the proxy and then
*scoring* it on the true objective is the cleanest possible illustration of the course's
lesson: the agent optimizes exactly what you reward, even when the reward is wrong.

Accounting convention (cost-at-order):

    reward = price * sales            (revenue earned on sales)
           - unit_cost * order_qty    (cash paid when the order is placed)
           - holding_cost * end_on_hand
           - stockout_penalty * unmet
           - order_cost * 1[order_qty > 0]
           + salvage * end_on_hand    (terminal period only)

Paying for orders at placement (rather than netting margin on sales) makes the terminal
salvage term coherent -- leftover stock represents cash already spent, partially recovered
at the horizon end -- and removes the end-of-episode sell-down artifact that would otherwise
make the learned policy diverge from a horizon-unaware base-stock near the final week.
The per-period ordering trade-off is still governed by the newsvendor costs
``Cu = (price - unit_cost) + stockout_penalty`` and ``Co = holding_cost`` (see ``config``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from inventory_rl.config import EnvParams


@dataclass(frozen=True)
class StepResult:
    """The physical outcome of one period, the input every reward function scores.

    Separating the physics (what happened in the warehouse) from the reward (how we value
    it) is what lets the same simulated episode be re-scored under a different reward -- the
    mechanism behind the reward-hacking experiment.
    """

    sales: int  # units sold this period
    end_on_hand: int  # on-hand inventory carried into the next period
    unmet: int  # demand that could not be served (lost, under lost-sales)
    order_qty: int  # units ordered this period (after capacity clamping)
    is_terminal: bool  # whether this is the final period of the episode


class RewardFn(Protocol):
    """Callable signature shared by every reward function."""

    def __call__(self, step: StepResult, params: EnvParams) -> float: ...


def reward_balanced(step: StepResult, params: EnvParams) -> float:
    """The true economic objective: profit net of every real cost.

    This is the reward the deployable agent should optimize and the reward every policy is
    *evaluated* under, regardless of what it was trained on.
    """
    reward = (
        params.price * step.sales
        - params.unit_cost * step.order_qty
        - params.holding_cost * step.end_on_hand
        - params.stockout_penalty * step.unmet
        - params.order_cost * (1.0 if step.order_qty > 0 else 0.0)
    )
    if step.is_terminal:
        # Recover part of the cash tied up in leftover stock so the agent is not punished
        # for inventory it could not have sold within the horizon.
        reward += params.salvage * step.end_on_hand
    return float(reward)


def reward_naive(step: StepResult, params: EnvParams) -> float:
    """A tempting but gameable proxy: minimize stockouts, ignore every cost.

    Optimizing this rewards ordering to the warehouse cap every period -- maximal service,
    ruinous holding. It is the inventory analog of "optimize engagement" or "optimize
    conversion": the proxy is satisfied and the business is not.
    """
    return float(-params.stockout_penalty * step.unmet)


# Registry so the CLI / experiments can select a reward by name and record which was used.
REWARD_FUNCTIONS: dict[str, RewardFn] = {
    "balanced": reward_balanced,
    "naive": reward_naive,
}
