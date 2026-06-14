"""The inventory-replenishment simulator: a periodic-review, lead-time, lost-sales MDP.

The environment is a small, inspectable Gymnasium ``Env`` so the same object trains the
tabular agent, trains PPO, and runs every evaluation. Two design points matter most:

1. **Hard safety constraints live here, not in the reward.** A single order is clamped to
   ``[0, q_max]`` and further clamped so inventory position can never exceed warehouse
   capacity ``i_max``. The agent cannot learn its way around these limits because the
   environment enforces them before the order is ever placed -- the course's "rules the
   policy cannot cross" principle, implemented.
2. **Demand is injected, not owned.** An episode runs against a demand sequence supplied at
   ``reset`` (for paired evaluation) or sampled from the configured process (for training).
   The environment never mixes demand randomness with the agent's exploration randomness.

Per-period timing (one decision per period):
    observe -> receive order placed L periods ago -> place (clamped) order ->
    demand realized -> sell what stock allows (rest is lost) -> charge reward.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from inventory_rl.config import EnvParams
from inventory_rl.demand import StationaryPoisson
from inventory_rl.rewards import RewardFn, StepResult, reward_balanced

# Observation layout: the three numbers a policy needs to decide an order.
OBS_ON_HAND = 0
OBS_PIPELINE = 1
OBS_WEEK = 2


class InventoryEnv(gym.Env):
    """Single-SKU periodic-review inventory environment with lost sales.

    Observation: ``[on_hand, pipeline_total, week]`` as ``float32``.
    Action: an index into ``params.action_menu`` (a discrete order quantity).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: EnvParams | None = None,
        demand_process: Any | None = None,
        reward_fn: RewardFn = reward_balanced,
    ) -> None:
        super().__init__()
        self.params = params or EnvParams()
        # Default to a stationary process at the configured-ish mean; callers usually inject
        # a demand sequence at reset, so this only matters for ad-hoc sampling.
        self.demand_process = demand_process or StationaryPoisson(10.0)
        self.reward_fn = reward_fn

        p = self.params
        self.action_space = spaces.Discrete(len(p.action_menu))
        high = np.array([p.i_max, p.q_max * p.lead_time, p.horizon], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(3,), dtype=np.float32)

        # Episode state (initialized in reset).
        self.on_hand: int = 0
        self.pipeline: deque[int] = deque()
        self.week: int = 0
        self._demand_sequence: np.ndarray = np.zeros(p.horizon, dtype=np.int64)
        self._lead_time: int = p.lead_time
        # Conservation counters, for the mass-balance invariant.
        self._total_arrivals: int = 0
        self._total_sales: int = 0

    # Gymnasium API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        ``options`` may carry ``demand_sequence`` (replay a fixed episode -- used by paired
        evaluation) and/or ``lead_time`` (a per-episode override -- used by the lead-time
        stress scenario). Without a supplied sequence, demand is sampled from
        ``self.demand_process`` using the seeded Gymnasium RNG.
        """
        super().reset(seed=seed)
        options = options or {}
        self._lead_time = int(options.get("lead_time", self.params.lead_time))

        seq = options.get("demand_sequence")
        if seq is not None:
            self._demand_sequence = np.asarray(seq, dtype=np.int64)
        else:
            self._demand_sequence = self.demand_process.sample_episode(
                self.params.horizon, self.np_random
            )

        self.on_hand = self.params.init_on_hand
        self.pipeline = deque([0] * self._lead_time)
        self.week = 0
        self._total_arrivals = 0
        self._total_sales = 0
        return self._obs(), {"demand_sequence_len": int(len(self._demand_sequence))}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one period: receive, order (clamped), serve demand, score reward."""
        p = self.params
        requested_order = int(p.action_menu[int(action)])

        # 1) Receive the order placed `lead_time` periods ago (front of the pipeline).
        arrival = self.pipeline.popleft()
        on_hand_after_receipt = self.on_hand + arrival
        self._total_arrivals += arrival

        # 2) Place the new order, enforcing the hard constraints HERE (not via reward):
        #    never exceed the per-order cap, and never let inventory position exceed capacity.
        pending = sum(self.pipeline)  # orders still in transit (excludes the one just received)
        inv_position = on_hand_after_receipt + pending
        max_orderable = max(0, p.i_max - inv_position)
        effective_order = min(requested_order, p.q_max)
        if p.enforce_capacity:
            # The hard capacity clamp: ordering can never push inventory position past i_max.
            # Disabled only by the safety ablation, to show what the limit prevents.
            effective_order = min(effective_order, max_orderable)
        approval_flagged = requested_order > p.approval_threshold
        self.pipeline.append(effective_order)  # arrives after `lead_time` periods

        # 3) Demand is realized; lost-sales: unmet demand is lost, not backordered.
        demand = int(self._demand_sequence[self.week])
        sales = min(on_hand_after_receipt, demand)
        unmet = demand - sales
        end_on_hand = on_hand_after_receipt - sales
        self._total_sales += sales

        self.week += 1
        is_terminal = self.week >= p.horizon
        result = StepResult(
            sales=sales,
            end_on_hand=end_on_hand,
            unmet=unmet,
            order_qty=effective_order,
            is_terminal=is_terminal,
        )
        reward = self.reward_fn(result, p)
        self.on_hand = end_on_hand

        info: dict[str, Any] = {
            "demand": demand,
            "sales": sales,
            "unmet": unmet,
            "end_on_hand": end_on_hand,
            "requested_order": requested_order,
            "effective_order": effective_order,
            "inventory_position": end_on_hand + sum(self.pipeline),
            "approval_flagged": approval_flagged,
            "capacity_violation": end_on_hand > p.i_max,  # must always be False
            "mass_balance_residual": self.mass_balance_residual(),
        }
        return self._obs(), reward, is_terminal, False, info

    # Helpers
    def _obs(self) -> np.ndarray:
        """Current observation as ``float32``: ``[on_hand, pipeline_total, week]``."""
        return np.array([self.on_hand, sum(self.pipeline), self.week], dtype=np.float32)

    def inventory_position(self) -> int:
        """On-hand plus everything in the pipeline -- the statistic base-stock acts on."""
        return self.on_hand + sum(self.pipeline)

    def mass_balance_residual(self) -> int:
        """Stock-conservation residual; must be exactly zero at every step.

        Lost-sales accounting: ``on_hand == init_on_hand + total_arrivals - total_sales``.
        Any non-zero value signals a bug in the receive/sell bookkeeping.
        """
        return self.on_hand - (self.params.init_on_hand + self._total_arrivals - self._total_sales)


def order_to_action_index(order_qty: int, action_menu: tuple[int, ...]) -> int:
    """Map a desired order quantity to the nearest index in the discrete action menu.

    Used so continuous-looking policies (e.g. base-stock's ``S - inventory_position``) can be
    compared fairly against the agent on the identical discrete action set.
    """
    arr = np.asarray(action_menu)
    return int(np.argmin(np.abs(arr - order_qty)))
