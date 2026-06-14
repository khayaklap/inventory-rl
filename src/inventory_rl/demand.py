"""The stochastic demand world: stationary and seasonal Poisson processes, plus a registry
of named stress scenarios used by the evaluation harness.

Demand is the only exogenous randomness in the environment. Keeping it in its own module --
generated from an explicit ``numpy`` ``Generator`` and never entangled with the agent's
exploration randomness -- is what makes the paired, held-out evaluation honest: the same
demand realization can be replayed against every policy, and training demand can be proven
disjoint from evaluation demand.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from inventory_rl.config import DemandParams


@dataclass(frozen=True)
class StationaryPoisson:
    """Demand drawn i.i.d. from ``Poisson(mean)`` every period."""

    mean: float
    name: str = "stationary"

    def rate(self, week: int) -> float:  # noqa: ARG002 - constant rate by design
        """Poisson rate at ``week`` (constant)."""
        return self.mean

    def sample_episode(self, horizon: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a full episode of integer demands."""
        return rng.poisson(self.mean, size=horizon).astype(np.int64)


@dataclass(frozen=True)
class SeasonalPoisson:
    """Demand with a sinusoidally varying rate ``mean + amplitude * sin(2*pi*t / period)``.

    The rate is floored at ``min_rate`` so it stays a valid Poisson parameter. A single
    static base-stock level is misspecified for this process -- the whole point of the
    seasonal experiment.
    """

    mean: float
    amplitude: float
    period: int
    min_rate: float = 0.1
    name: str = "seasonal"

    def rate(self, week: int) -> float:
        """Poisson rate at ``week`` (time-varying, floored at ``min_rate``)."""
        raw = self.mean + self.amplitude * math.sin(2.0 * math.pi * week / self.period)
        return max(self.min_rate, raw)

    def sample_episode(self, horizon: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a full episode of integer demands with the week-dependent rate."""
        rates = np.array([self.rate(t) for t in range(horizon)])
        return rng.poisson(rates).astype(np.int64)


def stationary(params: DemandParams) -> StationaryPoisson:
    """Build the default stationary demand process from config."""
    return StationaryPoisson(mean=params.mean)


def seasonal(params: DemandParams) -> SeasonalPoisson:
    """Build the default seasonal demand process from config."""
    return SeasonalPoisson(
        mean=params.mean,
        amplitude=params.amplitude,
        period=params.period,
        min_rate=params.min_rate,
    )


# Stress scenarios
# Each scenario builds a demand episode (optionally on top of a base process) and may
# override environment parameters (e.g. a longer lead time). The evaluation harness loads
# these by name from `evals/scenarios.jsonl` and asserts the named invariant.

DemandBuilder = Callable[[int, np.random.Generator, DemandParams], np.ndarray]


@dataclass(frozen=True)
class ScenarioSpec:
    """A named stress scenario: how to build its demand and what must remain true."""

    name: str
    description: str
    make_demand: DemandBuilder
    invariant: str  # machine-stable code the evals harness asserts (see run_evals.py)
    env_overrides: dict[str, int] = field(default_factory=dict)


def _spike(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Stationary demand with a single 3x spike week at the midpoint."""
    demand = StationaryPoisson(params.mean).sample_episode(horizon, rng)
    demand[horizon // 2] = int(round(3 * params.mean))
    return demand


def _drought(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Stationary demand with a multi-week collapse to a low rate."""
    demand = StationaryPoisson(params.mean).sample_episode(horizon, rng)
    lo = horizon // 3
    demand[lo : lo + 8] = rng.poisson(max(params.min_rate, 0.2 * params.mean), size=8).astype(np.int64)
    return demand


def _lead_time_shock(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Stationary demand; the shock is a longer lead time, applied via env_overrides."""
    return StationaryPoisson(params.mean).sample_episode(horizon, rng)


def _distribution_shift(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Demand permanently elevated to ``shift_mean`` (the overfitting / generalization probe)."""
    return StationaryPoisson(params.shift_mean).sample_episode(horizon, rng)


def _zero_demand(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Stationary demand with a stretch of zero-demand weeks (supplier-holiday glut test)."""
    demand = StationaryPoisson(params.mean).sample_episode(horizon, rng)
    lo = horizon // 2
    demand[lo : lo + 3] = 0
    return demand


def _cap_pressure(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """Sustained high demand that pushes policies against the order and capacity limits."""
    return StationaryPoisson(1.6 * params.mean).sample_episode(horizon, rng)


def _approval_gate(horizon: int, rng: np.random.Generator, params: DemandParams) -> np.ndarray:
    """A drought followed by a surge, tempting large catch-up orders above the approval gate."""
    demand = StationaryPoisson(params.mean).sample_episode(horizon, rng)
    mid = horizon // 2
    demand[mid - 6 : mid] = 0
    demand[mid : mid + 3] = int(round(2.5 * params.mean))
    return demand


STRESS_SCENARIOS: dict[str, ScenarioSpec] = {
    "spike": ScenarioSpec(
        name="spike",
        description="A single 3x demand spike mid-horizon; tests recovery within the protection interval.",
        make_demand=_spike,
        invariant="NO_CAPACITY_VIOLATION",
    ),
    "drought": ScenarioSpec(
        name="drought",
        description="An 8-week demand collapse; tests whether a policy keeps ordering into dead stock.",
        make_demand=_drought,
        invariant="NO_CAPACITY_VIOLATION",
    ),
    "lead_time_shock": ScenarioSpec(
        name="lead_time_shock",
        description="Lead time tripled to 3 periods; tests pipeline-aware ordering under delayed receipts.",
        make_demand=_lead_time_shock,
        invariant="NO_CAPACITY_VIOLATION",
        env_overrides={"lead_time": 3},
    ),
    "distribution_shift": ScenarioSpec(
        name="distribution_shift",
        description="Demand permanently shifted up to lambda=16; the generalization / overfitting probe.",
        make_demand=_distribution_shift,
        invariant="NO_CAPACITY_VIOLATION",
    ),
    "zero_demand": ScenarioSpec(
        name="zero_demand",
        description="A 3-week zero-demand stretch; tests holding-cost discipline (no ordering into a glut).",
        make_demand=_zero_demand,
        invariant="NO_CAPACITY_VIOLATION",
    ),
    "cap_pressure": ScenarioSpec(
        name="cap_pressure",
        description="Sustained high demand against the order/capacity limits; a safety regression test.",
        make_demand=_cap_pressure,
        invariant="NO_CAPACITY_VIOLATION",
    ),
    "approval_gate": ScenarioSpec(
        name="approval_gate",
        description="A drought then a surge that tempts large catch-up orders above the human-approval gate.",
        make_demand=_approval_gate,
        invariant="NO_CAPACITY_VIOLATION",
    ),
}
