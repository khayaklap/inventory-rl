"""Tests for the demand processes and the stress-scenario registry."""

from __future__ import annotations

import math

import numpy as np

from inventory_rl.config import DemandParams
from inventory_rl.demand import (
    STRESS_SCENARIOS,
    SeasonalPoisson,
    StationaryPoisson,
    seasonal,
    stationary,
)


def test_stationary_sample_mean_close_to_lambda() -> None:
    """Many stationary draws average to the configured rate."""
    rng = np.random.default_rng(0)
    demand = StationaryPoisson(10.0).sample_episode(20_000, rng)
    assert abs(demand.mean() - 10.0) < 0.3
    assert demand.dtype == np.int64
    assert (demand >= 0).all()


def test_seasonal_rate_is_periodic() -> None:
    """The seasonal rate repeats every `period` weeks and swings around the mean."""
    proc = SeasonalPoisson(mean=10.0, amplitude=6.0, period=52)
    assert math.isclose(proc.rate(0), proc.rate(52), abs_tol=1e-9)
    assert proc.rate(13) > proc.rate(0)  # quarter-year peak is above the mean
    assert proc.rate(39) < proc.rate(0)  # three-quarter trough is below the mean


def test_seasonal_rate_floored_nonnegative() -> None:
    """A large amplitude cannot drive the Poisson rate to or below zero."""
    proc = SeasonalPoisson(mean=2.0, amplitude=10.0, period=52, min_rate=0.1)
    rates = [proc.rate(t) for t in range(52)]
    assert min(rates) >= 0.1


def test_shifted_mean_exceeds_stationary() -> None:
    """The distribution-shift scenario draws from a higher mean than the training process."""
    params = DemandParams()
    rng = np.random.default_rng(1)
    base = stationary(params).sample_episode(10_000, rng)
    shifted = StationaryPoisson(params.shift_mean).sample_episode(10_000, rng)
    assert shifted.mean() > base.mean() + 3.0


def test_default_builders_use_config() -> None:
    """The stationary/seasonal builders read their parameters from config."""
    params = DemandParams()
    assert stationary(params).mean == params.mean
    s = seasonal(params)
    assert s.amplitude == params.amplitude and s.period == params.period


def test_stress_scenarios_reproducible_under_seed() -> None:
    """Every stress scenario produces the identical demand sequence for a fixed seed."""
    params = DemandParams()
    horizon = 52
    for spec in STRESS_SCENARIOS.values():
        a = spec.make_demand(horizon, np.random.default_rng(123), params)
        b = spec.make_demand(horizon, np.random.default_rng(123), params)
        assert np.array_equal(a, b)
        assert len(a) == horizon
        assert (a >= 0).all()


def test_lead_time_shock_declares_env_override() -> None:
    """The lead-time scenario carries the environment override the harness applies."""
    spec = STRESS_SCENARIOS["lead_time_shock"]
    assert spec.env_overrides.get("lead_time") == 3


def test_every_scenario_has_an_invariant_code() -> None:
    """Each scenario names a machine-stable invariant for the evals harness to assert."""
    for spec in STRESS_SCENARIOS.values():
        assert spec.invariant and spec.invariant.isupper()
