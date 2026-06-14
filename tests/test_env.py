"""Tests for the inventory environment: conservation, determinism, and hard constraints."""

from __future__ import annotations

import numpy as np

from inventory_rl.config import EnvParams
from inventory_rl.demand import StationaryPoisson
from inventory_rl.env import InventoryEnv, order_to_action_index


def _rollout(env: InventoryEnv, actions: list[int], seed: int) -> list[dict]:
    """Run one episode following a fixed action sequence; return per-step info dicts."""
    env.reset(seed=seed)
    infos = []
    for a in actions:
        _, _, terminated, truncated, info = env.step(a)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def test_mass_balance_holds_every_step() -> None:
    """Stock conservation (on_hand == init + arrivals - sales) holds at every step."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(10.0))
    rng = np.random.default_rng(7)
    actions = [int(rng.integers(0, env.action_space.n)) for _ in range(env.params.horizon)]
    for info in _rollout(env, actions, seed=7):
        assert info["mass_balance_residual"] == 0


def test_determinism_under_seed() -> None:
    """Identical seed + identical actions reproduce identical reward and observation streams."""
    actions = [3] * 52
    env_a = InventoryEnv(EnvParams(), StationaryPoisson(10.0))
    env_b = InventoryEnv(EnvParams(), StationaryPoisson(10.0))

    env_a.reset(seed=42)
    env_b.reset(seed=42)
    for a in actions:
        obs_a, r_a, term_a, _, _ = env_a.step(a)
        obs_b, r_b, term_b, _, _ = env_b.step(a)
        assert np.array_equal(obs_a, obs_b)
        assert r_a == r_b
        assert term_a == term_b


def test_capacity_constraint_never_violated() -> None:
    """Ordering the maximum every period never pushes on-hand above the warehouse cap."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(3.0))  # low demand => stock would pile up
    max_action = env.action_space.n - 1
    for info in _rollout(env, [max_action] * env.params.horizon, seed=3):
        assert info["capacity_violation"] is False
        assert info["end_on_hand"] <= env.params.i_max
        assert info["inventory_position"] <= env.params.i_max


def test_effective_order_clamped_to_capacity() -> None:
    """When near capacity, the placed order is clamped below the requested quantity."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(1.0))
    max_action = env.action_space.n - 1
    saw_clamp = False
    for info in _rollout(env, [max_action] * env.params.horizon, seed=11):
        assert info["effective_order"] <= info["requested_order"]
        if info["effective_order"] < info["requested_order"]:
            saw_clamp = True
    assert saw_clamp, "expected the capacity clamp to bind under near-zero demand"


def test_inventory_never_negative() -> None:
    """Lost-sales accounting keeps on-hand inventory non-negative."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(20.0))  # high demand => stockouts
    for info in _rollout(env, [0] * env.params.horizon, seed=5):  # never order
        assert info["end_on_hand"] >= 0
        assert info["unmet"] >= 0


def test_capacity_can_be_exceeded_when_unenforced() -> None:
    """The safety counterfactual: with the clamp disabled, on-hand breaches capacity.

    This is what the hard constraint prevents -- proving the guarantee comes from the
    environment, not from trusting the policy. The default (enforced) path is covered by
    test_capacity_constraint_never_violated.
    """
    params = EnvParams(enforce_capacity=False)
    env = InventoryEnv(params, StationaryPoisson(2.0))  # low demand => stock piles up
    max_action = env.action_space.n - 1
    breached = any(info["end_on_hand"] > params.i_max for info in _rollout(env, [max_action] * params.horizon, seed=3))
    assert breached, "expected capacity to be exceeded once the clamp is disabled"


def test_episode_terminates_at_horizon() -> None:
    """The episode terminates exactly at the configured horizon."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(10.0))
    env.reset(seed=1)
    steps = 0
    terminated = False
    while not terminated and steps < 1000:
        _, _, terminated, _, _ = env.step(1)
        steps += 1
    assert steps == env.params.horizon
    assert terminated


def test_order_to_action_index_snaps_to_nearest() -> None:
    """The order->action mapping picks the nearest menu quantity (fair discrete comparison)."""
    menu = EnvParams().action_menu
    assert order_to_action_index(0, menu) == 0
    assert order_to_action_index(13, menu) == menu.index(15)
    assert order_to_action_index(100, menu) == len(menu) - 1
