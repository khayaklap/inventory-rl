"""Tests for tabular Q-learning: update arithmetic, exploration schedule, discretization,
and convergence on a trivial MDP plus a fast structural check on the inventory problem."""

from __future__ import annotations

import numpy as np

from inventory_rl.config import EnvParams, QLearnParams
from inventory_rl.demand import StationaryPoisson
from inventory_rl.env import InventoryEnv
from inventory_rl.q_learning import (
    QLearningAgent,
    bellman_update,
    discretize,
    epsilon_at,
    n_ip_bins,
    state_space_shape,
)


def test_bellman_update_matches_hand_computation() -> None:
    """The Q-update equals Q + alpha*(r + gamma*max_a' Q' - Q) exactly."""
    # 2 + 0.5*(1 + 0.9*3 - 2) = 2 + 0.5*1.7 = 2.85
    assert bellman_update(2.0, 1.0, 3.0, 0.5, 0.9) == 2.85


def test_q_learning_converges_on_trivial_mdp() -> None:
    """On a 2-state MDP where 'move' pays 1 and 'stay' pays 0, the agent learns to move."""
    gamma, alpha = 0.9, 0.1
    q = np.zeros((1, 2))  # one decision state, two actions: 0=stay, 1=move-and-finish
    rng = np.random.default_rng(0)
    for _ in range(5_000):
        action = int(rng.integers(2))  # pure exploration; Q-learning is off-policy
        if action == 0:  # stay: no reward, self-loop
            reward, next_max = 0.0, float(np.max(q[0]))
        else:  # move: reward 1 and the episode ends (no future value)
            reward, next_max = 1.0, 0.0
        q[0][action] = bellman_update(float(q[0][action]), reward, next_max, alpha, gamma)
    assert int(np.argmax(q[0])) == 1


def test_epsilon_schedule_is_monotone_and_reaches_floor() -> None:
    """Exploration starts high, never increases, and settles at the floor."""
    qp = QLearnParams()
    n = 1_000
    eps = [epsilon_at(e, qp, n) for e in range(n)]
    assert eps[0] == qp.epsilon_start
    assert all(b <= a + 1e-12 for a, b in zip(eps, eps[1:], strict=False))
    assert abs(eps[-1] - qp.epsilon_end) < 1e-9


def test_discretization_is_total_and_in_bounds() -> None:
    """Every reachable observation maps to a valid index inside the state grid."""
    env_params = EnvParams()
    qp = QLearnParams(use_phase=True)
    shape = state_space_shape(env_params, qp)
    for on_hand in range(0, env_params.i_max + 1, 3):
        for pipeline in range(0, env_params.q_max + 1, 5):
            for week in range(0, env_params.horizon):
                obs = np.array([on_hand, pipeline, week], dtype=np.float32)
                state = discretize(obs, env_params, qp)
                assert len(state) == len(shape)
                assert all(0 <= idx < dim for idx, dim in zip(state, shape, strict=True))


def test_state_grid_is_small() -> None:
    """The default discretization keeps the table tiny (fast, inspectable)."""
    env_params = EnvParams()
    assert n_ip_bins(env_params, QLearnParams()) == 11  # inventory position 0..50, width 5
    assert int(np.prod(state_space_shape(env_params, QLearnParams()))) == 11


def test_agent_learns_base_stock_like_structure() -> None:
    """After short training, the greedy policy orders aggressively when empty and stops well
    above the base-stock level (a high-but-uncapped inventory position), recovering the
    monotone order-up-to shape from reward alone."""
    env = InventoryEnv(EnvParams(), StationaryPoisson(10.0))
    agent = QLearningAgent(env, QLearnParams(), seed=0)
    agent.train(StationaryPoisson(10.0), n_episodes=4_000, seed=1_000)
    policy = agent.greedy_policy()
    menu = env.params.action_menu

    def order_at(ip: int) -> int:
        return menu[policy(np.array([ip, 0, 0], dtype=np.float32))]

    # Inventory position 35 is far above the newsvendor level (~26) but below the cap (50),
    # so ordering nothing there is genuinely optimal and learnable.
    assert order_at(0) >= 15  # orders aggressively to refill an empty system
    assert order_at(0) > order_at(35)  # orders less as the system fills (monotone structure)
    assert order_at(35) == 0  # stops ordering well above the base-stock level


def test_save_and_load_roundtrip(tmp_path) -> None:
    """A saved Q-table reloads into a greedy policy that makes identical decisions."""
    from inventory_rl.q_learning import load_q_policy

    env = InventoryEnv(EnvParams(), StationaryPoisson(10.0))
    agent = QLearningAgent(env, QLearnParams(), seed=0)
    agent.train(StationaryPoisson(10.0), n_episodes=500, seed=1_000)
    path = tmp_path / "q_table.npz"
    agent.save(path)
    loaded = load_q_policy(path, env.params)
    original = agent.greedy_policy()
    for ip in range(0, 51, 5):
        obs = np.array([ip, 0, 0], dtype=np.float32)
        assert loaded(obs) == original(obs)
