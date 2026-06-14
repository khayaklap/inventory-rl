"""Figure generation. Reads evaluation results and policies, writes the committed PNGs.

Four figures carry the report's evidence:

* ``reward_curve`` -- training return vs episode (Q-learning and PPO): does learning converge?
* ``policy_comparison`` -- held-out return with 95% CIs per policy and regime: does RL beat
  the baseline, and by how much?
* ``policy_behavior`` -- the learned order-up-to behavior laid over the analytical base-stock:
  did the agent recover a sensible, base-stock-like structure (and slide it with the season)?
* ``reward_hacking`` -- service level vs economic return, plus a cost decomposition: the proxy
  agent wins on service and loses on profit.

The Agg backend is selected at import so figures render headless on any machine; matplotlib
is otherwise imported lazily inside functions so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: render to file without a display

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection
import numpy as np  # noqa: E402

from inventory_rl.config import Config
from inventory_rl.evaluation import PolicyEvalResult, bootstrap_mean_ci

DPI = 120


def _moving_average(values: list[float], window: int) -> np.ndarray:
    """Smooth a noisy training curve with a simple moving average."""
    if len(values) < window or window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def plot_reward_curve(
    training_logs: dict[str, list[float]],
    ppo_payload: dict[str, Any] | None,
    out_path: str | Path,
    window: int = 200,
) -> None:
    """Training return vs episode for the tabular agents and (if present) PPO seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, curve in training_logs.items():
        sm = _moving_average(curve, window)
        axes[0].plot(range(len(sm)), sm, label=name, linewidth=1.2)
    axes[0].set_title("Tabular Q-learning: training return")
    axes[0].set_xlabel(f"episode (moving avg, window={window})")
    axes[0].set_ylabel("episode return ($)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if ppo_payload and ppo_payload.get("stationary_seed_curves"):
        pwin = 20
        for i, curve in enumerate(ppo_payload["stationary_seed_curves"]):
            sm = _moving_average(curve, pwin)
            axes[1].plot(range(len(sm)), sm, label=f"PPO seed {i}", linewidth=1.0, alpha=0.8)
        axes[1].set_title("PPO (stationary): per-seed training return")
        axes[1].set_xlabel(f"episode (moving avg, window={pwin})")
        axes[1].set_ylabel("episode return ($)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "PPO curves unavailable\n(stretch deps / artifact missing)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_policy_comparison(
    results_by_regime: dict[str, dict[str, PolicyEvalResult]],
    baselines: dict[str, str],
    out_path: str | Path,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
) -> None:
    """Grouped bar chart of mean held-out return with 95% CIs, one panel per regime."""
    regimes = list(results_by_regime.keys())
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.5), squeeze=False)
    for ax, regime in zip(axes[0], regimes, strict=True):
        results = results_by_regime[regime]
        names = list(results.keys())
        means, errs = [], []
        for name in names:
            mean, low, high = bootstrap_mean_ci(results[name].returns, n_boot, ci_level)
            means.append(mean)
            errs.append([mean - low, high - mean])
        err_arr = np.asarray(errs).T
        colors = ["#c44" if n == baselines[regime] else "#369" for n in names]
        ax.bar(range(len(names)), means, yerr=err_arr, capsize=4, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{regime} demand (red = baseline anchor)")
        ax.set_ylabel("mean held-out return ($), 95% CI")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _greedy_order_curve(policy: Any, cfg: Config, week: int = 0) -> np.ndarray:
    """Greedy order quantity as a function of inventory position (pipeline assumed empty)."""
    menu = cfg.env.action_menu
    return np.array(
        [menu[policy(np.array([ip, 0, week], dtype=np.float32))] for ip in range(cfg.env.i_max + 1)]
    )


def _implied_order_up_to(policy: Any, cfg: Config, week: int) -> int:
    """The implied order-up-to level: the smallest inventory position with a zero greedy order."""
    menu = cfg.env.action_menu
    for ip in range(cfg.env.i_max + 1):
        if menu[policy(np.array([ip, 0, week], dtype=np.float32))] == 0:
            return ip
    return cfg.env.i_max


def _value_curve(policy: Any, cfg: Config) -> np.ndarray:
    """Learned state value V(s) = max_a Q(s, a) as a function of inventory position."""
    from inventory_rl.q_learning import discretize  # noqa: PLC0415 - local to keep plotting import light

    values = []
    for ip in range(cfg.env.i_max + 1):
        state = discretize(np.array([ip, 0, 0], dtype=np.float32), policy.env_params, policy.q_params)
        values.append(float(np.max(policy.q_table[state])))
    return np.asarray(values)


def plot_policy_behavior(
    cfg: Config,
    q_stationary: Any,
    base_stock_level: int,
    q_phase: Any,
    seasonal_levels: np.ndarray,
    out_path: str | Path,
) -> None:
    """Policy (order curve vs base-stock), the learned value function, and seasonal adaptivity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ip_grid = np.arange(cfg.env.i_max + 1)
    q_orders = _greedy_order_curve(q_stationary, cfg)
    bs_orders = np.clip(base_stock_level - ip_grid, 0, cfg.env.q_max)
    axes[0].step(ip_grid, q_orders, where="mid", label="Q-learning (learned)", linewidth=1.6)
    axes[0].step(ip_grid, bs_orders, where="mid", label=f"base-stock (S={base_stock_level})",
                 linewidth=1.4, linestyle="--")
    axes[0].axvline(base_stock_level, color="grey", alpha=0.4, linewidth=0.8)
    axes[0].set_title("Policy: order quantity vs inventory position")
    axes[0].set_xlabel("inventory position")
    axes[0].set_ylabel("order quantity")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # The learned value function V(s) = max_a Q(s, a): how good each inventory state is.
    values = _value_curve(q_stationary, cfg)
    axes[1].plot(ip_grid, values, color="#286", linewidth=1.6)
    axes[1].axvline(base_stock_level, color="grey", alpha=0.4, linewidth=0.8)
    axes[1].set_title("Learned value function  V(s) = max$_a$ Q(s, a)")
    axes[1].set_xlabel("inventory position")
    axes[1].set_ylabel("estimated value-to-go ($)")
    axes[1].grid(alpha=0.3)

    weeks = np.arange(cfg.demand.period)
    phases = cfg.qlearn.n_phases
    # Evaluate the phase agent at the representative week of each phase.
    q_levels = np.array([
        _implied_order_up_to(q_phase, cfg, int((p + 0.5) / phases * cfg.env.horizon))
        for p in range(phases)
    ])
    phase_weeks = np.array([(p + 0.5) / phases * cfg.demand.period for p in range(phases)])
    axes[2].plot(weeks, seasonal_levels, label="seasonal base-stock $S_t$", linewidth=1.4, linestyle="--")
    axes[2].plot(phase_weeks, q_levels, "o-", label="phase-Q implied order-up-to", linewidth=1.4)
    axes[2].set_title("Seasonal: order-up-to target slides with the season")
    axes[2].set_xlabel("week of year")
    axes[2].set_ylabel("order-up-to level")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_reward_hacking(
    metrics_by_policy: dict[str, dict[str, float]],
    out_path: str | Path,
) -> None:
    """Left: service level (bars) vs economic return (line). Right: cost decomposition."""
    names = list(metrics_by_policy.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    service = [metrics_by_policy[n]["service_level"] for n in names]
    returns = [metrics_by_policy[n]["total_return"] for n in names]
    ax_left = axes[0]
    bars = ax_left.bar(range(len(names)), service, color="#8bc", alpha=0.8, label="service level")
    ax_left.set_ylabel("service level (fill rate)")
    ax_left.set_ylim(0, 1.05)
    ax_left.set_xticks(range(len(names)))
    ax_left.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    for b, s in zip(bars, service, strict=True):
        ax_left.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=7)
    ax_right = ax_left.twinx()
    ax_right.plot(range(len(names)), returns, "o-", color="#c33", label="economic return")
    ax_right.set_ylabel("mean held-out return ($)", color="#c33")
    ax_left.set_title("The proxy wins on service and loses on profit")

    holding = [metrics_by_policy[n]["holding_cost"] for n in names]
    penalty = [metrics_by_policy[n]["penalty_cost"] for n in names]
    purchase = [metrics_by_policy[n].get("purchase_cost", 0.0) for n in names]
    x = np.arange(len(names))
    axes[1].bar(x, holding, label="holding", color="#e69")
    axes[1].bar(x, penalty, bottom=holding, label="stockout penalty", color="#69e")
    axes[1].bar(x, purchase, bottom=np.array(holding) + np.array(penalty), label="purchase", color="#9c6")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("total cost over horizon ($)")
    axes[1].set_title("Cost decomposition (holding explodes under the proxy)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
