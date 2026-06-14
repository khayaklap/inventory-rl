"""Orchestration: train the agents, run the honest policy x regime matrix, and run the
reward-hacking experiment. This is the engine behind the CLI -- it owns no argument parsing
and draws no plots, so the heavy logic stays testable and reusable.

The evaluation matrix is deliberately *honest* about when RL helps:

* Under **stationary** demand a correctly specified static base-stock is near-optimal, so the
  learned agents should only *tie* it -- the negative result that proves RL is not oversold.
* Under **seasonal** demand the fair competitor is a *time-indexed* ``S_t`` base-stock, not a
  handicapped static one. The phase-aware Q-agent should match that adaptive baseline while
  beating the static one. The value of RL is that it reaches the seasonal solution from
  reward alone, without being handed the seasonal model.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from inventory_rl import demand as demand_mod
from inventory_rl.baselines import (
    BaseStockPolicy,
    ConstantOrderPolicy,
    RandomPolicy,
    SeasonalBaseStockPolicy,
    seasonal_base_stock_levels,
    stationary_base_stock_level,
)
from inventory_rl.config import Config
from inventory_rl.env import InventoryEnv
from inventory_rl.evaluation import (
    Policy,
    PolicyEvalResult,
    build_report,
    evaluate_policies,
    generate_demand_matrix,
    rollout,
)
from inventory_rl.q_learning import GreedyQPolicy, QLearningAgent
from inventory_rl.rewards import reward_balanced, reward_naive

# Display names used consistently across the report, the CSV, and the figures.
NAME_RANDOM = "random"
NAME_BASE_STOCK = "base_stock"
NAME_STATIC_BASE_STOCK = "static_base_stock"
NAME_SEASONAL_BASE_STOCK = "seasonal_base_stock"
NAME_Q = "q_learning"
NAME_Q_IP = "q_learning_ip"
NAME_Q_PHASE = "q_learning_phase"
NAME_NAIVE = "q_learning_naive"
NAME_PPO = "ppo"


@dataclass
class TrainedAgents:
    """The learned greedy policies plus their training-return curves."""

    q_stationary: GreedyQPolicy
    q_seasonal_ip: GreedyQPolicy
    q_seasonal_phase: GreedyQPolicy
    q_naive: GreedyQPolicy
    training_logs: dict[str, list[float]] = field(default_factory=dict)


def _train_q(
    cfg: Config,
    demand_process: Any,
    *,
    use_phase: bool,
    reward_fn: Any = reward_balanced,
    n_episodes: int | None = None,
    name: str,
) -> tuple[GreedyQPolicy, list[float]]:
    """Train one tabular agent and return its greedy policy and training-return curve."""
    q_params = dataclasses.replace(cfg.qlearn, use_phase=use_phase)
    env = InventoryEnv(cfg.env, demand_process=demand_process, reward_fn=reward_fn)
    agent = QLearningAgent(env, q_params, seed=cfg.seed)
    episodes = n_episodes if n_episodes is not None else cfg.qlearn.n_episodes
    log = agent.train(demand_process, n_episodes=episodes, seed=cfg.eval.train_seed_base)
    return agent.greedy_policy(name=name), log["returns"]


def train_all_q_agents(cfg: Config, n_episodes: int | None = None) -> TrainedAgents:
    """Train every tabular agent the experiments need (stationary, seasonal x2, naive)."""
    stationary = demand_mod.stationary(cfg.demand)
    seasonal = demand_mod.seasonal(cfg.demand)
    logs: dict[str, list[float]] = {}

    q_stationary, logs[NAME_Q] = _train_q(
        cfg, stationary, use_phase=False, n_episodes=n_episodes, name=NAME_Q
    )
    q_seasonal_ip, logs[NAME_Q_IP] = _train_q(
        cfg, seasonal, use_phase=False, n_episodes=n_episodes, name=NAME_Q_IP
    )
    q_seasonal_phase, logs[NAME_Q_PHASE] = _train_q(
        cfg, seasonal, use_phase=True, n_episodes=n_episodes, name=NAME_Q_PHASE
    )
    # The naive agent is trained on the gameable proxy (stationary demand) for the
    # reward-hacking experiment; it will be SCORED on the true objective.
    q_naive, logs[NAME_NAIVE] = _train_q(
        cfg, stationary, use_phase=False, reward_fn=reward_naive, n_episodes=n_episodes, name=NAME_NAIVE
    )
    return TrainedAgents(
        q_stationary=q_stationary,
        q_seasonal_ip=q_seasonal_ip,
        q_seasonal_phase=q_seasonal_phase,
        q_naive=q_naive,
        training_logs=logs,
    )


def stationary_roster(cfg: Config, agents: TrainedAgents, ppo: Policy | None = None) -> list[Policy]:
    """Policies compared under stationary demand (base-stock is the near-optimal anchor)."""
    s = stationary_base_stock_level(cfg.env, cfg.demand.mean)
    roster: list[Policy] = [
        RandomPolicy(len(cfg.env.action_menu), seed=cfg.seed),
        BaseStockPolicy(s, cfg.env, name=NAME_BASE_STOCK),
        agents.q_stationary,
    ]
    if ppo is not None:
        roster.append(ppo)
    return roster


def seasonal_roster(cfg: Config, agents: TrainedAgents, ppo: Policy | None = None) -> list[Policy]:
    """Policies compared under seasonal demand (seasonal base-stock is the fair competitor)."""
    static_s = stationary_base_stock_level(cfg.env, cfg.demand.mean)
    seasonal_levels = seasonal_base_stock_levels(cfg.env, cfg.demand)
    roster: list[Policy] = [
        RandomPolicy(len(cfg.env.action_menu), seed=cfg.seed),
        BaseStockPolicy(static_s, cfg.env, name=NAME_STATIC_BASE_STOCK),
        SeasonalBaseStockPolicy(seasonal_levels, cfg.env, name=NAME_SEASONAL_BASE_STOCK),
        agents.q_seasonal_ip,
        agents.q_seasonal_phase,
    ]
    if ppo is not None:
        roster.append(ppo)
    return roster


@dataclass
class MatrixResult:
    """Per-regime evaluation results plus the assembled report dictionaries."""

    results_by_regime: dict[str, dict[str, PolicyEvalResult]]
    reports: dict[str, Any]
    baselines: dict[str, str]


def run_evaluation_matrix(
    cfg: Config,
    agents: TrainedAgents,
    ppo_policies: dict[str, Policy] | None = None,
) -> MatrixResult:
    """Run the full policy x regime evaluation on the held-out (out-of-sample) demand set."""
    ppo_policies = ppo_policies or {}
    stationary = demand_mod.stationary(cfg.demand)
    seasonal = demand_mod.seasonal(cfg.demand)

    eval_stationary = generate_demand_matrix(
        stationary, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )
    eval_seasonal = generate_demand_matrix(
        seasonal, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )

    rosters = {
        "stationary": (stationary_roster(cfg, agents, ppo_policies.get("stationary")), eval_stationary),
        "seasonal": (seasonal_roster(cfg, agents, ppo_policies.get("seasonal")), eval_seasonal),
    }
    baselines = {"stationary": NAME_BASE_STOCK, "seasonal": NAME_SEASONAL_BASE_STOCK}

    results_by_regime: dict[str, dict[str, PolicyEvalResult]] = {}
    reports: dict[str, Any] = {}
    for regime, (roster, matrix) in rosters.items():
        results = evaluate_policies(roster, matrix, cfg.env)
        results_by_regime[regime] = results
        meta = {
            "regime": regime,
            "n_eval_episodes": cfg.eval.n_eval_episodes,
            "eval_seed_base": cfg.eval.eval_seed_base,
            "train_seed_base": cfg.eval.train_seed_base,
            "horizon": cfg.env.horizon,
            "seed": cfg.seed,
            "baseline": baselines[regime],
            "base_stock_level": stationary_base_stock_level(cfg.env, cfg.demand.mean),
        }
        report = build_report(
            results, meta, baselines[regime], cfg.eval.n_bootstrap, cfg.eval.ci_level
        )
        reports[regime] = dataclasses.asdict(report)
    return MatrixResult(results_by_regime, reports, baselines)


@dataclass
class RewardHackingResult:
    """The naive-vs-balanced agents scored on the same true-objective held-out episodes."""

    balanced_metrics: dict[str, float]
    naive_metrics: dict[str, float]
    summary: dict[str, Any]


def run_reward_hacking_experiment(cfg: Config, agents: TrainedAgents) -> RewardHackingResult:
    """Score the balanced-trained and naive-trained agents on the true objective.

    Both agents are evaluated under the balanced reward on the held-out stationary set, so the
    contrast is apples-to-apples: the naive agent chased the service-only proxy, and we read
    off what that cost the business.
    """
    stationary = demand_mod.stationary(cfg.demand)
    matrix = generate_demand_matrix(
        stationary, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )
    balanced = evaluate_policies([agents.q_stationary], matrix, cfg.env)[NAME_Q]
    naive = evaluate_policies([agents.q_naive], matrix, cfg.env)[NAME_NAIVE]

    def mean_metrics(res: PolicyEvalResult) -> dict[str, float]:
        out = {"total_return": float(res.returns.mean())}
        out.update({k: float(v.mean()) for k, v in res.metrics.items()})
        return out

    bm, nm = mean_metrics(balanced), mean_metrics(naive)
    summary = {
        "service_gain_naive_minus_balanced": nm["service_level"] - bm["service_level"],
        "profit_loss_balanced_minus_naive": bm["total_return"] - nm["total_return"],
        "profit_loss_pct": (
            100.0 * (bm["total_return"] - nm["total_return"]) / abs(bm["total_return"])
            if bm["total_return"] != 0
            else 0.0
        ),
        "naive_avg_on_hand": nm["avg_on_hand"],
        "balanced_avg_on_hand": bm["avg_on_hand"],
        "naive_capacity_violations": nm["capacity_violations"],
    }
    return RewardHackingResult(balanced_metrics=bm, naive_metrics=nm, summary=summary)


@dataclass
class GeneralizationResult:
    """Overfitting probe: tuned-for-stationary policies scored in- and out-of-distribution."""

    summary: dict[str, float]


def run_generalization_experiment(cfg: Config, agents: TrainedAgents) -> GeneralizationResult:
    """Measure how the stationary-tuned agent and base-stock degrade under a demand shift.

    Both the Q-agent and the base-stock level were tuned for lambda=10. We score them on the
    held-out in-distribution set (lambda=10) and on a permanently shifted set (lambda=16), and
    compare each to an *oracle* base-stock recomputed for lambda=16 (the achievable best if the
    shift were known). The gap to the oracle measures how badly each policy generalizes -- the
    overfitting failure mode, made quantitative rather than asserted.
    """
    stationary = demand_mod.stationary(cfg.demand)
    shifted = demand_mod.StationaryPoisson(cfg.demand.shift_mean)
    in_matrix = generate_demand_matrix(
        stationary, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )
    shift_matrix = generate_demand_matrix(
        shifted, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )

    base_tuned = BaseStockPolicy(stationary_base_stock_level(cfg.env, cfg.demand.mean), cfg.env, NAME_BASE_STOCK)
    oracle_level = stationary_base_stock_level(cfg.env, cfg.demand.shift_mean)
    base_oracle = BaseStockPolicy(oracle_level, cfg.env, "base_stock_oracle")

    in_res = evaluate_policies([agents.q_stationary, base_tuned], in_matrix, cfg.env)
    shift_res = evaluate_policies([agents.q_stationary, base_tuned, base_oracle], shift_matrix, cfg.env)

    q_in, q_shift = float(in_res[NAME_Q].returns.mean()), float(shift_res[NAME_Q].returns.mean())
    bs_in = float(in_res[NAME_BASE_STOCK].returns.mean())
    bs_shift = float(shift_res[NAME_BASE_STOCK].returns.mean())
    oracle_shift = float(shift_res["base_stock_oracle"].returns.mean())

    summary = {
        "q_in_distribution": q_in,
        "q_shifted": q_shift,
        "q_gap_to_oracle": oracle_shift - q_shift,
        "base_stock_in_distribution": bs_in,
        "base_stock_shifted": bs_shift,
        "base_stock_gap_to_oracle": oracle_shift - bs_shift,
        "oracle_shifted": oracle_shift,
        "oracle_base_stock_level": float(oracle_level),
        "tuned_base_stock_level": float(stationary_base_stock_level(cfg.env, cfg.demand.mean)),
    }
    return GeneralizationResult(summary=summary)


@dataclass
class SafetyAblationResult:
    """Counterfactual showing the capacity constraint is enforced by code, not by the policy."""

    summary: dict[str, float]


def run_safety_ablation(cfg: Config) -> SafetyAblationResult:
    """Run a worst-case always-max-order policy with the capacity clamp on vs off.

    With the clamp on, on-hand can never exceed warehouse capacity (zero violations) no matter
    how pathological the policy; with it off, the same policy floods the warehouse far past the
    limit. This makes the "safety is enforced, not learned" claim empirical: the guarantee comes
    from the environment, not from trusting the agent.
    """
    max_index = len(cfg.env.action_menu) - 1
    demand = demand_mod.stationary(cfg.demand).sample_episode(
        cfg.env.horizon, np.random.default_rng(cfg.eval.eval_seed_base)
    )

    capped_policy = ConstantOrderPolicy(max_index, cfg.env, name="always_max")
    capped = rollout(capped_policy, demand, cfg.env)

    uncapped_env = dataclasses.replace(cfg.env, enforce_capacity=False)
    uncapped_policy = ConstantOrderPolicy(max_index, uncapped_env, name="always_max")
    uncapped = rollout(uncapped_policy, demand, uncapped_env)

    summary = {
        "i_max": float(cfg.env.i_max),
        "capped_peak_on_hand": float(max(capped.trajectory["on_hand"])),
        "capped_capacity_violations": float(capped.metrics["capacity_violations"]),
        "uncapped_peak_on_hand": float(max(uncapped.trajectory["on_hand"])),
        "uncapped_capacity_violations": float(uncapped.metrics["capacity_violations"]),
    }
    return SafetyAblationResult(summary=summary)


@dataclass
class RobustnessResult:
    """Sensitivity of the tabular result to its two main knobs: discretization and seed."""

    discretization: dict[str, dict[str, float]]  # bin_width -> {return, gap_to_base_stock, n_states}
    training_seed_returns: list[float]  # held-out return of the default agent across training seeds
    base_stock_return: float


def run_robustness_study(
    cfg: Config,
    n_episodes: int | None = None,
    bin_widths: tuple[int, ...] = (2, 5, 10, 25),
    extra_seeds: tuple[int, ...] = (1, 2),
) -> RobustnessResult:
    """Quantify how the headline tie depends on the discretization and the training seed.

    Two knobs a skeptical reviewer would poke: (1) the inventory-position bin width -- too
    coarse and the table cannot represent the base-stock staircase, so the gap to base-stock
    widens; (2) the training seed -- a single lucky seed is not evidence. We retrain the
    tabular agent across bin widths and seeds and report the held-out return for each, so the
    "within ~3% of base-stock" claim is shown to be a property of the method, not a fluke.
    """
    episodes = n_episodes if n_episodes is not None else cfg.qlearn.n_episodes
    stationary = demand_mod.stationary(cfg.demand)
    matrix = generate_demand_matrix(
        stationary, cfg.eval.n_eval_episodes, cfg.eval.eval_seed_base, cfg.env.horizon
    )
    base = BaseStockPolicy(stationary_base_stock_level(cfg.env, cfg.demand.mean), cfg.env, NAME_BASE_STOCK)
    base_return = float(evaluate_policies([base], matrix, cfg.env)[NAME_BASE_STOCK].returns.mean())

    def train_eval(bin_width: int, seed: int) -> float:
        q_params = dataclasses.replace(cfg.qlearn, ip_bin_width=bin_width)
        env = InventoryEnv(cfg.env, demand_process=stationary)
        agent = QLearningAgent(env, q_params, seed=seed)
        agent.train(stationary, episodes, cfg.eval.train_seed_base)
        policy = agent.greedy_policy()
        return float(evaluate_policies([policy], matrix, cfg.env)[policy.name].returns.mean())

    discretization: dict[str, dict[str, float]] = {}
    default_width_return: float | None = None
    for width in bin_widths:
        ret = train_eval(width, cfg.seed)
        discretization[str(width)] = {
            "return": ret,
            "gap_to_base_stock": ret - base_return,
            "n_states": float(cfg.env.i_max // width + 1),
        }
        if width == cfg.qlearn.ip_bin_width:
            default_width_return = ret

    if default_width_return is None:
        default_width_return = train_eval(cfg.qlearn.ip_bin_width, cfg.seed)
    seed_returns = [default_width_return]
    seed_returns += [train_eval(cfg.qlearn.ip_bin_width, cfg.seed + s) for s in extra_seeds]
    return RobustnessResult(
        discretization=discretization,
        training_seed_returns=seed_returns,
        base_stock_return=base_return,
    )


def run_ppo_study(
    cfg: Config,
    models_dir: str | Path,
    evidence_dir: str | Path,
    n_seeds: int | None = None,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """Train the PPO stretch and persist the committed models + the seed-variance curves.

    Stationary PPO is trained under several seeds (the failure-analysis variance study); the
    best is saved as the committed artifact. Seasonal PPO is trained once. The per-seed
    training curves are written to evidence for the reward-curve and variance plots.

    Importing this function never imports torch; the PPO dependencies load only when it runs.
    """
    from inventory_rl import ppo_agent as ppo  # noqa: PLC0415 - lazy: keeps torch out of the core path

    models_dir, evidence_dir = Path(models_dir), Path(evidence_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    stationary = demand_mod.stationary(cfg.demand)
    seasonal = demand_mod.seasonal(cfg.demand)

    study = ppo.train_ppo_multiseed(cfg, stationary, n_seeds=n_seeds, total_timesteps=total_timesteps)
    ppo.save_ppo(study["best_model"], models_dir / "ppo_stationary.zip")

    seasonal_model, seasonal_curve = ppo.train_ppo(
        cfg, seasonal, seed=cfg.seed, total_timesteps=total_timesteps
    )
    ppo.save_ppo(seasonal_model, models_dir / "ppo_seasonal.zip")

    payload = {
        "stationary_seed_curves": study["curves"],
        "stationary_final_returns": study["final_returns"],
        "stationary_best_index": study["best_index"],
        "seasonal_curve": seasonal_curve,
        "total_timesteps": total_timesteps if total_timesteps is not None else cfg.ppo.total_timesteps,
        "n_seeds": n_seeds if n_seeds is not None else cfg.ppo.n_seeds,
    }
    (evidence_dir / "ppo_training.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_ppo_policies(cfg: Config, models_dir: str | Path) -> dict[str, Policy]:
    """Load committed PPO policies per regime, if both the models and torch are available."""
    models_dir = Path(models_dir)
    paths = {"stationary": models_dir / "ppo_stationary.zip", "seasonal": models_dir / "ppo_seasonal.zip"}
    if not all(p.exists() for p in paths.values()):
        return {}
    try:
        from inventory_rl.ppo_agent import PPOPolicy  # noqa: PLC0415 - lazy torch import
    except SystemExit:
        return {}
    return {regime: PPOPolicy.load(path, name=NAME_PPO) for regime, path in paths.items()}


def sample_demo_episode(cfg: Config, policy: Policy, seed_offset: int = 0) -> dict[str, Any]:
    """Run one held-out episode for the CLI `demo` view (a real, inspectable trajectory)."""
    stationary = demand_mod.stationary(cfg.demand)
    seq = stationary.sample_episode(
        cfg.env.horizon, np.random.default_rng(cfg.eval.eval_seed_base + seed_offset)
    )
    result = rollout(policy, seq, cfg.env)
    return {"return": result.total_return, "metrics": result.metrics, "trajectory": result.trajectory}
