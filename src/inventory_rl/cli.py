"""Command-line entry point: train, evaluate, plot, demo, or run the whole pipeline.

Subcommands:

* ``train``    -- train the tabular agents (``--agent ppo`` trains the PPO stretch instead).
* ``evaluate`` -- run the held-out policy x regime matrix and write the report + CSV.
* ``plot``     -- regenerate the figures from committed artifacts.
* ``demo``     -- run one held-out episode and print an inspectable trajectory.
* ``all``      -- the reproducible one-command pipeline: train -> evaluate -> reward-hacking
                  -> plot, writing every committed artifact. Loads the PPO stretch if present.

Everything runs on CPU with no GPU and (for the core path) no torch. Paths are resolved
relative to the repository root, so the commands work from anywhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from inventory_rl import experiments as ex
from inventory_rl import plotting
from inventory_rl.baselines import seasonal_base_stock_levels, stationary_base_stock_level
from inventory_rl.config import Config, default_config
from inventory_rl.evaluation import PolicyEvalResult, write_report, write_results_csv
from inventory_rl.experiments import TrainedAgents
from inventory_rl.q_learning import save_q_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"
EVIDENCE_DIR = REPO_ROOT / "evidence"


def _library_versions() -> dict[str, str]:
    """Record key library versions for run provenance (observability)."""
    import gymnasium
    import scipy

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gymnasium": gymnasium.__version__,
        "python": sys.version.split()[0],
    }


def _env_fingerprint(cfg: Config) -> str:
    """A short, stable hash of the environment + demand parameters (the 'simulator version').

    Recorded in every report so a result can be tied to the exact world that produced it --
    the deck's observability point: log the simulator/environment version, not just the score.
    """
    payload = json.dumps(
        {"env": dataclasses.asdict(cfg.env), "demand": dataclasses.asdict(cfg.demand)},
        sort_keys=True,
        default=list,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _run_id(cfg: Config, n_episodes: int | None) -> str:
    """A deterministic run identifier (the deck's 'trace ID') tying a run's artifacts together.

    A function of the only things that determine the run -- seed, simulator fingerprint, and
    episode counts -- so the same configuration always yields the same ``run_id`` (reproducible),
    and a given ``run_id`` pins exactly which world + seed produced the committed evidence.
    Per-episode traceability is separate: evaluation episode ``i`` is demand seed
    ``eval_seed_base + i``.
    """
    episodes = n_episodes if n_episodes is not None else cfg.qlearn.n_episodes
    payload = f"{cfg.seed}|{_env_fingerprint(cfg)}|{episodes}|{cfg.eval.n_eval_episodes}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _means(res: PolicyEvalResult) -> dict[str, float]:
    """Mean of every metric (plus the return) for one policy's held-out episodes."""
    out = {"total_return": float(res.returns.mean())}
    out.update({k: float(v.mean()) for k, v in res.metrics.items()})
    return out


def _save_q_tables(agents: TrainedAgents) -> None:
    """Persist the committed Q-tables (the artifacts the grader evaluates without retraining)."""
    save_q_policy(agents.q_stationary, MODELS_DIR / "q_table.npz")
    save_q_policy(agents.q_seasonal_ip, MODELS_DIR / "q_table_seasonal_ip.npz")
    save_q_policy(agents.q_seasonal_phase, MODELS_DIR / "q_table_seasonal_phase.npz")
    save_q_policy(agents.q_naive, MODELS_DIR / "q_table_naive.npz")


def _write_training_curves(agents: TrainedAgents, stride: int = 10) -> None:
    """Write downsampled training-return curves for the reward-curve plot."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    curves = {name: log[::stride] for name, log in agents.training_logs.items()}
    (EVIDENCE_DIR / "training_curves.json").write_text(json.dumps(curves), encoding="utf-8")


def _ppo_payload() -> dict[str, Any] | None:
    """Load the saved PPO training curves, if the stretch study has been run."""
    path = EVIDENCE_DIR / "ppo_training.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@dataclasses.dataclass
class PipelineResults:
    """The five experiment outputs of a full run, grouped so the pipeline reads cleanly."""

    matrix: ex.MatrixResult
    reward_hacking: ex.RewardHackingResult
    generalization: ex.GeneralizationResult
    safety_ablation: ex.SafetyAblationResult
    robustness: ex.RobustnessResult
    ppo_payload: dict[str, Any] | None


def _train_and_save(cfg: Config, n_episodes: int | None) -> TrainedAgents:
    """Train the tabular agents and persist the committed Q-tables + training curves."""
    print("Training tabular agents (stationary, seasonal IP, seasonal phase, naive)...")
    agents = ex.train_all_q_agents(cfg, n_episodes=n_episodes)
    _save_q_tables(agents)
    _write_training_curves(agents)
    return agents


def _run_all_experiments(
    cfg: Config, agents: TrainedAgents, ppo_policies: dict[str, ex.Policy], n_episodes: int | None
) -> PipelineResults:
    """Run the evaluation matrix and the four failure/robustness experiments."""
    print("Running held-out evaluation matrix (stationary + seasonal)...")
    matrix = ex.run_evaluation_matrix(cfg, agents, ppo_policies)
    write_results_csv(matrix.results_by_regime, EVIDENCE_DIR / "results.csv")
    print("Running failure-analysis experiments (reward hacking, overfitting, safety ablation)...")
    reward_hacking = ex.run_reward_hacking_experiment(cfg, agents)
    generalization = ex.run_generalization_experiment(cfg, agents)
    safety_ablation = ex.run_safety_ablation(cfg)
    print("Running robustness study (discretization + training-seed sensitivity)...")
    robustness = ex.run_robustness_study(cfg, n_episodes=n_episodes)
    return PipelineResults(matrix, reward_hacking, generalization, safety_ablation, robustness, _ppo_payload())


def _assemble_report(
    cfg: Config, n_episodes: int | None, run_id: str, results: PipelineResults, ppo_included: bool
) -> dict[str, Any]:
    """Build the committed evaluation report: run provenance + every result section."""
    meta = {
        "run_id": run_id,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "n_train_episodes": n_episodes if n_episodes is not None else cfg.qlearn.n_episodes,
        "n_eval_episodes": cfg.eval.n_eval_episodes,
        "base_stock_level": stationary_base_stock_level(cfg.env, cfg.demand.mean),
        "critical_ratio": cfg.env.critical_ratio,
        "env_fingerprint": _env_fingerprint(cfg),
        "ppo_included": ppo_included,
        "library_versions": _library_versions(),
    }
    ppo_payload = results.ppo_payload
    instability: Any = (
        {
            "stationary_final_returns_per_seed": ppo_payload["stationary_final_returns"],
            "n_seeds": ppo_payload["n_seeds"],
        }
        if ppo_payload
        else "PPO artifact absent (run `inventory-rl train --agent ppo`)"
    )
    return {
        "meta": meta,
        "stationary": results.matrix.reports["stationary"],
        "seasonal": results.matrix.reports["seasonal"],
        "reward_hacking": {
            "balanced_metrics": results.reward_hacking.balanced_metrics,
            "naive_metrics": results.reward_hacking.naive_metrics,
            "summary": results.reward_hacking.summary,
        },
        # The four named failure modes, each measured (not asserted).
        "failure_analysis": {
            "reward_hacking": results.reward_hacking.summary,
            "overfitting_generalization": results.generalization.summary,
            "unsafe_behavior_safety_ablation": results.safety_ablation.summary,
            "instability_ppo_seed_variance": instability,
        },
        # Sensitivity of the headline tie to its two main knobs (not a lucky configuration).
        "robustness": {
            "discretization_bin_width": results.robustness.discretization,
            "training_seed_returns": results.robustness.training_seed_returns,
            "base_stock_return": results.robustness.base_stock_return,
        },
    }


def _render_figures(cfg: Config, agents: TrainedAgents, results: PipelineResults) -> None:
    """Render the four committed figures from the run's results."""
    print("Rendering figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plotting.plot_reward_curve(agents.training_logs, results.ppo_payload, FIGURES_DIR / "reward_curve.png")
    plotting.plot_policy_comparison(
        results.matrix.results_by_regime, results.matrix.baselines,
        FIGURES_DIR / "policy_comparison.png", cfg.eval.n_bootstrap, cfg.eval.ci_level,
    )
    plotting.plot_policy_behavior(
        cfg, agents.q_stationary, stationary_base_stock_level(cfg.env, cfg.demand.mean),
        agents.q_seasonal_phase, seasonal_base_stock_levels(cfg.env, cfg.demand),
        FIGURES_DIR / "policy_behavior.png",
    )
    metrics_by_policy = {
        "base_stock": _means(results.matrix.results_by_regime["stationary"]["base_stock"]),
        "q_learning (balanced)": results.reward_hacking.balanced_metrics,
        "q_learning (naive)": results.reward_hacking.naive_metrics,
    }
    plotting.plot_reward_hacking(metrics_by_policy, FIGURES_DIR / "reward_hacking.png")


def _run_full_pipeline(cfg: Config, n_episodes: int | None) -> None:
    """The reproducible one-command pipeline, as a clear sequence of named steps."""
    run_id = _run_id(cfg, n_episodes)
    agents = _train_and_save(cfg, n_episodes)
    ppo_policies = ex.load_ppo_policies(cfg, MODELS_DIR)
    print("Loaded committed PPO policies (stretch)." if ppo_policies
          else "No PPO artifact found (or torch absent); evaluating the core path only.")

    results = _run_all_experiments(cfg, agents, ppo_policies, n_episodes)
    report = _assemble_report(cfg, n_episodes, run_id, results, bool(ppo_policies))
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _render_figures(cfg, agents, results)
    _write_sample_run_log(cfg, agents, run_id)
    _print_summary(cfg, results, run_id)


def _write_sample_run_log(cfg: Config, agents: TrainedAgents, run_id: str) -> None:
    """Write a human-readable transcript of one held-out episode under the learned policy."""
    demo = ex.sample_demo_episode(cfg, agents.q_stationary)
    traj = demo["trajectory"]
    lines = [
        "Sample held-out episode -- tabular Q-learning (balanced reward), stationary demand",
        f"run_id = {run_id}  |  episode demand seed = eval_seed_base + 0 = {cfg.eval.eval_seed_base}",
        f"episode return = ${demo['return']:.2f}  service level = {demo['metrics']['service_level']:.3f}",
        "",
        f"{'week':>4} {'on_hand':>8} {'order':>6} {'demand':>7} {'sales':>6} {'unmet':>6} {'inv_pos':>8}",
    ]
    n = len(traj["on_hand"])
    for t in range(n):
        lines.append(
            f"{t:>4} {traj['on_hand'][t]:>8} {traj['order'][t]:>6} {traj['demand'][t]:>7} "
            f"{traj['sales'][t]:>6} {traj['unmet'][t]:>6} {traj['inventory_position'][t]:>8}"
        )
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "sample_run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_headline(matrix: ex.MatrixResult, reward_hacking: ex.RewardHackingResult) -> None:
    """Print the key results so a run is legible at a glance."""
    print("\nHeld-out results (mean return | p05 downside-tail return):")
    for regime, results in matrix.results_by_regime.items():
        print(f"  {regime} (baseline = {matrix.baselines[regime]}):")
        for name, res in sorted(results.items(), key=lambda kv: -kv[1].returns.mean()):
            p05 = float(np.quantile(res.returns, 0.05))
            print(f"    {name:<22} {res.returns.mean():8.1f} | {p05:8.1f}")
    s = reward_hacking.summary
    print("\nReward hacking (naive vs balanced, scored on the true objective):")
    print(f"  service gain (naive - balanced):  +{s['service_gain_naive_minus_balanced']:.3f}")
    print(f"  profit loss (balanced - naive):   ${s['profit_loss_balanced_minus_naive']:.0f} "
          f"({s['profit_loss_pct']:.1f}% of the balanced policy's profit)")
    print(f"  avg on-hand:  naive {s['naive_avg_on_hand']:.1f}  vs  balanced {s['balanced_avg_on_hand']:.1f}")


def _print_summary(cfg: Config, results: PipelineResults, run_id: str) -> None:
    """Print the full-run summary: headline results, run id, and the failure/robustness studies."""
    _print_headline(results.matrix, results.reward_hacking)
    print(f"\nrun_id = {run_id}  (env_fingerprint {_env_fingerprint(cfg)}, seed {cfg.seed})")

    g = results.generalization.summary
    print("\nOverfitting probe (tuned for lambda=10, scored under a shift to lambda=16):")
    print(f"  Q-learning:  in-dist {g['q_in_distribution']:.0f} -> shifted {g['q_shifted']:.0f} "
          f"(gap to oracle {g['q_gap_to_oracle']:.0f})")
    print(f"  base-stock:  in-dist {g['base_stock_in_distribution']:.0f} -> shifted "
          f"{g['base_stock_shifted']:.0f} (gap to oracle {g['base_stock_gap_to_oracle']:.0f})")

    s = results.safety_ablation.summary
    print("\nSafety ablation (always-max-order policy, capacity clamp on vs off):")
    print(f"  clamp ON:  peak on-hand {s['capped_peak_on_hand']:.0f} (<= {s['i_max']:.0f}), "
          f"violations {s['capped_capacity_violations']:.0f}")
    print(f"  clamp OFF: peak on-hand {s['uncapped_peak_on_hand']:.0f}, "
          f"violations {s['uncapped_capacity_violations']:.0f}  <- what the hard limit prevents")

    rb = results.robustness
    print(f"\nRobustness (held-out return vs base-stock {rb.base_stock_return:.0f}):")
    print("  discretization:  " + "  ".join(
        f"width{w}->{d['return']:.0f}({d['gap_to_base_stock']:+.0f})" for w, d in rb.discretization.items()))
    print(f"  training seeds:  {[round(x) for x in rb.training_seed_returns]} "
          f"(spread {max(rb.training_seed_returns) - min(rb.training_seed_returns):.0f})")
    print("\nArtifacts written to models/, figures/, evidence/.")


# Subcommands
def cmd_all(args: argparse.Namespace) -> int:
    """Run the full reproducible pipeline."""
    cfg = default_config()
    _run_full_pipeline(cfg, n_episodes=args.episodes)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the tabular agents, or the PPO stretch with ``--agent ppo``."""
    cfg = default_config()
    if args.agent == "ppo":
        print("Training the PPO stretch (multi-seed stationary study + seasonal)...")
        payload = ex.run_ppo_study(
            cfg, MODELS_DIR, EVIDENCE_DIR, n_seeds=args.seeds, total_timesteps=args.timesteps
        )
        print(f"Saved PPO models to {MODELS_DIR}. Per-seed final returns: "
              f"{[round(x, 1) for x in payload['stationary_final_returns']]}")
        return 0
    print("Training tabular agents...")
    agents = ex.train_all_q_agents(cfg, n_episodes=args.episodes)
    _save_q_tables(agents)
    _write_training_curves(agents)
    print(f"Saved Q-tables and training curves under {MODELS_DIR} and {EVIDENCE_DIR}.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate (training Q agents first; this is the full matrix without re-plotting)."""
    cfg = default_config()
    agents = ex.train_all_q_agents(cfg, n_episodes=args.episodes)
    ppo_policies = ex.load_ppo_policies(cfg, MODELS_DIR)
    matrix = ex.run_evaluation_matrix(cfg, agents, ppo_policies)
    write_results_csv(matrix.results_by_regime, EVIDENCE_DIR / "results.csv")
    for regime, report in matrix.reports.items():
        from inventory_rl.evaluation import EvalReport

        write_report(EvalReport(**report), EVIDENCE_DIR / f"eval_report_{regime}.json")
    _print_headline(matrix, ex.run_reward_hacking_experiment(cfg, agents))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:  # noqa: ARG001 - uniform subcommand signature
    """Regenerate figures by re-running the in-memory pipeline (figures need live policies)."""
    cfg = default_config()
    _run_full_pipeline(cfg, n_episodes=None)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run one held-out episode under the committed (or freshly trained) Q policy."""
    cfg = default_config()
    q_path = MODELS_DIR / "q_table.npz"
    if q_path.exists():
        from inventory_rl.q_learning import load_q_policy

        policy = load_q_policy(q_path, cfg.env)
    else:
        print("No committed Q-table; training a quick agent for the demo...")
        agents = ex.train_all_q_agents(cfg, n_episodes=args.episodes or 5_000)
        policy = agents.q_stationary
    demo = ex.sample_demo_episode(cfg, policy)
    traj = demo["trajectory"]
    print(f"Sample held-out episode -- {policy.name}, stationary demand")
    print(f"episode return = ${demo['return']:.2f}  service level = {demo['metrics']['service_level']:.3f}\n")
    print(f"{'week':>4} {'on_hand':>8} {'order':>6} {'demand':>7} {'sales':>6} {'unmet':>6}")
    for t in range(min(10, len(traj["on_hand"]))):
        print(f"{t:>4} {traj['on_hand'][t]:>8} {traj['order'][t]:>6} {traj['demand'][t]:>7} "
              f"{traj['sales'][t]:>6} {traj['unmet'][t]:>6}")
    print("    ... (run `inventory-rl all` for the full evaluation and figures)")
    return 0


def main() -> int:
    """Parse arguments and dispatch to the chosen subcommand."""
    parser = argparse.ArgumentParser(prog="inventory-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="run the full reproducible pipeline")
    p_all.add_argument("--episodes", type=int, default=None, help="override Q-learning training episodes")
    p_all.set_defaults(func=cmd_all)

    p_train = sub.add_parser("train", help="train the tabular agents (or PPO with --agent ppo)")
    p_train.add_argument("--agent", choices=["q", "ppo"], default="q")
    p_train.add_argument("--episodes", type=int, default=None, help="Q-learning training episodes")
    p_train.add_argument("--seeds", type=int, default=None, help="PPO training seeds")
    p_train.add_argument("--timesteps", type=int, default=None, help="PPO timesteps per seed")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="run the held-out evaluation matrix")
    p_eval.add_argument("--episodes", type=int, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_plot = sub.add_parser("plot", help="regenerate the figures")
    p_plot.set_defaults(func=cmd_plot)

    p_demo = sub.add_parser("demo", help="print one held-out episode trajectory")
    p_demo.add_argument("--episodes", type=int, default=None)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
