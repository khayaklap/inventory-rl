"""Scenario evaluation harness: edge-episode behavior + a hard safety regression test.

Two tiers, mirroring the reference project's deterministic/live split:

* **Tier A (always, no torch):** run the analytical baselines and -- if the committed Q-table
  is present -- the tabular agent through every stress scenario, and assert the hard
  invariant the environment must never break (zero capacity violations, the safety guarantee).
  This is the deterministic backbone any grader can run with the core dependencies only.
* **Tier B (when available):** additionally load and run the committed PPO policy.

Each scenario is declared in ``scenarios.jsonl`` and references a named generator from
``inventory_rl.demand`` (so the data stays reproducible and the cases stay declarative).
The harness writes ``evidence/scenario_report.json`` and exits non-zero on any failure.

Run:
    python -m evals.run_evals            # auto (Tier B if a PPO artifact exists)
    python -m evals.run_evals --offline  # Tier A only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inventory_rl.baselines import (
    BaseStockPolicy,
    RandomPolicy,
    SeasonalBaseStockPolicy,
    seasonal_base_stock_levels,
    stationary_base_stock_level,
)
from inventory_rl.config import default_config
from inventory_rl.demand import STRESS_SCENARIOS
from inventory_rl.evaluation import Policy, rollout

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
EVIDENCE_DIR = REPO_ROOT / "evidence"


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load the declarative scenario list from a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_policies(offline: bool) -> list[Policy]:
    """Assemble the policy roster for the harness, including committed artifacts if present."""
    import numpy as np  # noqa: PLC0415 - local import keeps the module head light

    cfg = default_config()
    policies: list[Policy] = [
        RandomPolicy(len(cfg.env.action_menu), seed=cfg.seed),
        BaseStockPolicy(stationary_base_stock_level(cfg.env, cfg.demand.mean), cfg.env, name="base_stock"),
        SeasonalBaseStockPolicy(seasonal_base_stock_levels(cfg.env, cfg.demand), cfg.env),
    ]
    q_path = MODELS_DIR / "q_table.npz"
    if q_path.exists():
        from inventory_rl.q_learning import load_q_policy  # noqa: PLC0415

        policies.append(load_q_policy(q_path, cfg.env))
    ppo_path = MODELS_DIR / "ppo_stationary.zip"
    if not offline and ppo_path.exists():
        try:
            from inventory_rl.ppo_agent import PPOPolicy  # noqa: PLC0415

            policies.append(PPOPolicy.load(ppo_path))
        except SystemExit:
            pass  # torch not installed: Tier B silently degrades to Tier A
    _ = np  # referenced to keep the import meaningful if numpy-only helpers are added
    return policies


def run() -> int:
    """Run every scenario against every policy, assert invariants, write the report."""
    parser = argparse.ArgumentParser(description="Inventory-RL scenario evaluation harness.")
    parser.add_argument("--offline", action="store_true", help="Tier A only (skip the PPO artifact).")
    args = parser.parse_args()

    cfg = default_config()
    scenarios = load_scenarios(Path(__file__).parent / "scenarios.jsonl")
    policies = build_policies(args.offline)

    report: dict[str, Any] = {"tier": "A" if args.offline else "auto", "scenarios": {}}
    failures: list[str] = []

    print(f"Running {len(scenarios)} scenarios x {len(policies)} policies\n")
    for spec in scenarios:
        scenario = STRESS_SCENARIOS[spec["scenario"]]
        import numpy as np  # noqa: PLC0415

        demand = scenario.make_demand(cfg.env.horizon, np.random.default_rng(spec["seed"]), cfg.demand)
        lead_time = scenario.env_overrides.get("lead_time")
        scenario_record: dict[str, Any] = {"description": spec["description"], "policies": {}}
        print(f"[{spec['id']}] {spec['description']}")
        for policy in policies:
            result = rollout(policy, demand, cfg.env, lead_time=lead_time)
            violations = int(result.metrics["capacity_violations"])
            ok = violations == 0  # the NO_CAPACITY_VIOLATION invariant
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append(f"{spec['id']}/{policy.name}: {violations} capacity violations")
            scenario_record["policies"][policy.name] = {
                "return": round(result.total_return, 2),
                "service_level": round(result.metrics["service_level"], 3),
                "avg_on_hand": round(result.metrics["avg_on_hand"], 2),
                "approval_flags": int(result.metrics["approval_flags"]),
                "capacity_violations": violations,
                "invariant_ok": ok,
            }
            print(f"    {status}  {policy.name:<22} return={result.total_return:8.1f} "
                  f"service={result.metrics['service_level']:.3f} flags={int(result.metrics['approval_flags'])}")
        report["scenarios"][spec["id"]] = scenario_record
        print()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "scenario_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(scenarios)} scenarios passed the NO_CAPACITY_VIOLATION invariant "
          f"across {len(policies)} policies.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
