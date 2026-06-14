"""Tests for the failure-analysis experiments that need no training (fast)."""

from __future__ import annotations

from inventory_rl.config import default_config
from inventory_rl.experiments import run_robustness_study, run_safety_ablation


def test_safety_ablation_shows_the_clamp_bounds_the_blast_radius() -> None:
    """With the capacity clamp on, a worst-case policy never breaches it; with it off, it does."""
    cfg = default_config()
    result = run_safety_ablation(cfg)
    s = result.summary
    # Enforced: the guarantee holds no matter how pathological the policy.
    assert s["capped_capacity_violations"] == 0
    assert s["capped_peak_on_hand"] <= s["i_max"]
    # Unenforced: the same policy floods the warehouse well past the limit.
    assert s["uncapped_peak_on_hand"] > s["i_max"]
    assert s["uncapped_capacity_violations"] > 0


def test_robustness_study_reports_discretization_and_seed_spread() -> None:
    """The robustness study returns a gap-to-base-stock per bin width and a per-seed return list."""
    cfg = default_config()
    # Tiny budget: this test checks structure/wiring, not convergence.
    result = run_robustness_study(cfg, n_episodes=200, bin_widths=(5, 10), extra_seeds=(1,))
    assert set(result.discretization) == {"5", "10"}
    assert result.discretization["5"]["n_states"] == 11  # i_max 50 / width 5 + 1
    assert result.discretization["10"]["n_states"] == 6
    for entry in result.discretization.values():
        assert "return" in entry and "gap_to_base_stock" in entry
    assert len(result.training_seed_returns) == 2  # default seed + one extra
    assert result.base_stock_return > 0
