"""Central configuration: the global seed and every tunable parameter as frozen dataclasses.

Keeping all parameters in one typed, immutable place is the project's "the small world is
data" decision (mirroring the reference course project's fixtures module): the simulator,
the baselines, the learners, and the evaluation all read from these objects, so a single
edit changes the whole system consistently and every run records exactly which numbers
produced it.

Parameter choices (see ``docs/DESIGN.md`` for the full justification):

* Economics ``p=10, c=4`` give a unit margin of 6; holding ``h=1`` and stockout penalty
  ``b=5`` make the *underage* cost ``Cu = (p - c) + b = 11`` dominate the *overage* cost
  ``Co = h = 1``. That asymmetry (critical ratio 11/12) is what makes a naive
  "avoid stockouts" reward tempting -- and gameable.
* Lead time ``L=1`` keeps the pipeline one-dimensional and the protection interval ``L+1=2``.
* The inventory cap ``i_max=50`` is the binding hard constraint that bounds the
  reward-hacking blast radius; ``q_max=30`` caps a single order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Global seed. Date-based and explicit so every stochastic step is reproducible and the
# provenance of a run is obvious. Per-stream seeds (training vs held-out evaluation) are
# derived from dedicated base offsets in ``EvalParams`` so they can never collide.
SEED: int = 20260614

# The discrete order menu the agent (and the menu-snapped baselines) may choose from.
ACTION_MENU: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)


@dataclass(frozen=True)
class EnvParams:
    """Economics, physics, and hard constraints of the inventory system.

    All costs are per unit. ``salvage`` is applied once, to leftover on-hand inventory at
    the end of the finite horizon, so the agent is not punished for stock it could not
    possibly have sold in time.
    """

    price: float = 10.0  # p: revenue per unit sold
    unit_cost: float = 4.0  # c: cost of goods per unit sold (margin = price - unit_cost)
    holding_cost: float = 1.0  # h: per unit of END-of-period on-hand (the overage cost Co)
    stockout_penalty: float = 5.0  # b: per unit of unmet demand (goodwill loss, on top of lost margin)
    order_cost: float = 0.0  # K: fixed cost charged once per non-zero order (0 => base-stock optimal)
    salvage: float = 4.0  # s_v: value recovered per leftover unit at the horizon end

    lead_time: int = 1  # L: periods between placing and receiving an order
    q_max: int = 30  # largest single order (safety cap)
    i_max: int = 50  # warehouse capacity: on-hand may never exceed this (hard constraint)
    horizon: int = 52  # H: periods per episode (one "year")

    init_on_hand: int = 10  # starting inventory (~one period of mean demand)
    approval_threshold: int = 25  # orders strictly above this are flagged for human approval (HITL analog)
    enforce_capacity: bool = True  # hard warehouse-capacity clamp; set False ONLY for the safety ablation

    action_menu: tuple[int, ...] = ACTION_MENU

    @property
    def margin(self) -> float:
        """Per-unit gross margin, ``price - unit_cost``."""
        return self.price - self.unit_cost

    @property
    def underage_cost(self) -> float:
        """Newsvendor underage cost ``Cu``: lost margin PLUS the explicit stockout penalty.

        Under lost sales a unit short forfeits the margin it would have earned *and* incurs
        the goodwill penalty -- a frequent source of error is to count only the penalty.
        """
        return self.margin + self.stockout_penalty

    @property
    def overage_cost(self) -> float:
        """Newsvendor overage cost ``Co``: the per-period holding cost."""
        return self.holding_cost

    @property
    def critical_ratio(self) -> float:
        """Newsvendor critical ratio ``Cu / (Cu + Co)`` -- the target service quantile."""
        return self.underage_cost / (self.underage_cost + self.overage_cost)

    @property
    def protection_interval(self) -> int:
        """Periods a single order must cover under periodic review: ``L + 1``."""
        return self.lead_time + 1


@dataclass(frozen=True)
class DemandParams:
    """Parameters of the stochastic demand world.

    Stationary demand is ``Poisson(mean)``. Seasonal demand modulates the rate as
    ``mean + amplitude * sin(2*pi*t / period)`` (floored at ``min_rate`` so it stays a
    valid Poisson rate). ``shift_mean`` is the permanently elevated rate used by the
    distribution-shift stress test.
    """

    mean: float = 10.0  # lambda for stationary demand
    amplitude: float = 6.0  # seasonal swing around the mean
    period: int = 52  # seasonal period in weeks
    min_rate: float = 0.1  # floor on the Poisson rate
    shift_mean: float = 16.0  # elevated stationary rate for the distribution-shift scenario


@dataclass(frozen=True)
class QLearnParams:
    """Tabular Q-learning hyperparameters and state discretization.

    ``alpha`` uses a Robbins-Monro schedule ``1 / (1 + visits(s, a))`` when ``robbins_monro``
    is true (more robust than a fixed rate); otherwise the fixed ``alpha`` is used.
    ``ip_bin_width`` discretizes inventory position into bins; ``use_phase`` adds a
    coarse season index so the agent can learn time-varying targets under seasonal demand.
    """

    gamma: float = 0.99
    alpha: float = 0.1
    robbins_monro: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    n_episodes: int = 30_000
    ip_bin_width: int = 5  # inventory-position bin width (matches the order-menu step)
    use_phase: bool = False  # include a season-phase feature in the state (seasonal runs)
    n_phases: int = 13  # number of season buckets when use_phase is true (52 weeks / 4)


@dataclass(frozen=True)
class PPOParams:
    """Stable-Baselines3 PPO hyperparameters for the stretch agent."""

    total_timesteps: int = 200_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: tuple[int, ...] = (64, 64)
    n_seeds: int = 5  # independent training seeds, to quantify seed variance


@dataclass(frozen=True)
class EvalParams:
    """Evaluation protocol: seed streams, episode counts, and bootstrap settings.

    The three seed streams are kept disjoint by construction so demand realizations used in
    training can never leak into the held-out evaluation set. "Out-of-sample" refers ONLY
    to ``eval_seed_base`` episodes; ``val_seed_base`` is an in-sample validation stream.
    """

    n_eval_episodes: int = 100
    # Seed streams are separated by large offsets so the training stream (which consumes
    # `train_seed_base .. train_seed_base + n_episodes`) can never overlap the validation or
    # held-out evaluation streams. Disjointness is asserted in `evaluation.seed_sets_disjoint`.
    train_seed_base: int = 1_000_000  # Stream B: training-demand seeds
    val_seed_base: int = 2_000_000  # Stream B': in-sample validation seeds (tuning)
    eval_seed_base: int = 3_000_000  # Stream C: held-out (out-of-sample) evaluation seeds
    n_bootstrap: int = 10_000
    ci_level: float = 0.95


@dataclass(frozen=True)
class Config:
    """Top-level bundle wiring every sub-config together, plus the global seed."""

    seed: int = SEED
    env: EnvParams = field(default_factory=EnvParams)
    demand: DemandParams = field(default_factory=DemandParams)
    qlearn: QLearnParams = field(default_factory=QLearnParams)
    ppo: PPOParams = field(default_factory=PPOParams)
    eval: EvalParams = field(default_factory=EvalParams)


def default_config() -> Config:
    """Return the default configuration used by the CLI and the committed artifacts."""
    return Config()
