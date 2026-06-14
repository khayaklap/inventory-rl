# Reflection & failure analysis

Honest notes on what broke, what improved, and what is still risky. The rubric rewards real
failure analysis, so this is written as a build log, not a victory lap. Several of these are
*design* bugs caught before they shipped — which is the point of building the evaluation before
trusting the agent.

## What I built (one paragraph)

A periodic-review, lost-sales inventory simulator (`env.py`) with hard capacity clamps; a
stationary/seasonal Poisson demand world (`demand.py`); a balanced reward and a deliberately
gameable naive reward (`rewards.py`); analytical baselines (random, newsvendor base-stock,
time-indexed seasonal base-stock); a tabular Q-learning agent and a PPO stretch; and a paired,
held-out evaluation with bootstrap CIs. 51 unit tests and the scenario harness run with no GPU and
no torch.

## What failed (and what I changed)

### 1. The newsvendor underage cost was nearly wrong (`Cu = b`, not `Cu = (p−c) + b`)

My first instinct was to set the underage cost to the stockout penalty `b = 5`. That is the most
common newsvendor error: under lost sales, a unit short forfeits the **margin** it would have
earned (`p − c = 6`) *and* the penalty. The correct `Cu = 11` gives a critical ratio of `11/12`
and an order-up-to level `S = 26`; the wrong `Cu = 5` gives `≈ 0.83` and `S = 24`, which
under-stocks relative to the correct level and would have biased the base-stock baseline downward —
making the agent's tie look like a win it had not earned. **Fix:** derived `Cu` from first
principles in `config.EnvParams.underage_cost`, with a unit test
(`test_underage_cost_includes_lost_margin`) that pins the formula. Lesson: get the baseline *right*
before claiming RL beats it.

### 2. An evaluation-seed collision would have leaked training demand into the test set

My first seed scheme used `train_seed_base = 1000` (training consumes `1000 .. 31000` across 30k
episodes) and `eval_seed_base = 9000` — squarely **inside** the training range. The held-out set
would have re-used demand the agent trained on. **Fix:** separated the three streams by 10⁶-sized
offsets (`config.EvalParams`) and added `test_training_and_eval_seeds_are_disjoint`, which asserts
disjointness and flags the original overlapping pair as a failure. Lesson: "out-of-sample" is a
claim you have to *prove*, not assume.

### 3. The "RL wins under seasonality" story was dishonest until I added a second baseline

My initial plan compared a season-aware agent against a *static* base-stock and declared victory.
A skeptic is right to object: a competent planner would just index the order-up-to level by week.
That comparison was rigged. **Fix:** added `SeasonalBaseStockPolicy` (`S_t` from the
protection-interval demand each week) as the fair competitor. The honest result is that the
phase-aware agent **ties** it (gap −$79/episode, CI [−91, −66]) while beating the static one by
~$500/episode. The defensible value of RL is reaching the seasonal solution *without being handed
the model* — not magic outperformance. Lesson: the most important baseline is the one that makes
your method look *least* impressive.

### 4. The base-stock-structure test failed at full capacity

`test_agent_learns_base_stock_like_structure` first probed inventory position 50 (the cap) and
asserted the greedy order was 0. It was not — but it did not matter, because at the cap the
environment clamps *every* order to 0, so all actions are equivalent and the learned Q-values there
are pure noise. The assertion was checking a meaningless state. **Fix:** probe inventory position
35 (above the base-stock level, below the cap), where ordering nothing is genuinely optimal and
learnable. Lesson: a green test on a degenerate state proves nothing.

### 5. State had to be inventory position, not on-hand

An early discretization on on-hand alone hid in-transit stock, so the agent kept re-ordering
against orders already on the way and lost to base-stock — not because RL is weak, but because the
state was not a sufficient statistic. **Fix:** discretize on inventory position (`on_hand +
pipeline`), the same quantity base-stock acts on. The learned order curve then traces the
base-stock staircase. Lesson: a representation bug masquerades as an algorithm failure.

### 6. PPO is the cautionary tale, not the hero

Across three training seeds PPO's final returns were 2303 / 2299 / 2369 — a visible spread, and all
*below* the tabular agent (2586) and base-stock (2674); on the harder seasonal task it landed far
lower (1929). A single PPO seed would have been untrustworthy either way. **Fix:** train multiple
seeds (`train_ppo_multiseed`), plot the spread, save the best as the artifact, and let PPO carry
*none* of the headline. Lesson: on a small problem, deep RL buys variance, not value — the baseline
ladder, lived.

### 7. The reward accounting had to be made coherent before salvage made sense

I first wrote margin accounting (`margin·sales − holding − …`), then tried to add a terminal
salvage — but in margin accounting leftover stock was never *paid for*, so salvaging it gave the
agent free money and an incentive to end the horizon full. **Fix:** switched to cost-at-order
accounting (pay `c·order` at placement, recover `salvage·on_hand` at the end). Salvage now recovers
cash actually spent, and the end-of-horizon sell-down artifact disappears. The per-period ordering
trade-off (`Cu`, `Co`) is unchanged, so `S = 26` still holds. Lesson: accounting choices have
behavioral consequences; write them down.

### 8. I over-predicted the reward-hacking damage

My planning note guessed the naive agent's true-objective return would go "near zero or negative."
The actual result is −38.5% (return 1591 vs 2586) — a severe, clear failure, but **positive**. The
reason: under cost-at-order, purchase cost is paid for whatever is eventually sold regardless of
policy, so the differentiating damage is *holding* (1513 vs ~470), not a total wipeout. **Fix:**
report the measured number, not the dramatic guess. Lesson: let the experiment correct the
narrative.

### 9. Tooling papercuts

- `scipy.stats.wilcoxon` returns **NaN** (not a `ValueError`) when all paired differences are zero
  (identical policies); handled explicitly so the report never carries a NaN p-value.
- A seasonal-periodicity test compared `sin(2π) == 0` exactly and failed on floating point; fixed
  with `math.isclose`.
- Python 3.14 was the only system interpreter; pinned **3.12** via the venv to avoid wheel risk
  (and installing Stable-Baselines3 pinned `gymnasium` to 1.2.x — recorded in the frozen lock).

### 10. Two of the four failure modes were asserted, not measured — so I measured them

A self-audit against the assignment's failure-analysis list (reward hacking, unsafe behavior,
instability, overfitting) found that only two were backed by evidence: reward hacking (the
experiment) and instability (the PPO seed spread). "Unsafe behavior" rested on *"the cap holds"*
and "overfitting" on a prose claim — neither was a number. An unmeasured claim is a defensibility
hole, not rigor. **Fix:** added two committed experiments. (a) A **safety ablation**
(`run_safety_ablation`): the same always-max-order policy floods the warehouse to a peak of
**1,023 units with the clamp off (49 violations)** but is bounded to **39 units, zero violations**
with it on — the "safety is enforced by code, not the policy" claim, now empirical. (b) A
**generalization probe** (`run_generalization_experiment`): score the λ=10-tuned Q-agent and
base-stock under a shift to λ=16, against an oracle base-stock recomputed for λ=16. The result
**corrected my prior**: I had assumed the analytical formula would generalize better, but the
*learned* policy tracks the oracle closer (gap **685** vs base-stock's **1,549**) because the
fixed `S=26` rule rigidly under-stocks once demand rises, while the Q-policy orders more
aggressively when depleted. Lesson (again): measure the failure mode, do not narrate it — and
report the result even when it contradicts your guess. (Caveat: this is one shift direction;
either policy should be re-fit when the regime changes.)

### 11. The discretization claim was also unmeasured — and the data sharpened it

I had written that "coarser bins erase the base-stock structure." True in spirit, but a robustness
study (`run_robustness_study`) showed the claim needed a *range*: bin widths **2, 5, 10 all tie
base-stock** (gaps −83 / −88 / −87), and only at **width 25 (3 states)** does the policy
**collapse** (return −2451) because three bins cannot express "order hard below S, stop above it".
So the honest statement is "robust across a wide range, with a clear collapse boundary," not
"sensitive." The same study retrains across three seeds (held-out returns 2586 / 2612 / 2621,
spread **$35**), so the headline tie is a property of the method, not a lucky seed. I also checked
the **finite-horizon tail**: because the stationary agent's state omits the week, it *cannot* taper
its last order to dodge the cost-at-order charge — and indeed its last-period order ($41.6 of waste)
sits within $2.40 of base-stock's ($39.2), so the tie is not a horizon artifact. Lesson: a reviewer
will poke the hyperparameters and the edge effects; measure them before they do.

## What improved

- **Reward ⟂ constraints.** Putting the objective in the reward and the safety limits in the
  environment is the single best decision: the reward-hacking agent demonstrates a catastrophic
  *economic* failure while provably never breaching a *safety* limit (zero capacity violations).
- **An analytical anchor.** Because base-stock is computable in closed form, "did the agent learn?"
  has a rigorous answer (within 3% of optimal), not a vibe.
- **Paired, held-out evaluation.** Same demand for every policy on unseen seeds turns noisy episode
  returns into tight, defensible CIs.

## What is still risky

- **Sim-to-real gap.** Poisson demand and a fixed lead time are gentle; real demand is correlated,
  censored, and promotion-driven. The agent's seasonal adaptivity could overfit a pattern that does
  not recur.
- **Reward is a model of the objective.** The balanced reward is *a* proxy too — better than the
  naive one, but the whole project is a warning that a wrong reward ships a confident, wrong agent.
- **Lost-sales base-stock is a heuristic.** Our anchor is near-optimal, not provably optimal, so the
  small stationary gap is reported honestly rather than explained away.

## Live results (final, from `evidence/eval_report.json`)

- **Unit tests:** 51 passed (no GPU, no torch). Full `inventory-rl all` pipeline: **~135 seconds**.
- **Scenario harness:** 7 / 7 scenarios pass the zero-capacity-violation invariant across 5 policies.
- **Stationary:** base-stock 2674 · Q-learning 2586 (−3.3%) · PPO 2539 · random 1793.
- **Seasonal:** seasonal base-stock 2667 · phase-Q 2588 (−3.0%) · IP-only Q 2253 · static base-stock
  2087 · PPO 1929 · random 1427.
- **Reward hacking:** naive service 0.997 vs balanced 0.989, but naive return 1591 vs 2586 (−38.5%),
  on-hand 29 vs 9, holding 1513 vs ~470, capacity violations 0.
- **Overfitting probe (λ=10 → λ=16):** gap-to-oracle 685 (Q-learning) vs 1,549 (static base-stock).
- **Safety ablation (always-max policy):** peak on-hand 39 / 0 violations (clamp on) vs 1,023 / 49
  (clamp off).
- **Robustness:** discretization gap-to-base-stock −83 / −88 / −87 at bin widths 2 / 5 / 10, then a
  collapse to −5,125 at width 25; training-seed held-out returns 2586 / 2612 / 2621 (spread $35).

The honest takeaway: the *agent* never beats the analytical baseline — it **ties** it, cheaply and
from interaction alone, and adapts to seasonality a static rule cannot. The value of the build is
the **harness around it** — the right baseline, the held-out evaluation, the hard safety clamps, and
the reward-hacking experiment — that turns "an RL agent ordered some stock" into an inspectable,
defensible decision.
