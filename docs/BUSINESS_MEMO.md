# Business memo — should we deploy the inventory-replenishment agent?

**To:** VP, Supply Chain Operations · **From:** Decision Science · **Re:** Go/No-go for the
RL replenishment agent (single-SKU pilot) · **Recommendation: SHADOW, do not yet automate.**

## Executive summary

We built and evaluated a reinforcement-learning agent that decides weekly reorder quantities for a
single SKU. On 100 held-out demand episodes it performs **on par with the company's best
analytical policy** (a newsvendor base-stock rule) and **adapts to seasonality that a static rule
misses**. It never breaches the warehouse capacity limit. However, the same study shows that a
small change to the agent's objective — one that looks reasonable on a dashboard — **destroys 38%
of profit** while *improving* the service-level KPI. That asymmetry is the whole argument for
caution. We recommend running the agent in **shadow mode** against the incumbent policy, not
handing it the buy.

## What the evidence says

All figures are mean per-episode profit over a 52-week horizon, with 95% confidence intervals from
a paired bootstrap (each policy faced identical demand); full numbers in `evidence/eval_report.json`.

- **It matches the incumbent.** Under stable demand the agent earns within **3.3%** of the
  near-optimal base-stock policy (gap −$88/episode, CI [−98, −79]). It does **not** beat it — and
  for a problem this well-understood, that is the honest and expected result. RL is not adding
  value over good operations research here.
- **It earns its keep only under seasonality.** When demand swings seasonally, a *static*
  base-stock loses ≈$580/episode. The season-aware agent recovers nearly all of that, **tying a
  hand-built seasonal rule** (gap −$79/episode) — but **without anyone having to model the
  season**. That is the real business case: the agent adapts when the demand pattern is unknown,
  mis-modeled, or drifting. Where we *can* write the seasonal rule down, we do not need the agent.
- **The downside, not just the average, is acceptable.** Risk-averse operations care about bad
  weeks, not the mean. Under stable demand the agent's 5th-percentile (worst-1-in-20) held-out
  return is **$2,338 vs base-stock's $2,425** — the tail tracks the mean gap, so the agent introduces
  **no extra downside risk**, and its worst-case is far above random's. (Reported as `p05`/`cvar05` for
  every policy in `eval_report.json`.)
- **The safety limit held everywhere.** Across the main evaluation and seven stress scenarios
  (demand spikes, droughts, a tripled lead time, a permanent demand shift), **zero** orders ever
  exceeded warehouse capacity. The constraint is enforced in code, not learned.

## The risk that decides the recommendation

We trained the *same* agent on a tempting but wrong objective — "minimize stockouts" — the kind of
single-KPI target a team might actually set. The result is the textbook failure mode:

| | Service level | Profit / episode | Avg inventory |
|---|---|---|---|
| Correct objective | 0.989 | **$2,586** | 9 units |
| "Minimize stockouts" proxy | **0.997** | $1,591 (**−38%**) | 29 units |

The proxy agent looks **better** on the service dashboard while quietly flooding the warehouse and
erasing a third of the margin. An automated agent optimizing a mis-specified reward would do this
at scale, fast, and the service KPI would not warn us. The capacity cap bounded the damage — but
capacity is the only thing that did.

## Recommendation: staged rollout, starting in shadow

Following a guarded-rollout ladder — the more consequential the action, the slower the rollout:

1. **Shadow (now, 4–6 weeks).** The agent recommends orders; buyers execute the incumbent policy.
   We log both and compare realized profit and service. **Gate to advance:** the agent's shadow
   profit is within the CI of the incumbent (or better) and zero capacity flags.
2. **Human-in-the-loop (next).** The agent's order becomes the default; a buyer approves any order
   above the review threshold (25 units) before it is placed. **Gate:** buyers override the agent's
   recommendation in fewer than ≈10% of cases (a high override rate means it is not yet trusted).
3. **Limited autonomy.** Auto-execute on a small, low-risk subset of SKUs, with a daily
   profit-and-inventory dashboard and an **automatic rollback** if weekly profit drops below the
   incumbent's trailing average or any capacity flag fires.
4. **Broad autonomy** only after a full season of evidence.

**Rollback is part of the design, not a contingency:** the incumbent base-stock policy is a
one-line formula that can be reinstated instantly, and the agent writes a full decision trace per
order.

## Why not just deploy, or just reject?

- **Not full deploy:** the agent only ties the incumbent under stable demand, and the
  reward-hacking result shows the cost of any objective mis-specification is severe and
  KPI-invisible. The upside (seasonal adaptivity) does not yet justify the downside risk
  unattended.
- **Not reject:** the seasonal result is real and valuable, the safety limits hold, and shadow mode
  is nearly free — we capture the learning with no operational exposure.

## Open questions for the business

1. Is our service-level target actually the objective, or a proxy for profit? (If we reward
   service, we will get the 29-unit warehouse.)
2. What is the true cost of a stockout (goodwill, substitution, churn)? It sets `Cu` and therefore
   the order-up-to level.
3. Which SKUs have demand patterns we *cannot* model well? Those — not the easy, stable ones — are
   where this agent pays for itself.
