# Design notes & rubric mapping

This document maps each design decision to (a) the grading rubric and (b) the specific
course-deck idea it applies. It is the "defend any decision in a one-minute review" cheat sheet.
Page references are to the course's reinforcement-learning deck (Module 3).

## The MDP & safety contract

| | |
|---|---|
| **State** `S` | inventory position (on-hand + pipeline); + a season phase for the seasonal agent |
| **Action** `A` | order quantity ∈ `{0, 5, 10, 15, 20, 25, 30}` |
| **Transition** `P` | receive order from `L` periods ago → place clamped order → `Poisson` demand → lost sales |
| **Reward** `R` | `price·sales − cost·order − holding·on_hand − penalty·unmet − fixed·1[order] (+salvage)` |
| **Discount** `γ` | 0.99 (finite 52-week horizon; a mild time preference, deck p.28–29) |
| **Horizon** | 52 weeks |

**Learned / coded / forbidden** (the deck's closing question, p.189):

| Learned | Coded | Forbidden |
|---|---|---|
| the ordering policy (how much, when to stop, how to ride a swing) | cost accounting, the newsvendor baseline, the held-out evaluation, the approval flag | order > capacity (`on_hand ≤ 50`), negative orders — clamped in `env.step`, never via reward |

The forbidden rules are **enforced by the environment before an order is placed** (deck p.153,
"rules the policy cannot cross … not allowed to discover them by running into the wall"). This is
why the reward-hacking agent over-orders yet records zero capacity violations — and the safety
ablation (`run_safety_ablation`) quantifies the guarantee: a worst-case always-max policy peaks at
**1,023 units with the clamp disabled** but is bounded to **39, zero violations** with it on.

## Architecture in one picture

```text
config (frozen params, SEED)
   │
   ▼
demand ──► env (simulator + hard clamps) ──► reward (balanced | naive)
                   │                               │
   baselines ──────┤                               │ trains
   q_learning ─────┤  policies (obs → order)       ▼
   ppo_agent ──────┘            │            q_learning / ppo agents
                                ▼
                   evaluation (paired, held-out, bootstrap CI, Wilcoxon)
                                │
                 experiments (matrix + reward-hacking)  ──►  plotting + reports
                                │
                          cli  /  evals.run_evals  ──►  models/ figures/ evidence/
```

The deliberate split: the **reward** carries the objective and the **environment** carries the
safety limits; the agents only choose actions. Everything safety-relevant is computed and
enforced in code, never trusted to the learned policy.

## Rubric mapping (25 pts)

| Category (pts) | Where it lives | Course-deck basis |
|---|---|---|
| **MDP & reward design (5)** | `env.py` (S, A, P, transition, horizon, clamps); `rewards.py` (`reward_balanced` vs `reward_naive`); `config.py` (the newsvendor `Cu = (p−c)+b`); README §3 | MDP design checklist (p.32); inventory MDP table — stock/forecast/lead-time/capacity → order → sales−holding−stockout (p.158); reward-design questions "what proxy could be gamed? what to constrain not optimize?" (p.146); discounting as time preference (p.28–29) |
| **Implementation & reproducibility (5)** | `pyproject.toml` (`[project.scripts]`, pytest/ruff/mypy), pinned `requirements-core/stretch.txt`, `.python-version`, committed `models/`+`figures/`, `cli all`, the no-GPU quality gate | practical RL stack — "the algorithm is one layer; the rest is where projects live or die" (p.143); "install once and forget" (p.188); observability — log seed / policy version / env version (p.152) |
| **Evaluation against baseline (5)** | `evaluation.py` (paired held-out eval, bootstrap CI, Wilcoxon); `experiments.py` (the policy×regime matrix + `run_robustness_study` for discretization/seed sensitivity); `evals/` (7 edge episodes, Tier A/B); `figures/`; `evidence/eval_report.json` | offline evaluation — baseline comparison / stress scenarios / constraint violations / seed sensitivity (p.150); baseline ladder, "climb only as far as the problem demands" (p.163); capstone "evidence: evaluation plan can catch failures" (p.182) |
| **Correct use of RL/DRL concepts (4)** | `q_learning.py` (`bellman_update`, ε-greedy decay, discretization, inventory-position state); `ppo_agent.py` (on-policy actor-critic via SB3) | Q-learning update + code (p.48–49); ε-greedy schedule (p.54); on/off-policy and model-free tables (p.57–58); tabular → function approximation (p.63–64); PPO "improves the policy and refuses to move too far" (p.102–105); algorithm cheat-sheet, discrete actions ⇒ Q-learning/DQN/PPO (p.60) |
| **Safety / reward-hacking / governance (3)** | `rewards.py` (the gameable proxy, demonstrated); `env.py` (hard clamps + approval flag); `experiments.run_reward_hacking_experiment`; `docs/REFLECTION.md`; this contract | reward hacking — "inventory over-orders to avoid stockouts … the agent doing exactly what you told it" (p.147); `reward_bad` vs `reward_better` (p.170); safety constraints enforced by code (p.153); learned/coded/forbidden (p.189) |
| **Business communication (3)** | `docs/BUSINESS_MEMO.md` (deploy vs shadow vs reject, tied to the eval CIs + tail risk); README §10–11 | guarded rollout — shadow → human review → limited → full, with rollback thresholds (p.151); business deployment risks — local optimization, feedback delay, confounding (p.162); "communication: assumptions and risks are named" (p.182) |

## The capstone's eight artifacts (p.181)

The deck's capstone says "if any of the eight is TBD, the project is not ready." Here is each one:

| # | Artifact | Where it lives |
|---|---|---|
| 1 | **MDP definition** | `env.py` + the contract table above + README §3 |
| 2 | **Baseline policy** | `baselines.py` (random, base-stock, seasonal `S_t`, `(s,S)`) |
| 3 | **Algorithm choice (and why)** | `q_learning.py` (primary) + `ppo_agent.py` (stretch); justified by the baseline ladder (p.163) and the discrete-action cheat-sheet (p.60) — see "Key decisions" |
| 4 | **Reward spec** | `rewards.py` (`reward_balanced` vs `reward_naive`) + config economics |
| 5 | **Simulator / data plan** | `env.py` + `demand.py` (stationary/seasonal/stress) + seeded `config.py` (p.148, "simulation first") |
| 6 | **Safety controls** | `env.py` clamps + approval flag; the learned/coded/forbidden split (p.153, p.189) |
| 7 | **Evaluation plan** | `evaluation.py` + `evals/scenarios.jsonl` + `evals/run_evals.py` (p.150) |
| 8 | **Rollout plan** | `docs/BUSINESS_MEMO.md` — staged shadow→limited→full with rollback thresholds (p.151) |

## Key decisions (and why)

**Why tabular Q-learning is the headline, not PPO.** The state space is tiny once you act on
inventory position, so a table converges in seconds, every value is inspectable, and the greedy
policy can be plotted against base-stock. PPO is the *stretch*: it is slower, seed-variable, and on
this problem lands *below* the table (mean held-out gap −$135 vs base-stock, vs −$88 for the
table). That is the baseline ladder (p.163) argued with numbers — climb only as far as the problem
needs.

**Why the underage cost is `Cu = (p−c) + b = 11`, not just `b`.** Under lost sales, being one unit
short forfeits the margin you would have earned *and* the goodwill penalty. Counting only the
penalty is the classic newsvendor error; it would understate the order-up-to level and make
base-stock look worse than it is. With `Cu = 11`, `Co = 1`, the critical ratio is `11/12` over the
two-week protection interval → `S = 26`.

**Why the seasonal comparison uses a time-indexed base-stock.** Comparing a season-aware agent to a
*static* base-stock would be rigging the match — a competent planner would just index `S_t` by
week. So the seasonal regime includes a seasonal base-stock as the fair competitor. The honest
claim is that the phase-aware agent **ties** it (within 3%) while beating the static one, and the
value of RL is reaching that solution *without being given the seasonal model*.

**The Markov property is why the seasonal agent needs a phase feature (deck p.31).** Under
*stationary* demand, inventory position is a sufficient statistic — the problem is Markov in `IP`,
so a `(IP)` table is enough. Under *seasonal* demand it is **not**: the demand rate depends on the
week, so `IP` alone is a partially-observable state and the deck's prescription applies — add a
time/belief feature. That is exactly the season `phase` index. This is not a footnote; it is
*measured*: the `IP`-only agent scores 2253 under seasonality while the phase-augmented agent scores
2588. The 335-point gap is a **partial-observability failure, not an algorithm failure** — the
clearest possible demonstration of the Markov-property lesson, and a caution against blaming the
learner when the real bug is the state representation (cf. REFLECTION #5, the on-hand-vs-IP bug).

**Why the contextual-bandit rung of the ladder is skipped (deck p.157, p.163).** A contextual bandit
is the right tool when actions do not move the future state; the deck is explicit that "full RL
matters when delayed effects matter" (p.157). Inventory has delayed effects by construction — an
order placed today arrives after a lead time, and stock carried today becomes a holding cost or a
markdown tomorrow (deck p.158, "an over-order today becomes a markdown tomorrow"). So a bandit would
mis-model the problem; the ladder correctly starts at a rule (base-stock) and climbs to tabular RL.

**Value functions are first-class, not implicit (deck p.34–44).** The agent learns `Q(s, a)`; the
deployed policy is its greedy projection `argmax_a Q`, and the learned **value function**
`V(s) = max_a Q(s, a)` is plotted (`figures/policy_behavior.png`, middle panel) — it peaks at
moderate inventory and falls off toward stockout and overstock, the shape the Bellman recursion
should produce. Showing `V` alongside the policy makes the value/policy distinction concrete.

**Why the reward is the experiment.** `reward_naive` ("minimize stockouts") is a deliberately
gameable proxy. Training on it and scoring on the balanced objective makes the reward-hacking
lesson empirical rather than rhetorical: +0.8 service points, −38.5% profit, 3.2× the inventory.
The DRL deck's reward-hacking analogy (p.147, p.170) is the spine of the whole project.

**Why "out-of-sample" is used carefully.** Training, validation, and held-out evaluation draw from
three seed streams separated by 10⁶-sized offsets (`config.EvalParams`), and `seed_sets_disjoint`
asserts they cannot overlap. Only the held-out stream is called out-of-sample; tuning would use the
validation stream.

## Methodology & defensibility (one place to check the rigor)

- **No data leakage.** Training, validation, and evaluation demand come from three seed streams
  separated by 10⁶-sized offsets (`config.EvalParams`); `seed_sets_disjoint` asserts it for the
  tabular stream, and the held-out evaluation matrix is generated from a *dedicated* RNG
  (`eval_seed_base`) that **no trainer ever draws from** — the tabular agent uses `train_seed_base`,
  and PPO uses Stable-Baselines3's own internal seeding, neither of which touches the eval seeds.
- **No tuning on the test set.** Every hyperparameter (γ, α schedule, ε schedule, bin width,
  episode count) is a fixed default in `config.py`, not selected on the held-out set; a separate
  `val_seed_base` stream exists for tuning if needed. The robustness study further shows the result
  is insensitive to the main knob (bin width 2–10), so it is not a tuned-to-the-seed artifact.
- **Fair baseline comparison.** Base-stock is evaluated on the **same discrete action menu** as the
  agent (orders are menu-snapped via `order_to_action_index`), so neither competitor has an action
  set the other lacks — the comparison isolates the policy, not the action granularity.
- **Paired, effect-size-first statistics.** Every policy faces the identical demand per eval episode
  (paired design); differences use a **paired bootstrap** (resampling episode indices preserves the
  pairing) for a 95% CI, with **Wilcoxon signed-rank** as the primary test and paired-t as a
  cross-check. The reported number is the **dollar effect size with its CI**, not just a p-value
  (every comparison is significant at p < 1e-14, most ≈1e-17, so multiple-comparison correction is moot).
- **Determinism.** Same seeds → identical held-out numbers (verified by re-running, and by a
  **clean-room reproduction**: a fresh venv from the frozen `requirements-core.txt` reproduces both
  the 51 tests and the exact headline return, 2585.6). The only non-reproducible field in
  `eval_report.json` is the `generated_utc` timestamp, which is provenance, not a result.
- **Traceability.** Every run stamps a deterministic **`run_id`** (a hash of seed + simulator
  fingerprint + episode counts) into `eval_report.json` and `sample_run_log.txt`, tying the
  artifacts together; the **`env_fingerprint`** records the exact simulator version; library
  versions and the seed are logged; and per-episode traceability is explicit — evaluation episode
  `i` is demand seed `eval_seed_base + i`. This is the deck's observability checklist (p.152),
  applied: state/action/reward (sample log), constraints + approvals (scenario report), simulator
  version + trace id (meta).
- **Honest framing.** The agent **ties** the analytical optimum, it does not beat it; all four
  failure modes are *measured*; and where a measurement contradicted a prior guess (the overfitting
  probe, the reward-hacking magnitude, the discretization range) the narrative was corrected to match
  the data (REFLECTION #8, #10, #11). The strongest baseline (seasonal `S_t`) was added precisely
  because it makes the method look *least* impressive.

## Course-content cross-reference

DRL deck (the whole project): MDP framing (p.30–32), value/Bellman (p.39–48), Q-learning (p.48–49,
p.70–84), policy gradients/PPO (p.85–113), reward hacking (p.147, p.170), practical system design
(p.140–153), retail applications (p.156–163), capstone (p.181–182), closing question (p.189). The
Multi-Agent deck is intentionally **not** applied — this is a single-agent control problem by scope.
