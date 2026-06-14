"""Single-SKU inventory-replenishment RL.

A small, inspectable reinforcement-learning project: a periodic-review inventory
simulator (`env`), a stochastic demand world (`demand`), analytical and learned
ordering policies (`baselines`, `q_learning`, `ppo_agent`), an explicit balanced-vs-naive
reward pair that demonstrates reward hacking (`rewards`), and a paired, held-out
evaluation harness with bootstrap confidence intervals (`evaluation`).

The package is deliberately structured so the entire tabular path runs on CPU with no
GPU and no heavyweight dependencies; the PPO stretch (`ppo_agent`) guards its torch import
so importing anything else never requires Stable-Baselines3.
"""

from __future__ import annotations

__version__ = "0.1.0"
