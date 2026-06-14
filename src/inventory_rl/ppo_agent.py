"""PPO stretch agent (Stable-Baselines3), with a guarded import so the core path never needs it.

PPO is the *stretch*, not the headline. On a problem this small a correctly specified
base-stock policy is the bar, tabular Q-learning already ties it, and -- as the failure
analysis shows -- PPO adds training-seed variance for no reproducibility benefit. It is
included to demonstrate that the same environment supports a modern deep-RL method and to
make the "climb the baseline ladder only as far as you must" argument empirically.

Every torch / Stable-Baselines3 import is deferred into the functions that need them, so
importing this module (and running the entire tabular path and the test suite) requires none
of the heavy stretch dependencies. Calling a PPO function without them raises a clear,
actionable error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inventory_rl.config import Config
from inventory_rl.env import InventoryEnv
from inventory_rl.rewards import reward_balanced

_MISSING_DEPS_MSG = (
    "The PPO stretch requires torch + stable-baselines3. Install them with:\n"
    "    pip install -r requirements-stretch.txt"
)


def _require_sb3() -> Any:
    """Import Stable-Baselines3 lazily; raise an actionable error if it is not installed."""
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415 - intentional lazy import
        from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415
        from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the stretch deps
        raise SystemExit(_MISSING_DEPS_MSG) from exc
    return PPO, BaseCallback, Monitor


def make_env(cfg: Config, demand_process: Any, reward_fn: Any = reward_balanced) -> InventoryEnv:
    """Build the inventory environment configured for a given demand process."""
    return InventoryEnv(cfg.env, demand_process=demand_process, reward_fn=reward_fn)


class PPOPolicy:
    """Adapter exposing a trained SB3 PPO model through the project's ``Policy`` interface."""

    def __init__(self, model: Any, name: str = "ppo") -> None:
        self._model = model
        self.name = name

    def __call__(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    @classmethod
    def load(cls, path: str | Path, name: str = "ppo") -> PPOPolicy:
        """Load a committed PPO model (no retraining needed to evaluate it)."""
        ppo_cls, _, _ = _require_sb3()
        return cls(ppo_cls.load(str(path)), name=name)


def train_ppo(
    cfg: Config,
    demand_process: Any,
    seed: int,
    total_timesteps: int | None = None,
) -> tuple[Any, list[float]]:
    """Train one PPO model; return the model and its per-episode training-return curve."""
    ppo_cls, base_callback, monitor = _require_sb3()
    env = monitor(make_env(cfg, demand_process))
    timesteps = total_timesteps if total_timesteps is not None else cfg.ppo.total_timesteps

    returns_curve: list[float] = []

    class _EpisodeReturnCallback(base_callback):  # type: ignore[misc, valid-type]
        """Collect each finished episode's return from the Monitor info buffer."""

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is not None:
                    returns_curve.append(float(episode["r"]))
            return True

    model = ppo_cls(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        policy_kwargs={"net_arch": list(cfg.ppo.net_arch)},
        verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=_EpisodeReturnCallback())
    return model, returns_curve


def save_ppo(model: Any, path: str | Path) -> None:
    """Persist a PPO model to the committed ``.zip`` artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def train_ppo_multiseed(
    cfg: Config,
    demand_process: Any,
    n_seeds: int | None = None,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """Train PPO under several seeds to quantify seed variance (the failure-analysis study).

    Returns every seed's training curve plus the trained models, so the caller can save the
    best as the committed artifact and plot the spread across seeds.
    """
    seeds = n_seeds if n_seeds is not None else cfg.ppo.n_seeds
    curves: list[list[float]] = []
    models: list[Any] = []
    final_returns: list[float] = []
    for s in range(seeds):
        model, curve = train_ppo(cfg, demand_process, seed=cfg.seed + s, total_timesteps=total_timesteps)
        curves.append(curve)
        models.append(model)
        # A cheap proxy for quality: the mean of the last 10% of training episode returns.
        tail = curve[max(0, len(curve) - max(1, len(curve) // 10)) :]
        final_returns.append(float(np.mean(tail)) if tail else float("-inf"))
    best = int(np.argmax(final_returns))
    return {
        "curves": curves,
        "models": models,
        "final_returns": final_returns,
        "best_index": best,
        "best_model": models[best],
    }
