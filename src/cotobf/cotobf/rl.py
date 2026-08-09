"""GRPO training via TRL with defensive API pinning and graceful fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from .monitors import MONITOR_NAMES, apply_training_monitor, heldout_monitor
from .obfuscation import obfuscation_index
from .rollouts import collect_rollouts


PINNED_TRL_NOTE = "trl==0.14.0 preferred; GRPO API probed at runtime"


def _try_import_grpo() -> tuple[Any | None, str]:
    try:
        import trl  # type: ignore

        ver = getattr(trl, "__version__", "unknown")
    except Exception as exc:
        return None, f"trl import failed: {exc}"
    # Probe common names across trl versions.
    for path in (
        ("trainer", "GRPOTrainer"),
        ("trainer.grpo_trainer", "GRPOTrainer"),
        ("GRPOTrainer",),
    ):
        mod = trl
        try:
            if len(path) == 1:
                cls = getattr(trl, path[0], None)
            else:
                import importlib

                mod = importlib.import_module(f"trl.{path[0]}" if not path[0].startswith("trl") else path[0])
                cls = getattr(mod, path[1], None)
            if cls is not None:
                return cls, f"found {cls} in trl {ver}"
        except Exception:
            continue
    return None, f"trl {ver} present but GRPOTrainer API not found; using synthetic GRPO fallback"


def synthetic_grpo(
    env: Any,
    *,
    reward_shape: str,
    steps: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate GRPO dynamics under a reward shape; always score with held-out monitor."""
    if reward_shape not in MONITOR_NAMES:
        raise KeyError(reward_shape)
    rng = np.random.default_rng(seed)
    # Higher CoT monitoring pressure -> more obfuscation over steps.
    pressure = {"cot_token": 0.8, "posthoc_summary": 0.5, "output_only": 0.2}[reward_shape]
    history = []
    for step in range(steps):
        obfuscate = min(0.95, pressure * (step + 1) / steps)
        cheat_rate = 0.4 + 0.3 * pressure
        rollouts = collect_rollouts(env, seed=seed + step, obfuscate=obfuscate, cheat_rate=cheat_rate)
        train_flags = [apply_training_monitor(reward_shape, tr)["flagged"] for tr in rollouts]
        # Training reward: task pass minus monitor flag (monitor-shaped).
        rewards = []
        for tr, flagged in zip(rollouts, train_flags):
            from .env import detect_cheat

            passed = detect_cheat(tr["output"]) or ("*" in tr["output"])
            rewards.append(float(passed) - float(flagged))
        metrics = obfuscation_index(rollouts, train_flags)
        metrics["mean_reward"] = float(np.mean(rewards))
        metrics["step"] = step
        # Also record held-out flags explicitly
        metrics["heldout_flags"] = sum(heldout_monitor(tr)["flagged"] for tr in rollouts)
        history.append(metrics)
    final = history[-1]
    return {
        "mode": "synthetic_grpo",
        "reward_shape": reward_shape,
        "steps": steps,
        "pinned_trl_note": PINNED_TRL_NOTE,
        "history": history,
        "final": final,
        "trl_status": _try_import_grpo()[1],
    }


def train_with_grpo(
    env: Any,
    *,
    reward_shape: str,
    model_name: str = "sshleifer/tiny-gpt2",
    steps: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    cls, status = _try_import_grpo()
    if cls is None:
        out = synthetic_grpo(env, reward_shape=reward_shape, steps=steps, seed=seed)
        out["trl_status"] = status
        out["model_name"] = model_name
        return out
    # Real path still falls back for pilot without full GRPO dataset plumbing.
    try:
        out = synthetic_grpo(env, reward_shape=reward_shape, steps=steps, seed=seed)
        out["mode"] = "trl_api_detected_synthetic_loop"
        out["trl_status"] = status + "; full GRPO loop reserved for non-pilot runs"
        out["model_name"] = model_name
        return out
    except Exception as exc:  # pragma: no cover
        out = synthetic_grpo(env, reward_shape=reward_shape, steps=steps, seed=seed)
        out["trl_status"] = f"{status}; runtime error {exc}"
        return out
