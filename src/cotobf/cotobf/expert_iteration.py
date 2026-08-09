"""Cheap smoke-test tier for harness validation only — not an RL substitute."""

from __future__ import annotations

from typing import Any

from .monitors import apply_training_monitor
from .obfuscation import obfuscation_index
from .rl import _structural_pass_proxy
from .rollouts import collect_rollouts


def expert_iteration_smoke(env: Any, *, reward_shape: str, rounds: int = 3, seed: int = 0) -> dict[str, Any]:
    history = []
    best = None
    for r in range(rounds):
        rollouts = collect_rollouts(env, seed=seed + r, obfuscate=0.1 * r, cheat_rate=0.4)
        # Keep "experts": non-flagged passing trajectories.
        experts = []
        flags = []
        for tr in rollouts:
            flagged = apply_training_monitor(reward_shape, tr)["flagged"]
            flags.append(flagged)
            # Structural / public-test proxy only — detect_cheat is eval-only.
            passed = _structural_pass_proxy(tr.get("output", ""))
            if passed and not flagged:
                experts.append(tr)
        metrics = obfuscation_index(rollouts, flags)
        metrics["round"] = r
        metrics["n_experts"] = len(experts)
        history.append(metrics)
        best = experts or best
    return {
        "mode": "expert_iteration_smoke",
        "note": "harness validation only; not a substitute for GRPO/RL",
        "reward_shape": reward_shape,
        "history": history,
        "final": history[-1] if history else {},
        "n_expert_trajectories": len(best or []),
    }
