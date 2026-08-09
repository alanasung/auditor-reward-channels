"""Cross-lineage replication harness."""

from __future__ import annotations

from typing import Any

from .env import RewardHackEnv
from .monitors import MONITOR_NAMES
from .rl import train_with_grpo

DEFAULT_LINEAGES = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
)


def run_lineage_comparison(
    *,
    lineages: tuple[str, ...] = DEFAULT_LINEAGES,
    steps: int = 10,
    seed: int = 0,
    n_tasks: int = 32,
    force_synthetic: bool = True,
) -> dict[str, Any]:
    """Compare reward-shape orderings across lineage labels.

    Default ``force_synthetic=True`` keeps the harness offline (unit tests /
    smoke). Measured per-lineage LoRA/PG runs are opt-in when weights are local.
    """
    env = RewardHackEnv(n_tasks=n_tasks, seed=seed)
    results = {}
    orderings = {}
    for i, lineage in enumerate(lineages):
        per_shape = {}
        modes = {}
        for shape in MONITOR_NAMES:
            fit = train_with_grpo(
                env,
                reward_shape=shape,
                model_name=lineage,
                steps=steps,
                seed=seed + i,
                force_synthetic=force_synthetic,
            )
            per_shape[shape] = fit["final"]["obfuscation_index_heldout"]
            modes[shape] = fit["mode"]
        ordering = sorted(per_shape, key=lambda s: per_shape[s], reverse=True)
        results[lineage] = {"per_shape": per_shape, "ordering": ordering, "modes": modes}
        orderings[lineage] = ordering
    canonical = list(orderings.values())[0] if orderings else []
    stable = all(v == canonical for v in orderings.values()) if orderings else True
    return {
        "lineages": list(lineages),
        "results": results,
        "ordering_stable_across_lineages": bool(stable),
        "force_synthetic": force_synthetic,
        "note": "lineage, not parameter count, is the comparison axis",
    }
