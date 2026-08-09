"""Cross-lineage replication harness."""

from __future__ import annotations

from typing import Any

from .env import RewardHackEnv
from .monitors import MONITOR_NAMES
from .rl import train_with_grpo

# Small open lineages preferred for local pilots (when weights are present).
DEFAULT_LINEAGES = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "openai-community/gpt2",
    "meta-llama/Llama-3.2-1B-Instruct",
)


def run_lineage_comparison(
    *,
    lineages: tuple[str, ...] = DEFAULT_LINEAGES,
    steps: int = 10,
    seed: int = 0,
    n_tasks: int = 32,
    force_synthetic: bool = True,
    fpr_matched: bool = False,
) -> dict[str, Any]:
    """Compare reward-shape orderings across lineage labels.

    Default ``force_synthetic=True`` keeps the harness offline (unit tests /
    smoke). Measured per-lineage runs are opt-in when weights are local.

    ``ordering_stable_across_lineages`` is only claimable when every lineage
    ran in a non-synthetic mode and monitors are FPR-matched; otherwise the
    flag is False and ``ordering_claim_ok`` explains why.
    """
    env = RewardHackEnv(n_tasks=n_tasks, seed=seed)
    results = {}
    orderings = {}
    any_synthetic = False
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
            mode = str(fit.get("mode", "synthetic"))
            modes[shape] = mode
            if (
                mode in {"synthetic", "synthetic_grpo"}
                or force_synthetic
                or bool(fit.get("is_synthetic"))
            ):
                any_synthetic = True
        ordering = sorted(per_shape, key=lambda s: per_shape[s], reverse=True)
        results[lineage] = {"per_shape": per_shape, "ordering": ordering, "modes": modes}
        orderings[lineage] = ordering
    canonical = list(orderings.values())[0] if orderings else []
    stable = all(v == canonical for v in orderings.values()) if orderings else True
    claim_ok = bool(stable and not any_synthetic and len(orderings) >= 2 and fpr_matched)
    return {
        "lineages": list(lineages),
        "results": results,
        "ordering_stable_across_lineages": bool(claim_ok),
        "ordering_raw_stable": bool(stable),
        "ordering_claim_ok": claim_ok,
        "any_lineage_synthetic": any_synthetic,
        "force_synthetic": force_synthetic,
        "fpr_matched": bool(fpr_matched),
        "note": (
            "lineage, not parameter count, is the comparison axis; "
            "ordering claims require measured modes on all lineages and fpr_matched=true"
        ),
    }
