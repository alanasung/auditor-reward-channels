"""Domain stages: reward-hack env, monitors, GRPO/synthetic RL, obfuscation curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .env import RewardHackEnv
from .expert_iteration import expert_iteration_smoke
from .lineages import run_lineage_comparison
from .monitors import MONITOR_NAMES, apply_training_monitor
from .obfuscation import obfuscation_index, validate_entailment_scorer
from .rl import train_with_grpo
from .rollouts import collect_rollouts
from ._util import ensure_dir, read_json, stage_result, write_json


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n(cfg: DictConfig) -> int:
    return max(24, int(getattr(cfg.data, "n_items", 64)))


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env = RewardHackEnv(n_tasks=_n(cfg), seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "env.json", env.to_dict())
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=env.n_tasks,
        metrics={"n_tasks": env.n_tasks, "cheat_detector": "programmatic"},
    )
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    # Collect rollouts under low/high obfuscation for entailment validation
    low = collect_rollouts(env, seed=_seed(cfg), obfuscate=0.1, cheat_rate=0.5)
    high = collect_rollouts(env, seed=_seed(cfg) + 1, obfuscate=0.8, cheat_rate=0.6)
    rollouts = low + high
    val = validate_entailment_scorer(rollouts)
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "rollouts.json", rollouts)
    write_json(out / "entailment_validation.json", val)
    payload = stage_result(
        task="collect",
        seed=_seed(cfg),
        n=len(rollouts),
        metrics={"entailment_validated": val["validated"], **{k: val[k] for k in ("mean_obfuscated", "mean_confessed", "mean_honest")}},
    )
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    model_name = str(getattr(cfg.model, "name", "sshleifer/tiny-gpt2"))
    per_shape = {}
    for shape in MONITOR_NAMES:
        fit = train_with_grpo(env, reward_shape=shape, model_name=model_name, steps=12, seed=_seed(cfg))
        per_shape[shape] = {
            "final": fit["final"],
            "trl_status": fit["trl_status"],
            "mode": fit["mode"],
        }
    smoke = expert_iteration_smoke(env, reward_shape="output_only", rounds=2, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "per_shape.json", per_shape)
    write_json(out / "expert_iteration_smoke.json", smoke)
    ordering = sorted(
        MONITOR_NAMES,
        key=lambda s: per_shape[s]["final"]["obfuscation_index_heldout"],
        reverse=True,
    )
    metrics = {
        "ordering_by_heldout_obfuscation": list(ordering),
        "heldout_index": {s: per_shape[s]["final"]["obfuscation_index_heldout"] for s in MONITOR_NAMES},
        "trl_status": per_shape[MONITOR_NAMES[0]]["trl_status"],
        "note": "all arms scored with fixed held-out monitor",
    }
    payload = stage_result(task="fit", seed=_seed(cfg), n=env.n_tasks, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    # Re-score a fresh rollout batch per shape with held-out monitor
    curves = {}
    for shape in MONITOR_NAMES:
        pressure = {"cot_token": 0.8, "posthoc_summary": 0.5, "output_only": 0.2}[shape]
        trs = collect_rollouts(env, seed=_seed(cfg) + 7, obfuscate=pressure, cheat_rate=0.5 + 0.2 * pressure)
        flags = [apply_training_monitor(shape, tr)["flagged"] for tr in trs]
        curves[shape] = obfuscation_index(trs, flags)
    lineages = run_lineage_comparison(steps=6, seed=_seed(cfg), n_tasks=min(32, env.n_tasks))
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "curves.json", curves)
    write_json(out / "lineages.json", lineages)
    metrics = {
        "obfuscation_index_heldout": {s: curves[s]["obfuscation_index_heldout"] for s in MONITOR_NAMES},
        "thought_output_diff": {s: curves[s]["thought_output_diff"] for s in MONITOR_NAMES},
        "ordering_stable_across_lineages": lineages["ordering_stable_across_lineages"],
    }
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=env.n_tasks, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    fit = read_json(run_dir / "artifacts" / "fit" / "results.json")
    ev = read_json(run_dir / "artifacts" / "evaluate" / "results.json")
    val = read_json(run_dir / "artifacts" / "collect" / "entailment_validation.json")
    metrics = {
        "ordering": fit["metrics"]["ordering_by_heldout_obfuscation"],
        "heldout_index": ev["metrics"]["obfuscation_index_heldout"],
        "entailment_validated": val.get("validated"),
        "lineage_stable": ev["metrics"]["ordering_stable_across_lineages"],
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=3, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
