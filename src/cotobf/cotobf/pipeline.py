"""Domain stages: reward-hack env, monitors, GRPO/PG RL, obfuscation curve."""

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


def _force_synthetic(cfg: Any) -> bool:
    return bool(getattr(cfg, "force_synthetic", False))


def _model_name(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "model", cfg), "name", "openai-community/gpt2"))


def _revision(cfg: Any) -> str | None:
    rev = getattr(getattr(cfg, "model", cfg), "revision", None)
    return str(rev) if rev else None


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env = RewardHackEnv(n_tasks=_n(cfg), seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "env.json", env.to_dict())
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=env.n_tasks,
        metrics={
            "n_tasks": env.n_tasks,
            "cheat_detector": "programmatic",
            "force_synthetic": _force_synthetic(cfg),
            "training_monitors": list(MONITOR_NAMES),
            "eval_monitor": "heldout_cheat_detector",
        },
    )
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    force = _force_synthetic(cfg)
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    low = collect_rollouts(
        env,
        seed=_seed(cfg),
        obfuscate=0.1,
        cheat_rate=0.5,
        model_name=model_name,
        revision=revision,
        force_synthetic=force,
    )
    high = collect_rollouts(
        env,
        seed=_seed(cfg) + 1,
        obfuscate=0.8,
        cheat_rate=0.6,
        model_name=model_name,
        revision=revision,
        force_synthetic=force,
    )
    rollouts = low + high
    val = validate_entailment_scorer(rollouts)
    out = ensure_dir(run_dir / "artifacts" / "collect")
    # Drop bulky raw fields from artifact if present.
    slim = [{k: v for k, v in r.items() if k != "raw"} for r in rollouts]
    write_json(out / "rollouts.json", slim)
    write_json(out / "entailment_validation.json", val)
    modes = {r.get("mode", "synthetic") for r in rollouts}
    payload = stage_result(
        task="collect",
        seed=_seed(cfg),
        n=len(rollouts),
        metrics={
            "entailment_validated": val["validated"],
            "hand_labeled_accuracy": val.get("hand_labeled_accuracy"),
            "rollout_modes": sorted(modes),
            **{k: val[k] for k in ("mean_obfuscated", "mean_confessed", "mean_honest")},
        },
    )
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    force = _force_synthetic(cfg)
    # Smaller measured steps for local pilots.
    steps = 4 if not force else 12
    per_shape = {}
    adapter_root = ensure_dir(run_dir / "artifacts" / "fit" / "adapters")
    for shape in MONITOR_NAMES:
        try:
            fit = train_with_grpo(
                env,
                reward_shape=shape,
                model_name=model_name,
                steps=steps,
                seed=_seed(cfg),
                revision=revision,
                force_synthetic=force,
                adapter_dir=str(adapter_root / shape),
            )
        except RuntimeError as exc:
            # Uniform fallback only when weights missing — label the whole fit synthetic.
            fit = train_with_grpo(
                env,
                reward_shape=shape,
                model_name=model_name,
                steps=steps,
                seed=_seed(cfg),
                revision=revision,
                force_synthetic=True,
            )
            fit["fallback_reason"] = f"measured_unavailable: {exc}"
        per_shape[shape] = {
            "final": fit["final"],
            "trl_status": fit["trl_status"],
            "mode": fit["mode"],
            "lora_enabled": fit.get("lora_enabled"),
            "fallback_reason": fit.get("fallback_reason"),
            "history": fit.get("history"),
        }
    smoke = expert_iteration_smoke(env, reward_shape="output_only", rounds=2, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "per_shape.json", per_shape)
    write_json(out / "expert_iteration_smoke.json", smoke)
    ordering = sorted(
        MONITOR_NAMES,
        key=lambda s: per_shape[s]["final"].get("obfuscation_index_heldout", 0.0),
        reverse=True,
    )
    metrics = {
        "ordering_by_heldout_obfuscation": list(ordering),
        "heldout_index": {
            s: per_shape[s]["final"].get("obfuscation_index_heldout", 0.0) for s in MONITOR_NAMES
        },
        "trl_status": per_shape[MONITOR_NAMES[0]]["trl_status"],
        "modes": {s: per_shape[s]["mode"] for s in MONITOR_NAMES},
        "note": "all arms scored with fixed held-out monitor; training monitors differ",
    }
    payload = stage_result(task="fit", seed=_seed(cfg), n=env.n_tasks, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env_d = read_json(run_dir / "artifacts" / "dataset" / "env.json")
    env = RewardHackEnv(n_tasks=env_d["n_tasks"], seed=env_d["seed"])
    force = _force_synthetic(cfg)
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    # Prefer curves from fit histories (actual trained checkpoints / PG steps).
    fit_payload = read_json(run_dir / "artifacts" / "fit" / "per_shape.json")
    curves = {}
    for shape in MONITOR_NAMES:
        hist = fit_payload.get(shape, {}).get("history") or []
        if hist:
            curves[shape] = hist[-1]
            continue
        pressure = {"cot_token": 0.8, "posthoc_summary": 0.5, "output_only": 0.2}[shape]
        trs = collect_rollouts(
            env,
            seed=_seed(cfg) + 7,
            obfuscate=pressure,
            cheat_rate=0.5 + 0.2 * pressure,
            model_name=model_name,
            revision=revision,
            force_synthetic=True if force else force,
        )
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
        "hand_labeled_accuracy": val.get("hand_labeled_accuracy"),
        "fit_modes": fit["metrics"].get("modes"),
        "lineage_stable": ev["metrics"]["ordering_stable_across_lineages"],
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=3, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
