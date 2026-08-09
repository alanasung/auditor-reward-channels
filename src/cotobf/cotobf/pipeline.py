"""Domain stages: reward-hack env, monitors, GRPO/PG RL, obfuscation curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ._util import ensure_dir, read_json, stage_result, write_json
from .env import RewardHackEnv
from .expert_iteration import expert_iteration_smoke
from .lineages import run_lineage_comparison
from .monitors import (
    MONITOR_NAMES,
    apply_training_monitor,
    fit_fpr_matched_thresholds,
    power_status,
)
from .obfuscation import obfuscation_index, obfuscation_over_training_curve, validate_entailment_scorer
from .rl import train_with_grpo
from .rollouts import collect_rollouts

_MEASURED_RL_MODES = frozenset({"local_policy_gradient", "trl_grpo"})


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


def _is_measured_rl(mode: str) -> bool:
    return str(mode) in _MEASURED_RL_MODES


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    env = RewardHackEnv(n_tasks=_n(cfg), seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "env.json", env.to_dict())
    n_tasks = env.n_tasks
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=n_tasks,
        metrics={
            "n_tasks": n_tasks,
            "cheat_detector": "programmatic",
            "force_synthetic": _force_synthetic(cfg),
            "training_monitors": list(MONITOR_NAMES),
            "eval_monitor": "heldout_cheat_detector",
            "power_status": power_status(n_tasks),
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
    # Fit per-surface thresholds on held-out rollouts to shared FPR.
    fpr_fit = fit_fpr_matched_thresholds(rollouts, target_fpr=0.10, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "collect")
    # Drop bulky raw fields from artifact if present.
    slim = [{k: v for k, v in r.items() if k != "raw"} for r in rollouts]
    write_json(out / "rollouts.json", slim)
    write_json(out / "entailment_validation.json", val)
    write_json(out / "fpr_matched_thresholds.json", fpr_fit)
    modes = {r.get("mode", "synthetic") for r in rollouts}
    payload = stage_result(
        task="collect",
        seed=_seed(cfg),
        n=len(rollouts),
        metrics={
            "entailment_validated": val["validated"],
            "hand_labeled_accuracy": val.get("hand_labeled_accuracy"),
            "rollout_modes": sorted(modes),
            "fpr_matched": bool(fpr_fit["fpr_matched"]),
            "monitor_thresholds": fpr_fit["thresholds"],
            "achieved_fpr": fpr_fit.get("achieved_fpr"),
            "target_fpr": fpr_fit.get("target_fpr"),
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
    fpr_fit = read_json(run_dir / "artifacts" / "collect" / "fpr_matched_thresholds.json")
    thresholds = fpr_fit.get("thresholds") or {}
    fpr_matched = bool(fpr_fit.get("fpr_matched"))
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
            fit["is_synthetic"] = True
            fit["claims_gated"] = True
        per_shape[shape] = {
            "final": fit["final"],
            "trl_status": fit["trl_status"],
            "mode": fit["mode"],
            "lora_enabled": fit.get("lora_enabled"),
            "fallback_reason": fit.get("fallback_reason"),
            "history": fit.get("history"),
            "is_synthetic": bool(fit.get("is_synthetic", fit["mode"] == "synthetic_grpo")),
            "claims_gated": bool(fit.get("claims_gated", fit["mode"] == "synthetic_grpo")),
        }
    smoke = expert_iteration_smoke(env, reward_shape="output_only", rounds=2, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "per_shape.json", per_shape)
    write_json(out / "expert_iteration_smoke.json", smoke)
    # Raw ordering from held-out index (diagnostic); claims require measured + FPR match.
    ordering = sorted(
        MONITOR_NAMES,
        key=lambda s: per_shape[s]["final"].get("obfuscation_index_heldout", 0.0),
        reverse=True,
    )
    any_synthetic = any(per_shape[s]["is_synthetic"] for s in MONITOR_NAMES)
    all_measured = all(_is_measured_rl(per_shape[s]["mode"]) for s in MONITOR_NAMES)
    ordering_claim_ok = bool(all_measured and fpr_matched and not any_synthetic)
    curves = {}
    evolves_any = False
    for shape in MONITOR_NAMES:
        hist = per_shape[shape].get("history") or []
        measured = _is_measured_rl(per_shape[shape]["mode"])
        curve = obfuscation_over_training_curve(
            hist, measured=measured, min_steps=3, seed=_seed(cfg)
        )
        curves[shape] = curve
        evolves_any = evolves_any or bool(curve["obfuscation_evolves"])
    write_json(out / "obfuscation_curves.json", curves)
    metrics = {
        "ordering_by_heldout_obfuscation": list(ordering) if ordering_claim_ok else [],
        "ordering_raw": list(ordering),
        "ordering_claim_ok": ordering_claim_ok,
        "fpr_matched": fpr_matched,
        "monitor_thresholds": thresholds,
        "heldout_index": {
            s: per_shape[s]["final"].get("obfuscation_index_heldout", 0.0) for s in MONITOR_NAMES
        },
        "trl_status": per_shape[MONITOR_NAMES[0]]["trl_status"],
        "modes": {s: per_shape[s]["mode"] for s in MONITOR_NAMES},
        "is_synthetic": {s: per_shape[s]["is_synthetic"] for s in MONITOR_NAMES},
        "any_synthetic_rl": any_synthetic,
        "obfuscation_evolves": bool(evolves_any and ordering_claim_ok),
        "obfuscation_curves": {s: {k: curves[s][k] for k in ("slope", "n_steps", "obfuscation_evolves", "measured")} for s in MONITOR_NAMES},
        "power_status": power_status(env.n_tasks),
        "note": (
            "arm ordering requires fpr_matched=true and measured RL modes; "
            "synthetic_grpo is smoke-only with claims_gated"
        ),
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
    fpr_fit = read_json(run_dir / "artifacts" / "collect" / "fpr_matched_thresholds.json")
    thresholds = fpr_fit.get("thresholds")
    fpr_matched = bool(fpr_fit.get("fpr_matched"))
    # Prefer curves from fit histories (actual trained checkpoints / PG steps).
    fit_payload = read_json(run_dir / "artifacts" / "fit" / "per_shape.json")
    fit_metrics = read_json(run_dir / "artifacts" / "fit" / "results.json")["metrics"]
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
        flags = [
            apply_training_monitor(
                shape, tr, thresholds=thresholds, fpr_matched=fpr_matched
            )["flagged"]
            for tr in trs
        ]
        curves[shape] = obfuscation_index(trs, flags)
    lineages = run_lineage_comparison(
        steps=6,
        seed=_seed(cfg),
        n_tasks=min(32, env.n_tasks),
        force_synthetic=True,
        fpr_matched=fpr_matched,
    )
    curve_stats = read_json(run_dir / "artifacts" / "fit" / "obfuscation_curves.json")
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "curves.json", curves)
    write_json(out / "lineages.json", lineages)
    ordering_claim_ok = bool(fit_metrics.get("ordering_claim_ok") and fpr_matched)
    metrics = {
        "obfuscation_index_heldout": {
            s: curves[s]["obfuscation_index_heldout"] for s in MONITOR_NAMES
        },
        "thought_output_diff": {s: curves[s]["thought_output_diff"] for s in MONITOR_NAMES},
        "ordering_stable_across_lineages": lineages["ordering_stable_across_lineages"],
        "ordering_claim_ok": ordering_claim_ok,
        "fpr_matched": fpr_matched,
        "obfuscation_evolves": bool(fit_metrics.get("obfuscation_evolves")),
        "obfuscation_curves": curve_stats,
        "any_synthetic_rl": bool(fit_metrics.get("any_synthetic_rl")),
        "power_status": power_status(env.n_tasks),
    }
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=env.n_tasks, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    fit = read_json(run_dir / "artifacts" / "fit" / "results.json")
    ev = read_json(run_dir / "artifacts" / "evaluate" / "results.json")
    val = read_json(run_dir / "artifacts" / "collect" / "entailment_validation.json")
    claim_ok = bool(fit["metrics"].get("ordering_claim_ok"))
    # Gate synthetic GRPO out of headlines.
    metrics = {
        "ordering": fit["metrics"]["ordering_by_heldout_obfuscation"] if claim_ok else [],
        "ordering_raw": fit["metrics"].get("ordering_raw"),
        "ordering_claim_ok": claim_ok,
        "fpr_matched": fit["metrics"].get("fpr_matched"),
        "heldout_index": ev["metrics"]["obfuscation_index_heldout"] if claim_ok else {},
        "entailment_validated": val.get("validated"),
        "hand_labeled_accuracy": val.get("hand_labeled_accuracy"),
        "fit_modes": fit["metrics"].get("modes"),
        "is_synthetic": fit["metrics"].get("is_synthetic"),
        "lineage_stable": ev["metrics"]["ordering_stable_across_lineages"] if claim_ok else False,
        "obfuscation_evolves": bool(ev["metrics"].get("obfuscation_evolves")) if claim_ok else False,
        "power_status": fit["metrics"].get("power_status") or ev["metrics"].get("power_status"),
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=3, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
