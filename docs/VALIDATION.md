# VALIDATION — cot-obfuscation-pressure

## Codex (p2)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants sandboxed execution with richer task diversity, matched-strength monitor family, multi-seed powered pilots, and measured lineage-by-default — beyond the local micro-pilot.
- Detail: `orchestration/out/validate/cot-obfuscation-pressure.json`

## Grok (p2 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Real chat-templated rollouts, TRL GRPO attempt + LoRA REINFORCE path (not a fake curve), held-out monitor independent of `detect_cheat`, entailment validation on hand-labeled pairs, `force_synthetic` smoke-only.
- Detail: `orchestration/out/grok/validate/cot-obfuscation-pressure.p2.md`

## KEY_FIXES (p2)
| Fix | Status |
|---|---|
| Real model rollouts + chat templates | OK (`rollouts.py`) |
| TRL GRPO attempt or local LoRA PG (real updates) | OK (`rl.py`) |
| REINFORCE sign corrected (`r * NLL` minimized) | OK |
| Training reward from public-test execution, not GT cheat | OK |
| Held-out monitor independent of `detect_cheat` | OK (`monitors.py`) |
| Hidden tests + subprocess execution in env | OK (`env.py`) |
| Entailment scorer validated on labeled pairs | OK (`obfuscation.py`) |
| Adapter checkpoints saved under fit artifacts | OK |
| `force_synthetic` smoke-only / pilot measured default | OK |
| Lineage harness synthetic by default (opt-in measured) | OK (labeled) |

## Remaining (compute / weights / scale — not empty stages)
- Lineage comparison defaults synthetic for offline harness; measured per-lineage adapters are opt-in when weights/credentials exist.
- Entailment scorer is lexical/heuristic with a small labeled set — not an NLI model.
- Pilot n≈32 tasks / few PG steps on Qwen-0.5B or gpt2 under MPS; not a frontier multi-seed study.
- Monitor arms still differ in detector surface (location + cues); full matched-classifier redesign is future work.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable RL/obfuscation core. Codex SERIOUS_PROBLEMS targets powered multi-lineage perfection — residual scale notes. Domain tests pass with Hub monkeypatched; synthetic is smoke-only.

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 rigor pass

| Fix | Status |
|---|---|
| Purge `detect_cheat` from train rewards (`_reward`, `synthetic_grpo`, expert-iter smoke) | OK — public-test execute / structural proxy only |
| Separate post-hoc summary generation (2nd call; CoT-only; surface stamp) | OK (`rollouts.py` `summary_surface=posthoc_from_cot`) |
| Entailment: expanded labeled pairs; threshold fit on held-out half; P/R reported | OK (`obfuscation.py`) |
| Domain P6 tests Hub-free | OK (`test_domain_p6_train_reward.py`) |

Residual: entailment remains lexical (not NLI); lineage measured adapters still opt-in.

## P7 rigor pass

| Fix | Status |
|---|---|
| FPR-matched per-surface thresholds on held-out rollouts; `fpr_matched` stamp | OK (`monitors.py` `fit_fpr_matched_thresholds`) |
| Arm ordering / report metrics require `fpr_matched=true` + measured RL modes | OK (`pipeline.py`, `lineages.py`) |
| Synthetic GRPO gated: `is_synthetic=true`, claims false; smoke retained | OK (`rl.py` / fit / report) |
| Obfuscation-over-training curve + `obfuscation_evolves` slope test (measured steps) | OK (`obfuscation.py`) |
| Honest pilot-n: `power_status=micro` stamps; README/DESIGN softened | OK |
| Domain P7 tests Hub-free | OK (`test_domain_p7_matched_claims.py`) |

Residual: micro-pilot n and synthetic lineage harness remain; measured multi-lineage adapters still opt-in.

## Codex problem-statement fit

Model: `gpt-5.6-sol` · gates: `match` + `validate` · 2026-08-09
Artifacts: `orchestration/out/match/cot-obfuscation-pressure.md`, `orchestration/out/validate/cot-obfuscation-pressure.md`

- **Match verdict:** `MATERIAL_DRIFT` · methods `mixed_proxy`
- **Validate overall:** `SERIOUS_PROBLEMS` · fit `MATERIAL_DRIFT` · feasibility `RUNNABLE_IF_SHRUNK`
- **Match summary:** The repo targets the right scientific question and contains a genuine local RL scaffold, but its calibrated rewards are not used in training, its preferred GRPO path collapses the three surfaces, and its required lineage comparison is synthetic, so it cannot yet answer the mentor's central question.
- **Validate summary:** A well-targeted and unusually honest scaffold, but the measured training/evaluation path is scientifically invalid and the pilot cannot yet answer the mentor's central reward-surface or lineage questions.

### Top drift / missing (match)
- FPR-matched thresholds are fitted and used for claim gating but are never passed into the rewards used for training
- the preferred TRL GRPO reward path maps one completion onto overlapping CoT, summary, and output fields, so it does not implement three distinct reward surfaces
- the TRL reward uses a structural pass proxy rather than the task's public-test execution
- stage_evaluate hard-codes force_synthetic=true for every lineage comparison

### Blocking (validate)
- `src/cotobf/cotobf/rl.py`: TRL evaluation reloads the untrained base model, may retain fabricated honest rollouts after failure, and still marks the result measured and claimable.
- `src/cotobf/cotobf/pipeline.py`: FPR-matched thresholds are read and reported but never passed into the reward functions used for training.
- `src/cotobf/cotobf/env.py`: The environment provides no strong incentive to misbehave; the honest solution is explicit and simpler than the shortcut.
- `src/cotobf/cotobf/obfuscation.py`: The primary misbehavior metric ignores public-pass/hidden-fail behavioral cheating and relies only on an overbroad lexical/AST detector.
