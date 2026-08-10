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

## Scientific validity notes (Codex)

Model: `gpt-5.6-sol`. Full artifacts live in the private orchestration tree.

- **Framing:** When the Auditor's Channel Becomes the Policy — reward-channel effects on reasoning-trace legibility.
- **Methods fidelity:** mixed measured / proxy paths; smoke stays synthetic.
- **Open scientific gaps:** protocol confounds, proxy corpora, and claim gating remain; do not present pilots as settled empirical results.

