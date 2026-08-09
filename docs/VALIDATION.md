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

## P5 rigor pass (measured mentor-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

