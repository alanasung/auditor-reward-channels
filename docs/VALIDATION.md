# VALIDATION — cot-obfuscation-pressure

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: This is polished shared infrastructure around an unwritten experiment: it cannot run, its primary comparison is confounded by monitor strength, and the advertised M4 pilot is neither specified nor credibly sized.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Reward-hack env, three monitors, held-out obfuscation scoring, lineage comparison, and GRPO/synthetic RL path are wired through stages. Engineering bar met with synthetic-training fallback notes.

### Remaining
- GRPO fit falls back to a synthetic RL loop when TRL/weights are unavailable; pilot does not require a measured training run.
- Model revisions mostly `main`.

## Reconciliation
v1 empty-experiment failure fixed; held-out monitor scoring and entailment validation present. Grok PASS_WITH_NOTES alongside Codex v2 analogue.
