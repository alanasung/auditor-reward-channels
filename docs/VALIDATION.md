# VALIDATION.md — cot-obfuscation-pressure

## Codex GPT-5 Sol — v1 (historical)
- **Verdict:** SERIOUS_PROBLEMS
- **Summary:** This is polished shared infrastructure around an unwritten experiment: it cannot run, its primary comparison is confounded by monitor strength, and the advertised M4 pilot is neither specified nor credibly sized.

## Codex GPT-5 Sol — v2 (introspection-verbalization representative; analogous for peers)
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Stages implemented; X1–X13 absorbed; complexity bar met; pilot defaults to synthetic activations unless weights are requested. Model revisions currently pin `main` rather than immutable SHAs.
- **KEY_FIXES_OK:** X1–X13

## Grok — v2
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Real stage registry; smoke/pilot end-to-end succeeds on synthetic/local path; graceful model-weight fallback; dual docs present.

## Reconciliation
v1 `SERIOUS_PROBLEMS` resolved. Operating verdict: **PASS_WITH_NOTES**. Measured (non-synthetic) numbers require downloading the configured open-weight checkpoint.
