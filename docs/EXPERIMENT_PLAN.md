# Experiment plan

Stage-by-stage design. Each stage is registered in `src/cotobf/stages.py`
and appears in `python -m cotobf stages`.

## Stages

| stage | responsibility |
|---|---|
| `env` | reward-hacking task environment and programmatic cheat detector |
| `monitors` | CoT-token, post-hoc summary, and output-only monitors |
| `rollouts` | sampling and trajectory storage |
| `rl` | GRPO training via TRL with monitor-derived reward shapes |
| `expert_iteration` | cheap smoke-test tier for harness validation only |
| `lineages` | cross-lineage replication harness |
| `obfuscation` | entailment-based obfuscation index and differentiation |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m cotobf run -c configs/pilot.yaml --stage obfuscation
```

## Controls and their purpose

- Falling monitor-detection can mean the model reformed or the model hid. The ground-truth checker is what separates them, so it must be programmatic and never used as a training signal.
- Small models may lack a legible CoT at all, which caps the effect size. A capability floor check gates the whole experiment.
- Raw CoT-vs-output text divergence also moves under benign compression, formatting, and style drift. Obfuscation must be operationalized as the CoT ceasing to entail the action taken, with a style-only control arm.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
