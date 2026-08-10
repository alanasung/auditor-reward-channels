# When the Auditor's Channel Becomes the Policy

**Research areas.** Chain of thought; Scalable oversight

## Summary

Hold the task and the misbehavior incentive fixed, vary only where the monitor reads, and measure how much each reward shape teaches the model to hide rather than to behave.

## Hypothesis

Where a training-time auditor reads (trace tokens, post-hoc summary, or final answer) changes how much of the policy stays legible to a separate grader.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. In an environment that strongly rewards a detectable shortcut, how do the three reward shapes differ in misbehavior rate versus detection rate under actual RL optimization?
2. Can we separate genuine behavioral improvement from obfuscation using an independent held-out monitor the model was never trained against?
3. Does the ordering of obfuscation pressure hold across distinct training lineages (Qwen, Llama, OLMo), or does it invert? Lineage, not parameter count, is the variable the prior work asked about.
4. Does the thought region differentiate from the output region over the course of RL, which is the motivating stated stretch goal?

## Method

1. Build a reward-hacking environment where the shortcut is unambiguous and programmatically checkable, such as passing unit tests by special-casing the test inputs instead of implementing the function.
2. Implement three monitors: CoT-token, post-hoc output summary, output-only.
3. Train with GRPO via TRL, an open-source RL framework, using LoRA on small models so that genuine RL dynamics are what is being studied. Expert iteration is retained only as a cheap smoke-test tier for the harness, explicitly not as the substitute for RL.
4. Replicate the reward-shape comparison across three distinct lineages at comparable scale, so lineage effects are separable from scale effects.
5. Score every checkpoint with the held-out ground-truth cheat detector.
6. Track the obfuscation index and thought-vs-output differentiation across RL steps, not just at the endpoint.

## Measurements

- misbehavior rate under a ground-truth programmatic checker
- training-monitor detection rate
- obfuscation index: misbehaves AND evades the training monitor
- thought-region differentiation: does the CoT still entail the action actually taken, scored by an entailment check rather than raw text divergence, so benign summarization is not miscounted as obfuscation
- per-lineage ordering of the three reward shapes

## Threats to validity

- Falling monitor-detection can mean the model reformed or the model hid. The ground-truth checker is what separates them, so it must be programmatic and never used as a training signal.
- Small models may lack a legible CoT at all, which caps the effect size. A capability floor check gates the whole experiment.
- Raw CoT-vs-output text divergence also moves under benign compression, formatting, and style drift. Obfuscation must be operationalized as the CoT ceasing to entail the action taken, with a style-only control arm.

## Training lineages compared

- Qwen/Qwen2.5-0.5B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- allenai/OLMo-2-0425-1B-Instruct

## Literature engagement

docs/RELATED_WORK.md must contain a direct methodological critique of arXiv:2511.11584, stating what this repo replicates, what it considers under-specified, and what it changes. the motivating application question asks for exactly this.

## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
