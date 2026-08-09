# Related work

This note situates **A Monitor-Reward Taxonomy and the Obfuscation Curve** against the mentor-linked literature for
[An Exploration of What Kinds of Training Pressure Cause COT Obfuscation](https://sparai.org/projects/f26/recaZTXfK7ua29Izm).

## Positioning

Hold the task and the misbehavior incentive fixed, vary only where the monitor reads, and measure how much each reward shape teaches the model to hide rather than to behave.

The design hypothesis is: Obfuscation pressure is not binary in whether the monitor touches the chain of thought. Post-hoc summarization of reasoning in output space induces measurable obfuscation too, placing it between direct CoT-token monitoring and output-only monitoring. Whether that ordering is stable across training lineages is the open question, not an assumption: lineage-dependent reversals are a live possible outcome and would be a more interesting result than confirmation.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the mentor's main citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Mentors and affiliations

- Mentor(s): Cody Wild
- Affiliation(s): Google DeepMind

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
