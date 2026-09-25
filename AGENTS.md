# WhenToReflect — Agent Instructions

This repository implements a research project on **state-level reflection utility in reasoning language models**.

## Before changing experiment logic

Read:

1. `docs/RESEARCH_PLAN.md`
2. `docs/EXPERIMENT_SPEC.md`
3. `docs/EVALUATION_PLAN.md`

These files define the scientific protocol.

## Scientific invariants

Do not silently change:

- the definition of a reasoning state;
- the reflect-vs-continue branching semantics;
- checkpoint-selection rules;
- rollout count or decoding settings;
- dataset splits;
- answer-grading logic;
- utility definition;
- evaluation metrics;
- probe targets;
- controller baselines.

If a requested code change would alter one of these, stop and state the methodological implication.

## Branching requirement

Reflect and continue branches must originate from the same pre-intervention reasoning prefix/state.

Pre-treatment activations used for probing must be captured before branch-specific instructions are introduced.

## Data leakage

All states and rollouts derived from one source problem must remain in the same train/dev/test split.

## Reproducibility

Every experimental run must record enough information to recover:

- code commit;
- model/revision;
- dataset/split;
- seed;
- prompt version;
- checkpoint-selection rule;
- decoding configuration;
- device;
- dtype;
- output schema.

## Probe discipline

Linear probes are the primary representation analysis.

Do not replace them with a higher-capacity nonlinear model without preserving the linear baseline and documenting the reason.

## Development environment

- Local Apple Silicon/MPS runs are for development, smoke testing, and correctness checks.
- Final reported experiments run on CUDA.
- Do not silently replace unsupported MPS operations with scientifically different approximations.

## Scope control

Do not add:
- head-level circuit analysis;
- complex nonlinear probes;
- new datasets;
- new model families;
- training-from-scratch components

unless explicitly requested.

The immediate goal is always the smallest implementation that faithfully answers the next planned research question.
