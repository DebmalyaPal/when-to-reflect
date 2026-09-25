# WhenToReflect — Experiment Specification

> **Purpose:** Authoritative specification for implementing the experiments.  
> Code should implement this document faithfully. Scientific definitions must not be changed implicitly.

## 1. Experimental Unit

A state-level observation is defined by:

\[
(\text{problem}, \text{model}, \text{base trajectory}, \text{checkpoint}, \text{generation config})
\]

The core comparison branches from the **same reasoning prefix**.

## 2. Reasoning State

A reasoning state \(s\) is:

> the question plus all model-generated reasoning up to a selected valid step boundary.

A valid checkpoint must:
- occur after at least some model-generated reasoning;
- occur before the final answer state;
- be selected using a fixed rule defined before inspecting branch outcomes.

Candidate step boundaries may include:
- paragraph breaks;
- model-specific reasoning delimiters;
- other natural reasoning boundaries.

The initial prompt-only state and final answer state are excluded.

## 3. Base Trajectory Generation

For each source problem:

1. Apply a fixed chat template.
2. Use a fixed, logged generation configuration.
3. Sample one or more base reasoning traces.
4. Segment each trace into valid reasoning-step boundaries.
5. Select a small number of checkpoints per trace.

Recommended checkpoint strategy for the initial pilot:
- early;
- middle;
- late;
- optionally one random valid boundary.

Checkpoint-selection rules must be fixed before branch outcomes are examined.

## 4. Pre-Treatment Activation Logging

At each checkpoint, before any intervention:

- record the residual-stream hidden state from every layer;
- use the final prefix token / step-boundary token;
- treat this hidden state as the pre-treatment representation.

The hidden state used for probing must be captured **before** reflect-vs-continue branching.

Recommended storage:
- compact tensor arrays;
- float16 where scientifically acceptable;
- metadata stored separately in structured records.

## 5. Branching Protocol

From the exact same reasoning state \(s\), create at least two primary branches.

### Branch A — Direct Continuation

Continue generation normally from \(s\) without an explicit reflection instruction.

### Branch B — Reflection Continuation

Apply a standardized reflection instruction that asks the model to review the reasoning-so-far for possible errors before continuing.

### Branch C — Neutral Control (Ablation)

Use a length-matched neutral instruction that adds an instruction/pass without specifically requesting reflection.

Purpose:
- distinguish reflection-specific effects from generic extra-token or extra-instruction effects.

## 6. Scientific Invariants for Branching

The reflect and continue branches must begin from the same pre-intervention state.

Unless a specific ablation changes one of these factors, branches must match on:

- model weights;
- tokenizer;
- base question;
- full prefix text;
- selected checkpoint;
- decoding configuration;
- temperature;
- top-p / top-k settings;
- max generation length;
- generation implementation;
- device/dtype within a run.

All branch-specific differences must be explicit and logged.

## 7. Repeated Futures

For each checkpoint and each action:

- sample \(k\) continuations;
- use identical decoding parameters across actions;
- estimate expected outcome from repeated futures rather than a single completion.

Initial pilot:
- \(k = 4\).

Possible final-test increase:
- \(k = 8\) if variance is high.

## 8. Outcome Scoring

### Math

Primary reward:
- exact-answer or verifier-based correctness.

### GPQA-Diamond

Primary reward:
- answer-choice accuracy.

### Additional logged outcomes

For each continuation record:
- final correctness;
- generated token count;
- optional verifier score;
- whether a previously correct trajectory becomes incorrect;
- whether an incorrect trajectory becomes correct.

## 9. Reflection Utility

For each state \(s\), estimate:

\[
\hat{U}(s)
=
\frac{1}{k}\sum_{i=1}^{k} R^{(i)}_{\text{reflect}}
-
\frac{1}{k}\sum_{i=1}^{k} R^{(i)}_{\text{continue}}
\]

where \(R\) is primarily binary final-answer correctness.

Preserve both:

1. **continuous utility** \(\hat{U}(s)\);
2. **categorical utility**:
   - helpful;
   - neutral;
   - harmful.

The categorical threshold or confidence-interval rule must be frozen before final evaluation.

## 10. Dataset Construction Procedure

For every source problem:

1. Generate base trajectory/trajectories.
2. Identify valid reasoning boundaries.
3. Select checkpoints.
4. Log pre-treatment hidden activations.
5. Fork the exact state.
6. Generate repeated direct futures.
7. Generate repeated reflection futures.
8. Optionally generate neutral-control futures.
9. Score final outcomes.
10. Estimate utility.
11. Write state-level record and rollout-level records.

No manually authored reflection paths are required.

## 11. Required State-Level Record

Each state-level row should contain at least:

| Field | Purpose |
|---|---|
| `problem_id` | Stable problem identifier |
| `dataset` | Source dataset |
| `domain` | Domain/category for stratified analysis |
| `model_id` | Exact model identifier |
| `model_revision` | Model revision/commit when available |
| `seed` | Reproducibility |
| `decoding_config` | Full generation configuration |
| `prefix_text` | Exact state text |
| `step_index` | Step number in trace |
| `relative_position` | Early/middle/late or normalized position |
| `hidden_state[layer]` | Pre-reflection hidden activation |
| `uncertainty_features` | Entropy/confidence-related features |
| `reflect_rewards` | Outcomes for repeated reflect rollouts |
| `continue_rewards` | Outcomes for repeated direct rollouts |
| `neutral_rewards` | Outcomes for neutral-control rollouts if run |
| `U_hat` | Estimated reflection utility |
| `utility_class` | Helpful / neutral / harmful |
| `token_cost_reflect` | Reflection token cost |
| `token_cost_continue` | Direct token cost |
| `native_reflection_indicator` | Optional spontaneous-reflection signal |
| `git_commit` | Code version |
| `device` | MPS/CUDA/CPU |
| `dtype` | Numeric precision |

## 12. Uncertainty Features

Candidate uncertainty/confidence features include:

- token entropy;
- probability margin / log-probability;
- self-consistency;
- verbalized confidence;
- optional internal-confidence probe.

These are comparison features, not substitutes for the reflection-utility target.

## 13. Data Splitting

**Split by problem, not by state.**

All states and all rollouts derived from the same source problem must remain in the same split.

Reason:
- prevents near-duplicate prefixes from leaking across probe training and evaluation.

Probe hyperparameters and controller thresholds must be selected on development problems only.

Final test sets remain untouched until the design is frozen.

## 14. Initial Pilot Configuration

Initial target:

- model: Qwen3-8B;
- data: 50–100 MATH training problems;
- checkpoints: 3–4 per problem;
- branch samples: \(k=4\) per action;
- actions:
  - continue;
  - reflect;
  - neutral control for prompt ablation;
- full metadata logging;
- answer grading;
- step segmentation;
- hidden-state logging.

Primary pilot goal:
- verify that non-trivial helpful and harmful utility states exist;
- verify the pipeline is reproducible;
- verify the branch construction is scientifically valid.

## 15. Scaling Sequence

Only after the pilot is stable:

1. MATH-500.
2. GPQA-Diamond.
3. Qwen3-14B.
4. DeepSeek-R1-Distill-Llama-8B.
5. Optional larger model extension.

## 16. Controls and Ablations

### 16.1 Prompt-control ablation

Compare reflection with a length-matched neutral instruction.

### 16.2 Problem-level split

No source problem may appear across train/dev/test partitions.

### 16.3 Repeated branch sampling

Do not label utility from a single lucky/unlucky continuation.

### 16.4 Position and difficulty controls

Check whether prediction is reducible to:
- trace position;
- problem difficulty;
- prefix length.

### 16.5 Length control

Compare branches at matched token budgets where possible, or report token differences explicitly.

### 16.6 Correct-to-wrong analysis

Measure how often reflection damages an otherwise correct trajectory.

### 16.7 Probe-capacity control

Linear probes are primary.

Nonlinear probes are diagnostic and must not silently replace the primary evidence.

### 16.8 Polysemanticity caution

Interpret a predictive direction as a low-dimensional carrier of information, not as a pure semantic concept unless stronger causal evidence supports that interpretation.

## 17. Reproducibility Requirements

Every executable run should record:

- code commit;
- config file;
- model ID and revision;
- dataset split;
- seeds;
- device;
- dtype;
- tokenizer/chat template version where relevant;
- decoding parameters;
- checkpoint-selection rule;
- prompt version;
- output location.

## 18. Development vs Final Compute

### Local development

Apple Silicon / MPS is used for:
- smoke tests;
- correctness checks;
- small-model validation;
- schema validation;
- unit/integration tests.

### Final experiments

CUDA systems are used for:
- Qwen3-8B and larger;
- full hidden-state logging;
- repeated rollouts;
- final reported results.

MPS-only behavior must not be used as the scientific basis for final reported performance.
