# WhenToReflect — Research Plan

> **Status:** Working research plan.  
> **Important:** Hypotheses, success criteria, and table/figure templates in this repository are planned analyses, not observed findings.

## 1. Working Title

**Beyond Uncertainty: Predicting the Utility of Reflection in Large Language Models**

Repository/project name: **WhenToReflect**

## 2. Core Thesis

A reasoning model may be uncertain without being recoverable through reflection, and it may sometimes benefit from reflection even when its confidence is relatively high.

The central question is:

> **Does the model's pre-reflection internal state contain information that predicts the expected benefit of reflecting from that exact reasoning state?**

The project shifts the control question from:

> "Is the model uncertain?"

to:

> **"From this exact reasoning state, is reflection worth doing?"**

## 3. Motivation

Reflection can have three qualitatively different effects:

1. **Helpful:** repair an error or recover a better reasoning trajectory.
2. **Neutral:** repeat or rephrase reasoning without changing the outcome.
3. **Harmful:** derail a correct or promising trajectory.

Common routing signals such as confidence, token entropy, problem difficulty, or reflective-language propensity are related to the decision, but they do not directly estimate the value of the reflection action itself.

This project therefore treats **reflection utility** as a state-level counterfactual quantity.

## 4. Unit of Analysis

The unit of analysis is an **intermediate reasoning state**, not an entire question.

Let \(s\) denote the full prefix available to the model at a selected reasoning-step boundary.

From the same state \(s\), compare two actions:

- **Continue:** proceed normally from \(s\).
- **Reflect:** induce review of the reasoning-so-far, then continue.

Primary task reward is final-answer correctness.

Define reflection utility as:

\[
U(s) = \mathbb{E}[R \mid \text{reflect}, s]
-
\mathbb{E}[R \mid \text{continue}, s]
\]

where \(R\) is primarily final-answer correctness.

Interpretation:

- \(U(s) > 0\): reflection is helpful.
- \(U(s) \approx 0\): reflection is neutral.
- \(U(s) < 0\): reflection is harmful.

## 5. Main Research Questions

### RQ1 — Phenomenon

**How does the utility of self-reflection vary across reasoning states, tasks, and language models?**

Primary test:
- estimate paired reflect-vs-continue utility from the same prefix using repeated rollouts;
- characterize the distribution of utility;
- replicate across math/science domains and multiple open-weight model families.

What a positive result would establish:
- reflection has heterogeneous treatment effects;
- always-reflect and never-reflect policies leave exploitable structure.

---

### RQ2 — Representation Beyond Uncertainty

**To what extent is reflection utility encoded in internal model representations, independently of uncertainty or confidence, and where does this information emerge?**

Primary test:
- train layerwise probes for continuous and categorical utility;
- compare with entropy/confidence baselines;
- test combined models;
- perform matched-uncertainty controls;
- map performance across layers, token positions, and trajectory stages.

What a positive result would establish:
- pre-reflection hidden states contain information about future reflection value;
- this information is not reducible to conventional uncertainty alone;
- the signal can be localized by layer/position.

---

### RQ3 — Decision Value

**Can internal estimates of reflection utility enable selective reflection that improves reasoning performance under a fixed computational budget?**

Primary test:
- use the utility predictor as a controller;
- compare against never-reflect, always-reflect, random, uncertainty-triggered, and native-reflection baselines;
- evaluate matched-compute accuracy and accuracy-vs-compute curves.

What a positive result would establish:
- the decoded signal has practical value for adaptive inference;
- selective routing can reduce wasted or harmful reflection.

## 6. Working Hypotheses

### H1 — Heterogeneous utility

Reflection utility has substantial state-level variance, including genuinely helpful and genuinely harmful states.

### H2 — Pre-action decodability

Utility is decodable from pre-reflection hidden states above chance on held-out problems.

### H3 — Incremental value beyond uncertainty

Activation-based utility prediction retains predictive value after uncertainty/confidence features are included.

### H4 — Layer/trajectory structure

Utility information is strongest at particular middle-to-late layers and changes over the course of a reasoning trajectory rather than residing in one fixed "reflection layer."

### H5 — Selective-control advantage

A utility-based controller produces a better accuracy-compute trade-off than always-reflect and uncertainty-threshold policies.

## 7. Conceptual Distinctions

### Reflection utility is not the same as uncertainty

Two states may be equally uncertain while differing in recoverability:

| Model state | Reflection can help | Reflection is unlikely to help |
|---|---|---|
| High uncertainty | A local mistake is detectable and recoverable. | The model lacks needed knowledge or is trapped in a bad representation. |
| Low uncertainty | The model is confidently following a flawed path that a check could expose. | The trajectory is already correct and stable. |

Therefore, uncertainty is a useful baseline, but not the target variable.

### Decodability is not causality

A successful probe supports the claim that utility-related information is **accessible in the representation**.

It does **not** by itself establish:
- a pure semantic "reflection utility direction";
- that the model causally uses the decoded feature;
- conscious intention;
- an internal mechanism of deliberative control.

Stronger causal claims require intervention evidence.

## 8. Scope

### Main models

| Role | Model | Purpose | Priority |
|---|---|---|---|
| Main development model | Qwen3-8B | Open weights, activation access, explicit thinking capability, manageable experimentation | Start here |
| Scale check | Qwen3-14B | Same family at larger scale | After pipeline is stable |
| Cross-family check | DeepSeek-R1-Distill-Llama-8B | Different backbone/training lineage | Strong validation model |
| Optional extension | Qwen3-32B or QwQ-32B | Larger-scale extension if compute permits | Last |

Open-weight models are required for the representation analyses.

### Datasets

| Dataset / split | Role |
|---|---|
| MATH training data | Build state-level train/development examples |
| MATH-500 | Primary held-out in-domain evaluation |
| GPQA-Diamond | Cross-domain science evaluation |
| AIME / GSM8K | Optional difficulty stress tests only if needed |

## 9. Contribution Ladder

The project is organized as a sequence of increasingly strong claims.

### Stage 1 — Phenomenon

Does reflection have heterogeneous effects across reasoning states?

### Stage 2 — Representation

Can pre-reflection activations predict whether reflection will help?

### Stage 3 — Beyond uncertainty

Does the representation retain predictive value after controlling for uncertainty/confidence?

### Stage 4 — Decision policy

Can the signal route reflection selectively?

### Stage 5 — Generality

Does the pattern persist across domains and model families?

## 10. Scope Boundaries and Sequencing

### Tier 1 — Phenomenon

Must contain:
- branching dataset;
- utility estimates;
- never-reflect and always-reflect baselines.

Defer:
- large model sweep;
- head-level analysis.

### Tier 2 — Representation

Must contain:
- layerwise linear probes;
- uncertainty comparisons;
- strict problem-level splits.

Defer:
- complex nonlinear probes;
- circuit-level claims.

### Tier 3 — Decision Value

Must contain:
- utility controller;
- accuracy-compute curves.

Defer:
- training a new reasoning model from scratch.

### Tier 4 — Depth

Must contain:
- cross-domain/model replication;
- matched-uncertainty controls;
- optional ReflCtrl-style intervention.

Defer unless the signal is strong:
- attention-head/circuit localization;
- TransformerLens-level mechanistic decomposition.

## 11. Immediate Experimental Sequence

1. Implement a reproducible Qwen3-8B runner with:
   - generation;
   - answer grading;
   - step segmentation;
   - hidden-state logging.
2. Pilot on **50–100 MATH training problems**.
3. Branch **3–4 checkpoints per problem**.
4. Use **\(k=4\)** continuations per action initially.
5. Verify that both positive-utility and negative-utility states exist.
6. Freeze reflection and neutral-control prompts after a small qualitative audit.
7. Train layerwise linear probes with problem-level train/dev splits.
8. Compare against entropy/log-probability and simple confidence baselines.
9. If RQ1–RQ2 are supported, run the utility controller and produce accuracy-vs-token curves.
10. Only then scale to:
    - MATH-500;
    - GPQA-Diamond;
    - Qwen3-14B;
    - DeepSeek-R1-Distill-Llama-8B.
11. Add component/head-level analysis only if it helps explain an already-established representation result.

## 12. Strongest Intended Contribution

If the full pattern holds, the paper can make three compact contributions:

1. **Counterfactual definition and dataset construction** for reflection utility at intermediate reasoning states.
2. **Representation evidence** that reflection utility is predictable from internal states beyond conventional uncertainty signals.
3. **Adaptive reflection control** that converts the decoded signal into a better accuracy-compute trade-off.

A weaker but still informative outcome is possible:
- utility may be heterogeneous but difficult to decode linearly;
- the signal may be domain-specific or model-specific;
- selective routing may not beat always-reflect at every budget.

The paper should state the strongest claim supported by the evidence rather than force a stronger narrative.
