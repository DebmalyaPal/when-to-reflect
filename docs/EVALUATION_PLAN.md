# WhenToReflect — Evaluation Plan

> **Purpose:** Pre-specify the analyses that answer RQ1–RQ3 before final test results are examined.

## 1. Evaluation Philosophy

Every experiment should answer the same decision question:

> **From this exact reasoning state, is reflection worth doing?**

The evaluation proceeds from:
1. behavioral phenomenon;
2. representation;
3. incremental value beyond uncertainty;
4. decision/control value;
5. generalization.

## 2. RQ1 — Reflection Utility and Scope

### Primary quantities

For each state \(s\):

\[
\hat{U}(s)
=
\bar{R}_{\text{reflect}}
-
\bar{R}_{\text{continue}}
\]

Report:

- distribution of \(\hat{U}\);
- mean utility;
- state-level variance;
- percentage helpful;
- percentage neutral;
- percentage harmful;
- reflect-vs-continue accuracy difference;
- correct-to-wrong rate after reflection;
- incorrect-to-correct rescue rate.

### Stratifications

Report core metrics by:
- dataset;
- model;
- trajectory stage;
- difficulty bucket where available;
- seed.

### Evidence target

Compelling evidence would be:
- non-trivial helpful and harmful regions;
- replication across seeds/problems;
- directionally consistent evidence on MATH-500 and GPQA-Diamond;
- evidence on at least two model families;
- measurable room between always-reflect and an ideal selective policy.

## 3. RQ2 — Internal Representation of Reflection Utility

### 3.1 Linear probe

Primary representation test:
- one linear/logistic probe per layer.

Categorical targets:
- helpful vs not-helpful;
- optional 3-way helpful/neutral/harmful.

Continuous target:
- \(\hat{U}(s)\).

Recommended primary methods:
- logistic regression for categorical utility;
- ridge/linear regression for continuous utility.

### 3.2 Low-capacity nonlinear probe

Only after the linear result is known:
- small MLP;
- kernel baseline.

Purpose:
- diagnose whether a weak linear result reflects nonlinear encoding.

It must not replace the linear probe as the primary interpretability result.

## 4. Uncertainty Baselines

Construct uncertainty-only predictors using combinations of:

- entropy;
- probability/log-probability margin;
- verbal confidence;
- self-consistency;
- optional internal-confidence probe.

Primary question:

> Does the hidden-state utility representation add predictive information beyond these features?

## 5. Combined Model

Compare:

1. uncertainty-only predictor;
2. hidden-state utility predictor;
3. uncertainty + hidden-state predictor.

The key quantity is incremental predictive value from the hidden-state features.

## 6. Matched-Uncertainty Analysis

Construct subsets/pairs of states with similar uncertainty but different estimated reflection utility.

Purpose:
- show that recoverability is not reducible to generic confidence.

Possible presentation:
- high-utility vs low-utility states within narrow entropy/confidence bands;
- conditional plots of utility probe score at matched uncertainty.

## 7. Layer and Trajectory-Position Mapping

Evaluate probe performance as a function of:

- layer;
- token/checkpoint position;
- normalized trajectory stage.

Planned outputs:
- layerwise AUROC / \(R^2\) curve;
- layer × trajectory-stage heatmap.

This analysis tests whether the signal:
- emerges gradually;
- peaks in particular regions;
- changes over reasoning time.

## 8. RQ2 Metrics

### Categorical utility

Primary candidates:
- AUROC;
- balanced accuracy;
- macro-F1;
- AUPRC where class imbalance matters.

### Continuous utility

Primary candidates:
- Spearman correlation;
- \(R^2\);
- optional MAE.

### Incremental value beyond uncertainty

Report:
- \(\Delta\) AUROC;
- \(\Delta R^2\);
- nested-model comparisons;
- matched-uncertainty separation.

### Calibration

Report when relevant:
- Brier score;
- expected calibration error (ECE).

## 9. RQ3 — Selective Reflection Controller

At each eligible checkpoint, a controller estimates expected reflection utility and triggers reflection only when the score exceeds a threshold chosen on development data.

Primary output:
- a curve over decision thresholds;
- accuracy vs compute/tokens.

## 10. Controller Baselines

| Method | Decision rule | Purpose |
|---|---|---|
| Never reflect | Always continue | Lower-compute reference |
| Always reflect / double pass | Reflect at every eligible opportunity or perform full second pass | Tests whether selective routing is necessary |
| Random at matched budget | Reflect on same fraction of states as controller, chosen randomly | Controls for spending more tokens |
| Uncertainty threshold | Reflect when entropy/low-confidence passes tuned threshold | Main adaptive baseline |
| Native reflection | Leave model unconstrained and detect spontaneous reflection | Tests natural routing behavior |
| ReflCtrl-style control | Use published reflection direction/suppression where implementation permits | Representation-engineering baseline |
| Utility controller | Reflect when predicted \(U(s)\) exceeds tuned threshold | Proposed method |
| Oracle utility | Reflect only when empirical counterfactual utility is positive | Non-deployable upper bound |

## 11. RQ3 Metrics

Report:

- final-answer accuracy;
- generated reasoning tokens;
- latency only if measured reliably;
- accuracy at matched token budget;
- token savings at matched accuracy;
- area under the accuracy-compute curve;
- correct-to-wrong rate;
- incorrect-to-correct rescue rate;
- missed-helpful-state rate.

### Evidence target

The proposed controller should improve or extend the accuracy-compute Pareto frontier relative to:
- uncertainty-triggered policies;
- always-reflect policies.

## 12. Statistical Reporting

### Primary resampling unit

Use the **problem** as the primary resampling unit.

Reason:
- multiple states and rollouts from one problem are correlated.

### Confidence intervals

Use problem-level bootstrap confidence intervals for:
- accuracy;
- utility;
- controller comparisons.

### Paired comparisons

For paired final-answer comparisons:
- paired bootstrap differences;
- McNemar tests where appropriate.

### Model selection discipline

Probe hyperparameters and controller thresholds are selected using development problems only.

Final test sets remain untouched until:
- analysis definitions;
- metrics;
- thresholds;
- prompt versions;
- checkpoint rules

are frozen.

## 13. Failure-Avoidance Analyses

Always report:

- correct-to-wrong reflection rate;
- incorrect-to-correct rescue rate;
- missed-helpful-state rate;
- harmful reflection frequency.

A controller that improves mean accuracy but introduces avoidable harmful reflection should be examined explicitly.

## 14. Interpretation Rules

### Pattern: reflection helps some states and hurts others

Supported:
- reflection has heterogeneous treatment effects.

Do not claim:
- the model itself knows which states are recoverable.

### Pattern: linear probe predicts utility well

Supported:
- pre-reflection activations contain linearly accessible information about reflection utility.

Do not claim:
- a pure semantic "utility concept";
- causal use by the model.

### Pattern: nonlinear probe works, linear probe is weak

Supported:
- utility information may be distributed or nonlinearly encoded.

Do not claim:
- a simple steerable direction.

### Pattern: utility probe beats uncertainty baselines

Supported:
- recoverability contains information not captured by the tested uncertainty measures.

Do not claim:
- uncertainty is irrelevant.

### Pattern: native reflection tracks utility

Supported:
- spontaneous reflection policy is partially aligned with future benefit.

Do not claim:
- full optimality;
- deliberate conscious planning.

### Pattern: native reflection is weakly related to utility

Supported:
- there may be a routing gap between available information and natural reflection decisions.

Do not claim:
- a causal internal mechanism without intervention evidence.

### Pattern: utility controller improves matched-compute accuracy

Supported:
- decoded signal has practical decision value.

Do not claim:
- the controller reveals the model's natural mechanism.

### Pattern: always-reflect matches/beats controller at every budget

Supported:
- selective routing has limited practical value under that operator/model regime.

Do not force:
- an efficiency claim.

### Pattern: math works, GPQA fails

Supported:
- signal/operator may be domain-sensitive.

Do not claim:
- a domain-general metacognitive mechanism.

### Pattern: model families differ strongly

Supported:
- utility representation is model/training dependent.

Do not claim:
- a universal representation.

## 15. Optional Mechanistic Depth

Only after the main representation result is established:

- attention-output vs MLP-output decomposition;
- head-level analysis;
- TransformerLens-style localization;
- causal interventions.

These analyses are extensions, not prerequisites for the core paper.
