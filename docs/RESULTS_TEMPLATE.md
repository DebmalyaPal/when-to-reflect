# WhenToReflect — Results and Figure Templates

> **Important:** All tables below are intentionally blank.  
> They specify planned analyses and reporting structure; they are not observed results.

---

## Table 1 — Reflection Utility: Main Behavioral Result

| Model | Dataset | Continue Accuracy | Reflect Accuracy | Δ Accuracy | Mean \(\hat{U}\) | % Helpful | % Neutral | % Harmful | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-8B | MATH-500 |  |  |  |  |  |  |  |  |
| Qwen3-8B | GPQA-Diamond |  |  |  |  |  |  |  |  |
| Qwen3-14B | MATH-500 |  |  |  |  |  |  |  |  |
| Qwen3-14B | GPQA-Diamond |  |  |  |  |  |  |  |  |
| DeepSeek-R1-Distill-Llama-8B | MATH-500 |  |  |  |  |  |  |  |  |
| DeepSeek-R1-Distill-Llama-8B | GPQA-Diamond |  |  |  |  |  |  |  |  |

---

## Table 2 — Helpful vs Harmful Reflection Transitions

| Model | Dataset | Incorrect → Correct | Correct → Incorrect | Correct → Correct | Incorrect → Incorrect |
|---|---|---:|---:|---:|---:|
|  |  |  |  |  |  |

---

## Table 3 — Layerwise Utility Decoding

| Model | Dataset | Layer | AUROC | AUPRC | Balanced Accuracy | Macro-F1 | Spearman \(\rho\) | \(R^2\) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |

---

## Table 4 — Utility Prediction vs Uncertainty Baselines

| Model | Dataset | Predictor | AUROC | AUPRC | \(R^2\) | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|
|  |  | Entropy only |  |  |  |  |  |
|  |  | Log-probability / margin |  |  |  |  |  |
|  |  | Self-consistency |  |  |  |  |  |
|  |  | Confidence features combined |  |  |  |  |  |
|  |  | Hidden-state utility probe |  |  |  |  |  |
|  |  | Hidden state + uncertainty |  |  |  |  |  |

---

## Table 5 — Incremental Value Beyond Uncertainty

| Model | Dataset | Uncertainty Baseline AUROC | Combined AUROC | Δ AUROC | Uncertainty \(R^2\) | Combined \(R^2\) | Δ \(R^2\) |
|---|---|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |

---

## Table 6 — Matched-Uncertainty Analysis

| Model | Dataset | Uncertainty Band | Mean Utility: High-Utility Group | Mean Utility: Low-Utility Group | Probe Separation | CI / p-value |
|---|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |  |

---

## Table 7 — Probe Performance by Trajectory Stage

| Model | Dataset | Stage | Best Layer | AUROC | \(R^2\) | N States |
|---|---|---|---:|---:|---:|---:|
|  |  | Early |  |  |  |  |
|  |  | Middle |  |  |  |  |
|  |  | Late |  |  |  |  |

---

## Table 8 — Controller Comparison

| Model | Dataset | Policy | Accuracy | Reasoning Tokens | Δ Tokens vs Never Reflect | Correct→Wrong Rate | Incorrect→Correct Rate |
|---|---|---|---:|---:|---:|---:|---:|
|  |  | Never reflect |  |  |  |  |  |
|  |  | Always reflect |  |  |  |  |  |
|  |  | Random matched budget |  |  |  |  |  |
|  |  | Uncertainty threshold |  |  |  |  |  |
|  |  | Native reflection |  |  |  |  |  |
|  |  | ReflCtrl-style control |  |  |  |  |  |
|  |  | Utility controller |  |  |  |  |  |
|  |  | Oracle utility |  |  |  |  |  |

---

## Table 9 — Accuracy at Matched Compute

| Model | Dataset | Token Budget | Never Reflect | Always Reflect | Uncertainty Policy | Utility Policy | Oracle Utility |
|---|---|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |

---

## Table 10 — Token Savings at Matched Accuracy

| Model | Dataset | Target Accuracy | Always Reflect Tokens | Uncertainty Policy Tokens | Utility Policy Tokens | Utility Savings vs Always |
|---|---|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |

---

## Table 11 — Prompt-Control Ablation

| Model | Dataset | Condition | Accuracy | Mean Utility | Token Cost | Correct→Wrong Rate |
|---|---|---|---:|---:|---:|---:|
|  |  | Direct continue |  |  |  |  |
|  |  | Neutral instruction |  |  |  |  |
|  |  | Reflection instruction |  |  |  |  |

---

## Table 12 — Position / Difficulty / Prefix-Length Controls

| Model | Dataset | Control Variable | Controlled Probe AUROC | Controlled \(R^2\) | Interpretation |
|---|---|---|---:|---:|---|
|  |  | Relative trajectory position |  |  |  |
|  |  | Problem difficulty |  |  |  |
|  |  | Prefix length |  |  |  |

---

## Table 13 — Linear vs Nonlinear Probe Capacity

| Model | Dataset | Probe | Parameters / Capacity | AUROC | \(R^2\) |
|---|---|---|---|---:|---:|
|  |  | Linear / logistic |  |  |  |
|  |  | Ridge / linear regression |  |  |  |
|  |  | Small MLP |  |  |  |
|  |  | Kernel baseline |  |  |  |

---

# Planned Figures

## Figure 1 — Experimental Schematic

**Purpose:** Introduce the counterfactual state-level setup.

Planned structure:

```text
same reasoning prefix s
        |
        +-------------------+
        |                   |
        v                   v
   continue             reflect
        |                   |
        v                   v
 repeated futures      repeated futures
        |                   |
        +---------+---------+
                  |
                  v
             U_hat(s)
```

Include:
- pre-treatment hidden state capture;
- identical branch point;
- repeated rollouts;
- final reward difference.

---

## Figure 2 — Distribution of Reflection Utility

Plot:
- histogram / density of \(\hat{U}(s)\);
- fraction helpful/neutral/harmful;
- optionally split by trajectory stage.

Question answered:
> Does reflection have heterogeneous effects?

---

## Figure 3 — Layerwise Decodability

Plot:
- layer on x-axis;
- AUROC and/or \(R^2\) on y-axis.

Question answered:
> Where is reflection-utility information accessible?

---

## Figure 4 — Layer × Trajectory-Stage Heatmap

Axes:
- model layer;
- early/middle/late reasoning stage.

Cell:
- probe performance.

Question answered:
> Does the representation evolve over both depth and reasoning time?

---

## Figure 5 — Reflection Utility vs Uncertainty

Possible views:
- scatter of utility predictor score vs entropy/confidence;
- conditional utility curves;
- matched-uncertainty groups.

Question answered:
> Is recoverability distinct from generic uncertainty?

---

## Figure 6 — Incremental Prediction Beyond Uncertainty

Plot:
- uncertainty-only;
- hidden-state-only;
- combined predictor.

Metric:
- AUROC / \(R^2\) with confidence intervals.

Question answered:
> Does the representation add information beyond uncertainty?

---

## Figure 7 — Accuracy vs Compute Pareto Curve

Policies:
- never reflect;
- always reflect;
- random matched budget;
- uncertainty policy;
- utility controller;
- oracle.

Axes:
- x: reasoning tokens / compute;
- y: final-answer accuracy.

Question answered:
> Does selective reflection improve the accuracy-compute trade-off?

---

## Figure 8 — Rescue vs Harm Breakdown

Plot:
- incorrect→correct rescue rate;
- correct→incorrect harm rate;
- by controller/baseline.

Question answered:
> Does the controller avoid harmful reflection while preserving recoverable cases?

---

## Figure 9 — Cross-Model / Cross-Domain Summary

Plot or compact matrix:
- rows: models;
- columns: datasets;
- cells: mean utility, best probe AUROC, controller gain.

Question answered:
> How general is the phenomenon?

---

# Planned Result Interpretation Checklist

Before writing conclusions, answer:

- [ ] Are helpful and harmful states both present?
- [ ] Do effects replicate across seeds/problems?
- [ ] Is utility decodable on held-out problems?
- [ ] Does hidden-state information add value beyond uncertainty?
- [ ] Is the layer/position pattern stable rather than a single noisy peak?
- [ ] Does a utility controller improve the accuracy-compute frontier?
- [ ] Does the result replicate across at least one additional domain/model family?
- [ ] Have correct→wrong failures been reported?
- [ ] Are all stronger causal/mechanistic claims supported by intervention evidence?
