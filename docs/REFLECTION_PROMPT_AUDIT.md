# Reflection-Prompt Fidelity Audit

> **Status:** Development-time prompt tooling, not a research result.
> **Date:** 2026-09-24
> **Purpose:** Decide which reflection instruction best implements the intended
> intervention. Selection is by intervention fidelity, never by accuracy or utility.

The intended intervention is:

> Review the reasoning-so-far, make a local correction if necessary, and
> continue from the current state.

It should **not** normally cause the model to restart the problem, discard the
prefix, or reproduce all previous reasoning.

## Prompt definitions

| Variant | Instruction | Intervention tokens |
|---|---|---:|
| `baseline` | `"\n\nCarefully review the reasoning so far for mistakes, then continue the solution."` | 16 |
| `local_review` | `"Review the reasoning above for any mistakes. Do not restart the problem from the beginning. If the reasoning so far is correct, continue from the current point. If you find an error, briefly correct it and continue from the current point."` | 47 |
| `concise_local_review` | `"Check the reasoning so far for errors, then continue from the current point. Do not restart the solution."` | 21 |

Stored in `configs/reflection_prompt_audit.yaml`; none are hard-coded in Python.

Note the formatting asymmetry: `baseline` carries a leading blank line that the
other two do not. Stored prefixes already end in a paragraph break. This was
tested as a confound; see *Supplementary check* below.

## Audit setup

- States: the 18 eligible checkpoints already stored in
  `outputs/dev/math_utility_dataset.json`. **No base trace was regenerated**, so
  the reflection wording is the only treatment variable.
- Each state contributes its identical stored `shared_prefix_token_ids`.
- One reflection continuation per prompt per state; **no direct branches**.
- seed = 42 for every variant at every state; temperature 0.7, top_p 0.95,
  max_new_tokens 1024, `do_sample: true`.
- 18 states x 3 variants x 1 continuation = **54 generations**, runtime **2:54**.
- Model: Qwen/Qwen2.5-0.5B-Instruct, bfloat16, MPS.

**Invariants: 54/54 `shared_prefix_preserved`, 54/54
`intervention_after_shared_prefix`.**

Consistency check: the `baseline` variant at seed 42 reproduced the
already-stored reflect rollout token-for-token at **18/18** states, confirming
the audit replays the same states under the same decoding path.

Fidelity labels were assigned by direct human inspection of the saved
continuations. No classifier was trained and no model was used as a judge.
Labels are persisted in the `fidelity_label` field of
`outputs/dev/reflection_prompt_audit.json`.

## Aggregate fidelity table

| Prompt | Continue | Local correction | Full restart | Repetition | Empty | Uncertain | Fidelity (C+LC) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 9 | 1 | **8** | 0 | 0 | 0 | **10/18 (56%)** |
| `local_review` | 2 | 0 | 0 | 0 | **16** | 0 | 2/18 (11%) |
| `concise_local_review` | 6 | 1 | 2 | 0 | **9** | 0 | 7/18 (39%) |

### Secondary descriptive statistics

| Prompt | Mean tokens | Completion rate | Answered | No answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| `baseline` | 379 | 18/18 | 18/18 | 0/18 | 7/18 |
| `local_review` | 34 | 18/18 | 2/18 | 16/18 | 0/18 |
| `concise_local_review` | 161 | 18/18 | 8/18 | 10/18 | 1/18 |

Accuracy is descriptive context only and was **not** used for selection. It is
in any case confounded: `baseline` is the only variant that reliably generates
anything at all.

**The completion rate is uninformative here.** `generation_complete` means
"terminated by EOS", and an immediate EOS satisfies it. All three variants score
18/18 while two of them mostly produce nothing.

## Representative examples

### `baseline` — full restart (8/18)

Self-announced restarts that re-derive material already established in the prefix:

- `math_train_00000 late` (pos 0.786), prefix ends `... -4 - b = -7.`:
  > " Let's restate the steps:\n\n1. **Define the function:**\n \\[ f(x) = ... \\]"
- `math_train_00001 middle` (pos 0.448):
  > " I'll restate and correct my steps:\n\nGiven the conditions, let's break down the logic more clearly:\n\n1. There are fewer than 100 members. ..."
- `math_train_00004 random` (pos 0.607), prefix ends mid-substitution:
  > " ... We start again ensuring accuracy.\n\nLet's restate the steps correctly:\n\n1. \\( 60x - 30y = 660 \\)\n2. \\( x + y = 20 \\)"
- `math_train_00002 late` (pos 0.822):
  > " Let's restate the steps clearly:\n- Start with the given polynomial: \\( P(x) = 4 + 5x^3 + 100 + ... \\)"

### `baseline` — local correction (1/18)

- `math_train_00001 random` (pos 0.660). The prefix wrongly asserts that
  `8(8+4) = 96` is "greater than 100". The continuation targets exactly that step:
  > " We know from our earlier calculations: ... This indicates an error in the initial assumption. Rechecking confirms that for \\( m = 8 \\) we have: \\[ 8(8+4) = 8 \\times 12 = 96 < 100 \\]"

  This is the intended operator: local error identified, corrected, continued.

### `baseline` — continue (9/18)

- `math_train_00004 late` (pos 0.787), prefix ends `x = 14`:
  > " Substitute \\( x = 14 \\) back to find \\( y \\): \\[ y = 20 - 14 = 6 \\] ..."

### `local_review` — empty / immediate EOS (16/18)

16 of 18 continuations are the single token `151645` (`<|im_end|>`), decoding to
the empty string. The intervention produced no continuation at all.

The 2 substantive cases do not act on the instruction either; they **continue
writing instruction text**:

- `math_train_00000 early`:
  > " Once corrected or concluded, return to the previous line of reasoning for clarity.\n\nContinuing from the previous thought: ..."
- `math_train_00001 middle`:
  > " Once corrected, re-run the process.\nLet's try \\( m = 98 \\): ..."

### `concise_local_review` — full restart (2/18) and the negation failure

9 of 18 are immediate EOS. Of the 9 that generate, **almost every one opens with
the literal phrase "Start again"** — despite the instruction ending
`"Do not restart the solution."`:

- `math_train_00000 random`: `" Start again:\n\n1. **Identify the discontinuity points:** ..."` (full restart)
- `math_train_00001 early`: `" Start again from:\n\nWe start by noting that the total number of band members can be represented as \\( m \\times r \\) ..."` (full restart)
- `math_train_00004 middle`: `" Start again from where I left off:\nLet's solve the system of equations step-by-step: ..."` (continue)

The prohibition appears to **prime the behaviour it forbids**.

### `concise_local_review` — local correction (1/18)

- `math_train_00001 late` (pos 0.763):
  > " ... let's verify if there might be an error: \n\nChecking further, \\( m=8 \\) would be: \\[ 8(8+4)=8 \\cdot 12 = 96 \\] which is correct."

## Supplementary check: was the leading blank line the cause?

`baseline` carried a leading `\n\n` that the two new prompts lacked. To test
whether the immediate-EOS collapse was a formatting artifact rather than a
wording effect, the identical 18 states were rerun with a matched leading
separator (`configs/reflection_prompt_audit_separator.yaml`,
`outputs/dev/reflection_prompt_audit_separator.json`, 54 generations, 2:31).

| Variant | Empty/EOS, no separator | Empty/EOS, separator matched |
|---|---:|---:|
| `baseline` (control) | 0/18 | 0/18 |
| `local_review` | 16/18 | **17/18** |
| `concise_local_review` | 9/18 | **12/18** |

**The hypothesis is rejected.** Matching the separator did not help and was
marginally worse. The collapse is intrinsic to the wording, not to formatting.
The "Start again" opening persists under the separator-matched condition too.

Only the objective empty/EOS counts are reported for this supplementary run;
full fidelity labels were assigned only for the primary 54.

## Conclusions

### 1. Which prompt best implements the intended intervention

**None of the three is acceptable, and I recommend freezing none of them.**

On the stated criteria, `baseline` is the least-bad: it has the highest
`continue + local_correction` proportion (10/18) and a zero empty rate. But its
44% full-restart rate is precisely the defect that motivated this audit, so
"least-bad" is not "good".

`local_review` posts a 0/18 full-restart rate, but this is a **degenerate win**:
it never restarts because it produces nothing 89% of the time. A prompt that
emits an empty continuation is not implementing "review locally, correct
locally, then continue" — it implements no intervention whatsoever. Under the
selection criteria, an empty continuation is a total fidelity failure, not a
neutral outcome.

`concise_local_review` sits between the two and fails both ways: 50% empty and
an explicit negation that backfires.

### 2. Why, on fidelity rather than accuracy

The accuracy ordering (`baseline` 7/18, `concise` 1/18, `local_review` 0/18)
matches the fidelity ordering here, but that is a coincidence of the failure
mode: variants that generate nothing cannot answer. No selection weight was
placed on accuracy, token count, or utility. The decision rests on the
restart/empty/continue counts above.

### 3. Remaining ambiguity and failure modes

- **Immediate EOS after an instruction-shaped sentence.** Both new prompts read
  as a complete second-person command, and the model treats the instruction as
  the end of its own assistant turn. Not a formatting artifact (tested above).
- **Negation priming.** `"Do not restart the solution."` reliably produces
  `"Start again:"`. Negative imperatives should be avoided.
- **Instruction echo.** The model sometimes continues writing *instructions*
  (`"Once corrected, re-run the process."`) rather than reasoning. The
  intervention is inserted as raw text inside the assistant turn with no role
  delimiter — required by the exact-prefix invariant — so a second-person
  command is out of register. First-person monologue phrasing is untested and is
  the most promising next direction.
- **Label boundary cases.** A few `baseline` continuations open with
  restart-sounding language but then genuinely continue (for example
  `math_train_00003 early`: *"Let's walk through it again"* followed by step 2
  reusing the prefix result). These were labelled `continue_current_path`
  because the criterion is whether prefix work is discarded, not the opening
  phrasing. Reviewers may reasonably disagree on 2-3 of the 18 baseline labels;
  none would change the ordering.
- **Single sample per state.** k=1 at one seed. Rates are indicative, not
  precise.
- **Small model.** Qwen2.5-0.5B-Instruct may be unusually sensitive to
  instruction register. These conclusions should be re-checked on the target
  model before any prompt is frozen for reported experiments.

### 4. Freezing recommendation

**Do not freeze any of these three.** Keep `baseline` in place as the incumbent
so the main configuration is unchanged, and run a second audit round before
freezing.

Consequence if `local_review` had been adopted without this audit: 16 of 18
reflect rollouts would have been `no_answer`, scoring reward 0 under the frozen
taxonomy, and every state's `U_hat` would have collapsed toward -1 for a purely
artifactual reason.

Suggested next-round candidates, for approval rather than adoption — all
first-person, positively phrased, and ending on a continuation cue:

- `"\n\nWait, let me check the last few steps before continuing.\n\n"`
- `"\n\nBefore continuing, let me verify the steps above.\n\n"`
- `"\n\nLet me double-check the work so far, then keep going from here.\n\n"`

The main experiment's reflection prompt has **not** been modified.

---

# Part 2 — Intervention-Format Audit

> **Status:** Development-time prompt tooling, not a research result.
> **Follows from Part 1:** none of the three wording variants was frozen. Part 1
> showed that explicit second-person local-review instructions collapse into
> immediate EOS, and that this was not a separator artifact, pointing at
> intervention *register/format* rather than wording alone.

## Audit setup

- The same 18 stored eligible states from the 5-problem development run. **No
  base trace regenerated**, no direct branch generated.
- 4 conditions x 18 states x 2 seeds = **144 generations**, runtime **13:19**.
- seeds `[42, 43]`, temperature 0.7, top_p 0.95, max_new_tokens 1024, identical
  across all conditions.
- Config: `configs/intervention_format_audit.yaml`. Output:
  `outputs/dev/intervention_format_audit.json`.

### Conditions

| Condition | Format | Instruction | Intervention tokens |
|---|---|---|---:|
| `baseline` | `assistant_text` | `"\n\nCarefully review the reasoning so far for mistakes, then continue the solution."` | 16 |
| `assistant_monologue_minimal` | `assistant_text` | `"Let me verify the reasoning so far and continue carefully from here."` | 13 |
| `assistant_monologue_local` | `assistant_text` | `"Let me check the reasoning so far for a local mistake. If I find one, I'll correct it and continue from here."` | 26 |
| `user_turn_local` | `user_turn` | `"Briefly check the reasoning so far for any local mistake, then continue from the current point."` | 29 |

### Invariants

| Check | Result |
|---|---|
| `shared_prefix_preserved` | **144/144** |
| `intervention_after_shared_prefix` | **144/144** |
| `input_length_matches` (nothing inserted before the boundary) | **144/144** |
| `stored_states_unmutated` | **True** |
| `baseline` seed 42 reproduces the Part 1 wording audit | **18/18** |

Together the first three establish that the pre-treatment prefix is
integer-identical and that every intervention token, role-transition tokens
included, occurs strictly after the boundary.

**Role-transition tokens are derived from the tokenizer's own chat template**,
never hand-written: a probe conversation is rendered with and without the extra
user turn and the difference is taken. Two guards run before use — the extended
rendering must extend the assistant turn in place, and the derived tail must
tokenize identically standalone and in context. The result is 29 tokens
beginning `<|im_end|> \n <|im_start|> user`, verified stable across probe
contents including LaTeX and trailing blank lines.

## Aggregate fidelity table (pooled over both seeds, n=36 per condition)

| Condition | Continue | Local correction | Full restart | Repetition | Empty | Other | Instruction echo | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 21 | 1 | **14** | 0 | 0 | 0 | 0 | 22/36 (61%) |
| `assistant_monologue_minimal` | 25 | 1 | 6 | 0 | 0 | 4 | 0 | 26/36 (72%) |
| `assistant_monologue_local` | 25 | 2 | **5** | 0 | 0 | 4 | 0 | **27/36 (75%)** |
| `user_turn_local` | 15 | 2 | **18** | 1 | 0 | 0 | 0 | 17/36 (47%) |

**Immediate EOS is eliminated in all four conditions (0/144).** The Part 1
collapse does not reappear once the instruction is either in first-person
monologue register or in a properly role-delimited user turn.

**Instruction echo is also eliminated (0/144).** The Part 1 failure where the
model continued writing *instructions* did not recur in any condition.

## Seed-level breakdown

| Condition | Seed | Continue | Local corr | Full restart | Rep | Other | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 42 | 9 | 1 | 8 | 0 | 0 | 10/18 |
| `baseline` | 43 | 12 | 0 | 6 | 0 | 0 | 12/18 |
| `assistant_monologue_minimal` | 42 | 11 | 1 | 2 | 0 | 4 | 12/18 |
| `assistant_monologue_minimal` | 43 | 14 | 0 | 4 | 0 | 0 | 14/18 |
| `assistant_monologue_local` | 42 | 10 | 1 | 3 | 0 | 4 | 11/18 |
| `assistant_monologue_local` | 43 | 15 | 1 | 2 | 0 | 0 | 16/18 |
| `user_turn_local` | 42 | 7 | 1 | 10 | 0 | 0 | 8/18 |
| `user_turn_local` | 43 | 8 | 1 | 8 | 1 | 0 | 9/18 |

**The ordering is stable across seeds.** Both monologue conditions beat
`baseline` on full-restart rate at both seeds (2-4 vs 6-8), and `user_turn_local`
is worst at both seeds (10 and 8). The two monologue conditions are within noise
of each other and swap places between seeds.

## Completion and token statistics (descriptive only)

| Condition | Mean tokens | Empty | Answered | Correct |
|---|---:|---:|---:|---:|
| `baseline` | 337 | 0/36 | 34/36 | 10/36 |
| `assistant_monologue_minimal` | 334 | 0/36 | 32/36 | 10/36 |
| `assistant_monologue_local` | 337 | 0/36 | 34/36 | 8/36 |
| `user_turn_local` | 399 | 0/36 | 27/36 | 10/36 |

Accuracy spans 8-10 of 36 across all four conditions and played no part in
selection.

## Representative examples

### `user_turn_local` — the new turn resets the frame (18/36 full restarts)

Opening a user turn makes the model treat the request as a fresh problem:

- `math_train_00001 early` seed 42: *"Let's restate the problem clearly and carefully examine the process:\n\nThe band formation is described as having \( m \) members in each row of \( r \) rows..."*
- `math_train_00004 early` seed 42: *"Let's restate the problem clearly and carefully examine the process:\n\n1. Sam is hired for a 20-day period.\n2. He earns $60 per day he works..."*
- `math_train_00000 early` seed 43: *"To find \(a + b\) for the piecewise function to be continuous, we need to ensure that the value of the function at \(x = -2\) matches..."* — restarts from the problem goal, discarding the prefix's limit work.

It also produced the audit's only `repetition_no_progress`
(`math_train_00002 late` seed 43), which restates the method and ends on the
non-answer `\boxed{\text{the maximum degree}}`.

Where it does behave, it can affirm the prefix cleanly — `math_train_00001 random`
seed 43: *"The reasoning so far was correct. We found that \( m(m+4) < 100 \) and tested successive values..."*

### `assistant_monologue_local` — continuation and local correction

- `math_train_00004 random` seed 42 (continue): *"Let's use substitution or elimination to solve the system correctly.\n\nStarting from:\n\[2x - y = 22\]\n\[y = 20 - x\]"* — picks up exactly where the prefix stopped.
- `math_train_00001 late` seed 42 (local correction): the prefix wrongly says `8(8+4) = 96` is "greater than 100"; the continuation replies *"let's try \( m = 8 \) again: ... 96, which is still not greater than 100"*, correcting that step and continuing.
- `math_train_00002 random` seed 43 (local correction): *"The key is to identify which term contributes to the highest degree, not just look at the overall sum of their contributions."* — targets a method error rather than re-deriving.
- `math_train_00004 late` seed 43 (continue): *"But let's verify the final answer by checking if \( x = 14 \) satisfies both conditions..."*

### `assistant_monologue_minimal` — clean continuation

- `math_train_00000 late` seed 42: *"I'll use this equation to find the relationship between \(a\) and \(b\): ... \[-4 - b = -7.\] Solving for \(b\): \[b = 3.\]"* — uses the prefix's own equation.
- `math_train_00001 late` seed 42 (local correction): *"For \( m = 8 \): \[8(8+4) = 96,\] which is within the limit but not larger than 100"* — silently contradicts the prefix's error.

### New failure mode: hallucinated code blocks

Both monologue conditions produced continuations whose substance is a Python
block the model cannot execute (4/36 each, 8/72 = 11% of monologue generations;
0/36 in `baseline` and `user_turn_local`). Labelled `other_or_uncertain`:

- `assistant_monologue_local`, `math_train_00001 early` seed 42: *"Let's use Python code to verify the steps and find the largest possible value of \( m \).\n```python\n# We will start checking from the largest possible value of r..."* — ends inside the code fence with no answer.
- `assistant_monologue_minimal`, `math_train_00002 middle` seed 42: *"I'll use Python code for precise verification.\n\nHere's the Python code using SymPy..."*

All 8 occurred at seed 42 and none at seed 43, so the rate is unstable and may
be specific to this small model's tool-use priors.

## Model-specific formatting observations

- Qwen2.5-0.5B-Instruct's template injects a default system prompt; the stored
  prefixes therefore already contain one `<|im_end|>`, so "no intervention
  tokens before the boundary" is established by exact prefix equality plus
  length matching rather than by scanning for control-token ids.
- The role-transition tail is context-independent for this template, which is
  what makes the `user_turn` format safe to build once and reuse.
- Assistant-turn text interventions need no separator: stored prefixes already
  end at a paragraph break (a consequence of the paragraph-boundary
  segmentation rule).

## Recommendation

**Freeze `assistant_monologue_local` (condition C) for the next development run.**

Why, on fidelity rather than accuracy:

1. **Lowest full-restart rate** (5/36, 14%) against `baseline`'s 14/36 (39%) —
   this was the defect that motivated both audits, and it is cut by roughly
   two-thirds.
2. **Highest combined continue + local-correction rate** (27/36, 75%).
3. **Zero empty/immediate-EOS and zero instruction echo**, the two failure modes
   that disqualified every Part 1 candidate.
4. **Most local corrections** (2/36) and the only condition producing them at
   both seeds — it is the one wording that names the intended operator
   ("check for a local mistake, correct it, continue").
5. Accuracy is 8/36, the *lowest* of the four. It was not used, and the
   recommendation stands against it.

**Honest caveat:** `assistant_monologue_local` and `assistant_monologue_minimal`
are within noise of each other (27 vs 26 fidelity, 5 vs 6 restarts, identical
code-block rate) and they swap rank between seeds. The tie is broken on the
qualitative grounds in point 4, not on the counts. Either would be defensible;
if the next run shows the code-block failure growing, `minimal` is the natural
fallback.

**Remaining failure modes:** the 11% hallucinated-code rate in monologue
conditions is new and should be monitored — it yields `no_answer` under the
frozen reward taxonomy. Restarts are reduced but not eliminated (5/36). One
sample per state per seed at k=2 means these rates carry real uncertainty.
All of this is on a 0.5B model and must be re-checked on the target model before
any prompt is frozen for reported experiments.

### Note on the neutral control

`user_turn_local` did **not** win, so the neutral control can stay an
assistant-text intervention, length-matched in tokens to
`assistant_monologue_local` (26 tokens). Had the role-delimited condition won,
the neutral control would have had to use a **matched role-delimited user turn**
so that the generic effect of introducing a new conversational turn was
controlled rather than confounded with reflection. That requirement is recorded
here in case the format decision is revisited.

**`configs/math_dev_utility.yaml` has not been modified.** This is a
recommendation only, pending approval.
