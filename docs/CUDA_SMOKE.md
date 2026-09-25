# Qwen3-8B CUDA smoke (UCSD A5000)

The first commands to run after moving this repository to the A5000 box. They
exercise the whole pipeline — base trace, segmentation, checkpoint selection,
exact-prefix branching, paired rollouts, grading, `U_hat`, pre-treatment
activation capture — on `Qwen/Qwen3-8B`, with **no source changes**.

Device selection is automatic: `select_device()` prefers CUDA, then MPS, then
CPU, so no config names a backend.

## Setup

```bash
./setup.sh                                          # venv + dependencies
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python scripts/prepare_math_data.py       # writes data/raw/math_train
```

`pyproject.toml` pins only `torch>=2.5`, and on Linux the default PyPI wheel is
CUDA-enabled — but check the line above prints `True` before going further. If
it prints `False` the runs will silently fall back to CPU, which defines
different states.

## 1. Utility smoke

```bash
.venv/bin/python scripts/run_utility_dataset.py configs/math_dev_utility_qwen3_8b_cuda.yaml
```

Writes `outputs/cuda_smoke/qwen3_8b/math_utility_dataset_smoke.json`.

Expect: `device: cuda`, `dtype: torch.bfloat16` in the record's provenance, a
base trace that terminates on EOS, four selected checkpoints, and a `U_hat` per
state. The run fails loudly if any configured decoding setting is not the one
`generate` applies.

## 2. Activation smoke

```bash
.venv/bin/python scripts/extract_activations.py configs/activations_qwen3_8b_cuda.yaml
```

Writes `outputs/cuda_smoke/qwen3_8b/activations/{activations.npz,manifest.json}`.

Expect `hidden_states` of shape `[num_states, 37, 4096]` — 36 transformer
layers plus the embedding output — at roughly 0.3 MB per state.

The extractor refuses to run if the states in the input record were generated
on a different backend, so run step 2 on the same machine as step 1.

## 3. Checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

## Notes

- `num_rollouts: 2` in the smoke config is a **compatibility setting only**.
  Restore `4` before any run whose utilities are meant to be read.
- Memory on one A5000 (24 GB): ~16.4 GB of bfloat16 weights plus a small KV
  cache. `load_causal_lm` materializes the weights on the host before moving
  them to the GPU, so ~17 GB of free system RAM is also needed.
- Everything scientific is shared with the Qwen3-0.6B development configs:
  checkpoint policy and eligibility band, the frozen
  `assistant_monologue_local_v1` operator, `enable_thinking: false`, and the
  explicit decoding parameters.
- Traces, checkpoints, branches, utilities and activations must all be produced
  on the same backend. CUDA states are not interchangeable with the MPS
  development states.
