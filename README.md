# WhenToReflect

Studying when language models should reflect, and whether reflection improves
downstream reasoning outcomes.

## Setup

Use Python 3.11 or newer. Create an isolated environment and install the
project with development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch selects the available accelerator at runtime. Development is intended
for Apple Silicon with MPS; final experiment runs should use a CUDA-enabled
PyTorch installation on the target machine.

## Layout

- `src/when_to_reflect/`: importable project code.
- `configs/`: versioned experiment settings, one file per runnable entry point.
- `scripts/`: command-line entry points as experiments are added.
- `tests/`: automated checks.

## Checks

```bash
pytest
ruff check .
python scripts/smoke.py
python scripts/run_utility_dataset.py
```

`scripts/smoke.py` reads `configs/smoke.yaml`, reports accelerator support,
performs a tensor operation, and runs one deterministic Qwen generation. It
verifies the runtime; it does not implement an experiment.

`scripts/prepare_math_data.py` downloads and prepares the local MATH training
data under `data/raw/math_train`.

## Experiments

`scripts/run_utility_dataset.py` is the single experiment entry point. It reads
`configs/math_dev_utility.yaml` by default, or a config path given as its first
argument. For each source problem it generates a deterministic base trajectory,
segments it into token-preserving reasoning states, selects early/middle/late
and one seeded-random checkpoint from the eligible interior candidates, forks
each state into matched-seed direct and reflection rollouts, grades them, and
estimates state-level reflection utility.

Key properties:

- direct and reflection branches start from the identical stored token prefix;
- rollout `i` of both actions uses the same seed, so the actions are compared
  under common random numbers;
- `U_hat` is a state-level quantity and is never aggregated to problem level;
- a state's `U_hat` is null unless every planned rollout yielded a reward, and
  each unavailable rollout records an explicit outcome.

Set `num_problems` in the config to change how many problems are run.
