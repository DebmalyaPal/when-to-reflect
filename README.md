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
- `configs/`: versioned experiment settings; `base.yaml` holds shared defaults.
- `scripts/`: command-line entry points as experiments are added.
- `tests/`: automated checks.

## Checks

```bash
pytest
ruff check .
```

No experiment implementation is included yet.
