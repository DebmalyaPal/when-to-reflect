#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " WhenToReflect — Initial Setup"
echo "=========================================="

cd "${PROJECT_ROOT}"

# --------------------------------------------------
# 1. Check Python
# --------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found."
    exit 1
fi

echo "Python:"
python3 --version


# --------------------------------------------------
# 2. Create virtual environment
# --------------------------------------------------

if [[ ! -d "${VENV_DIR}" ]]; then
    echo
    echo "Creating virtual environment at .venv ..."
    python3 -m venv "${VENV_DIR}"
else
    echo
    echo ".venv already exists — reusing it."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Using Python:"
which python


# --------------------------------------------------
# 3. Upgrade packaging tools
# --------------------------------------------------

echo
echo "Upgrading pip/setuptools/wheel ..."

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


# --------------------------------------------------
# 4. Install repository
# --------------------------------------------------

echo
echo "Installing WhenToReflect dependencies ..."

python -m pip install -e '.[dev]'


# --------------------------------------------------
# 5. Verify required packages
# --------------------------------------------------

echo
echo "Verifying core dependencies ..."

python - <<'PY'
import torch
import transformers
import datasets
import yaml
import numpy

print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"datasets     : {datasets.__version__}")
print(f"numpy        : {numpy.__version__}")
print("Core dependencies OK.")
PY


# --------------------------------------------------
# 6. Prepare directories
# --------------------------------------------------

echo
echo "Creating local data directories ..."

mkdir -p \
    data/raw \
    data/processed \
    outputs \
    logs


# --------------------------------------------------
# 7. Download and prepare MATH training data
# --------------------------------------------------

echo
echo "Downloading/preparing MATH training data ..."

python scripts/prepare_math_data.py


# --------------------------------------------------
# 8. Run tests
# --------------------------------------------------

echo
echo "Running tests ..."

python -m pytest -q


# --------------------------------------------------
# 9. Finished
# --------------------------------------------------

echo
echo "=========================================="
echo " Setup complete"
echo "=========================================="
echo
echo "Activate the environment with: source .venv/bin/activate"
echo
echo "Prepared MATH data: data/raw/math_train"
echo
