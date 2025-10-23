#!/usr/bin/env bash
set -e

# Create a local venv in backend/.venv and install requirements
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Desired Python version for the project
PY_VER="3.11.4"
VENV_DIR=".venv"

# If the user provided PYTHON env var, respect it; otherwise try pyenv, then system python
if [ -n "$PYTHON" ]; then
  PYTHON_CMD="$PYTHON"
else
  if command -v pyenv >/dev/null 2>&1; then
    echo "pyenv found — ensuring Python $PY_VER is installed..."
    # install if missing (-s means skip if already installed)
    pyenv install -s "$PY_VER" || true
    # set a local project python which creates backend/.python-version
    echo "$PY_VER" > .python-version
    # determine python executable from pyenv
    PYTHON_CMD="$(pyenv prefix "$PY_VER")/bin/python3"
    if [ ! -x "$PYTHON_CMD" ]; then
      PYTHON_CMD="$(pyenv which python3 2>/dev/null || true)"
    fi
  else
    echo "pyenv not found. Falling back to system python. To get a local Python 3.11 install, install pyenv (recommended)." >&2
    echo "Install pyenv: https://github.com/pyenv/pyenv#installation" >&2
    PYTHON_CMD="python3"
  fi
fi

echo "Using python: ${PYTHON_CMD}"

if [ ! -x "$PYTHON_CMD" ]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_CMD" >&2
  echo "Please install Python $PY_VER or set the PYTHON env var to a Python 3.11 executable." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# activate and install
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
if python -m pip install -r requirements.txt; then
  echo "Requirements installed successfully."
else
  echo "Requirements installation failed. If the failure is about torch, try installing a CPU torch wheel manually:" >&2
  echo "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "Then re-run: python -m pip install -r requirements.txt" >&2
  exit 1
fi

echo "Backend venv ready at ${SCRIPT_DIR}/${VENV_DIR}"
echo "Activate with: cd ${SCRIPT_DIR} && source ${VENV_DIR}/bin/activate"
