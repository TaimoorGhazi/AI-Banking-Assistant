#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv

if [ -f ".venv/Scripts/activate" ]; then
  # Windows Git Bash / MSYS2 layout
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
else
  # POSIX layout
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm || true
