#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Preparing GUM's private Python environment (first launch only)..."
  python3 -m venv .venv
fi

if [ ! -f .venv/gum-ready ]; then
  echo "Installing GUM's declared dependencies (first launch only)..."
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
  touch .venv/gum-ready
fi

echo "Starting GUM Studio. Press Ctrl+C to stop it."
.venv/bin/python -m gum --workspace .gum-workspace serve --release . --open-browser
