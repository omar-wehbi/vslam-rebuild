#!/usr/bin/env bash
# Run a command with LD_LIBRARY_PATH set so loopclosuretoolbox can load libVLAD.so and libDBoW3.so.
# Usage: ./scripts/run_with_bow.sh python main.py --loop_detection_strategy bow ...
# Or from project root: ./scripts/run_with_bow.sh python -c "import loopclosuretoolbox.dbow; print('OK')"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Use site-packages so we don't need to import (import fails until LD_LIBRARY_PATH is set)
PKG_DIR=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -z "$PKG_DIR" ]] || [[ ! -f "$PKG_DIR/libVLAD.so" ]]; then
  echo "Error: loopclosuretoolbox libs not found. Run ./scripts/install_loopclosure_toolbox.sh first." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${PKG_DIR}:${LD_LIBRARY_PATH:-}"

exec "$@"
