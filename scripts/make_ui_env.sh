#!/usr/bin/env bash
# Build .venv-ui: the SAME pinned libraries as the verification environment (requirements.lock)
# plus the real Pangolin bindings, so a viewer run uses the verified configuration.
#
#   bash scripts/make_ui_env.sh
#   .venv-ui/bin/python main.py --dataset ... --default_k --ground_truth ... --timestamp_file ...
#
# The verifier's .venv keeps its headless pypangolin stub on purpose (it must never open a window);
# this environment is for the human's eyes only and nothing in the loop uses it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[[ -d .venv-ui ]] || "$PY" -m venv .venv-ui
. .venv-ui/bin/activate
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.lock
SITE=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
# Real Pangolin: the system build at /usr/local (pypangolin.cpython-3XY-*.so). Link it in.
PANGO=$(ls /usr/local/lib/python3*/dist-packages/pypangolin*.so 2>/dev/null | head -1 || true)
if [[ -z "$PANGO" ]]; then echo "real pypangolin not found under /usr/local/lib/python3*/dist-packages"; exit 2; fi
PYTAG=$(python -c "import sys;print(f'{sys.version_info.major}{sys.version_info.minor}')")
if [[ "$PANGO" != *"cpython-${PYTAG}-"* ]]; then
  echo "pypangolin at $PANGO is built for a different Python than .venv-ui ($PYTAG); set PYTHON to match"; exit 2
fi
rm -f "$SITE"/pypangolin.py
ln -sf "$PANGO" "$SITE/$(basename "$PANGO")"
# BoW native libs, as in the verification env.
if ! grep -q "loop-closure-toolbox" .venv-ui/bin/activate 2>/dev/null; then
  printf '\n# loop-closure-toolbox native libs (BoW loop detection)\nexport LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$SITE" >> .venv-ui/bin/activate
fi
python - <<'EOF'
import numpy, cv2, scipy, pypangolin
print("ui env ok:", "numpy", numpy.__version__, "cv2", cv2.__version__, "scipy", scipy.__version__, "| pangolin:", pypangolin.__file__)
EOF
