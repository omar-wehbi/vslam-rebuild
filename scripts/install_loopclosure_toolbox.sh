#!/usr/bin/env bash
# Install loopclosuretoolbox from source for BoW loop closure (--loop_detection_strategy bow).
# Requires: git, CMake 3.12+, C++ compiler, Python dev headers. Optional: ninja (pip install ninja).
#
# Run from project root with your venv activated:
#   ./scripts/install_loopclosure_toolbox.sh
# Or clone manually and pip install -e . from the repo.

set -e
REPO="${1:-https://github.com/HyeonJaeGil/loop-closure-toolbox.git}"
DIR="${2:-$(dirname "$0")/../.deps/loop-closure-toolbox}"

echo "Cloning loop-closure-toolbox (with submodules) into $DIR ..."
mkdir -p "$(dirname "$DIR")"
if [[ -d "$DIR/.git" ]]; then
  (cd "$DIR" && git pull && git submodule update --init --recursive)
else
  git clone --recurse-submodules "$REPO" "$DIR"
fi

# Fix DBoW3 build on non-Windows: BowVector.h needs <cstdint> for uint32_t/uint64_t and
# <iostream>/<string> for std::ostream, std::istream, std::string (only <cstdint> was guarded by _WIN32).
BOW_H="$DIR/src/DBoW3/include/BowVector.h"
if [[ -f "$BOW_H" ]]; then
  if grep -q '#if _WIN32' "$BOW_H"; then
    echo "Patching $BOW_H for Linux/macOS (add missing includes) ..."
    perl -i -0pe 's/#if _WIN32\n#include <cstdint>\n#endif/#include <cstdint>\n#include <iostream>\n#include <string>/' "$BOW_H"
  elif grep -q '#include <cstdint>' "$BOW_H" && ! grep -q '#include <iostream>' "$BOW_H"; then
    echo "Patching $BOW_H (add <iostream> and <string>) ..."
    perl -i -pe 's/(#include <cstdint>)/$1\n#include <iostream>\n#include <string>/' "$BOW_H"
  fi
fi

# Use absolute path so cmake run from BUILD_DIR finds the source tree
DIR="$(cd "$DIR" && pwd)"

echo "Installing loopclosuretoolbox into current Python environment ..."
pip install -e "$DIR"

# The Python extension links to libVLAD.so and libDBoW3.so; pip does not install them.
# Build in-tree so we can copy those libs next to the extension for the loader to find.
echo "Building shared libs (libVLAD.so, libDBoW3.so) and copying next to Python extension ..."
BUILD_DIR="$DIR/build"
mkdir -p "$BUILD_DIR"
(cd "$BUILD_DIR" && cmake "$DIR" -DBUILD_PYBIND=ON -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build . -j "$(nproc 2>/dev/null || echo 2)")

# Find site-packages (we can't import loopclosuretoolbox yet because the .so deps are missing)
PKG_DIR=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -z "$PKG_DIR" ]]; then
  PKG_DIR=$(python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])" 2>/dev/null || true)
fi
if [[ -n "$PKG_DIR" ]]; then
  for lib in "$BUILD_DIR/src/VLAD/libVLAD.so" "$BUILD_DIR/src/DBoW3/libDBoW3.so"; do
    if [[ -f "$lib" ]]; then
      cp -f "$lib" "$PKG_DIR/"
      echo "  Copied $(basename "$lib") to $PKG_DIR"
    fi
  done
  echo ""
  echo "To load the toolbox at runtime, set LD_LIBRARY_PATH before running Python:"
  echo "  export LD_LIBRARY_PATH=\"$PKG_DIR\":\$LD_LIBRARY_PATH"
  echo "Or use: ./scripts/run_with_bow.sh python main.py ..."
else
  echo "Could not find site-packages; copy libVLAD.so and libDBoW3.so from $BUILD_DIR/src/*/ into your Python site-packages and set LD_LIBRARY_PATH to that directory."
fi

echo ""
echo "Done. Use --loop_detection_strategy bow and --vocabulary_path <path> to use BoW loop closure."
