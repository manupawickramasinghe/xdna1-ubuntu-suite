#!/usr/bin/env bash
# Clone aristocratos/btop at a known-good commit, apply the XDNA NPU patch,
# refresh the vendored telemetry headers from this repo, and build.
#
#   btop-patch/build_btop.sh            # build into ~/.cache/xdna1-btop/build/btop
#   btop-patch/build_btop.sh --install  # also `sudo cmake --install` (to /usr/local)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SRC="${BTOP_SRC:-$HOME/.cache/xdna1-btop}"
BTOP_COMMIT="${BTOP_COMMIT:-119c8d83e1331ac6fbc9aef0fe78c7fadda9b391}"

# btop needs std::ranges::to (GCC >= 14 or Clang >= 17). Use matching C/C++ compilers (LTO).
if [[ -z "${CXX:-}" ]]; then
	if command -v g++-14 >/dev/null && command -v gcc-14 >/dev/null; then export CC=gcc-14 CXX=g++-14
	elif command -v clang++-18 >/dev/null; then export CC=clang-18 CXX=clang++-18
	else echo "Need g++-14 (sudo apt install g++-14) or clang++-18" >&2; exit 1; fi
fi

if [[ ! -d "$SRC/.git" ]]; then
	git clone https://github.com/aristocratos/btop.git "$SRC"
fi
cd "$SRC"
git fetch -q origin
git checkout -q --detach "$BTOP_COMMIT"
git apply --check "$HERE/0001-add-amdxdna-npu-support.patch"
git apply "$HERE/0001-add-amdxdna-npu-support.patch"
cp "$REPO/include/xdna_telemetry.hpp" "$REPO/include/amdxdna_accel.h" src/linux/xdna/

cmake -S . -B build -DBTOP_GPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "Built: $SRC/build/btop"

# Leave the checkout clean so the script can be re-run.
git checkout -q -- . && git clean -qfd src/linux/xdna

if [[ "${1:-}" == "--install" ]]; then
	sudo cmake --install build
fi
