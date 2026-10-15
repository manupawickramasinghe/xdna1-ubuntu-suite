#!/usr/bin/env bash
# Rebuild the npu-face matmul kernels with mlir-aie (IRON whole_array, bf16 -> f32, npu1, 4 columns).
#
# Needs an mlir-aie checkout at tag v1.3.4 with its wheels installed in a venv:
#   python3 -m venv ironenv && . ironenv/bin/activate
#   pip install mlir_aie==1.3.4 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.4
#   pip install -r utils/peano-requirements.txt -r python/requirements.txt
#   # if `import aie.iron` fails on aie.extras: cp -rn site-packages/aie/extras/. site-packages/mlir_aie/python/aie/extras/
# and XRT installed (xclbinutil), e.g. via scripts/install_xdna1_ubuntu.sh.
#
#   MLIR_AIE=~/.cache/mlir-aie scripts/build_npu_kernels.sh
set -euo pipefail
MLIR_AIE="${MLIR_AIE:-$HOME/.cache/mlir-aie}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/npu-face-rs/kernels"

# M K N core-m (core tile k=64, n=32). M % (m*4) == 0, K % 64 == 0, N % 128 == 0.
# 64x64x64 core tiles don't fit L1 with f32 output.
KERNELS=(
  "512 1152 128 64"
  "256 2304 256 32"
  "256 4608 256 32"
)

# shellcheck disable=SC1091
source "$MLIR_AIE/ironenv/bin/activate"
set +u; source /opt/xilinx/xrt/setup.sh >/dev/null; set -u
MLIR_AIE_INSTALL_DIR=$(pip show mlir_aie | awk '/^Location:/{print $2"/mlir_aie"}')
PEANO_INSTALL_DIR=$(pip show llvm-aie | awk '/^Location:/{print $2"/llvm-aie"}')
export MLIR_AIE_INSTALL_DIR PEANO_INSTALL_DIR
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH" PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

cd "$MLIR_AIE/programming_examples/basic/matrix_multiplication/whole_array"
mkdir -p "$OUT"
for spec in "${KERNELS[@]}"; do
  read -r M K N m <<<"$spec"
  tag="${M}x${K}x${N}_${m}x64x32_4c"
  make NPU2=0 devicename=npu dtype_in=bf16 dtype_out=f32 M="$M" K="$K" N="$N" m="$m" k=64 n=32 \
    "build/final_${tag}.xclbin" "build/insts_${tag}.bin"
  cp "build/final_${tag}.xclbin" "$OUT/mm_bf16_f32_${M}x${K}x${N}.xclbin"
  cp "build/insts_${tag}.bin" "$OUT/mm_bf16_f32_${M}x${K}x${N}.insts.bin"
  echo "built mm_bf16_f32_${M}x${K}x${N}"
done
echo "Verify each with npu-face-rs/tools/kernel_check, then: npu-face selftest --model <onnx>"
