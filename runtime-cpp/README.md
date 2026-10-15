# runtime-cpp: XRT runner and inference daemon

**Status:** compiles against XRT `1ea5176` headers (`-fsyntax-only`, see CI). It hasn't run on
hardware yet because XRT isn't installed on the dev box; run
`scripts/install_xdna1_ubuntu.sh` first.

* `xdna_runner.hpp`: `xdna::XdnaRunner` holds `xrt::device` → `xrt::hw_context` → `xrt::kernel`,
  the instruction BO, and pre-allocated input/output BOs. Each call does only BO sync + dispatch.
  It follows the mlir-aie host ABI: `opcode=3, instr_bo, instr_words, in..., out`, with
  `group_id(1)` for the instruction BO and `group_id(3+i)` for data BOs.
* `xdna_daemon.cpp`: serves that kernel over a Unix socket, or runs a load generator.
* `xdna_client.py`: Python client for the socket protocol.

## Getting a kernel

XRT doesn't ship generic NPU kernels. Build one with [mlir-aie](https://github.com/Xilinx/mlir-aie)
for `npu1` (Phoenix/Hawk Point), e.g. `programming_examples/basic/vector_scalar_mul`:

```bash
make NPU2=0           # → build/final.xclbin, build/insts.bin
```

vector_scalar_mul (int16, 4096 elements) takes one 8192-byte input and one 4-byte scale factor,
and produces an 8192-byte output.

## Build

```bash
cmake -S . -B build && cmake --build build -j   # runtime builds automatically once /opt/xilinx/xrt exists
```

## Load generator (watch btop / xdna_stat)

```bash
build/xdna_daemon --xclbin final.xclbin --insts insts.bin --in 8192,4 --out 8192 \
    --benchmark-loop --duty 50 --period-ms 200
```

## Daemon

```bash
build/xdna_daemon --xclbin final.xclbin --insts insts.bin --in 8192,4 --out 8192
python3 runtime-cpp/xdna_client.py --in 8192,4 -n 1000
```

Protocol (little-endian): request `u32 'XDNQ' | u32 n | u64 size[n] | payloads`; response
`u32 'XDNR' | i32 status | u64 len | payload` (an error message when status ≠ 0). Input sizes must
exactly match `--in`. Payloads are read straight into the pinned input BOs, with no intermediate
copy. The default socket is `$XDG_RUNTIME_DIR/xdna_infer.sock`, or `/run/xdna_infer.sock` when
running as root.
