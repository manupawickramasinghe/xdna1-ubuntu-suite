# runtime-cpp: XRT runner and inference daemon

**Status:** ✅ verified on a Ryzen 5 8600G with XRT 2.27.0 (built by `scripts/install_xdna1_ubuntu.sh`),
running mlir-aie v1.3.4's `vector_scalar_mul`. See `docs/results/` for the burn test.

Two lessons from bring-up:
* **The output BO's argument slot depends on the design.** IRON designs built with
  `transform` declare `(in, out, *params)`, so vector_scalar_mul is `(A, C, factor)` → `--out-pos 1`.
  The wrong slot makes the NPU write results into an input buffer.
* **Flush CPU writes to the output BO before a run.** `XdnaRunner::run()` syncs it `TO_DEVICE`;
  otherwise dirty cache lines get written back over the NPU's results.

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
make NPU2=0           # npu1 → build/final_8192.xclbin, build/insts_8192.bin
```

vector_scalar_mul (default `int_bit_width=16`, 4096 elements, kernel `MLIR_AIE`) takes one 8192-byte input and one 4-byte scale factor,
and produces an 8192-byte output.

## Build

```bash
cmake -S . -B build && cmake --build build -j   # → build/runtime-cpp/{xdna_daemon,xdna_burn} once /opt/xilinx/xrt exists
```

## Benchmark + burn test

```bash
build/runtime-cpp/xdna_burn --xclbin final_8192.xclbin --insts insts_8192.bin \
    --contexts 5 --minutes 10 --json report.json
```

Every run gets fresh random inputs and a poisoned output, and every element is verified. Exits
non-zero on any mismatch, timeout, exception or amdxdna kernel-log error.

## Load generator (watch btop / xdna_stat)

```bash
build/xdna_daemon --xclbin final_8192.xclbin --insts insts_8192.bin --in 8192,4 --out 8192 --out-pos 1 \
    --benchmark-loop --duty 50 --period-ms 200
```

## Daemon

```bash
build/xdna_daemon --xclbin final_8192.xclbin --insts insts_8192.bin --in 8192,4 --out 8192 --out-pos 1
python3 runtime-cpp/xdna_client.py --in 8192,4 -n 1000
```

Protocol (little-endian): request `u32 'XDNQ' | u32 n | u64 size[n] | payloads`; response
`u32 'XDNR' | i32 status | u64 len | payload` (an error message when status ≠ 0). Input sizes must
exactly match `--in`. Payloads are read straight into the pinned input BOs, with no intermediate
copy. The default socket is `$XDG_RUNTIME_DIR/xdna_infer.sock`, or `/run/xdna_infer.sock` when
running as root.
