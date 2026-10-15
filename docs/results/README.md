# NPU benchmark + burn test: Ryzen 5 8600G (XDNA1, RyzenAI-npu1)

**2026-10-15 · Pop!_OS 24.04 · kernel 7.1.5 (in-tree amdxdna) · fw 1.5.5.391 · XRT 2.27.0 (`1ea5176`) ·
mlir-aie v1.3.4 `vector_scalar_mul` (4096 × int16, 1 column per context)**

## Result: ✅ PASS

| | |
| --- | --- |
| Burn duration | 600 s, 5 concurrent hardware contexts |
| Runs | **6,328,771**, every output element verified against `in × scale` |
| Data verified | 51.85 GB |
| Mismatches / timeouts / exceptions | **0 / 0 / 0** |
| Throughput | 10,540 runs/s, stable from first to last minute (10.3k–10.6k) |
| H-clock | 1800 MHz throughout (no throttling) |
| CPU package Tctl | 64–71 °C |
| amdxdna / IOMMU kernel log during run | clean |

Raw output: [`burn-8600g-2026-10-15.log`](burn-8600g-2026-10-15.log), [`.json`](burn-8600g-2026-10-15.json).

## Benchmarks

`xrt-smi validate` (AMD's built-in tests): latency **72 µs**, throughput **37,886 ops/s**. Both PASSED.

Single context, end to end (input sync + dispatch + wait + output sync), 5000 runs:

| min | p50 | avg | p99 | max | runs/s | host↔NPU |
| --- | --- | --- | --- | --- | --- | --- |
| 70.9 µs | 100.0 µs | 102.9 µs | 151.8 µs | 1.23 ms | 8,282 | 135.7 MB/s |

Hardware-context creation (xclbin load + partition): **27.6 ms**.

Scaling with parallel contexts (20 s each, all verified, 0 errors):

| contexts | runs/s | p99 latency | columns busy |
| --- | --- | --- | --- |
| 1 | 8,087 | 202 µs | 1 |
| 2 | 11,087 | 248 µs | 2 |
| 3 | 12,628 | 341 µs | 3 |
| 4 | **13,570** | 479 µs | 4 |
| 5 | 10,589 | 697 µs | 4 (ctx 5 time-shares col 1) |

## Findings

* **Column 0 is never assigned to user contexts.** Partitions land on columns 1–4, so 4 is the
  maximum useful parallelism on npu1, and a 5th context costs ~22% throughput through time-sharing.
* This workload is dispatch/sync-bound (8 KB per run), so these numbers measure the runtime
  and driver path, not AIE compute TOPS.
* There's no NPU temperature or power sensor on this platform (`QUERY_SENSORS` → ENODEV), so CPU Tctl
  is the only thermal signal.

## Reproduce

```bash
scripts/install_xdna1_ubuntu.sh                                   # XRT + plugin
# mlir-aie venv: pip install mlir_aie==1.3.4 + utils/peano-requirements.txt, repo at tag v1.3.4
make -C mlir-aie/programming_examples/basic/vector_scalar_mul NPU2=0
build/runtime-cpp/xdna_burn --xclbin .../final_8192.xclbin --insts .../insts_8192.bin \
    --contexts 5 --minutes 10 --json report.json
```
