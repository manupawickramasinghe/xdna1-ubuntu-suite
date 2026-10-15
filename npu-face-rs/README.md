# npu-face: face embeddings on the AMD XDNA NPU (Rust)

Runs ONNX face-embedding CNNs (SFace, Frigate's ArcFace) with every regular convolution and
fully-connected layer executed as **bf16 matmuls on the XDNA1 NPU** through XRT, and serves them to
Frigate over TCP.

* `kernels/` has three fixed-shape mlir-aie (`whole_array`, v1.3.4, 4 columns) bf16×bf16→f32 matmul
  kernels sized for ArcFace's ResNet stages: `512x1152x128`, `256x2304x256`, `256x4608x256`. Each is
  exact against a double-precision reference (error ≈ 1e-7·√K, i.e. f32 accumulation) and runs
  ~0.7–1.0 TFLOP/s. The host picks the cheapest kernel per layer (calls × (measured µs + host
  overhead)), packs that layer's weights once into pinned NPU buffers, and keeps two tiles in flight
  (double buffering). `kernels-generic/` holds smaller general-purpose tiles.
* `kernels-unsafe/512x576x128` is exact alone but **corrupts tiles when it shares the NPU with
  `256x2304x256`** (up to 61–89% max error in some K=576 layers). Not loaded.
* `src/model.rs` is a small ONNX interpreter. Conv (group = 1) runs via im2col on the NPU. Depthwise
  conv, PReLU, BatchNorm (folded into convs at load), Add/Mul/Sub/... run on the CPU (rayon).
* `shim/xmm_shim.cpp` is a C ABI over the XRT runner from `runtime-cpp/`.

```bash
CC=gcc-14 CXX=g++-14 cargo build --release
target/release/npu-face selftest --model arcface.onnx   # NPU vs bf16-emulated and exact-f32 CPU
target/release/npu-face bench    --model arcface.onnx
target/release/npu-face serve    --model arcface.onnx --listen 127.0.0.1:5800
```

`serve` and `selftest` validate **every NPU layer** against a CPU recomputation with the same bf16
numerics (per-layer cosine ≥ 0.9999 and max error ≤ 1% of the layer's range) and refuse to run
otherwise. A whole-network cosine check alone passed (0.9995) while the faulty kernel combination
was corrupting layers, so it isn't enough.

While serving it also:
* **spot-checks one rotating NPU layer per embedding** against the CPU (~10 ms), covering all 104
  ArcFace layers every 104 faces;
* **fully re-validates every 10 minutes**;
* on any mismatch, returns an error to Frigate (so no wrong embedding is stored) and exits, and
  systemd restarts it and re-validates.

Observed on npu1: the first validation right after a restart can fail (identically: cosine 0.999968,
max error 24% at `conv M=784 K=1152 N=256`) while the previous process's NPU state is being torn
down. The next start passes, so a restart takes ~10–20 s. While the service holds its 3 contexts,
other processes can't create NPU contexts (`ENODEV`), so they can't co-reside and corrupt it.

Protocol: request `"XFQ1" | u32 n,c,h,w | f32[n*c*h*w]`; response `"XFR1" | i32 status | u32 n,dim |
f32[n*dim]` (an error message when status ≠ 0). The Frigate side is `NpuServiceRunner`
(`device: npu` in `face_recognition`).

## Results (Ryzen 5 8600G)

| Model | NPU vs onnxruntime | Latency |
| --- | --- | --- |
| SFace (OpenCV zoo) | cosine ≥ 0.9999 on 0–255 inputs | ~40 ms |
| **ArcFace IResNet-100 (Frigate `arcface.onnx`)** | NPU vs bf16 CPU 0.99999, vs f32 CPU 0.99997; worst layer error 1.8e-6 | **127 ms p50**, 144 ms with per-request spot-check (onnxruntime CPU: 70 ms) |

Optimisation history for ArcFace: 418 ms (one 512×256×256 kernel) → 280 ms (4 general kernels, best
per layer) → 193 ms (pinned weight buffers + double buffering) → 127 ms (stage-sized full-K kernels:
229 NPU calls/face instead of 1423).
