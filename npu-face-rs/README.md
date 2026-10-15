# npu-face: face embeddings on the AMD XDNA NPU (Rust)

Runs ONNX face-embedding CNNs (SFace, Frigate's ArcFace) with every regular convolution and
fully-connected layer executed as **bf16 matmuls on the XDNA1 NPU** through XRT, and serves them to
Frigate over TCP.

* `kernels/mm_bf16_f32_512x256x256.{xclbin,insts.bin}` is one fixed-shape mlir-aie (`whole_array`,
  v1.3.4, 4 columns, core tiles 64×64×32) bf16×bf16→f32 matmul kernel. It is exact in 20/20 random
  runs and takes 167 µs/tile (~400 GFLOP/s). The host splits any matmul into these tiles and
  accumulates K partial sums in f32, so there's one hardware context and no per-shape kernels.
* `src/model.rs` is a small ONNX interpreter. Conv (group = 1) runs via im2col on the NPU. Depthwise
  conv, PReLU, BatchNorm (folded into convs at load), Add/Mul/Sub/... run on the CPU (rayon).
* `shim/xmm_shim.cpp` is a C ABI over the XRT runner from `runtime-cpp/`.

```bash
CC=gcc-14 CXX=g++-14 cargo build --release
target/release/npu-face selftest --model arcface.onnx   # NPU vs bf16-emulated and exact-f32 CPU
target/release/npu-face bench    --model arcface.onnx
target/release/npu-face serve    --model arcface.onnx --listen 127.0.0.1:5800
```

`serve` refuses to start if the NPU disagrees with the bf16-emulated CPU path (cosine < 0.999).

Protocol: request `"XFQ1" | u32 n,c,h,w | f32[n*c*h*w]`; response `"XFR1" | i32 status | u32 n,dim |
f32[n*dim]` (an error message when status ≠ 0). The Frigate side is `NpuServiceRunner`
(`device: npu` in `face_recognition`).

## Results (Ryzen 5 8600G)

| Model | NPU vs onnxruntime | Latency |
| --- | --- | --- |
| SFace (OpenCV zoo) | cosine ≥ 0.9999 on 0–255 inputs | 92 ms in process, 112 ms via TCP |
