`512x576x128` is exact on its own, but on npu1 (fw 1.5.5.391, kernel 7.1.5 amdxdna, XRT 2.27.0) it
returns corrupted tiles (per-layer max error up to 89% of the layer's range) when it shares the NPU
with the `256x2304x256` kernel's hardware context. Other pairs and the three kernels in `kernels/` are
exact. Don't load it until that's understood.
