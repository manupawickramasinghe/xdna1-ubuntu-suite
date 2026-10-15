# Hardware findings — Ryzen 5 8600G, kernel 7.1.5, in-tree amdxdna

Date: 2026-10-15. Device `0000:10:00.1`, `1022:1502`, vbnv `RyzenAI-npu1`, fw `1.5.5.391`.

## GET_INFO params

| param | result |
| --- | --- |
| `QUERY_AIE_STATUS` (0) | ok, `cols_filled=0` when idle |
| `QUERY_AIE_METADATA` (1) | cols=5 rows=6 col_size=504, core rows 4 (start 2), mem 1, shim 1, AIE 1.1 |
| `QUERY_CLOCK_METADATA` (3) | "MP-NPU Clock" 960 MHz, "H Clock" 1800 MHz |
| `QUERY_SENSORS` (4) | **ENODEV**; kernel reports required size 2688 = 16 × sizeof(sensor) |
| `QUERY_HW_CONTEXTS` (5) | returns 0 but leaves `buffer_size` unchanged, so the count can't be derived. Use `GET_ARRAY / HW_CONTEXT_ALL` instead |
| `QUERY_FIRMWARE_VERSION` (8) | 1.5.5.391 |
| `GET_POWER_MODE` (9) | 0 (default) |
| `QUERY_RESOURCE_INFO` (12) | clk_max 1600, tops_max 10, task_max 6, tops_curr 17, task_curr 0 |

Note: `tops_curr` (17) > `tops_max` (10) and H-clock (1800) > `npu_clk_max` (1600) are reported
as-is by the driver.

## GET_ARRAY

* `HW_CONTEXT_ALL`: element_size 144 (`struct amdxdna_drm_hwctx_entry`), 0 entries when idle.
* `BO_USAGE` for a pid: works without root for the caller's own processes. A 64 MiB
  `AMDXDNA_BO_DEV_HEAP` shows up as `total=internal=64 MiB`.

## fdinfo

`/proc/<pid>/fdinfo/<fd>` for an open `/dev/accel/accel0` contains only `pos/flags/mnt_id/ino`;
no `drm-driver`, `drm-total-memory` or `drm-engine-*` keys.

## Runtime PM

* `power/autosuspend_delay_ms` = 5000.
* `open()` of the node does **not** resume the device; any `GET_INFO` ioctl does.
* BOs alone do not hold a runtime-PM reference (`runtime_usage` stays 0).
* Naive 1 Hz polling keeps the device `active` forever; with the back-off in
  `xdna_telemetry.hpp` it suspends after ~9 s and stays suspended.

## Corrections to the original spec

* Enum names are `DRM_AMDXDNA_QUERY_SENSORS`, `DRM_AMDXDNA_QUERY_CLOCK_METADATA`,
  `DRM_AMDXDNA_QUERY_HW_CONTEXTS`; there are no `AMDXDNA_INFO_*` names.
* `amdxdna_drm_query_sensor.unitm` is `__s8` and `label/status/units` are `__u8[]`.
* Clocks come from `QUERY_CLOCK_METADATA`, not `QUERY_AIE_METADATA`.
* Per-process memory comes from `GET_ARRAY / DRM_AMDXDNA_BO_USAGE`, not fdinfo.
* The PCI device ID `1502` covers Phoenix and Hawk Point; `17f0` is Strix/Krackan (XDNA2).

## Under real XRT load (2026-10-15)

* `GET_ARRAY / HW_CONTEXT_ALL` works as expected: per-context `start_col/num_col`, and
  submissions/completions advance, so busy detection and cmd/s are correct.
* **User contexts are placed on columns 1–4 only; column 0 is never assigned.** A 5th
  single-column context shares column 1. Column-occupancy utilisation therefore tops out at 80%.
* Malformed `CREATE_HWCTX` (no XRT partition params) logs
  `amdxdna_gem_heap_alloc: Invalid dev heap userptr` and fails with EINVAL; that's harmless.
* Burn test results: [results/README.md](results/README.md).

## IREE + AMD-AIE on npu1 (2026-10-15)

Built `nod-ai/iree-amd-aie` @ `fddfec1b` (IREE 3.12.0.dev `2608330`) from source with the
`amd_aie` compiler plugin and `amdxdna` HAL driver. Peano: `llvm-aie 22.0.0.2026101201`.

* `iree-run-module --list_devices=amdxdna` → `amdxdna://`; the HAL driver works with the
  in-tree kernel 7.1 amdxdna driver.
* **bf16 MatMul from ONNX runs on the NPU and is correct.** 128×256 · 256×128, flags:
  `--iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu1_4col
  --iree-amdaie-device-hal=amdxdna --iree-amdaie-stack-size=2048` (default 1024 B stack is
  too small with this Peano). Max error 1 bf16 ULP vs numpy; **2.06 ms per call** end to end.
* **CNNs do not compile.** SFace (MobileFaceNet-style, 27 Conv incl. depthwise + Gemm), with
  contraction inputs demoted to bf16: 20 dispatches fail with `Unhandled pass pipeline in
  setRootConfig` (conv ops arrive as `linalg.generic`); with `--iree-amdaie-tile-pipeline=
  conv-decompose`, 21 fail. Upstream's own CI says the same: named conv ops lowered to
  `linalg.generic` are unsupported, and depthwise conv tests are disabled.
* Consequence: face (SFace/ArcFace) and plate OCR (PaddleOCR det/rec, LPRNet) can't run on
  npu1 through IREE today. Only matmul-dominated graphs are candidates, and per-dispatch overhead
  (~2 ms) rules out small models anyway.
