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
