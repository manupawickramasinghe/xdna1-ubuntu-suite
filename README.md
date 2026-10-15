# xdna1-ubuntu-suite

Tooling for the **1st-gen AMD XDNA NPU** (Ryzen 7040 / 8040 / 8000G "Phoenix / Hawk Point",
PCI `1022:1502`, `RyzenAI-npu1`) on Ubuntu / Pop!_OS 24.04:

| Component | Status | What it is |
| --- | --- | --- |
| `include/xdna_telemetry.hpp` + `tools/xdna_stat` | ✅ working, tested on 8600G | Header-only C++20 telemetry reader for the in-tree `amdxdna` driver |
| `btop-patch/` | ✅ working, tested on 8600G | Adds the NPU as a panel in [btop](https://github.com/aristocratos/btop): `btop-patch/build_btop.sh` |
| `scripts/install_xdna1_ubuntu.sh` | 🧪 preflight tested; build/install need sudo | Staged XRT + `xrt_plugin-amdxdna` builder/installer with a Pop!_OS flavor shim |
| `runtime-cpp/` | 🧪 compiles against XRT headers; not yet run on hardware | XRT `.xclbin` runner + Unix-socket daemon with `--benchmark-loop` |
| `monitor-rs/` | 🚧 | Standalone Rust TUI (`xdna-top`) |

## What the kernel actually exposes (verified on kernel 7.1, Ryzen 5 8600G)

Measured with `xdna_stat` against the real driver; see [docs/FINDINGS.md](docs/FINDINGS.md).

| Data | Source | On 8600G |
| --- | --- | --- |
| Array topology | `GET_INFO / QUERY_AIE_METADATA` | 5 cols × 6 rows (4 core, 1 mem, 1 shim), AIE v1.1 |
| Clocks | `GET_INFO / QUERY_CLOCK_METADATA` | MP-NPU 960 MHz, H-clock 1800 MHz |
| Firmware | `GET_INFO / QUERY_FIRMWARE_VERSION` | 1.5.5.391 |
| Power / per-column utilisation | `GET_INFO / QUERY_SENSORS` | **`ENODEV`**: no amd-pmf NPU metrics on desktop APUs |
| Hardware contexts | `GET_ARRAY / HW_CONTEXT_ALL` | ✅ (pid, column partition, state, cmd counters) |
| Per-process BO memory | `GET_ARRAY / BO_USAGE` | ✅ |
| `/proc/<pid>/fdinfo` `drm-*` keys | fdinfo | ❌ not emitted by `amdxdna` on 7.1 |

Because the sensors aren't available, utilisation falls back to **column occupancy**: the
percentage of the 5 AIE columns owned by contexts that are `ACTIVE`, have commands in flight, or
completed commands since the last sample. That measures partition occupancy, not ALU
utilisation.

### Runtime PM

The NPU runtime-suspends about 5 s after its last ioctl, and **any** `GET_INFO` ioctl wakes it.
`xdna::Device::sample()` reads `power/runtime_status` first and issues no ioctls while the
device is suspended. After 3 idle samples it re-queries only every 10 s, so a 1 Hz monitor
doesn't keep an idle NPU powered.

## Build

```bash
cmake -S . -B build && cmake --build build -j
./build/xdna_stat            # one sample (won't wake a suspended NPU)
./build/xdna_stat -w         # wake and query
./build/xdna_stat -i 1000    # live, 1 s interval
./build/xdna_stat --json --scan-pids
```

Your user must be in the `render` group (`/dev/accel/accel0` is `root:render 0660`).

## Layout

```text
include/      amdxdna_accel.h (vendored UAPI), xdna_telemetry.hpp
tools/        xdna_stat.cpp
scripts/      install_xdna1_ubuntu.sh
runtime-cpp/  XRT runner + daemon
btop-patch/   patch for aristocratos/btop
monitor-rs/   Rust TUI
docs/         findings and design notes
```
