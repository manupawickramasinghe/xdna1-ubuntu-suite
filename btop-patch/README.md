# btop NPU patch

`0001-add-amdxdna-npu-support.patch` adds a `Gpu::Xdna` collector to
[aristocratos/btop](https://github.com/aristocratos/btop) (based on commit `119c8d8`). The NPU
then shows up as an extra GPU panel:

```text
╭─┐AMD Ryzen AI NPU (XDNA1)┌──────────────────────────┐1800 MHz┌╮
│GPU ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■    0%│
│VRAM usage:                                             0 Byte │
╰───────────────────────────────────────────────────────────────╯
```

(Captured from a real 8600G. btop's panel labels are generic: "GPU" is NPU utilisation and
"VRAM usage" is the summed per-process BO memory.)

| btop field | NPU source |
| --- | --- |
| name | PCI ID → `AMD Ryzen AI NPU (XDNA1)` / `(XDNA2)` |
| utilisation | `QUERY_SENSORS` column utilisation if available, else column occupancy from `GET_ARRAY / HW_CONTEXT_ALL` |
| clock | H-clock from `QUERY_CLOCK_METADATA` (0 while runtime-suspended) |
| memory used | Σ `GET_ARRAY / BO_USAGE` over processes owning contexts |
| power | `QUERY_SENSORS` power sensor; hidden when unsupported (ENODEV on desktop APUs) |

It's enabled when `shown_gpus` contains `amd` (the default) or `npu`. Add `gpu1` (or whichever
index the NPU gets) to `shown_boxes`, or press <kbd>5</kbd>–<kbd>0</kbd> in btop.

## Build

```bash
btop-patch/build_btop.sh             # → ~/.cache/xdna1-btop/build/btop
btop-patch/build_btop.sh --install   # also install to /usr/local
```

No-sudo install (`~/.local/bin` is ahead of `/usr/local/bin` in a default Ubuntu PATH):

```bash
install -Dm755 ~/.cache/xdna1-btop/build/btop ~/.local/bin/btop && hash -r
```

**The NPU won't appear if `shown_boxes` in `~/.config/btop/btop.conf` lacks its `gpuN` entry.**
On an 8600G the iGPU is `gpu0` and the NPU is `gpu1`:
`shown_boxes = "proc mem gpu0 gpu1 cpu net"`. You can also press <kbd>6</kbd> in btop.

Needs `g++-14` (btop uses `std::ranges::to`). The script copies the current
`include/xdna_telemetry.hpp` over the copy embedded in the patch.
