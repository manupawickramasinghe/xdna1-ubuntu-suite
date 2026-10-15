# xdna-top

A standalone Rust TUI for the AMD XDNA NPU, using `nix` ioctls and `ratatui`. No XRT needed.

```bash
cargo run --release              # q quit · w wake+query · +/- interval
cargo run --release -- --once    # one text sample
```

```text
┌ AIE array (░ free ▒ owned █ busy) ───────┐┌ utilisation 0% (column occupancy) · BO memory 0 B · tasks 0/6 ┐
│        col0  col1  col2  col3  col4      ││100│                                                            │
│core r5  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│core r4  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│core r3  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│core r2  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│mem  r1  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│shim r0  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░     ││   │                                                            │
│busy      ·     ·     ·     ·     ·       ││0  │                                                            │
```

* **AIE array:** all 5 columns × (4 core + 1 mem + 1 shim) rows from `QUERY_AIE_METADATA`.
  Columns are coloured by the hardware context that owns them, and bold when that context is
  busy. With PMF column sensors, per-column % replaces the busy dots.
* **Chart:** utilisation history (sensors, or column occupancy without them).
* **Tables:** hardware contexts (`GET_ARRAY / HW_CONTEXT_ALL`: cols, state, cmd/s, in-flight,
  errors) and per-process BO memory (`GET_ARRAY / BO_USAGE`).

Struct layouts are checked at compile time against the sizes measured on kernel 7.1. The
runtime-PM policy matches `xdna_telemetry.hpp`: polling never wakes a suspended NPU.
