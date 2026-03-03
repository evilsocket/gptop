# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```sh
cargo build              # Debug build (macOS or Linux, platform-detected)
cargo build --release    # Release build (LTO, strip, panic=abort)
cargo run                # Run TUI in debug mode
cargo run -- --json      # Single JSON sample
```

No test suite exists yet. No linter or formatter config beyond default `rustfmt`/`clippy`:

```sh
cargo fmt --check
cargo clippy
```

## Architecture

gptop is a cross-platform GPU monitor TUI supporting Apple Silicon (macOS) and NVIDIA (Linux).

### Backend Trait Pattern

`backend::GpuBackend` is the core abstraction. Each platform implements `name()`, `devices()`, and `sample(duration_ms)`. Backends are cfg-gated in `backend/mod.rs`:

- `#[cfg(target_os = "macos")] pub mod apple` — IOReport for GPU stats, SMC/HID for temperature, CoreFoundation FFI
- `#[cfg(target_os = "linux")] pub mod nvidia` — NVML via `nvml-wrapper` crate

Backend detection happens in `main.rs::detect_backend()` which tries each platform backend in order.

### Shared Types (`backend/mod.rs`)

`DeviceInfo` (static per-device), `GpuMetrics` (per-sample per-device), `GpuProcess`, `SystemMetrics`, and `SampleResult` are the data types that flow from backends through the app to the UI. All rendering code depends only on these types, never on backend internals.

### Multi-threaded TUI Event Loop (`app.rs`)

Three threads communicate via `mpsc::channel<AppEvent>`:
- **Input thread**: polls `crossterm` keyboard events
- **Sampler thread**: calls `backend.sample()` in a loop, sends `AppEvent::Sample`
- **Main thread**: renders UI with `ratatui`, processes events

`App` struct holds all mutable state including circular history buffers (`VecDeque`, max 300 entries) for GPU utilization, memory, and power per device.

### UI Layout (`ui/mod.rs`)

Vertical layout: GPU charts (11 lines) → Memory charts (7 lines) → Info bar (1 line) → Process table (flexible) → Footer (1 line). For multi-GPU, each chart row is split horizontally into equal columns per device.

- `ui/charts.rs` — GPU/memory line charts (Braille markers) and info bar (MHz, temp, power, TFLOPS)
- `ui/processes.rs` — Sortable process table with 9 columns, DEV column auto-hidden for single-GPU

### Platform-Specific Code in `app.rs`

`get_exe_path()` and `get_cwd()` are cfg-gated: macOS uses `proc_pidpath`/`lsof`, Linux reads `/proc/<pid>/exe` and `/proc/<pid>/cwd`. The `ps` utility calls in `ps_field()` work on both platforms.

### Configuration (`config.rs`)

Persisted to `~/.config/gptop.json`. Saved automatically when the user changes settings via keybindings. Fields: `update_interval_ms`, `accent_color_idx`, `sort_column`, `sort_ascending`.

## Platform Dependencies

macOS-only deps (`core-foundation`, `core-foundation-sys`) and Linux-only deps (`nvml-wrapper`) are declared under target-specific sections in `Cargo.toml`. When adding new platform code, always use `#[cfg(target_os = "...")]` and the corresponding target dependency section.

## Adding a New Backend

1. Create `src/backend/<name>.rs` implementing `GpuBackend`
2. Add `#[cfg(...)] pub mod <name>` in `backend/mod.rs`
3. Add target-specific dependencies in `Cargo.toml`
4. Add detection block in `main.rs::detect_backend()`
5. If the platform needs different process inspection, add cfg-gated variants of `get_exe_path()`/`get_cwd()` in `app.rs`
