<div align="center">

# `gptop`

A cross-platform GPU monitor TUI with support for both Apple Silicon and NVIDIA GPUs.

[![Documentation](https://img.shields.io/badge/docs-blue)](https://github.com/evilsocket/gptop/blob/main/docs/index.md)
[![Release](https://img.shields.io/github/release/evilsocket/gptop.svg?style=flat-square)](https://github.com/evilsocket/gptop/releases/latest)
[![Rust Report](https://rust-reportcard.xuri.me/badge/github.com/evilsocket/gptop)](https://rust-reportcard.xuri.me/report/github.com/evilsocket/gptop)
[![CI](https://img.shields.io/github/actions/workflow/status/evilsocket/gptop/ci.yml)](https://github.com/evilsocket/gptop/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square)](https://github.com/evilsocket/gptop/blob/master/LICENSE.md)

</div>

<p align="center">
  <a href="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/apple.png"><img src="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/apple.png" width="32%"/></a>
  <a href="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/nvidia.png"><img src="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/nvidia.png" width="32%"/></a>
  <a href="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/nvidia_dual.png"><img src="https://raw.githubusercontent.com/evilsocket/gptop/main/docs/images/nvidia_dual.png" width="32%"/></a>
</p>

## Features

- Apple Silicon and NVIDIA supported, more to come.
- Real-time GPU and memory utilization charts
- Live stats: clock speed, temperature, power draw, FP32 TFLOPS
- Per-process GPU usage, memory and CPU breakdown
- Process inspector with detailed info (path, CWD, command line, etc.)
- Kill processes directly from the TUI
- Multi-GPU support with side-by-side charts
- JSON output mode for scripting and monitoring pipelines
- Configurable update interval and accent colors (persisted across sessions)

## Quick Start

Download one of the precompiled binaries from the [project latest release page](https://github.com/evilsocket/gptopp/releases/latest), or if you're a **Homebrew** user, you can install it with a custom tap:

```bash
brew tap evilsocket/gptop https://github.com/evilsocket/gptop
brew install evilsocket/gptop/gptop
```

You are now ready to go! 🚀

### From source

```
cargo install --path .
```

## Usage

```
# Launch the TUI
gptop

# Output a single JSON snapshot and exit
gptop --json

# JSON with custom sample interval (ms)
gptop --json --interval 2000
```

## Keybindings

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `Up` / `Down` | Select process |
| `Enter` | Process details |
| `k` | Kill selected process |
| `Left` / `Right` | Change sort column |
| `+` / `-` | Toggle sort order (asc/desc) |
| `c` | Cycle accent color |
| `e` / `d` | Increase / decrease update interval |

## Contributors

<a href="https://github.com/evilsocket/gptop/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=evilsocket/gptop" alt="gptop project contributors" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/gptop&type=Timeline)](https://www.star-history.com/#evilsocket/gptop&Timeline)

## License

gptop is released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.