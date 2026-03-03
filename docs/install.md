# Installation

## Precompiled Binaries

Download the latest binary for your platform from the [releases page](https://github.com/evilsocket/gptop/releases/latest).

## Homebrew (macOS)

```sh
brew tap evilsocket/gptop https://github.com/evilsocket/gptop
brew install evilsocket/gptop/gptop
```

## Debian/Ubuntu (Linux)

A `.deb` package is available from the [releases page](https://github.com/evilsocket/gptop/releases/latest), or you can build it locally:

```sh
cargo install cargo-deb
cargo deb --install
```

## From Source

```sh
git clone https://github.com/evilsocket/gptop.git
cd gptop
cargo build --release
```

The binary will be at `target/release/gptop`.

## Requirements

| Platform | Requirements |
|----------|-------------|
| macOS | Apple Silicon (M1 or later) |
| Linux | NVIDIA GPU with proprietary drivers (NVML) |
