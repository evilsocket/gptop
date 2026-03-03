# Installation

## Homebrew (macOS)

```sh
brew tap evilsocket/gptop https://github.com/evilsocket/gptop
brew install evilsocket/gptop/gptop
```

## Cargo

If you have [Cargo](https://www.rust-lang.org/tools/install) installed, you can install directly from [crates.io](https://crates.io/crates/gptop):

```sh
cargo install gptop
```

## Debian/Ubuntu (Linux)

You can build and install a `.deb` package locally:

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
