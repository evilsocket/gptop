mod app;
mod backend;
mod config;
mod ui;

use anyhow::{anyhow, Result};
use clap::Parser;

#[derive(Parser)]
#[command(name = "gptop", about = "GPU/Accelerator Monitor TUI")]
struct Cli {
    /// Output a single JSON sample and exit
    #[arg(long)]
    json: bool,

    /// Sample duration in milliseconds (for JSON mode)
    #[arg(long, default_value = "1000")]
    interval: u64,
}

fn detect_backend() -> Result<Box<dyn backend::GpuBackend>> {
    // Try Apple Silicon backend on macOS
    #[cfg(target_os = "macos")]
    {
        match backend::apple::AppleBackend::new() {
            Ok(b) => return Ok(Box::new(b)),
            Err(e) => eprintln!("Apple backend init failed: {}", e),
        }
    }

    // Try NVIDIA backend on Linux
    #[cfg(target_os = "linux")]
    {
        match backend::nvidia::NvidiaBackend::new() {
            Ok(b) => return Ok(Box::new(b)),
            Err(e) => eprintln!("NVIDIA backend init failed: {}", e),
        }
    }

    Err(anyhow!(
        "No supported GPU backend found. Currently supported: Apple Silicon (macOS), NVIDIA (Linux)"
    ))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let backend = detect_backend()?;

    if cli.json {
        app::run_json(backend, cli.interval)
    } else {
        app::run_tui(backend)
    }
}
