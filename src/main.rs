mod app;
mod backend;
mod benchmark;
mod config;
mod ui;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gptop", version, about = "GPU/Accelerator Monitor TUI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Output a single JSON sample and exit
    #[arg(long)]
    json: bool,

    /// Sample duration in milliseconds (for JSON mode)
    #[arg(long, default_value = "1000")]
    interval: u64,
}

#[derive(Subcommand)]
enum Commands {
    /// GPU benchmark and performance testing
    #[command(name = "benchmark", visible_alias = "bench", visible_alias = "b")]
    Benchmark {
        /// Benchmark duration in seconds (required unless --list or --cmp)
        #[arg(required_unless_present_any = &["list", "cmp"])]
        duration: Option<u64>,

        /// Benchmark type: comprehensive, efficiency, compute, memory, thermal
        #[arg(short, long, default_value = "comprehensive")]
        type_: String,

        /// Save report to file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// List saved benchmark reports
        #[arg(long, conflicts_with_all = &["duration", "cmp", "output", "no_save"])]
        list: bool,

        /// Limit number of results when listing
        #[arg(long, requires = "list", value_name = "N")]
        limit: Option<usize>,

        /// Compare two benchmarks (comma-separated IDs: id1,id2)
        #[arg(long, conflicts_with_all = &["duration", "list", "no_save"], value_name = "ID1,ID2")]
        cmp: Option<String>,

        /// Do not auto-save the benchmark report
        #[arg(long, conflicts_with_all = &["list", "cmp"])]
        no_save: bool,
    },
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

    match cli.command {
        Some(Commands::Benchmark {
            duration,
            type_,
            output,
            list,
            limit,
            cmp,
            no_save,
        }) => {
            let backend = detect_backend()?;

            if list {
                // List saved benchmarks
                benchmark::list_benchmarks(backend, limit)
            } else if let Some(cmp_ids) = cmp {
                // Compare two benchmarks
                let ids: Vec<&str> = cmp_ids.split(',').collect();
                if ids.len() != 2 {
                    return Err(anyhow!(
                        "--cmp requires exactly two benchmark IDs (comma-separated)"
                    ));
                }
                benchmark::compare_benchmarks(backend, ids[0], ids[1], output)
            } else if let Some(duration) = duration {
                // Run new benchmark
                let benchmark_type = type_.parse::<benchmark::BenchmarkType>()?;
                benchmark::run_benchmark(backend, duration, benchmark_type, output, no_save)
            } else {
                unreachable!() // Required by clap validation
            }
        }
        None => {
            // Original behavior: TUI or JSON mode
            let backend = detect_backend()?;

            if cli.json {
                app::run_json(backend, cli.interval)
            } else {
                app::run_tui(backend)
            }
        }
    }
}