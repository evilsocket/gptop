pub mod compare;
pub mod kernels;
pub mod metrics;
pub mod printer;
pub mod runner;
pub mod storage;
pub mod types;

#[cfg(target_os = "macos")]
pub mod metal_kernels;

pub use compare::compare_reports;
pub use printer::BenchmarkPrinter;
pub use runner::BenchmarkRunner;
pub use storage::BenchmarkStorage;
pub use types::*;

use anyhow::Result;

/// Main entry point for benchmark subcommand
pub fn run_benchmark(
    backend: Box<dyn crate::backend::GpuBackend>,
    duration: u64,
    benchmark_type: BenchmarkType,
    output: Option<std::path::PathBuf>,
    no_save: bool,
) -> Result<()> {
    let mut runner = BenchmarkRunner::new(backend)?;
    
    let mut report = runner.run(duration, benchmark_type)?;
    
    // Auto-save unless --no-save is specified
    if !no_save {
        runner.save_report(&mut report)?;
        println!("Report saved with ID: {}", report.metadata.id);
    }
    
    // Save to output file if specified
    if let Some(path) = output {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&path, json)?;
        println!("Report exported to: {}", path.display());
    }
    
    // Print the report
    BenchmarkPrinter::print_report(&report);
    
    Ok(())
}

/// List saved benchmarks
pub fn list_benchmarks(
    backend: Box<dyn crate::backend::GpuBackend>,
    limit: Option<usize>,
) -> Result<()> {
    let runner = BenchmarkRunner::new(backend)?;
    let entries = runner.list_reports(limit)?;
    BenchmarkPrinter::print_list(&entries, limit);
    Ok(())
}

/// Compare two benchmarks
pub fn compare_benchmarks(
    backend: Box<dyn crate::backend::GpuBackend>,
    id1: &str,
    id2: &str,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let runner = BenchmarkRunner::new(backend)?;
    
    // Find full IDs from prefixes
    let full_id1 = runner.find_report(id1)?.ok_or_else(|| {
        anyhow::anyhow!("Benchmark '{}' not found", id1)
    })?;
    
    let full_id2 = runner.find_report(id2)?.ok_or_else(|| {
        anyhow::anyhow!("Benchmark '{}' not found", id2)
    })?;
    
    let baseline = runner.load_report(&full_id1)?;
    let current = runner.load_report(&full_id2)?;
    
    let comparison = compare_reports(&baseline, &current);
    
    // Export comparison if output path specified
    if let Some(path) = output {
        runner.export_comparison(&comparison, &path)?;
        println!("Comparison exported to: {}", path.display());
    }
    
    // Print comparison
    BenchmarkPrinter::print_comparison(&comparison);
    
    Ok(())
}