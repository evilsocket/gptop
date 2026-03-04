use super::types::*;
use std::io::Write;

pub struct BenchmarkPrinter;

impl BenchmarkPrinter {
    pub fn print_report(report: &BenchmarkReport) {
        let mut stdout = std::io::stdout();

        println!();
        Self::print_header(&mut stdout, "GPU BENCHMARK REPORT");
        println!();

        // Metadata
        Self::print_section(&mut stdout, "Metadata");
        println!("  Report ID:    {}", report.metadata.id);
        println!(
            "  Timestamp:    {}",
            report.metadata.timestamp.format("%Y-%m-%d %H:%M:%S")
        );
        println!("  Hostname:     {}", report.metadata.hostname);
        println!(
            "  Duration:     {} seconds",
            report.metadata.duration_seconds
        );
        println!(
            "  Type:         {}",
            report.metadata.benchmark_type.as_str()
        );
        println!("  Version:      {}", report.metadata.gptop_version);
        println!();

        // System Info
        Self::print_section(&mut stdout, "System Information");
        println!("  OS:           {}", report.system_info.os);
        println!("  GPU Vendor:   {}", report.system_info.gpu_vendor);
        println!("  GPU Name:     {}", report.system_info.gpu_name);
        if let Some(cores) = report.system_info.gpu_cores {
            println!("  GPU Cores:    {}", cores);
        }
        println!(
            "  Total Memory: {}",
            Self::format_bytes(report.system_info.total_memory)
        );
        println!();

        // Summary
        Self::print_section(&mut stdout, "Overall Summary");
        let reset = "\x1b[0m";
        let grade_color = report.summary.grade.color_code();
        println!(
            "  Grade:        {}{}{}",
            grade_color,
            report.summary.grade.as_str(),
            reset
        );
        println!("  Score:        {:.1}/100", report.summary.score);
        println!();

        // Devices
        for (i, device) in report.devices.iter().enumerate() {
            Self::print_section(
                &mut stdout,
                &format!("Device {}: {}", i, device.device_name),
            );
            Self::print_device_metrics(&device.overall_metrics);

            // Kernel results
            if !device.kernel_results.is_empty() {
                println!();
                println!("  Kernel Results:");
                println!();
                Self::print_kernel_table(&device.kernel_results);
            }
            println!();
        }

        // Strengths
        if !report.summary.strengths.is_empty() {
            Self::print_section(&mut stdout, "Strengths");
            for strength in &report.summary.strengths {
                println!("  ✓ {}", strength);
            }
            println!();
        }

        // Weaknesses
        if !report.summary.weaknesses.is_empty() {
            Self::print_section(&mut stdout, "Weaknesses");
            for weakness in &report.summary.weaknesses {
                println!("  ✗ {}", weakness);
            }
            println!();
        }

        // Recommendations
        if !report.summary.recommendations.is_empty() {
            Self::print_section(&mut stdout, "Recommendations");
            for rec in &report.summary.recommendations {
                println!("  → {}", rec);
            }
            println!();
        }

        Self::print_footer(&mut stdout);
    }

    pub fn print_list(entries: &[BenchmarkIndexEntry], limit: Option<usize>) {
        if entries.is_empty() {
            println!("No benchmarks found.");
            return;
        }

        let mut stdout = std::io::stdout();
        Self::print_header(&mut stdout, "SAVED BENCHMARKS");
        println!();

        // Table header
        println!(
            "  {:<20} {:<19} {:<15} {:<8} {:<10} {:<6} {:<8}",
            "ID", "Timestamp", "Hostname", "Type", "Duration", "Grade", "Score"
        );
        println!("  {}", "─".repeat(95));

        let reset = "\x1b[0m";

        for (i, entry) in entries.iter().enumerate() {
            if let Some(limit) = limit {
                if i >= limit {
                    break;
                }
            }

            let grade_color = entry.grade.color_code();

            println!(
                "  {:<20} {:<19} {:<15} {:<8} {:<10} {}{:<5}{} {:<7.1}",
                entry.id,
                entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                Self::truncate(&entry.hostname, 14),
                match entry.benchmark_type {
                    BenchmarkType::Comprehensive => "full",
                    BenchmarkType::Efficiency => "eff",
                    BenchmarkType::Compute => "comp",
                    BenchmarkType::Memory => "mem",
                    BenchmarkType::Thermal => "therm",
                },
                format!("{}s", entry.duration_seconds),
                grade_color,
                entry.grade.as_str(),
                reset,
                entry.score
            );
        }

        println!();
        println!("  Total: {} benchmark(s)", entries.len());
        println!();
    }

    pub fn print_comparison(comparison: &ComparisonResult) {
        let mut stdout = std::io::stdout();
        Self::print_header(&mut stdout, "BENCHMARK COMPARISON");
        println!();

        let reset = "\x1b[0m";
        let verdict_color = comparison.verdict.color_code();

        println!(
            "  Verdict:      {}{}{}",
            verdict_color,
            comparison.verdict.as_str(),
            reset
        );
        println!();

        Self::print_section(&mut stdout, "Comparing");
        println!(
            "  Baseline:     {} ({})",
            comparison.baseline_id,
            comparison.baseline_timestamp.format("%Y-%m-%d %H:%M:%S")
        );
        println!(
            "  Current:      {} ({})",
            comparison.current_id,
            comparison.current_timestamp.format("%Y-%m-%d %H:%M:%S")
        );
        println!();

        Self::print_section(&mut stdout, "Delta Analysis (% change)");

        Self::print_delta_line("Sustained TFLOPS", comparison.delta_pct.sustained_tflops);
        Self::print_delta_line("Peak TFLOPS", comparison.delta_pct.peak_tflops);
        Self::print_delta_line("Efficiency (GF/W)", comparison.delta_pct.avg_efficiency);
        Self::print_delta_line("Avg Power", comparison.delta_pct.avg_power);
        Self::print_delta_line("Peak Temp", comparison.delta_pct.peak_temp);

        if let Some(bw) = comparison.delta_pct.memory_bandwidth {
            Self::print_delta_line("Memory BW", bw);
        }

        if comparison.delta_pct.throttle_events != 0 {
            let color = if comparison.delta_pct.throttle_events > 0 {
                "\x1b[31m" // Red for more throttling
            } else {
                "\x1b[32m" // Green for less
            };
            println!(
                "  {:<20} {}{:+}{}",
                "Throttle Events:", color, comparison.delta_pct.throttle_events, reset
            );
        }

        println!();

        // Improvements
        if !comparison.improvements.is_empty() {
            Self::print_section(&mut stdout, "Improvements");
            for imp in &comparison.improvements {
                println!("  \x1b[32m+\x1b[0m {}", imp);
            }
            println!();
        }

        // Regressions
        if !comparison.regressions.is_empty() {
            Self::print_section(&mut stdout, "Regressions");
            for reg in &comparison.regressions {
                println!("  \x1b[31m-\x1b[0m {}", reg);
            }
            println!();
        }

        // Unchanged
        if !comparison.unchanged.is_empty() {
            Self::print_section(&mut stdout, "Unchanged");
            let unchanged_str = comparison.unchanged.join(", ");
            println!("  {}", unchanged_str);
            println!();
        }

        Self::print_footer(&mut stdout);
    }

    fn print_header(stdout: &mut std::io::Stdout, title: &str) {
        let width = 80;
        let padding = (width - title.len() - 2) / 2;
        println!("\x1b[1;36m{:=^width$}\x1b[0m", "", width = width);
        print!("\x1b[1;36m");
        for _ in 0..padding {
            print!("=");
        }
        print!(" {} ", title);
        for _ in 0..(width - padding - title.len() - 2) {
            print!("=");
        }
        println!("\x1b[0m");
        println!("\x1b[1;36m{:=^width$}\x1b[0m", "", width = width);
    }

    fn print_footer(stdout: &mut std::io::Stdout) {
        let width = 80;
        println!("\x1b[1;36m{:=^width$}\x1b[0m", "", width = width);
        println!();
    }

    fn print_section(stdout: &mut std::io::Stdout, title: &str) {
        println!("\x1b[1;33m▶ {}\x1b[0m", title);
    }

    fn print_device_metrics(metrics: &OverallMetrics) {
        println!("    Average Metrics:");
        println!("      GPU Utilization:  {:.1}%", metrics.avg_utilization);
        println!("      Power Draw:       {:.1} W", metrics.avg_power_watts);
        println!("      Temperature:      {:.1} °C", metrics.avg_temp_celsius);
        println!("    Peak Metrics:");
        println!("      GPU Utilization:  {:.1}%", metrics.peak_utilization);
        println!("      Power Draw:       {:.1} W", metrics.peak_power_watts);
        println!(
            "      Temperature:      {:.1} °C",
            metrics.peak_temp_celsius
        );
        println!("    Performance:");
        println!("      Peak TFLOPS:      {:.2}", metrics.peak_tflops);
        println!("      Sustained TFLOPS: {:.2}", metrics.sustained_tflops);
        println!(
            "      Efficiency:       {:.1} GFLOPS/Watt",
            metrics.avg_efficiency
        );
        if let Some(bw) = metrics.memory_bandwidth_gbps {
            println!("      Memory BW:        {:.0} GB/s", bw);
        }
        if metrics.total_throttle_events > 0 {
            println!(
                "    \x1b[31m⚠ Thermal throttling: {} events\x1b[0m",
                metrics.total_throttle_events
            );
        }
    }

    fn print_kernel_table(results: &std::collections::HashMap<KernelType, KernelResult>) {
        let mut kernels: Vec<_> = results.iter().collect();
        kernels.sort_by_key(|(k, _)| match k {
            KernelType::MatMulSmall => 0,
            KernelType::MatMulMedium => 1,
            KernelType::MatMulLarge => 2,
            KernelType::ElementWise => 3,
            KernelType::Bandwidth => 4,
            KernelType::ReadHeavy => 5,
            KernelType::WriteHeavy => 6,
            KernelType::Sustained => 7,
        });

        println!(
            "    {:<20} {:<10} {:<12} {:<12} {:<10}",
            "Kernel", "TFLOPS", "Power (W)", "Temp (°C)", "Eff (GF/W)"
        );
        println!("    {}", "─".repeat(68));

        for (kernel_type, result) in kernels {
            let eff_str = if result.efficiency_gflops_per_watt > 0.0 {
                format!("{:.1}", result.efficiency_gflops_per_watt)
            } else {
                "-".to_string()
            };

            println!(
                "    {:<20} {:<10.2} {:<12.1} {:<12.1} {:<10}",
                kernel_type.name(),
                result.tflops,
                result.avg_power_watts,
                result.avg_temp_celsius,
                eff_str
            );
        }
    }

    fn print_delta_line(label: &str, value: f64) {
        let reset = "\x1b[0m";
        let color = if value > 5.0 {
            "\x1b[32m" // Green for improvement
        } else if value > 0.0 {
            "\x1b[36m" // Cyan for slight improvement
        } else if value < -5.0 {
            "\x1b[31m" // Red for regression
        } else if value < 0.0 {
            "\x1b[33m" // Yellow for slight regression
        } else {
            "\x1b[37m" // White for no change
        };

        println!("  {:<20} {}{:>+.1}%{}", label, color, value, reset);
    }

    fn format_bytes(bytes: u64) -> String {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        const MIB: f64 = 1024.0 * 1024.0;
        const KIB: f64 = 1024.0;

        let b = bytes as f64;
        if b >= GIB {
            format!("{:.1} GiB", b / GIB)
        } else if b >= MIB {
            format!("{:.1} MiB", b / MIB)
        } else if b >= KIB {
            format!("{:.1} KiB", b / KIB)
        } else {
            format!("{} B", bytes)
        }
    }

    fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }
}
