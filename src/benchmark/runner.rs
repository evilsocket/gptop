use super::kernels::{GpuContext, KernelRunner, KernelStats};
use super::metrics::{
    compute_grade, generate_recommendations, generate_strengths, generate_weaknesses,
    MetricsCollector,
};
use super::storage::BenchmarkStorage;
use super::types::*;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

pub struct BenchmarkRunner {
    #[cfg(target_os = "macos")]
    metal_context: Option<super::metal_kernels::MetalContext>,
    #[cfg(target_os = "macos")]
    wgpu_context: Option<Arc<GpuContext>>,
    #[cfg(not(target_os = "macos"))]
    context: Arc<GpuContext>,
    storage: BenchmarkStorage,
    backend: Box<dyn crate::backend::GpuBackend>,
}

impl BenchmarkRunner {
    pub fn new(backend: Box<dyn crate::backend::GpuBackend>) -> Result<Self> {
        let storage = BenchmarkStorage::new()?;

        #[cfg(target_os = "macos")]
        {
            // Use wgpu backend (Metal has issues)
            println!("Using wgpu backend for cross-platform compatibility");
            let context = Arc::new(pollster::block_on(GpuContext::new())?);
            Ok(Self {
                metal_context: None,
                wgpu_context: Some(context),
                storage,
                backend,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let context = Arc::new(pollster::block_on(GpuContext::new())?);
            Ok(Self {
                context,
                storage,
                backend,
            })
        }
    }

    #[cfg(target_os = "macos")]
    fn run_kernel(&self, kernel_type: KernelType, duration_ms: u64) -> Result<KernelStats> {
        if let Some(ref ctx) = self.metal_context {
            let metal_stats = match kernel_type {
                KernelType::MatMulSmall => ctx.run_matmul(512, duration_ms),
                KernelType::MatMulMedium => ctx.run_matmul(1024, duration_ms),
                KernelType::MatMulLarge => ctx.run_matmul(1536, duration_ms),
                KernelType::ElementWise => ctx.run_element_wise(duration_ms),
                KernelType::Bandwidth => ctx.run_bandwidth(duration_ms),
                _ => ctx.run_matmul(1024, duration_ms), // Default fallback
            }?;

            // Convert metal stats to kernel stats
            Ok(KernelStats {
                duration_ms: metal_stats.duration_ms,
                operations: metal_stats.operations,
                tflops: metal_stats.tflops,
                bandwidth_gbps: metal_stats.bandwidth_gbps,
                dispatches: metal_stats.dispatches,
            })
        } else if let Some(ref ctx) = self.wgpu_context {
            let runner = KernelRunner::new(ctx.clone());
            pollster::block_on(runner.run_kernel(kernel_type, duration_ms))
        } else {
            Err(anyhow::anyhow!("No compute backend available"))
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn run_kernel(&self, kernel_type: KernelType, duration_ms: u64) -> Result<KernelStats> {
        let runner = KernelRunner::new(self.context.clone());
        pollster::block_on(runner.run_kernel(kernel_type, duration_ms))
    }

    pub fn run(
        &mut self,
        duration_seconds: u64,
        benchmark_type: BenchmarkType,
    ) -> Result<BenchmarkReport> {
        println!(
            "Starting {} benchmark for {} seconds...",
            benchmark_type.as_str(),
            duration_seconds
        );
        println!();

        let kernels = benchmark_type.kernels();
        let kernel_duration_ms = (duration_seconds * 1000) / kernels.len().max(1) as u64;

        let devices_info = self.backend.devices().to_vec();
        let mut device_benchmarks = Vec::new();

        // Get system info
        #[cfg(target_os = "macos")]
        let (gpu_name, total_memory) = if let Some(ref ctx) = self.metal_context {
            ctx.gpu_info()
        } else {
            (String::new(), 0)
        };

        #[cfg(not(target_os = "macos"))]
        let (gpu_name, total_memory) = self.context.gpu_info();

        let os = std::env::consts::OS.to_string();
        let vendor = if os == "macos" { "Apple" } else { "NVIDIA/AMD" }.to_string();

        let system_info = SystemInfo {
            os: os.clone(),
            gpu_vendor: vendor.clone(),
            gpu_name: if gpu_name.is_empty() {
                devices_info
                    .first()
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string())
            } else {
                gpu_name
            },
            gpu_cores: devices_info.first().and_then(|d| d.core_count),
            total_memory: devices_info
                .first()
                .map(|d| d.total_memory)
                .unwrap_or(total_memory),
        };

        for device in &devices_info {
            println!("Benchmarking device: {}", device.name);

            let mut collector = MetricsCollector::new();
            let mut kernel_results = HashMap::new();
            let mut total_bandwidth = 0.0;
            let mut bandwidth_count = 0;

            for kernel_type in &kernels {
                print!("  Running {}... ", kernel_type.name());
                std::io::Write::flush(&mut std::io::stdout()).unwrap();

                // Run the kernel
                let kernel_stats = self.run_kernel(*kernel_type, kernel_duration_ms)?;

                // Collect GPU metrics during kernel execution
                let sample = self.backend.sample(100)?; // 100ms sample
                for metrics in sample.gpu_metrics {
                    if metrics.device_id == device.id {
                        collector.add_sample(metrics);
                    }
                }

                // Convert to kernel result
                let kernel_result =
                    collector.kernel_result_from_samples(*kernel_type, &kernel_stats);

                if let Some(bw) = kernel_result.bandwidth_gbps {
                    total_bandwidth += bw;
                    bandwidth_count += 1;
                }

                kernel_results.insert(*kernel_type, kernel_result);

                println!("Done (TFLOPS: {:.2})", kernel_stats.tflops);
            }

            // Compute overall metrics
            let mut overall_metrics = collector.compute_overall_metrics();
            if bandwidth_count > 0 {
                overall_metrics.memory_bandwidth_gbps =
                    Some(total_bandwidth / bandwidth_count as f64);
            }

            device_benchmarks.push(DeviceBenchmark {
                device_id: device.id,
                device_name: device.name.clone(),
                kernel_results,
                overall_metrics,
            });

            println!();
        }

        // Compute summary from first device (or aggregate if multi-GPU)
        let summary = if let Some(first_device) = device_benchmarks.first() {
            let (grade, score) = compute_grade(&first_device.overall_metrics, benchmark_type);

            BenchmarkSummary {
                grade,
                score,
                strengths: generate_strengths(&first_device.overall_metrics),
                weaknesses: generate_weaknesses(&first_device.overall_metrics),
                recommendations: generate_recommendations(
                    &first_device.overall_metrics,
                    benchmark_type,
                ),
            }
        } else {
            BenchmarkSummary {
                grade: Grade::F,
                score: 0.0,
                strengths: vec![],
                weaknesses: vec!["No devices found".to_string()],
                recommendations: vec!["Check GPU availability".to_string()],
            }
        };

        let report = BenchmarkReport {
            metadata: BenchmarkMetadata {
                id: String::new(), // Will be generated on save
                timestamp: chrono::Local::now(),
                hostname: hostname::get()
                    .map(|h| h.to_string_lossy().to_string())
                    .unwrap_or_else(|_| "unknown".to_string()),
                gptop_version: env!("CARGO_PKG_VERSION").to_string(),
                duration_seconds,
                benchmark_type,
                kernels_run: kernels,
            },
            system_info,
            devices: device_benchmarks,
            summary,
        };

        Ok(report)
    }

    pub fn save_report(&self, report: &mut BenchmarkReport) -> Result<()> {
        self.storage.save(report)
    }

    pub fn list_reports(&self, limit: Option<usize>) -> Result<Vec<BenchmarkIndexEntry>> {
        self.storage.list(limit)
    }

    pub fn load_report(&self, id: &str) -> Result<BenchmarkReport> {
        self.storage.load(id)
    }

    pub fn find_report(&self, prefix: &str) -> Result<Option<String>> {
        self.storage.find_by_prefix(prefix)
    }

    pub fn export_comparison(
        &self,
        comparison: &ComparisonResult,
        path: &std::path::Path,
    ) -> Result<()> {
        self.storage.export_comparison(comparison, path)
    }
}
