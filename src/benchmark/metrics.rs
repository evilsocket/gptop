use super::kernels::KernelStats;
use super::types::*;
use crate::backend::GpuMetrics;

pub struct MetricsCollector {
    samples: Vec<SamplePoint>,
}

#[derive(Debug, Clone)]
pub struct SamplePoint {
    pub timestamp_ms: u64,
    pub gpu_metrics: GpuMetrics,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, gpu_metrics: GpuMetrics) {
        self.samples.push(SamplePoint {
            timestamp_ms: self.samples.len() as u64 * 1000, // Approximate
            gpu_metrics,
        });
    }

    pub fn compute_overall_metrics(&self) -> OverallMetrics {
        if self.samples.is_empty() {
            return OverallMetrics::default();
        }

        let utilizations: Vec<f32> = self
            .samples
            .iter()
            .map(|s| s.gpu_metrics.utilization_pct)
            .collect();
        let powers: Vec<f32> = self
            .samples
            .iter()
            .map(|s| s.gpu_metrics.power_watts)
            .collect();
        let temps: Vec<f32> = self
            .samples
            .iter()
            .map(|s| s.gpu_metrics.temp_celsius)
            .collect();
        let tflops: Vec<f32> = self
            .samples
            .iter()
            .filter_map(|s| s.gpu_metrics.fp32_tflops)
            .collect();
        let efficiencies: Vec<f32> = self
            .samples
            .iter()
            .filter_map(|s| s.gpu_metrics.efficiency_gflops_per_watt)
            .collect();

        OverallMetrics {
            avg_utilization: average_f32(&utilizations),
            peak_utilization: max_f32(&utilizations),
            avg_power_watts: average_f32(&powers),
            peak_power_watts: max_f32(&powers),
            avg_temp_celsius: average_f32(&temps),
            peak_temp_celsius: max_f32(&temps),
            sustained_tflops: average_f32(&tflops) as f64,
            peak_tflops: max_f32(&tflops) as f64,
            avg_efficiency: average_f32(&efficiencies) as f64,
            memory_bandwidth_gbps: None, // Set from kernel results
            total_throttle_events: self.count_throttle_events(),
        }
    }

    fn count_throttle_events(&self) -> u32 {
        self.samples
            .iter()
            .filter(|s| s.gpu_metrics.throttling_reason.is_some())
            .count() as u32
    }

    pub fn kernel_result_from_samples(
        &self,
        kernel_type: KernelType,
        kernel_stats: &KernelStats,
    ) -> KernelResult {
        let start_idx = self
            .samples
            .len()
            .saturating_sub((kernel_stats.duration_ms / 100).max(1) as usize);
        let relevant_samples = &self.samples[start_idx..];

        let powers: Vec<f32> = relevant_samples
            .iter()
            .map(|s| s.gpu_metrics.power_watts)
            .collect();
        let temps: Vec<f32> = relevant_samples
            .iter()
            .map(|s| s.gpu_metrics.temp_celsius)
            .collect();
        let efficiencies: Vec<f32> = relevant_samples
            .iter()
            .filter_map(|s| s.gpu_metrics.efficiency_gflops_per_watt)
            .collect();

        KernelResult {
            kernel_type,
            duration_ms: kernel_stats.duration_ms,
            operations: kernel_stats.operations,
            tflops: kernel_stats.tflops,
            bandwidth_gbps: kernel_stats.bandwidth_gbps,
            avg_power_watts: average_f32(&powers),
            peak_power_watts: max_f32(&powers),
            avg_temp_celsius: average_f32(&temps),
            peak_temp_celsius: max_f32(&temps),
            efficiency_gflops_per_watt: average_f32(&efficiencies) as f64,
            throttle_events: relevant_samples
                .iter()
                .filter(|s| s.gpu_metrics.throttling_reason.is_some())
                .count() as u32,
        }
    }
}

fn average_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn max_f32(values: &[f32]) -> f32 {
    values.iter().cloned().fold(0.0, f32::max)
}

pub fn compute_grade(metrics: &OverallMetrics, benchmark_type: BenchmarkType) -> (Grade, f64) {
    let score = match benchmark_type {
        BenchmarkType::Comprehensive => compute_comprehensive_score(metrics),
        BenchmarkType::Efficiency => compute_efficiency_score(metrics),
        BenchmarkType::Compute => compute_compute_score(metrics),
        BenchmarkType::Memory => compute_memory_score(metrics),
        BenchmarkType::Thermal => compute_thermal_score(metrics),
    };

    (Grade::from_score(score), score)
}

fn compute_comprehensive_score(metrics: &OverallMetrics) -> f64 {
    let mut score = 0.0;

    // Compute performance (40%)
    score += (metrics.peak_tflops.min(20.0) / 20.0) * 40.0;

    // Sustained performance (25%)
    score += (metrics.sustained_tflops.min(20.0) / 20.0) * 25.0;

    // Efficiency (20%)
    score += (metrics.avg_efficiency.min(10.0) / 10.0) * 20.0;

    // Thermal stability (15%) - penalize high temps and throttling
    let peak_temp = metrics.peak_temp_celsius as f64;
    let thermal_score = if peak_temp > 90.0 {
        0.0
    } else if peak_temp > 80.0 {
        50.0 - (peak_temp - 80.0) * 5.0
    } else {
        100.0
    };
    score += thermal_score * 0.15;

    // Penalize throttling
    score -= metrics.total_throttle_events as f64 * 2.0;

    score.clamp(0.0, 100.0)
}

fn compute_efficiency_score(metrics: &OverallMetrics) -> f64 {
    let mut score = 0.0;

    // Efficiency is the primary metric (70%)
    score += (metrics.avg_efficiency.min(15.0) as f64 / 15.0) * 70.0;

    // Power efficiency at different loads (20%)
    let power_score = if metrics.avg_power_watts > 0.0 {
        let perf_per_watt = metrics.sustained_tflops / metrics.avg_power_watts as f64;
        (perf_per_watt.min(1.0) / 1.0) * 20.0
    } else {
        0.0_f64
    };
    score += power_score;

    // Thermal efficiency (10%)
    let avg_temp = metrics.avg_temp_celsius as f64;
    let thermal_score = if avg_temp < 70.0 {
        100.0
    } else if avg_temp < 85.0 {
        100.0 - (avg_temp - 70.0) * 6.67
    } else {
        0.0
    };
    score += thermal_score * 0.10;

    score.clamp(0.0, 100.0)
}

fn compute_compute_score(metrics: &OverallMetrics) -> f64 {
    let mut score = 0.0;

    // Peak compute (50%)
    score += (metrics.peak_tflops.min(25.0) / 25.0) * 50.0;

    // Sustained compute (35%)
    score += (metrics.sustained_tflops.min(25.0) / 25.0) * 35.0;

    // Utilization (15%)
    score += (metrics.avg_utilization as f64 / 100.0) * 15.0;

    score.clamp(0.0, 100.0)
}

fn compute_memory_score(metrics: &OverallMetrics) -> f64 {
    let mut score = 0.0;

    // Memory bandwidth (60%)
    if let Some(bw) = metrics.memory_bandwidth_gbps {
        score += (bw.min(1000.0) / 1000.0) * 60.0;
    }

    // Compute efficiency with memory (40%)
    score += (metrics.avg_efficiency.min(10.0) / 10.0) * 40.0;

    score.clamp(0.0, 100.0)
}

fn compute_thermal_score(metrics: &OverallMetrics) -> f64 {
    let mut score: f64 = 0.0;

    // Temperature stability (60%)
    let peak_temp = metrics.peak_temp_celsius as f64;
    let temp_score = if peak_temp < 80.0 {
        100.0
    } else if peak_temp < 95.0 {
        100.0 - (peak_temp - 80.0) * 6.67
    } else {
        0.0
    };
    score += temp_score * 0.60;

    // Sustained performance under thermal load (30%)
    let sustained_ratio = if metrics.peak_tflops > 0.0 {
        (metrics.sustained_tflops / metrics.peak_tflops).min(1.0)
    } else {
        0.0
    };
    score += sustained_ratio * 30.0;

    // No throttling (10%)
    score += if metrics.total_throttle_events == 0 {
        10.0
    } else {
        0.0
    };

    score.clamp(0.0, 100.0)
}

pub fn generate_recommendations(
    metrics: &OverallMetrics,
    benchmark_type: BenchmarkType,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Throttling detection
    if metrics.total_throttle_events > 0 {
        recommendations.push(format!(
            "Thermal throttling detected ({} events). Consider improving cooling or reducing load.",
            metrics.total_throttle_events
        ));
    }

    // Temperature warnings
    if metrics.peak_temp_celsius > 85.0 {
        recommendations.push(format!(
            "Peak temperature reached {:.0}°C. Risk of thermal throttling.",
            metrics.peak_temp_celsius
        ));
    }

    // Performance vs sustained gap
    if metrics.peak_tflops > 0.0 {
        let ratio = metrics.sustained_tflops / metrics.peak_tflops;
        if ratio < 0.7 {
            recommendations.push(format!(
                "Sustained performance is {:.0}% of peak. Thermal or power limiting may be occurring.",
                ratio * 100.0
            ));
        }
    }

    // Efficiency suggestions
    if metrics.avg_efficiency < 3.0 {
        recommendations.push(
            "Low power efficiency. Consider workload optimization or power profile adjustment."
                .to_string(),
        );
    }

    // Type-specific recommendations
    match benchmark_type {
        BenchmarkType::Comprehensive => {
            if metrics.avg_utilization < 50.0 {
                recommendations.push(
                    "GPU underutilized. Workload may be CPU-bound or insufficiently parallel."
                        .to_string(),
                );
            }
        }
        BenchmarkType::Memory => {
            if metrics.memory_bandwidth_gbps.map_or(true, |bw| bw < 200.0) {
                recommendations.push(
                    "Memory bandwidth appears limited. Check memory access patterns.".to_string(),
                );
            }
        }
        BenchmarkType::Thermal => {
            if metrics.avg_temp_celsius > 75.0 {
                recommendations.push(
                    "Average temperature is high. Consider active cooling solutions.".to_string(),
                );
            }
        }
        _ => {}
    }

    recommendations
}

pub fn generate_strengths(metrics: &OverallMetrics) -> Vec<String> {
    let mut strengths = Vec::new();

    if metrics.peak_tflops > 10.0 {
        strengths.push(format!(
            "Strong peak compute: {:.1} TFLOPS",
            metrics.peak_tflops
        ));
    }

    if metrics.avg_efficiency > 5.0 {
        strengths.push(format!(
            "Excellent efficiency: {:.1} GFLOPS/Watt",
            metrics.avg_efficiency
        ));
    }

    if metrics.total_throttle_events == 0 && metrics.peak_temp_celsius < 80.0 {
        strengths.push("Excellent thermal management".to_string());
    }

    if metrics.sustained_tflops > 0.0 && metrics.peak_tflops > 0.0 {
        let ratio = metrics.sustained_tflops / metrics.peak_tflops;
        if ratio > 0.9 {
            strengths.push("Consistent sustained performance".to_string());
        }
    }

    if let Some(bw) = metrics.memory_bandwidth_gbps {
        if bw > 400.0 {
            strengths.push(format!("High memory bandwidth: {:.0} GB/s", bw));
        }
    }

    strengths
}

pub fn generate_weaknesses(metrics: &OverallMetrics) -> Vec<String> {
    let mut weaknesses = Vec::new();

    if metrics.total_throttle_events > 0 {
        weaknesses.push(format!(
            "{} thermal throttle events",
            metrics.total_throttle_events
        ));
    }

    if metrics.peak_temp_celsius > 85.0 {
        weaknesses.push(format!(
            "High peak temperature: {:.0}°C",
            metrics.peak_temp_celsius
        ));
    }

    if metrics.peak_tflops > 0.0 {
        let ratio = metrics.sustained_tflops / metrics.peak_tflops;
        if ratio < 0.7 {
            weaknesses.push(format!("Sustained/peak ratio: {:.0}%", ratio * 100.0));
        }
    }

    if metrics.avg_efficiency < 2.0 {
        weaknesses.push(format!(
            "Low efficiency: {:.1} GFLOPS/Watt",
            metrics.avg_efficiency
        ));
    }

    weaknesses
}

impl Default for OverallMetrics {
    fn default() -> Self {
        Self {
            avg_utilization: 0.0,
            peak_utilization: 0.0,
            avg_power_watts: 0.0,
            peak_power_watts: 0.0,
            avg_temp_celsius: 0.0,
            peak_temp_celsius: 0.0,
            sustained_tflops: 0.0,
            peak_tflops: 0.0,
            avg_efficiency: 0.0,
            memory_bandwidth_gbps: None,
            total_throttle_events: 0,
        }
    }
}
