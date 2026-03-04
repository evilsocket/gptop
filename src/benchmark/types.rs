use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkType {
    Comprehensive,
    Efficiency,
    Compute,
    Memory,
    Thermal,
}

impl BenchmarkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BenchmarkType::Comprehensive => "comprehensive",
            BenchmarkType::Efficiency => "efficiency",
            BenchmarkType::Compute => "compute",
            BenchmarkType::Memory => "memory",
            BenchmarkType::Thermal => "thermal",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            BenchmarkType::Comprehensive => {
                "Full spectrum testing covering compute, memory, and thermal characteristics"
            }
            BenchmarkType::Efficiency => {
                "Power efficiency focus: GFLOPS/Watt under varying workloads"
            }
            BenchmarkType::Compute => {
                "Compute-heavy workloads: matrix multiplication, element-wise ops"
            }
            BenchmarkType::Memory => "Memory bandwidth and throughput testing",
            BenchmarkType::Thermal => "Thermal stability: sustained load, throttling detection",
        }
    }

    pub fn kernels(&self) -> Vec<KernelType> {
        match self {
            BenchmarkType::Comprehensive => vec![
                KernelType::MatMulMedium,
                KernelType::MatMulSmall,
                KernelType::ElementWise,
                KernelType::Bandwidth,
            ],
            BenchmarkType::Efficiency => vec![KernelType::MatMulSmall],
            BenchmarkType::Compute => vec![
                KernelType::MatMulMedium,
                KernelType::MatMulSmall,
                KernelType::ElementWise,
            ],
            BenchmarkType::Memory => vec![
                KernelType::Bandwidth,
                KernelType::ReadHeavy,
                KernelType::WriteHeavy,
            ],
            BenchmarkType::Thermal => vec![KernelType::Sustained],
        }
    }
}

impl std::str::FromStr for BenchmarkType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "comprehensive" | "full" => Ok(BenchmarkType::Comprehensive),
            "efficiency" | "eff" => Ok(BenchmarkType::Efficiency),
            "compute" | "comp" => Ok(BenchmarkType::Compute),
            "memory" | "mem" => Ok(BenchmarkType::Memory),
            "thermal" | "temp" => Ok(BenchmarkType::Thermal),
            _ => Err(anyhow::anyhow!("Unknown benchmark type: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelType {
    MatMulSmall,  // 512x512 matrices
    MatMulMedium, // 1024x1024 matrices
    MatMulLarge,  // 1536x1536 matrices - high compute intensity
    ElementWise,  // Element-wise operations
    Bandwidth,    // Memory bandwidth test
    ReadHeavy,    // Memory read bound
    WriteHeavy,   // Memory write bound
    Sustained,    // Sustained compute for thermal testing
}

impl KernelType {
    pub fn name(&self) -> &'static str {
        match self {
            KernelType::MatMulSmall => "MatMul 512x512",
            KernelType::MatMulMedium => "MatMul 1024x1024",
            KernelType::MatMulLarge => "MatMul 1536x1536",
            KernelType::ElementWise => "Element-wise ops",
            KernelType::Bandwidth => "Bandwidth",
            KernelType::ReadHeavy => "Read-heavy",
            KernelType::WriteHeavy => "Write-heavy",
            KernelType::Sustained => "Sustained load",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            KernelType::MatMulSmall => "Small matrix multiplication",
            KernelType::MatMulMedium => "Medium matrix multiplication",
            KernelType::MatMulLarge => "Large matrix multiplication - high compute intensity",
            KernelType::ElementWise => "Element-wise vector operations",
            KernelType::Bandwidth => "Global memory bandwidth test",
            KernelType::ReadHeavy => "Memory read bandwidth",
            KernelType::WriteHeavy => "Memory write bandwidth",
            KernelType::Sustained => "Continuous compute for thermal analysis",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub metadata: BenchmarkMetadata,
    pub system_info: SystemInfo,
    pub devices: Vec<DeviceBenchmark>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub id: String,
    pub timestamp: DateTime<Local>,
    pub hostname: String,
    pub gptop_version: String,
    pub duration_seconds: u64,
    pub benchmark_type: BenchmarkType,
    pub kernels_run: Vec<KernelType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub gpu_vendor: String,
    pub gpu_name: String,
    pub gpu_cores: Option<u32>,
    pub total_memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceBenchmark {
    pub device_id: usize,
    pub device_name: String,
    pub kernel_results: HashMap<KernelType, KernelResult>,
    pub overall_metrics: OverallMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelResult {
    pub kernel_type: KernelType,
    pub duration_ms: u64,
    pub operations: u64,
    pub tflops: f64,
    pub bandwidth_gbps: Option<f64>,
    pub avg_power_watts: f32,
    pub peak_power_watts: f32,
    pub avg_temp_celsius: f32,
    pub peak_temp_celsius: f32,
    pub efficiency_gflops_per_watt: f64,
    pub throttle_events: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    pub avg_utilization: f32,
    pub peak_utilization: f32,
    pub avg_power_watts: f32,
    pub peak_power_watts: f32,
    pub avg_temp_celsius: f32,
    pub peak_temp_celsius: f32,
    pub sustained_tflops: f64,
    pub peak_tflops: f64,
    pub avg_efficiency: f64,
    pub memory_bandwidth_gbps: Option<f64>,
    pub total_throttle_events: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub grade: Grade,
    pub score: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    APlus,
    A,
    AMinus,
    BPlus,
    B,
    BMinus,
    CPlus,
    C,
    CMinus,
    D,
    F,
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 97.0 => Grade::APlus,
            s if s >= 93.0 => Grade::A,
            s if s >= 90.0 => Grade::AMinus,
            s if s >= 87.0 => Grade::BPlus,
            s if s >= 83.0 => Grade::B,
            s if s >= 80.0 => Grade::BMinus,
            s if s >= 77.0 => Grade::CPlus,
            s if s >= 73.0 => Grade::C,
            s if s >= 70.0 => Grade::CMinus,
            s if s >= 60.0 => Grade::D,
            _ => Grade::F,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Grade::APlus => "A+",
            Grade::A => "A",
            Grade::AMinus => "A-",
            Grade::BPlus => "B+",
            Grade::B => "B",
            Grade::BMinus => "B-",
            Grade::CPlus => "C+",
            Grade::C => "C",
            Grade::CMinus => "C-",
            Grade::D => "D",
            Grade::F => "F",
        }
    }

    pub fn color_code(&self) -> &'static str {
        match self {
            Grade::APlus | Grade::A | Grade::AMinus => "\x1b[32m", // Green
            Grade::BPlus | Grade::B | Grade::BMinus => "\x1b[36m", // Cyan
            Grade::CPlus | Grade::C | Grade::CMinus => "\x1b[33m", // Yellow
            Grade::D => "\x1b[35m",                                // Magenta
            Grade::F => "\x1b[31m",                                // Red
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub baseline_id: String,
    pub current_id: String,
    pub baseline_timestamp: DateTime<Local>,
    pub current_timestamp: DateTime<Local>,
    pub delta_pct: DeltaMetrics,
    pub improvements: Vec<String>,
    pub regressions: Vec<String>,
    pub unchanged: Vec<String>,
    pub verdict: ComparisonVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMetrics {
    pub sustained_tflops: f64,
    pub peak_tflops: f64,
    pub avg_efficiency: f64,
    pub avg_power: f64,
    pub peak_temp: f64,
    pub memory_bandwidth: Option<f64>,
    pub throttle_events: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonVerdict {
    SignificantImprovement,
    Improvement,
    Neutral,
    Regression,
    SignificantRegression,
}

impl ComparisonVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComparisonVerdict::SignificantImprovement => "SIGNIFICANT IMPROVEMENT",
            ComparisonVerdict::Improvement => "IMPROVEMENT",
            ComparisonVerdict::Neutral => "NEUTRAL",
            ComparisonVerdict::Regression => "REGRESSION",
            ComparisonVerdict::SignificantRegression => "SIGNIFICANT REGRESSION",
        }
    }

    pub fn color_code(&self) -> &'static str {
        match self {
            ComparisonVerdict::SignificantImprovement => "\x1b[32m", // Green
            ComparisonVerdict::Improvement => "\x1b[92m",            // Light green
            ComparisonVerdict::Neutral => "\x1b[36m",                // Cyan
            ComparisonVerdict::Regression => "\x1b[33m",             // Yellow
            ComparisonVerdict::SignificantRegression => "\x1b[31m",  // Red
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkIndexEntry {
    pub id: String,
    pub timestamp: DateTime<Local>,
    pub hostname: String,
    pub benchmark_type: BenchmarkType,
    pub duration_seconds: u64,
    pub gpu_name: String,
    pub grade: Grade,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkIndex {
    pub entries: Vec<BenchmarkIndexEntry>,
}
