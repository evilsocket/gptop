#![allow(dead_code)]

#[cfg(target_os = "macos")]
pub mod apple;
#[cfg(target_os = "linux")]
pub mod nvidia;

/// Information about a GPU device (static, collected once)
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub id: usize,
    pub name: String,
    pub vendor: String,
    pub total_memory: u64,
    pub core_count: Option<u32>,
}

/// Per-GPU metrics snapshot (collected each sample)
#[derive(Clone, Debug)]
pub struct GpuMetrics {
    pub device_id: usize,
    pub utilization_pct: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub freq_mhz: u32,
    pub freq_max_mhz: u32,
    pub power_watts: f32,
    pub power_limit_watts: Option<f32>,
    pub temp_celsius: f32,
    pub fp32_tflops: Option<f32>,
    pub encoder_pct: Option<f32>,
    pub decoder_pct: Option<f32>,
    pub fan_speed_pct: Option<u32>,
    pub throttling_reason: Option<String>,
}

/// A process using the GPU
#[derive(Clone, Debug)]
pub struct GpuProcess {
    pub pid: u32,
    pub user: String,
    pub device_id: usize,
    pub name: String,
    pub gpu_usage_pct: f32,
    pub gpu_memory: u64,
    pub cpu_usage_pct: f32,
    pub host_memory: u64,
    pub process_type: String,
}

/// Aggregated system-level metrics
#[derive(Clone, Debug, Default)]
pub struct SystemMetrics {
    pub cpu_power: Option<f32>,
    pub ane_power: Option<f32>,
    pub package_power: Option<f32>,
    pub ram_used: u64,
    pub ram_total: u64,
    pub swap_used: u64,
    pub swap_total: u64,
    pub hostname: String,
    pub uptime_secs: u64,
    pub external_ip: Option<String>,
}

/// Result of a single sampling interval
#[derive(Clone, Debug)]
pub struct SampleResult {
    pub gpu_metrics: Vec<GpuMetrics>,
    pub processes: Vec<GpuProcess>,
    pub system: SystemMetrics,
}

/// The backend trait — implement for each GPU vendor
pub trait GpuBackend: Send {
    fn name(&self) -> &str;
    fn devices(&self) -> &[DeviceInfo];
    fn sample(&mut self, duration_ms: u64) -> anyhow::Result<SampleResult>;
}
