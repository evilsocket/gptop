use anyhow::{anyhow, Result};
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use super::{DeviceInfo, GpuBackend, GpuMetrics, GpuProcess, SampleResult, SystemMetrics};

pub struct NvidiaBackend {
    nvml: Nvml,
    devices: Vec<DeviceInfo>,
}

impl NvidiaBackend {
    pub fn new() -> Result<Self> {
        let nvml = Nvml::init().map_err(|e| anyhow!("Failed to initialize NVML: {}", e))?;
        let count = nvml
            .device_count()
            .map_err(|e| anyhow!("Failed to get device count: {}", e))?;

        if count == 0 {
            return Err(anyhow!("No NVIDIA devices found"));
        }

        let mut devices = Vec::new();
        for i in 0..count {
            let device = nvml
                .device_by_index(i)
                .map_err(|e| anyhow!("Failed to get device {}: {}", i, e))?;

            let name = device
                .name()
                .unwrap_or_else(|_| "Unknown NVIDIA GPU".into());
            let total_memory = device.memory_info().map(|m| m.total).unwrap_or(0);
            let core_count = device.num_cores().ok();

            devices.push(DeviceInfo {
                id: i as usize,
                name,
                vendor: "NVIDIA".to_string(),
                total_memory,
                core_count,
            });
        }

        Ok(Self { nvml, devices })
    }
}

fn used_gpu_memory_bytes(mem: UsedGpuMemory) -> u64 {
    match mem {
        UsedGpuMemory::Used(bytes) => bytes,
        UsedGpuMemory::Unavailable => 0,
    }
}

/// Helper: return 0/default on NotSupported, propagate other errors
fn ok_or_unsupported<T: Default>(result: Result<T, NvmlError>) -> Result<T, NvmlError> {
    match result {
        Ok(v) => Ok(v),
        Err(NvmlError::NotSupported) => Ok(T::default()),
        Err(e) => Err(e),
    }
}

impl GpuBackend for NvidiaBackend {
    fn name(&self) -> &str {
        "NVIDIA"
    }

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    fn sample(&mut self, duration_ms: u64) -> Result<SampleResult> {
        // NVML queries are instantaneous, so sleep first to match the sampling interval
        thread::sleep(Duration::from_millis(duration_ms));

        let mut gpu_metrics = Vec::new();
        let mut all_processes: Vec<GpuProcess> = Vec::new();

        for dev_info in &self.devices {
            let device = self.nvml.device_by_index(dev_info.id as u32)?;

            // GPU utilization
            let (gpu_util, _mem_util) = match device.utilization_rates() {
                Ok(u) => (u.gpu, u.memory),
                Err(NvmlError::NotSupported) => (0, 0),
                Err(e) => return Err(e.into()),
            };

            // Memory
            let mem_info = device.memory_info()?;

            // Clocks
            let freq_mhz = ok_or_unsupported(device.clock_info(Clock::Graphics)).unwrap_or(0);
            let freq_max_mhz =
                ok_or_unsupported(device.max_clock_info(Clock::Graphics)).unwrap_or(0);

            // Power (NVML returns milliwatts)
            let power_mw = ok_or_unsupported(device.power_usage()).unwrap_or(0);
            let power_watts = power_mw as f32 / 1000.0;
            let power_limit_watts = device
                .power_management_limit()
                .ok()
                .map(|mw| mw as f32 / 1000.0);

            // Temperature
            let temp = ok_or_unsupported(device.temperature(TemperatureSensor::Gpu)).unwrap_or(0);

            // Fan speed (fan index 0)
            let fan_speed_pct = device.fan_speed(0).ok();

            // Thermal throttling - check if temp is high (approximation since API not available)
            let throttling_reason = if temp > 85 {
                Some("HighTemp".to_string())
            } else {
                None
            };

            // Encoder/decoder utilization
            let encoder_pct = device
                .encoder_utilization()
                .ok()
                .map(|e| e.utilization as f32);
            let decoder_pct = device
                .decoder_utilization()
                .ok()
                .map(|d| d.utilization as f32);

            gpu_metrics.push(GpuMetrics {
                device_id: dev_info.id,
                utilization_pct: gpu_util as f32,
                memory_used: mem_info.used,
                memory_total: mem_info.total,
                freq_mhz,
                freq_max_mhz,
                power_watts,
                power_limit_watts,
                temp_celsius: temp as f32,
                // NVIDIA CUDA cores: 2 FP32 ops/clock (FMA)
                fp32_tflops: dev_info
                    .core_count
                    .map(|cores| cores as f32 * freq_mhz as f32 * 2.0 / 1e6),
                encoder_pct,
                decoder_pct,
                fan_speed_pct,
                throttling_reason,
            });

            // Collect processes
            let mut pid_gpu_mem: HashMap<u32, u64> = HashMap::new();

            if let Ok(procs) = device.running_compute_processes() {
                for p in procs {
                    *pid_gpu_mem.entry(p.pid).or_insert(0) +=
                        used_gpu_memory_bytes(p.used_gpu_memory);
                }
            }
            if let Ok(procs) = device.running_graphics_processes() {
                for p in procs {
                    *pid_gpu_mem.entry(p.pid).or_insert(0) +=
                        used_gpu_memory_bytes(p.used_gpu_memory);
                }
            }

            // Per-process GPU utilization
            let mut pid_gpu_pct: HashMap<u32, u32> = HashMap::new();
            if let Ok(stats) = device.process_utilization_stats(None) {
                for s in stats {
                    let entry = pid_gpu_pct.entry(s.pid).or_insert(0);
                    *entry = (*entry).max(s.sm_util);
                    // Also ensure this PID appears in our map
                    pid_gpu_mem.entry(s.pid).or_insert(0);
                }
            }

            // Check if we got any real per-process utilization data
            let have_per_process_util = pid_gpu_pct.values().any(|&v| v > 0);

            for (pid, gpu_memory) in &pid_gpu_mem {
                let pid = *pid;
                let gpu_pct = pid_gpu_pct.get(&pid).copied().unwrap_or(0) as f32;

                let (name, user, cpu_pct, host_memory) = read_proc_info(pid);

                all_processes.push(GpuProcess {
                    pid,
                    user,
                    device_id: dev_info.id,
                    name,
                    gpu_usage_pct: gpu_pct,
                    gpu_memory: *gpu_memory,
                    cpu_usage_pct: cpu_pct,
                    host_memory,
                    process_type: "GPU".to_string(),
                });
            }

            // Fallback: if NVML didn't report per-process utilization,
            // distribute overall GPU utilization proportionally by GPU memory.
            if !have_per_process_util && gpu_util > 0 {
                let total_mem: u64 = all_processes
                    .iter()
                    .filter(|p| p.device_id == dev_info.id)
                    .map(|p| p.gpu_memory)
                    .sum();
                if total_mem > 0 {
                    for proc in all_processes
                        .iter_mut()
                        .filter(|p| p.device_id == dev_info.id)
                    {
                        proc.gpu_usage_pct =
                            (proc.gpu_memory as f64 / total_mem as f64 * gpu_util as f64) as f32;
                    }
                }
            }
        }

        let system = read_system_metrics();

        Ok(SampleResult {
            gpu_metrics,
            processes: all_processes,
            system,
        })
    }
}

/// Read process info from /proc on Linux
fn read_proc_info(pid: u32) -> (String, String, f32, u64) {
    let name = std::fs::read_to_string(format!("/proc/{}/comm", pid))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let mut user = String::new();
    let mut vm_rss: u64 = 0;

    if let Ok(status) = std::fs::read_to_string(format!("/proc/{}/status", pid)) {
        for line in status.lines() {
            if let Some(val) = line.strip_prefix("Uid:") {
                let uid_str = val.split_whitespace().next().unwrap_or("0");
                let uid: u32 = uid_str.parse().unwrap_or(0);
                user = resolve_username(uid);
            } else if let Some(val) = line.strip_prefix("VmRSS:") {
                let kb_str = val.split_whitespace().next().unwrap_or("0");
                vm_rss = kb_str.parse::<u64>().unwrap_or(0) * 1024;
            }
        }
    }

    // CPU usage from /proc/<pid>/stat
    let cpu_pct = read_cpu_pct(pid);

    (name, user, cpu_pct, vm_rss)
}

/// Approximate CPU% by reading /proc/<pid>/stat utime+stime and /proc/uptime
fn read_cpu_pct(pid: u32) -> f32 {
    let stat = match std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };

    // Fields after the comm (which is in parens)
    let after_comm = match stat.rfind(')') {
        Some(pos) => &stat[pos + 2..],
        None => return 0.0,
    };

    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    if fields.len() < 20 {
        return 0.0;
    }

    // utime = field[11] (index 11 after comm), stime = field[12]
    // But in after_comm, index 0 = state, index 1 = ppid, ..., index 11 = utime, index 12 = stime
    let utime: u64 = fields[11].parse().unwrap_or(0);
    let stime: u64 = fields[12].parse().unwrap_or(0);
    let starttime: u64 = fields[19].parse().unwrap_or(0);

    let uptime = std::fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(|v| v.to_string()))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);

    let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    let total_time = (utime + stime) as f64 / clk_tck;
    let process_uptime = uptime - (starttime as f64 / clk_tck);

    if process_uptime > 0.0 {
        (total_time / process_uptime * 100.0) as f32
    } else {
        0.0
    }
}

fn resolve_username(uid: u32) -> String {
    std::fs::read_to_string("/etc/passwd")
        .ok()
        .and_then(|content| {
            content
                .lines()
                .find(|line| {
                    let mut parts = line.split(':');
                    parts.nth(2).and_then(|u| u.parse::<u32>().ok()) == Some(uid)
                })
                .and_then(|line| line.split(':').next().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| uid.to_string())
}

/// Read system memory info from /proc/meminfo
fn read_system_metrics() -> SystemMetrics {
    let mut ram_total: u64 = 0;
    let mut ram_available: u64 = 0;
    let mut swap_total: u64 = 0;
    let mut swap_free: u64 = 0;

    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or("");
            let val: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let val_bytes = val * 1024; // /proc/meminfo values are in kB

            match key {
                "MemTotal:" => ram_total = val_bytes,
                "MemAvailable:" => ram_available = val_bytes,
                "SwapTotal:" => swap_total = val_bytes,
                "SwapFree:" => swap_free = val_bytes,
                _ => {}
            }
        }
    }

    // Hostname
    let hostname = std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Uptime from /proc/uptime
    let uptime_secs = std::fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|s| {
            let first = s.split_whitespace().next()?;
            first.parse::<f64>().ok()
        })
        .map(|v| v as u64)
        .unwrap_or(0);

    // External IP (try to fetch)
    let external_ip = std::process::Command::new("sh")
        .args(["-c", "curl -s https://api.ipify.org 2>/dev/null || echo ''"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

    SystemMetrics {
        cpu_power: None,
        ane_power: None,
        package_power: None,
        ram_used: ram_total.saturating_sub(ram_available),
        ram_total,
        swap_used: swap_total.saturating_sub(swap_free),
        swap_total,
        hostname,
        uptime_secs,
        external_ip,
    }
}
