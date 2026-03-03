pub mod coreutils;
pub mod hid;
pub mod ioreport;
pub mod memory;
pub mod smc;

use anyhow::{anyhow, Result};
use std::process::Command;

use super::{DeviceInfo, GpuBackend, GpuMetrics, GpuProcess, SampleResult, SystemMetrics};
use ioreport::{ChannelData, IOReportSubscription};
use smc::SmcConnection;

/// SoC information detected at startup.
struct SocInfo {
    chip_name: String,
    gpu_cores: u32,
    gpu_freq_max_mhz: u32,
    gpu_pstate_freqs: Vec<u32>, // P1, P2, P3, ... frequencies in MHz (highest first)
}

/// Apple Silicon backend.
pub struct AppleBackend {
    devices: Vec<DeviceInfo>,
    soc: SocInfo,
    ioreport: IOReportSubscription,
    smc: Option<SmcConnection>,
    hid: Option<hid::HidTempReader>,
    total_ram: u64,
}

impl AppleBackend {
    pub fn new() -> Result<Self> {
        let soc = detect_soc()?;
        let total_ram = memory::total_ram()?;

        let ioreport = IOReportSubscription::new()?;
        let smc = SmcConnection::new().ok();
        let hid = hid::HidTempReader::new().ok();

        let devices = vec![DeviceInfo {
            id: 0,
            name: format!("{} GPU", soc.chip_name),
            vendor: "Apple".to_string(),
            total_memory: total_ram, // unified memory
            core_count: Some(soc.gpu_cores),
        }];

        Ok(Self {
            devices,
            soc,
            ioreport,
            smc,
            hid,
            total_ram,
        })
    }
}

impl GpuBackend for AppleBackend {
    fn name(&self) -> &str {
        "Apple Silicon"
    }

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    fn sample(&mut self, duration_ms: u64) -> Result<SampleResult> {
        // Take IOReport delta sample
        let samples = self.ioreport.sample_delta(duration_ms)?;

        // Parse GPU metrics from IOReport samples
        let mut gpu_active_residency: i64 = 0;
        let mut gpu_total_residency: i64 = 0;
        let mut gpu_power_watts: f64 = 0.0;
        let mut cpu_power_watts: f64 = 0.0;
        let mut ane_power_watts: f64 = 0.0;
        let _package_power_watts: f64 = 0.0;

        let duration_s = duration_ms as f64 / 1000.0;

        // GPU P-state frequency lookup for this SoC
        let _gpu_pstate_freqs = &self.soc.gpu_pstate_freqs;

        for sample in &samples {
            match (sample.group.as_str(), sample.subgroup.as_str()) {
                ("GPU Stats", "GPU Performance States") => {
                    // GPUPH channel has states like OFF, P1, P2, P3, P4, P5
                    if sample.channel_name == "GPUPH" {
                        if let ChannelData::State(ref states) = sample.data {
                            for state in states {
                                let name = &state.name;
                                if name == "OFF" || name.contains("IDLE") {
                                    gpu_total_residency += state.residency;
                                    continue;
                                }
                                // P-states: P1 is highest freq, P2 next, etc.
                                gpu_active_residency += state.residency;
                                gpu_total_residency += state.residency;
                            }
                        }
                    }
                }
                ("Energy Model", _) => {
                    if let ChannelData::Integer(energy) = sample.data {
                        if energy <= 0 {
                            continue;
                        }
                        let channel = sample.channel_name.as_str();

                        // Simple aggregate channels (e.g., "GPU", "ECPU", "PCPU") are in mJ.
                        // "X Energy" channels are in nJ. We use the simple ones.
                        // CPU total = ECPU + PCPU (efficiency + performance clusters)
                        match channel {
                            "GPU" => {
                                gpu_power_watts = energy as f64 / (duration_s * 1e3);
                            }
                            "CPU Energy" => {
                                // This aggregate is actually in mJ too
                                cpu_power_watts = energy as f64 / (duration_s * 1e3);
                            }
                            "ANE" => {
                                ane_power_watts = energy as f64 / (duration_s * 1e3);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Calculate GPU utilization and frequency
        let gpu_utilization = if gpu_total_residency > 0 {
            (gpu_active_residency as f64 / gpu_total_residency as f64 * 100.0) as f32
        } else {
            0.0
        };

        // Estimate GPU frequency: when active, assume highest P-state freq as approximation
        // (proper freq requires weighting individual P-state residencies with DVFS freq table)
        let gpu_freq_mhz = if gpu_active_residency > 0 {
            self.soc.gpu_freq_max_mhz
        } else {
            0
        };

        // Temperature: try SMC first, then HID fallback
        let gpu_temp = self
            .smc
            .as_ref()
            .and_then(|s| s.read_gpu_temp())
            .or_else(|| self.hid.as_ref().and_then(|h| h.read_gpu_temp()))
            .unwrap_or(0.0);

        // System power from SMC (if available, may be more accurate than IOReport)
        let smc_system_power = self.smc.as_ref().and_then(|s| s.read_system_power());

        // Memory
        let ram_used = memory::used_ram().unwrap_or(0);
        let (swap_used, swap_total) = memory::swap_usage().unwrap_or((0, 0));

        let gpu_metrics = vec![GpuMetrics {
            device_id: 0,
            utilization_pct: gpu_utilization.clamp(0.0, 100.0),
            memory_used: ram_used,
            memory_total: self.total_ram,
            freq_mhz: gpu_freq_mhz,
            freq_max_mhz: self.soc.gpu_freq_max_mhz,
            power_watts: gpu_power_watts as f32,
            power_limit_watts: None,
            temp_celsius: gpu_temp,
            // Apple GPU cores have 128 ALUs each, 2 FP32 ops/clock (FMA)
            fp32_tflops: if gpu_freq_mhz > 0 {
                Some(self.soc.gpu_cores as f32 * 128.0 * gpu_freq_mhz as f32 * 2.0 / 1e6)
            } else {
                None
            },
            encoder_pct: None,
            decoder_pct: None,
        }];

        let system = SystemMetrics {
            cpu_power: if cpu_power_watts > 0.0 {
                Some(cpu_power_watts as f32)
            } else {
                None
            },
            ane_power: if ane_power_watts > 0.0 {
                Some(ane_power_watts as f32)
            } else {
                None
            },
            package_power: smc_system_power.or({
                let total = cpu_power_watts + gpu_power_watts + ane_power_watts;
                if total > 0.0 { Some(total as f32) } else { None }
            }),
            ram_used,
            ram_total: self.total_ram,
            swap_used,
            swap_total,
        };

        // Processes: try to get from IOKit accelerator clients
        let processes = get_gpu_processes().unwrap_or_default();

        Ok(SampleResult {
            gpu_metrics,
            processes,
            system,
        })
    }
}

/// Get known GPU P-state frequencies for a chip (highest first).
fn get_gpu_pstate_freqs(chip_name: &str, max_freq: u32) -> Vec<u32> {
    // Known DVFS tables for Apple Silicon GPUs
    if chip_name.contains("M4") {
        vec![1580, 1398, 1200, 1000, 728]
    } else if chip_name.contains("M3") || chip_name.contains("M2") {
        vec![1398, 1200, 1000, 728, 444]
    } else if chip_name.contains("M1") {
        vec![1278, 1086, 900, 720, 444]
    } else {
        // Generic fallback
        vec![max_freq]
    }
}

/// Detect SoC information using system_profiler and sysctl.
fn detect_soc() -> Result<SocInfo> {
    // Get chip name from sysctl
    let chip_name = sysctl_string("machdep.cpu.brand_string")
        .or_else(|_| {
            // Fallback: try system_profiler
            let output = Command::new("system_profiler")
                .args(["SPHardwareDataType", "-json"])
                .output()?;
            let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            json["SPHardwareDataType"][0]["chip_type"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow!("chip_type not found"))
        })
        .unwrap_or_else(|_| "Apple Silicon".to_string());

    // Clean up chip name
    let chip_name = if chip_name.contains("Apple") {
        chip_name
            .split_whitespace()
            .skip_while(|w| *w != "Apple")
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        chip_name
    };

    // GPU core count from IOKit
    let gpu_cores = detect_gpu_core_count().unwrap_or(8);

    // Max GPU frequency from IOKit DVFS tables
    let gpu_freq_max_mhz = detect_gpu_max_freq().unwrap_or(1398);

    // Known P-state frequencies for common chips (highest first)
    let gpu_pstate_freqs = get_gpu_pstate_freqs(&chip_name, gpu_freq_max_mhz);

    Ok(SocInfo {
        chip_name,
        gpu_cores,
        gpu_freq_max_mhz,
        gpu_pstate_freqs,
    })
}

fn sysctl_string(name: &str) -> Result<String> {
    let cname = std::ffi::CString::new(name)?;
    let mut size: usize = 0;
    let ret = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size as *mut usize as *mut libc::size_t,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || size == 0 {
        return Err(anyhow!("sysctlbyname {} failed", name));
    }
    let mut buf = vec![0u8; size];
    let ret = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size as *mut usize as *mut libc::size_t,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return Err(anyhow!("sysctlbyname {} read failed", name));
    }
    // Remove trailing null byte
    if buf.last() == Some(&0) {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn detect_gpu_core_count() -> Option<u32> {
    // Try system_profiler
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let displays = json["SPDisplaysDataType"].as_array()?;
    for display in displays {
        if let Some(cores) = display["sppci_cores"].as_str() {
            if let Ok(n) = cores.parse::<u32>() {
                return Some(n);
            }
        }
        // Try numeric field
        if let Some(n) = display["sppci_cores"].as_u64() {
            return Some(n as u32);
        }
    }
    None
}

fn detect_gpu_max_freq() -> Option<u32> {
    // Try to read from IOKit performance state tables via ioreg
    let output = Command::new("ioreg")
        .args(["-r", "-c", "AGXAcceleratorG13X", "-d", "1"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Look for gpu-max-freq or similar key
    for line in stdout.lines() {
        let line = line.trim();
        // Try parsing "gpu-core-count" or frequency-related fields
        if line.contains("gpu-perf-state") || line.contains("max-freq") {
            // Try to extract a number
            let num: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(freq) = num.parse::<u32>() {
                if freq > 100 && freq < 5000 {
                    return Some(freq);
                }
            }
        }
    }

    // Fallback: try known chip frequencies
    let brand = sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
    if brand.contains("M4") {
        Some(1580)
    } else if brand.contains("M3") || brand.contains("M2") {
        Some(1398)
    } else if brand.contains("M1") {
        Some(1278)
    } else {
        None
    }
}

// proc_pid_rusage FFI for getting phys_footprint (includes GPU memory)
extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u8) -> i32;
}

/// Get phys_footprint for a process via proc_pid_rusage (RUSAGE_INFO_V4).
/// This includes GPU/IOAccelerator memory mapped into the process.
fn get_phys_footprint(pid: u32) -> u64 {
    // rusage_info_v4 is ~296 bytes; phys_footprint is at offset 72
    let mut buf = [0u8; 512];
    let ret = unsafe { proc_pid_rusage(pid as i32, 4, buf.as_mut_ptr()) };
    if ret != 0 {
        return 0;
    }
    u64::from_le_bytes(buf[72..80].try_into().unwrap_or_default())
}

/// Collect GPU-using process PIDs from IOKit AGXDeviceUserClient entries.
fn get_gpu_client_pids() -> Vec<u32> {
    let output = match Command::new("ioreg")
        .args(["-r", "-c", "AGXDeviceUserClient"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut pids = Vec::new();
    for line in stdout.lines() {
        // Match: "IOUserClientCreator" = "pid 12080, cake"
        if let Some(pos) = line.find("\"IOUserClientCreator\"") {
            let rest = &line[pos..];
            if let Some(pid_pos) = rest.find("pid ") {
                let after_pid = &rest[pid_pos + 4..];
                let num_str: String = after_pid.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(pid) = num_str.parse::<u32>() {
                    pids.push(pid);
                }
            }
        }
    }

    pids.sort_unstable();
    pids.dedup();
    pids
}

fn get_gpu_processes() -> Result<Vec<GpuProcess>> {
    // Step 1: Get all PIDs that have an AGXDeviceUserClient (= GPU connection)
    let gpu_pids = get_gpu_client_pids();
    if gpu_pids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Get process info for those PIDs via ps
    let mut processes = Vec::new();
    let output = Command::new("ps")
        .args(["-eo", "pid=,user=,%cpu=,rss=,command="])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let pid: u32 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(p) => p,
            None => continue,
        };

        // Only include processes that have a GPU client
        if gpu_pids.binary_search(&pid).is_err() {
            continue;
        }

        let user = match parts.next() {
            Some(u) => u.to_string(),
            None => continue,
        };
        let cpu_pct: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let rss_kb: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let args: String = parts.collect::<Vec<_>>().join(" ");
        if args.is_empty() {
            continue;
        }

        // Extract executable name
        let exe_path = args.split_whitespace().next().unwrap_or(&args);
        let name = exe_path.rsplit('/').next().unwrap_or(exe_path).to_string();

        // Classify process type
        let process_type = if name == "WindowServer"
            || args.contains("--type=gpu")
            || name.contains("Wallpaper")
            || name.contains("Dock")
            || name.contains("Finder")
            || name.contains("SystemUI")
            || name.contains("ControlCenter")
            || name.contains("loginwindow")
            || name.contains("Notification")
        {
            "Graphics"
        } else {
            "Compute"
        };

        // Friendly display name: extract app name from .app bundle path
        let display_name = if let Some(app_pos) = args.find(".app/") {
            let before_app = &args[..app_pos];
            before_app
                .rsplit('/')
                .next()
                .unwrap_or(&name)
                .to_string()
        } else {
            name
        };

        // Get phys_footprint which includes GPU memory
        let footprint = get_phys_footprint(pid);

        processes.push(GpuProcess {
            pid,
            user,
            device_id: 0,
            name: display_name,
            gpu_usage_pct: 0.0,
            gpu_memory: footprint,
            cpu_usage_pct: cpu_pct,
            host_memory: rss_kb * 1024,
            process_type: process_type.to_string(),
        });
    }

    // Sort by CPU usage descending as a proxy for GPU activity
    processes.sort_by(|a, b| {
        b.cpu_usage_pct
            .partial_cmp(&a.cpu_usage_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(processes)
}
