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
        // Take IOReport delta sample (sleeps for duration_ms)
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
                if total > 0.0 {
                    Some(total as f32)
                } else {
                    None
                }
            }),
            ram_used,
            ram_total: self.total_ram,
            swap_used,
            swap_total,
        };

        // Processes: get from IOKit accelerator clients
        let mut processes = get_gpu_processes().unwrap_or_default();

        // Distribute overall GPU utilization proportionally by GPU memory footprint.
        // macOS doesn't expose per-process GPU time, so this is the best approximation.
        let total_footprint: u64 = processes.iter().map(|p| p.gpu_memory).sum();
        if total_footprint > 0 {
            for proc in &mut processes {
                proc.gpu_usage_pct = (proc.gpu_memory as f64 / total_footprint as f64
                    * gpu_utilization as f64) as f32;
            }
        }

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
    // Try IOKit first: query AGXAccelerator service for "gpu-core-count" property
    let result = unsafe { detect_gpu_core_count_iokit() };
    if result.is_some() {
        return result;
    }

    // Fallback: system_profiler
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
        if let Some(n) = display["sppci_cores"].as_u64() {
            return Some(n as u32);
        }
    }
    None
}

/// Query IOKit for GPU core count from AGXAccelerator* service properties.
unsafe fn detect_gpu_core_count_iokit() -> Option<u32> {
    // Try multiple AGXAccelerator class names across chip generations
    let class_names = [
        "AGXAcceleratorG13X",
        "AGXAcceleratorG14X",
        "AGXAcceleratorG15X",
        "AGXAccelerator",
    ];

    for class in &class_names {
        let cname = std::ffi::CString::new(*class).ok()?;
        let matching = smc::IOServiceMatching(cname.as_ptr());
        if matching.is_null() {
            continue;
        }

        let mut iterator: u32 = 0;
        let kr = smc::IOServiceGetMatchingServices(0, matching, &mut iterator);
        if kr != 0 || iterator == 0 {
            continue;
        }

        let service = smc::IOIteratorNext(iterator);
        if service != 0 {
            let mut props: core_foundation_sys::dictionary::CFMutableDictionaryRef =
                std::ptr::null_mut();
            let kr =
                smc::IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);

            if kr == 0 && !props.is_null() {
                let dict_ref = props as core_foundation_sys::dictionary::CFDictionaryRef;
                if let Some(val) = coreutils::cfdict_get_value(dict_ref, "gpu-core-count") {
                    let result =
                        coreutils::cfnum_to_i64(val as core_foundation_sys::number::CFNumberRef)
                            .map(|n| n as u32);
                    coreutils::safe_cfrelease(val);
                    core_foundation_sys::base::CFRelease(
                        props as core_foundation_sys::base::CFTypeRef,
                    );
                    smc::IOObjectRelease(service);
                    smc::IOObjectRelease(iterator);
                    if result.is_some() {
                        return result;
                    }
                } else {
                    core_foundation_sys::base::CFRelease(
                        props as core_foundation_sys::base::CFTypeRef,
                    );
                }
            }

            smc::IOObjectRelease(service);
        }

        smc::IOObjectRelease(iterator);
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

/// Public wrapper for proc_pid_rusage FFI (used by app.rs for native process details).
pub unsafe fn proc_pid_rusage_raw(pid: i32, flavor: i32, buffer: *mut u8) -> i32 {
    proc_pid_rusage(pid, flavor, buffer)
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

/// Collect GPU-using process PIDs from IOKit AGXDeviceUserClient entries
/// using direct IOKit API (no ioreg subprocess).
/// Extract PID from IOUserClientCreator property of an IOKit child entry.
fn extract_pid_from_entry(entry: u32) -> Option<u32> {
    unsafe {
        let mut props: core_foundation_sys::dictionary::CFMutableDictionaryRef =
            std::ptr::null_mut();
        let kr = smc::IORegistryEntryCreateCFProperties(entry, &mut props, std::ptr::null(), 0);
        if kr != 0 || props.is_null() {
            return None;
        }

        let dict_ref = props as core_foundation_sys::dictionary::CFDictionaryRef;
        let result = coreutils::cfdict_get_value(dict_ref, "IOUserClientCreator").and_then(|val| {
            let cf_str = val as core_foundation_sys::string::CFStringRef;
            let creator = coreutils::from_cfstring(cf_str);
            coreutils::safe_cfrelease(val);
            creator.and_then(|s| {
                s.find("pid ").and_then(|pos| {
                    let after = &s[pos + 4..];
                    let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                    num.parse::<u32>().ok()
                })
            })
        });

        core_foundation_sys::base::CFRelease(props as core_foundation_sys::base::CFTypeRef);
        result
    }
}

fn get_gpu_client_pids() -> Vec<u32> {
    let mut pids = Vec::new();

    // AGXDeviceUserClient entries are children of the AGXAccelerator service,
    // not independently registered. Find the accelerator, then iterate children.
    let accelerator_classes = [
        "AGXAcceleratorG13X",
        "AGXAcceleratorG14X",
        "AGXAcceleratorG15X",
        "AGXAccelerator",
    ];

    unsafe {
        let io_service_plane = std::ffi::CString::new("IOService").unwrap();

        for class in &accelerator_classes {
            let cname = match std::ffi::CString::new(*class) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let matching = smc::IOServiceMatching(cname.as_ptr());
            if matching.is_null() {
                continue;
            }

            let mut accel_iter: u32 = 0;
            let kr = smc::IOServiceGetMatchingServices(0, matching, &mut accel_iter);
            if kr != 0 || accel_iter == 0 {
                continue;
            }

            // Iterate all accelerator services
            loop {
                let accel = smc::IOIteratorNext(accel_iter);
                if accel == 0 {
                    break;
                }

                // Get children of this accelerator in the IOService plane
                let mut child_iter: u32 = 0;
                let kr = smc::IORegistryEntryGetChildIterator(
                    accel,
                    io_service_plane.as_ptr(),
                    &mut child_iter,
                );

                if kr == 0 && child_iter != 0 {
                    loop {
                        let child = smc::IOIteratorNext(child_iter);
                        if child == 0 {
                            break;
                        }
                        if let Some(pid) = extract_pid_from_entry(child) {
                            pids.push(pid);
                        }
                        smc::IOObjectRelease(child);
                    }
                    smc::IOObjectRelease(child_iter);
                }

                smc::IOObjectRelease(accel);
            }

            smc::IOObjectRelease(accel_iter);

            // If we found PIDs with this accelerator class, no need to try others
            if !pids.is_empty() {
                break;
            }
        }
    }

    pids.sort_unstable();
    pids.dedup();
    pids
}

// proc_pidinfo / proc_pidpath FFI (shared with app.rs declarations)
extern "C" {
    fn proc_pidpath(pid: i32, buffer: *mut u8, buffersize: u32) -> i32;
    fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut u8, buffersize: i32) -> i32;
}

extern "C" {
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

/// proc_pidinfo flavor constants
const PROC_PIDTBSDINFO: i32 = 3;
pub(crate) const PROC_PIDTASKINFO: i32 = 4;
const PROC_BSDINFO_SIZE: i32 = 136;
pub(crate) const PROC_TASKINFO_SIZE: i32 = 96;

/// proc_taskinfo offsets (from XNU bsd/sys/proc_info.h):
///   pti_virtual_size:   0 (u64)
///   pti_resident_size:  8 (u64)
///   pti_total_user:    16 (u64) — Mach absolute time
///   pti_total_system:  24 (u64) — Mach absolute time
pub(crate) mod taskinfo_offsets {
    pub const PTI_VIRTUAL_SIZE: usize = 0;
    pub const PTI_RESIDENT_SIZE: usize = 8;
    pub const PTI_TOTAL_USER: usize = 16;
    pub const PTI_TOTAL_SYSTEM: usize = 24;
}

/// proc_bsdinfo offsets (same as in app.rs)
mod bsdinfo_offsets {
    pub const PBI_UID: usize = 20;
    pub const PBI_START_TVSEC: usize = 120;
}

/// Get the executable path for a PID via proc_pidpath.
fn native_proc_pidpath(pid: u32) -> String {
    let mut buf = vec![0u8; 4096];
    let ret = unsafe { proc_pidpath(pid as i32, buf.as_mut_ptr(), buf.len() as u32) };
    if ret > 0 {
        String::from_utf8_lossy(&buf[..ret as usize]).to_string()
    } else {
        String::new()
    }
}

/// Get the full command line (with arguments) for a PID via sysctl KERN_PROCARGS2.
pub(crate) fn native_procargs(pid: u32) -> String {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid as i32];
    let mut size: usize = 0;

    // First call to get buffer size
    let ret = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || size == 0 {
        return String::new();
    }

    let mut buf = vec![0u8; size];
    let ret = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return String::new();
    }

    // Layout: argc (i32), exec_path (null-terminated), padding nulls, argv[0..argc] (null-separated)
    if size < 4 {
        return String::new();
    }
    let argc = i32::from_le_bytes(buf[0..4].try_into().unwrap_or_default()) as usize;

    // Find end of exec_path (first null after offset 4)
    let exec_end = match buf[4..size].iter().position(|&b| b == 0) {
        Some(pos) => 4 + pos,
        None => return String::new(),
    };

    // Skip padding nulls after exec_path
    let args_start = match buf[exec_end..size].iter().position(|&b| b != 0) {
        Some(pos) => exec_end + pos,
        None => return String::new(),
    };

    // Collect argc arguments (null-separated)
    let mut args = Vec::with_capacity(argc);
    let mut pos = args_start;
    for _ in 0..argc {
        let end = buf[pos..size]
            .iter()
            .position(|&b| b == 0)
            .map(|p| pos + p)
            .unwrap_or(size);
        let arg = String::from_utf8_lossy(&buf[pos..end]).to_string();
        args.push(arg);
        pos = end + 1;
        if pos >= size {
            break;
        }
    }

    args.join(" ")
}

/// Get username from UID using getpwuid.
fn uid_to_username(uid: u32) -> String {
    unsafe {
        let pw = libc::getpwuid(uid);
        if pw.is_null() {
            return uid.to_string();
        }
        std::ffi::CStr::from_ptr((*pw).pw_name)
            .to_string_lossy()
            .to_string()
    }
}

/// Get Mach timebase ratio (numer/denom) for converting absolute time to nanoseconds.
pub(crate) fn mach_timebase_ratio() -> f64 {
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    unsafe {
        mach_timebase_info(&mut info);
    }
    if info.denom == 0 {
        1.0
    } else {
        info.numer as f64 / info.denom as f64
    }
}

/// Native process info for a single PID. Returns (user, cpu_pct, rss_bytes, args_string).
fn native_process_info(pid: u32, timebase: f64, now: f64) -> Option<(String, f32, u64, String)> {
    // Get bsdinfo for uid and start time
    let mut bsd_buf = vec![0u8; PROC_BSDINFO_SIZE as usize];
    let ret = unsafe {
        proc_pidinfo(
            pid as i32,
            PROC_PIDTBSDINFO,
            0,
            bsd_buf.as_mut_ptr(),
            PROC_BSDINFO_SIZE,
        )
    };
    if ret <= 0 {
        return None;
    }

    let uid = u32::from_le_bytes(
        bsd_buf[bsdinfo_offsets::PBI_UID..bsdinfo_offsets::PBI_UID + 4]
            .try_into()
            .ok()?,
    );
    let start_tvsec = u64::from_le_bytes(
        bsd_buf[bsdinfo_offsets::PBI_START_TVSEC..bsdinfo_offsets::PBI_START_TVSEC + 8]
            .try_into()
            .ok()?,
    );
    let user = uid_to_username(uid);

    // Get taskinfo for CPU times and resident size
    let mut task_buf = vec![0u8; PROC_TASKINFO_SIZE as usize];
    let ret = unsafe {
        proc_pidinfo(
            pid as i32,
            PROC_PIDTASKINFO,
            0,
            task_buf.as_mut_ptr(),
            PROC_TASKINFO_SIZE,
        )
    };

    let (cpu_pct, rss) = if ret > 0 {
        let resident_size = u64::from_le_bytes(
            task_buf[taskinfo_offsets::PTI_RESIDENT_SIZE..taskinfo_offsets::PTI_RESIDENT_SIZE + 8]
                .try_into()
                .unwrap_or_default(),
        );
        let total_user = u64::from_le_bytes(
            task_buf[taskinfo_offsets::PTI_TOTAL_USER..taskinfo_offsets::PTI_TOTAL_USER + 8]
                .try_into()
                .unwrap_or_default(),
        );
        let total_system = u64::from_le_bytes(
            task_buf[taskinfo_offsets::PTI_TOTAL_SYSTEM..taskinfo_offsets::PTI_TOTAL_SYSTEM + 8]
                .try_into()
                .unwrap_or_default(),
        );

        // Convert Mach absolute time to seconds
        let cpu_time_s = (total_user + total_system) as f64 * timebase / 1_000_000_000.0;
        let wall_time_s = now - start_tvsec as f64;
        let pct = if wall_time_s > 0.0 {
            (cpu_time_s / wall_time_s * 100.0) as f32
        } else {
            0.0
        };

        (pct, resident_size)
    } else {
        (0.0, 0)
    };

    // Get full command line with arguments
    let args = native_procargs(pid);
    // Fallback to proc_pidpath if procargs failed
    let args = if args.is_empty() {
        native_proc_pidpath(pid)
    } else {
        args
    };

    Some((user, cpu_pct, rss, args))
}

fn get_gpu_processes() -> Result<Vec<GpuProcess>> {
    // Step 1: Get all PIDs that have an AGXDeviceUserClient (= GPU connection)
    let gpu_pids = get_gpu_client_pids();
    if gpu_pids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Get process info natively (no subprocess)
    let timebase = mach_timebase_ratio();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    let mut processes = Vec::new();

    for &pid in &gpu_pids {
        let (user, cpu_pct, rss, args) = match native_process_info(pid, timebase, now) {
            Some(info) => info,
            None => continue, // process may have exited
        };

        if args.is_empty() {
            continue;
        }

        // Extract executable name from args or path
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
            before_app.rsplit('/').next().unwrap_or(&name).to_string()
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
            host_memory: rss,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_get_gpu_client_pids() {
        let pids = get_gpu_client_pids();
        // On macOS with a display, there should always be at least WindowServer
        assert!(
            !pids.is_empty(),
            "get_gpu_client_pids() should find at least one GPU client (e.g., WindowServer)"
        );
        // Verify PIDs are valid (all > 0)
        for pid in &pids {
            assert!(*pid > 0, "PID should be > 0");
        }
    }

    #[test]
    fn test_native_get_gpu_processes() {
        let procs = get_gpu_processes().expect("get_gpu_processes should succeed");
        // Should find at least WindowServer
        assert!(
            !procs.is_empty(),
            "get_gpu_processes() should find at least one process"
        );
        // Verify all processes have valid PIDs and names
        for p in &procs {
            assert!(p.pid > 0, "PID should be > 0");
            assert!(!p.name.is_empty(), "name should not be empty");
            assert!(!p.user.is_empty(), "user should not be empty");
        }
    }

    #[test]
    fn test_native_process_info_self() {
        let pid = std::process::id();
        let timebase = mach_timebase_ratio();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let (user, _cpu_pct, rss, args) =
            native_process_info(pid, timebase, now).expect("should get info for own process");
        assert!(!user.is_empty(), "user should not be empty");
        assert!(rss > 0, "rss should be > 0 for own process");
        assert!(!args.is_empty(), "args should not be empty");
    }

    #[test]
    fn test_native_detect_gpu_core_count() {
        let cores = detect_gpu_core_count();
        assert!(cores.is_some(), "should detect GPU core count");
        let n = cores.unwrap();
        assert!(
            (6..=80).contains(&n),
            "GPU core count {} should be in reasonable range",
            n
        );
    }
}
