use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::style::Color;
use ratatui::widgets::TableState;
use ratatui::Terminal;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::backend::{DeviceInfo, GpuBackend, GpuProcess, SampleResult};
use crate::config::Config;
use crate::ui;

pub struct ProcessDetail {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    pub user: String,
    pub uid: u32,
    pub gid: u32,
    pub state: String,
    pub nice: i32,
    pub started: String,
    pub cpu_time: String,
    pub cpu_pct: f32,
    pub mem_pct: f32,
    pub rss: u64,
    pub vsz: u64,
    pub gpu_memory: u64,
    pub gpu_usage_pct: f32,
    pub path: String,
    pub cwd: String,
    pub command: String,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pidpath(pid: i32, buffer: *mut u8, buffersize: u32) -> i32;
    fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut u8, buffersize: i32) -> i32;
}

/// Flavor constant for proc_pidinfo to get vnode path info (includes cwd).
#[cfg(target_os = "macos")]
const PROC_PIDVNODEPATHINFO: i32 = 9;

/// Flavor constant for proc_pidinfo to get BSD info (ppid, uid, gid, nice, status, etc).
#[cfg(target_os = "macos")]
const PROC_PIDTBSDINFO: i32 = 3;

/// Size of proc_vnodepathinfo struct (2352 bytes on macOS).
/// Layout: pvi_cdir (vnode_info_path, 1176 bytes) + pvi_rdir (vnode_info_path, 1176 bytes).
/// Each vnode_info_path = vnode_info (152 bytes) + vip_path ([c_char; 1024]).
/// CWD path is in pvi_cdir.vip_path at offset 152.
#[cfg(target_os = "macos")]
const PROC_VNODEPATHINFO_SIZE: i32 = 2352;

/// CWD path offset within proc_vnodepathinfo struct (pvi_cdir.vip_path).
#[cfg(target_os = "macos")]
const PROC_VNODEPATHINFO_CWD_OFFSET: usize = 152;

/// Size of proc_bsdinfo struct (136 bytes on macOS).
#[cfg(target_os = "macos")]
const PROC_BSDINFO_SIZE: i32 = 136;

/// proc_bsdinfo field offsets (from XNU sys/proc_info.h):
///   pbi_flags:    0 (u32)
///   pbi_status:   4 (u32)
///   pbi_xstatus:  8 (u32)
///   pbi_pid:     12 (u32)
///   pbi_ppid:    16 (u32)
///   pbi_uid:     20 (u32)
///   pbi_gid:     24 (u32)
///   pbi_ruid:    28 (u32)
///   pbi_rgid:    32 (u32)
///   pbi_svuid:   36 (u32)
///   pbi_svgid:   40 (u32)
///   rfu_1:       44 (u32)
///   pbi_comm:    48 ([c_char; 16])
///   pbi_name:    64 ([c_char; 32])
///   pbi_nfiles:  96 (u32)
///   pbi_pgid:   100 (u32)
///   pbi_pjobc:  104 (u32)
///   e_tdev:     108 (u32)
///   e_tpgid:    112 (u32)
///   pbi_nice:   116 (i32)
///   pbi_start_tvsec:  120 (u64)
///   pbi_start_tvusec: 128 (u64)
///   Total: 136 bytes
#[cfg(target_os = "macos")]
mod bsdinfo_offsets {
    pub const PBI_STATUS: usize = 4;
    pub const PBI_PPID: usize = 16;
    pub const PBI_UID: usize = 20;
    pub const PBI_GID: usize = 24;
    pub const PBI_NICE: usize = 116;
}

/// Get CWD for a process using proc_pidinfo (PROC_PIDVNODEPATHINFO).
#[cfg(target_os = "macos")]
fn native_get_cwd(pid: u32) -> String {
    let mut buf = vec![0u8; PROC_VNODEPATHINFO_SIZE as usize];
    let ret = unsafe {
        proc_pidinfo(
            pid as i32,
            PROC_PIDVNODEPATHINFO,
            0,
            buf.as_mut_ptr(),
            PROC_VNODEPATHINFO_SIZE,
        )
    };
    if ret <= 0 {
        return String::new();
    }
    // CWD path is a null-terminated C string at the CWD offset
    let cwd_bytes = &buf[PROC_VNODEPATHINFO_CWD_OFFSET..];
    let end = cwd_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(cwd_bytes.len());
    String::from_utf8_lossy(&cwd_bytes[..end]).to_string()
}

/// Get BSD info for a process using proc_pidinfo (PROC_PIDTBSDINFO).
/// Returns (ppid, uid, gid, nice, status_code).
#[cfg(target_os = "macos")]
fn native_bsdinfo(pid: u32) -> Option<(u32, u32, u32, i32, u32)> {
    let mut buf = vec![0u8; PROC_BSDINFO_SIZE as usize];
    let ret = unsafe {
        proc_pidinfo(
            pid as i32,
            PROC_PIDTBSDINFO,
            0,
            buf.as_mut_ptr(),
            PROC_BSDINFO_SIZE,
        )
    };
    if ret <= 0 {
        return None;
    }
    let ppid = u32::from_le_bytes(
        buf[bsdinfo_offsets::PBI_PPID..bsdinfo_offsets::PBI_PPID + 4]
            .try_into()
            .ok()?,
    );
    let uid = u32::from_le_bytes(
        buf[bsdinfo_offsets::PBI_UID..bsdinfo_offsets::PBI_UID + 4]
            .try_into()
            .ok()?,
    );
    let gid = u32::from_le_bytes(
        buf[bsdinfo_offsets::PBI_GID..bsdinfo_offsets::PBI_GID + 4]
            .try_into()
            .ok()?,
    );
    let nice = i32::from_le_bytes(
        buf[bsdinfo_offsets::PBI_NICE..bsdinfo_offsets::PBI_NICE + 4]
            .try_into()
            .ok()?,
    );
    let status = u32::from_le_bytes(
        buf[bsdinfo_offsets::PBI_STATUS..bsdinfo_offsets::PBI_STATUS + 4]
            .try_into()
            .ok()?,
    );
    Some((ppid, uid, gid, nice, status))
}

/// Convert a process status code (from proc_bsdinfo) to a ps-style state string.
#[cfg(target_os = "macos")]
fn status_to_state(status: u32) -> String {
    // XNU SIDL=1, SRUN=2, SSLEEP=3, SSTOP=4, SZOMB=5
    match status {
        1 => "I".to_string(), // Idle (being created)
        2 => "R".to_string(), // Running
        3 => "S".to_string(), // Sleeping
        4 => "T".to_string(), // Stopped
        5 => "Z".to_string(), // Zombie
        _ => "?".to_string(),
    }
}

/// Get username from UID using libc::getpwuid.
#[cfg(target_os = "macos")]
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

fn ps_field(pid: u32, field: &str) -> String {
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", &format!("{}=", field)])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

#[cfg(target_os = "macos")]
fn get_exe_path(pid: u32) -> String {
    let mut buf = vec![0u8; 4096];
    let ret = unsafe { proc_pidpath(pid as i32, buf.as_mut_ptr(), buf.len() as u32) };
    if ret > 0 {
        String::from_utf8_lossy(&buf[..ret as usize]).to_string()
    } else {
        String::new()
    }
}

#[cfg(target_os = "linux")]
fn get_exe_path(pid: u32) -> String {
    std::fs::read_link(format!("/proc/{}/exe", pid))
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

#[cfg(target_os = "macos")]
fn get_cwd(pid: u32) -> String {
    native_get_cwd(pid)
}

#[cfg(target_os = "linux")]
fn get_cwd(pid: u32) -> String {
    std::fs::read_link(format!("/proc/{}/cwd", pid))
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

#[cfg(target_os = "windows")]
fn get_exe_path(_pid: u32) -> String {
    String::new()
}

#[cfg(target_os = "windows")]
fn get_cwd(_pid: u32) -> String {
    String::new()
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .output();
}

#[cfg(target_os = "macos")]
fn get_process_detail(proc: &GpuProcess) -> ProcessDetail {
    let pid = proc.pid;

    // Native syscalls for ppid, uid, gid, nice, state (1 syscall instead of 5 ps spawns)
    let (ppid, uid, gid, nice, state) = native_bsdinfo(pid)
        .map(|(ppid, uid, gid, nice, status)| (ppid, uid, gid, nice, status_to_state(status)))
        .unwrap_or((0, 0, 0, 0, "?".to_string()));

    // Native RSS from proc_pid_rusage (already have the FFI in backend/apple/mod.rs)
    let rss = {
        // rusage_info_v4: phys_footprint at offset 72, but we want ri_resident_size at offset 64
        let mut buf = [0u8; 512];
        let ret =
            unsafe { crate::backend::apple::proc_pid_rusage_raw(pid as i32, 4, buf.as_mut_ptr()) };
        if ret == 0 {
            u64::from_le_bytes(buf[64..72].try_into().unwrap_or_default())
        } else {
            0
        }
    };

    // Still use ps for fields that are hard to replicate natively
    let started = ps_field(pid, "lstart");
    let cpu_time = ps_field(pid, "time");
    let cpu_pct = ps_field(pid, "%cpu").parse::<f32>().unwrap_or(0.0);
    let mem_pct = ps_field(pid, "%mem").parse::<f32>().unwrap_or(0.0);
    let vsz = ps_field(pid, "vsz").parse::<u64>().unwrap_or(0) * 1024;
    let command = ps_field(pid, "command");

    let path = get_exe_path(pid);
    let cwd = get_cwd(pid);

    ProcessDetail {
        pid,
        ppid,
        name: proc.name.clone(),
        user: uid_to_username(uid),
        uid,
        gid,
        state,
        nice,
        started,
        cpu_time,
        cpu_pct,
        mem_pct,
        rss,
        vsz,
        gpu_memory: proc.gpu_memory,
        gpu_usage_pct: proc.gpu_usage_pct,
        path,
        cwd,
        command,
    }
}

#[cfg(not(target_os = "macos"))]
fn get_process_detail(proc: &GpuProcess) -> ProcessDetail {
    let pid = proc.pid;

    let ppid = ps_field(pid, "ppid").parse::<u32>().unwrap_or(0);
    let uid = ps_field(pid, "uid").parse::<u32>().unwrap_or(0);
    let gid = ps_field(pid, "gid").parse::<u32>().unwrap_or(0);
    let state = ps_field(pid, "state");
    let nice = ps_field(pid, "nice").parse::<i32>().unwrap_or(0);
    let started = ps_field(pid, "lstart");
    let cpu_time = ps_field(pid, "time");
    let cpu_pct = ps_field(pid, "%cpu").parse::<f32>().unwrap_or(0.0);
    let mem_pct = ps_field(pid, "%mem").parse::<f32>().unwrap_or(0.0);
    let rss = ps_field(pid, "rss").parse::<u64>().unwrap_or(0) * 1024;
    let vsz = ps_field(pid, "vsz").parse::<u64>().unwrap_or(0) * 1024;
    let command = ps_field(pid, "command");

    let path = get_exe_path(pid);
    let cwd = get_cwd(pid);

    ProcessDetail {
        pid,
        ppid,
        name: proc.name.clone(),
        user: proc.user.clone(),
        uid,
        gid,
        state,
        nice,
        started,
        cpu_time,
        cpu_pct,
        mem_pct,
        rss,
        vsz,
        gpu_memory: proc.gpu_memory,
        gpu_usage_pct: proc.gpu_usage_pct,
        path,
        cwd,
        command,
    }
}

const ACCENT_COLORS: &[Color] = &[
    Color::Green,
    Color::Cyan,
    Color::Blue,
    Color::Magenta,
    Color::Yellow,
    Color::Red,
];

const MAX_HISTORY: usize = 300;

enum AppEvent {
    Key(event::KeyEvent),
    Sample(SampleResult),
    #[allow(dead_code)]
    Tick,
}

pub struct KillConfirm {
    pub pid: u32,
    pub name: String,
}

pub struct App {
    pub config: Config,
    pub devices: Vec<DeviceInfo>,
    pub gpu_util_history: Vec<VecDeque<(f64, f64)>>,
    pub mem_util_history: Vec<VecDeque<(f64, f64)>>,
    pub power_history: Vec<VecDeque<(f64, f64)>>,
    pub latest: Option<SampleResult>,
    pub process_sort_col: usize,
    pub process_sort_asc: bool,
    pub table_state: TableState,
    pub sorted_processes: Vec<GpuProcess>,
    pub process_detail: Option<ProcessDetail>,
    pub process_gpu_history: HashMap<u32, VecDeque<(f64, f64)>>,
    pub kill_confirm: Option<KillConfirm>,
    should_quit: bool,
    start_time: Instant,
    accent_idx: usize,
}

impl App {
    pub fn new(devices: Vec<DeviceInfo>, config: Config) -> Self {
        let device_count = devices.len();
        Self {
            accent_idx: config.accent_color_idx,
            process_sort_col: config.sort_column,
            process_sort_asc: config.sort_ascending,
            config,
            devices,
            gpu_util_history: vec![VecDeque::new(); device_count],
            mem_util_history: vec![VecDeque::new(); device_count],
            power_history: vec![VecDeque::new(); device_count],
            latest: None,
            table_state: TableState::default(),
            sorted_processes: Vec::new(),
            process_detail: None,
            process_gpu_history: HashMap::new(),
            kill_confirm: None,
            should_quit: false,
            start_time: Instant::now(),
        }
    }

    pub fn accent_color(&self) -> Color {
        ACCENT_COLORS[self.accent_idx % ACCENT_COLORS.len()]
    }

    fn update_history(&mut self, sample: &SampleResult) {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        for metrics in &sample.gpu_metrics {
            let id = metrics.device_id;
            if id < self.gpu_util_history.len() {
                self.gpu_util_history[id].push_back((elapsed, metrics.utilization_pct as f64));
                if self.gpu_util_history[id].len() > MAX_HISTORY {
                    self.gpu_util_history[id].pop_front();
                }

                let mem_pct = if metrics.memory_total > 0 {
                    metrics.memory_used as f64 / metrics.memory_total as f64 * 100.0
                } else {
                    0.0
                };
                self.mem_util_history[id].push_back((elapsed, mem_pct));
                if self.mem_util_history[id].len() > MAX_HISTORY {
                    self.mem_util_history[id].pop_front();
                }

                let power_pct = match metrics.power_limit_watts {
                    Some(limit) if limit > 0.0 => {
                        (metrics.power_watts as f64 / limit as f64 * 100.0).clamp(0.0, 100.0)
                    }
                    _ => 0.0,
                };
                self.power_history[id].push_back((elapsed, power_pct));
                if self.power_history[id].len() > MAX_HISTORY {
                    self.power_history[id].pop_front();
                }
            }
        }

        // Track per-process GPU usage history
        let active_pids: std::collections::HashSet<u32> =
            sample.processes.iter().map(|p| p.pid).collect();
        self.process_gpu_history
            .retain(|pid, _| active_pids.contains(pid));
        for proc in &sample.processes {
            let hist = self.process_gpu_history.entry(proc.pid).or_default();
            hist.push_back((elapsed, proc.gpu_usage_pct as f64));
            if hist.len() > MAX_HISTORY {
                hist.pop_front();
            }
        }
    }

    fn handle_key(&mut self, key: event::KeyEvent) {
        // Kill confirmation dialog takes priority
        if self.kill_confirm.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    if let Some(ref confirm) = self.kill_confirm {
                        kill_process(confirm.pid);
                    }
                    self.kill_confirm = None;
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.should_quit = true;
                }
                _ => {
                    self.kill_confirm = None;
                }
            }
            return;
        }

        // When detail modal is open, only handle Escape/q to close it
        if self.process_detail.is_some() {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.process_detail = None;
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.should_quit = true;
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('c') => {
                self.accent_idx = (self.accent_idx + 1) % ACCENT_COLORS.len();
                self.config.accent_color_idx = self.accent_idx;
                let _ = self.config.save();
            }
            KeyCode::Enter => {
                if let Some(idx) = self.table_state.selected() {
                    if let Some(proc) = self.sorted_processes.get(idx) {
                        self.process_detail = Some(get_process_detail(proc));
                    }
                }
            }
            KeyCode::Char('k') => {
                if let Some(idx) = self.table_state.selected() {
                    if let Some(proc) = self.sorted_processes.get(idx) {
                        self.kill_confirm = Some(KillConfirm {
                            pid: proc.pid,
                            name: proc.name.clone(),
                        });
                    }
                }
            }
            KeyCode::Up => {
                let i = self.table_state.selected().unwrap_or(0);
                if i > 0 {
                    self.table_state.select(Some(i - 1));
                }
            }
            KeyCode::Down => {
                let i = self.table_state.selected().unwrap_or(0);
                let max = self
                    .latest
                    .as_ref()
                    .map(|s| s.processes.len().saturating_sub(1))
                    .unwrap_or(0);
                if i < max {
                    self.table_state.select(Some(i + 1));
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.process_sort_asc = true;
                self.config.sort_ascending = true;
                let _ = self.config.save();
            }
            KeyCode::Char('-') => {
                self.process_sort_asc = false;
                self.config.sort_ascending = false;
                let _ = self.config.save();
            }
            KeyCode::Left => {
                let col_count = crate::ui::processes::SORT_COLUMN_COUNT;
                self.process_sort_col = (self.process_sort_col + col_count - 1) % col_count;
                self.config.sort_column = self.process_sort_col;
                let _ = self.config.save();
            }
            KeyCode::Right => {
                let col_count = crate::ui::processes::SORT_COLUMN_COUNT;
                self.process_sort_col = (self.process_sort_col + 1) % col_count;
                self.config.sort_column = self.process_sort_col;
                let _ = self.config.save();
            }
            KeyCode::F(6) | KeyCode::F(2) | KeyCode::Tab => {
                let col_count = crate::ui::processes::SORT_COLUMN_COUNT;
                self.process_sort_col = (self.process_sort_col + 1) % col_count;
                self.config.sort_column = self.process_sort_col;
                let _ = self.config.save();
            }
            KeyCode::Char('e') => {
                self.config.update_interval_ms = (self.config.update_interval_ms + 250).min(5000);
                let _ = self.config.save();
            }
            KeyCode::Char('d') => {
                self.config.update_interval_ms =
                    self.config.update_interval_ms.saturating_sub(250).max(250);
                let _ = self.config.save();
            }
            _ => {}
        }
    }

    fn refresh_process_detail(&mut self) {
        if let Some(ref detail) = self.process_detail {
            let pid = detail.pid;
            if let Some(proc) = self.sorted_processes.iter().find(|p| p.pid == pid) {
                self.process_detail = Some(get_process_detail(proc));
            }
        }
    }
}

pub fn run_tui(mut backend: Box<dyn GpuBackend>) -> Result<()> {
    let config = Config::load();
    let devices = backend.devices().to_vec();
    let mut app = App::new(devices, config);

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let term_backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(term_backend)?;

    // Terminal cleanup guard
    struct CleanupGuard;
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        }
    }
    let _guard = CleanupGuard;

    // Channel for events
    let (tx, rx) = mpsc::channel::<AppEvent>();

    // Input thread
    let tx_input = tx.clone();
    std::thread::spawn(move || loop {
        if event::poll(Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if tx_input.send(AppEvent::Key(key)).is_err() {
                    break;
                }
            }
        }
    });

    // Sampler thread
    let tx_sample = tx.clone();
    let initial_interval = app.config.update_interval_ms;
    std::thread::spawn(move || {
        let interval = initial_interval;
        loop {
            match backend.sample(interval) {
                Ok(result) => {
                    if tx_sample.send(AppEvent::Sample(result)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Sample error: {}", e);
                }
            }
        }
    });

    // Main render loop
    loop {
        terminal.draw(|f| {
            ui::render(f, &mut app);
        })?;

        // Wait for event with timeout
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(AppEvent::Key(key)) => app.handle_key(key),
            Ok(AppEvent::Sample(sample)) => {
                app.update_history(&sample);
                app.latest = Some(sample);
                app.refresh_process_detail();
            }
            Ok(AppEvent::Tick) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Run in JSON mode: print one sample and exit.
pub fn run_json(mut backend: Box<dyn GpuBackend>, duration_ms: u64) -> Result<()> {
    let sample = backend.sample(duration_ms)?;
    let devices = backend.devices();

    let output = serde_json::json!({
        "devices": devices.iter().map(|d| serde_json::json!({
            "id": d.id,
            "name": &d.name,
            "vendor": &d.vendor,
            "total_memory": d.total_memory,
            "core_count": d.core_count,
        })).collect::<Vec<_>>(),
        "gpu_metrics": sample.gpu_metrics.iter().map(|m| serde_json::json!({
            "device_id": m.device_id,
            "utilization_pct": m.utilization_pct,
            "memory_used": m.memory_used,
            "memory_total": m.memory_total,
            "freq_mhz": m.freq_mhz,
            "freq_max_mhz": m.freq_max_mhz,
            "power_watts": m.power_watts,
            "temp_celsius": m.temp_celsius,
        })).collect::<Vec<_>>(),
        "processes": sample.processes.iter().map(|p| serde_json::json!({
            "pid": p.pid,
            "user": &p.user,
            "name": &p.name,
            "gpu_usage_pct": p.gpu_usage_pct,
            "gpu_memory": p.gpu_memory,
            "cpu_usage_pct": p.cpu_usage_pct,
            "host_memory": p.host_memory,
            "process_type": &p.process_type,
        })).collect::<Vec<_>>(),
        "system": {
            "cpu_power": sample.system.cpu_power,
            "ane_power": sample.system.ane_power,
            "package_power": sample.system.package_power,
            "ram_used": sample.system.ram_used,
            "ram_total": sample.system.ram_total,
            "swap_used": sample.system.swap_used,
            "swap_total": sample.system.swap_total,
        },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn test_native_get_cwd_matches_lsof() {
        let pid = std::process::id();
        let native = native_get_cwd(pid);
        // Verify against std::env (more reliable than lsof)
        let expected = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        assert_eq!(native, expected, "native_get_cwd should match current_dir");
    }

    #[test]
    fn test_native_bsdinfo_ppid() {
        let pid = std::process::id();
        let (ppid, _, _, _, _) = native_bsdinfo(pid).expect("bsdinfo should succeed for own pid");
        let ps_ppid: u32 = ps_field(pid, "ppid").parse().unwrap_or(0);
        assert_eq!(ppid, ps_ppid, "native ppid should match ps ppid");
    }

    #[test]
    fn test_native_bsdinfo_uid() {
        let pid = std::process::id();
        let (_, uid, _, _, _) = native_bsdinfo(pid).expect("bsdinfo should succeed");
        let ps_uid: u32 = ps_field(pid, "uid").parse().unwrap_or(0);
        assert_eq!(uid, ps_uid, "native uid should match ps uid");
    }

    #[test]
    fn test_native_bsdinfo_gid() {
        let pid = std::process::id();
        let (_, _, gid, _, _) = native_bsdinfo(pid).expect("bsdinfo should succeed");
        let ps_gid: u32 = ps_field(pid, "gid").parse().unwrap_or(0);
        assert_eq!(gid, ps_gid, "native gid should match ps gid");
    }

    #[test]
    fn test_native_bsdinfo_nice() {
        let pid = std::process::id();
        let (_, _, _, nice, _) = native_bsdinfo(pid).expect("bsdinfo should succeed");
        let ps_nice: i32 = ps_field(pid, "nice").parse().unwrap_or(0);
        assert_eq!(nice, ps_nice, "native nice should match ps nice");
    }

    #[test]
    fn test_native_bsdinfo_state() {
        let pid = std::process::id();
        let (_, _, _, _, status) = native_bsdinfo(pid).expect("bsdinfo should succeed");
        let state = status_to_state(status);
        // Our process should be Running (R) or Sleeping (S) — ps may show
        // R or S depending on timing, so just verify we get a valid state
        assert!(
            state == "R" || state == "S",
            "native state should be R or S, got: {}",
            state
        );
    }

    #[test]
    fn test_uid_to_username() {
        let uid = unsafe { libc::getuid() };
        let name = uid_to_username(uid);
        let ps_user = ps_field(std::process::id(), "user");
        assert_eq!(name, ps_user, "uid_to_username should match ps user");
    }
}
