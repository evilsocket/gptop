use anyhow::{anyhow, Result};
use libc::{self, c_int, c_void, size_t};
use std::mem;

/// Get total physical RAM in bytes via sysctl hw.memsize.
pub fn total_ram() -> Result<u64> {
    let mut size: u64 = 0;
    let mut len = mem::size_of::<u64>();
    let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr() as *mut c_int,
            2,
            &mut size as *mut u64 as *mut c_void,
            &mut len as *mut usize as *mut size_t,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return Err(anyhow!("sysctl hw.memsize failed"));
    }
    Ok(size)
}

/// Get used RAM in bytes via host_statistics64.
pub fn used_ram() -> Result<u64> {
    let total = total_ram()?;

    // Use vm_statistics64 to compute used memory
    let mut vm_stats: libc::vm_statistics64 = unsafe { mem::zeroed() };
    let mut count = (mem::size_of::<libc::vm_statistics64>() / mem::size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;

    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    let ret = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut vm_stats as *mut _ as *mut libc::integer_t,
            &mut count,
        )
    };

    if ret != 0 {
        return Err(anyhow!("host_statistics64 failed: {}", ret));
    }

    let page_size = unsafe { libc::vm_page_size } as u64;
    // Free + purgeable + external (file-backed) is what macOS considers "available"
    let available = (vm_stats.free_count as u64
        + vm_stats.purgeable_count as u64
        + vm_stats.external_page_count as u64)
        * page_size;

    Ok(total.saturating_sub(available))
}

/// Swap usage via sysctl vm.swapusage.
#[repr(C)]
#[derive(Default)]
struct XswUsage {
    xsu_total: u64,
    xsu_avail: u64,
    xsu_used: u64,
    xsu_pagesize: u32,
    xsu_encrypted: bool,
}

pub fn swap_usage() -> Result<(u64, u64)> {
    let mut usage = XswUsage::default();
    let mut len = mem::size_of::<XswUsage>();

    let name = std::ffi::CString::new("vm.swapusage").unwrap();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut usage as *mut XswUsage as *mut c_void,
            &mut len as *mut usize as *mut size_t,
            std::ptr::null_mut(),
            0,
        )
    };

    if ret != 0 {
        return Err(anyhow!("sysctlbyname vm.swapusage failed"));
    }

    Ok((usage.xsu_used, usage.xsu_total))
}
