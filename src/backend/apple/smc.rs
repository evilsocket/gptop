use anyhow::{anyhow, Result};
use std::ffi::CString;
use std::mem;

// SMC key types
const SMC_KEY_TYPE_FLT: u32 = u32::from_be_bytes(*b"flt ");
const SMC_KEY_TYPE_SP78: u32 = u32::from_be_bytes(*b"sp78");
const SMC_KEY_TYPE_UI8: u32 = u32::from_be_bytes(*b"ui8 ");
const SMC_KEY_TYPE_UI16: u32 = u32::from_be_bytes(*b"ui16");
const SMC_KEY_TYPE_UI32: u32 = u32::from_be_bytes(*b"ui32");

// SMC selectors
const SMC_CMD_READ_KEY_INFO: u8 = 9;
const SMC_CMD_READ_BYTES: u8 = 5;

// IOKit constants
const KERNEL_INDEX_SMC: u32 = 2;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcKeyData {
    key: u32,
    vers: [u8; 6],
    p_limit_data: [u8; 16],
    key_info: SmcKeyInfoData,
    result: u8,
    status: u8,
    data8: u8,
    data32: u32,
    bytes: [u8; 32],
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcKeyInfoData {
    data_size: u32,
    data_type: u32,
    data_attributes: u8,
}

fn str_to_smc_key(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut key_bytes = [0x20u8; 4]; // space-padded
    for (i, &b) in bytes.iter().enumerate().take(4) {
        key_bytes[i] = b;
    }
    u32::from_be_bytes(key_bytes)
}

/// SMC connection for reading Apple system sensors.
pub struct SmcConnection {
    conn: u32, // io_connect_t
}

unsafe impl Send for SmcConnection {}

impl SmcConnection {
    pub fn new() -> Result<Self> {
        let conn = unsafe { open_smc_connection()? };
        Ok(Self { conn })
    }

    /// Read a floating-point value from an SMC key.
    pub fn read_key_float(&self, key: &str) -> Result<f32> {
        let smc_key = str_to_smc_key(key);

        // First get key info
        let mut input = SmcKeyData::default();
        input.key = smc_key;
        input.data8 = SMC_CMD_READ_KEY_INFO;

        let output = self.call_smc(&input)?;
        let data_type = output.key_info.data_type;
        let data_size = output.key_info.data_size;

        if data_size == 0 {
            return Err(anyhow!("SMC key {} has zero size", key));
        }

        // Now read the actual value
        let mut input2 = SmcKeyData::default();
        input2.key = smc_key;
        input2.key_info.data_size = data_size;
        input2.data8 = SMC_CMD_READ_BYTES;

        let output2 = self.call_smc(&input2)?;
        let bytes = &output2.bytes[..data_size as usize];

        decode_smc_float(data_type, bytes)
    }

    /// Read GPU temperature (tries multiple sensor keys).
    pub fn read_gpu_temp(&self) -> Option<f32> {
        // GPU temperature sensor keys across Apple Silicon generations
        let keys = [
            "Tg0f", "Tg0j", "Tg0D", "Tg0d",
            "Tg05", "Tg0L", "Tg0T",
            "Tg1f", "Tg1j", "Tg1D",
        ];
        for key in &keys {
            if let Ok(val) = self.read_key_float(key) {
                if val > 0.0 && val < 150.0 {
                    return Some(val);
                }
            }
        }
        None
    }

    /// Read CPU temperature.
    pub fn read_cpu_temp(&self) -> Option<f32> {
        let keys = [
            "Tp0f", "Tp0j", "Tp0D", "Tp09", "Tp01",
            "Tp05", "Tp0L", "Tp0T",
            "Tp1f", "Tp1j", "Tp1D",
        ];
        for key in &keys {
            if let Ok(val) = self.read_key_float(key) {
                if val > 0.0 && val < 150.0 {
                    return Some(val);
                }
            }
        }
        None
    }

    /// Try to read a list of keys and return the first valid temperature.
    pub fn read_temp_keys(&self, keys: &[&str]) -> Option<f32> {
        for key in keys {
            if let Ok(val) = self.read_key_float(key) {
                if val > 0.0 && val < 150.0 {
                    return Some(val);
                }
            }
        }
        None
    }

    /// Read system total power from PSTR key.
    pub fn read_system_power(&self) -> Option<f32> {
        self.read_key_float("PSTR").ok().filter(|&v| v > 0.0)
    }

    fn call_smc(&self, input: &SmcKeyData) -> Result<SmcKeyData> {
        let mut output = SmcKeyData::default();
        let mut output_size = mem::size_of::<SmcKeyData>();

        let result = unsafe {
            IOConnectCallStructMethod(
                self.conn,
                KERNEL_INDEX_SMC,
                input as *const SmcKeyData as *const u8,
                mem::size_of::<SmcKeyData>(),
                &mut output as *mut SmcKeyData as *mut u8,
                &mut output_size,
            )
        };

        if result != 0 {
            return Err(anyhow!("SMC call failed with code {}", result));
        }

        Ok(output)
    }
}

impl Drop for SmcConnection {
    fn drop(&mut self) {
        unsafe {
            IOServiceClose(self.conn);
        }
    }
}

fn decode_smc_float(data_type: u32, bytes: &[u8]) -> Result<f32> {
    match data_type {
        SMC_KEY_TYPE_FLT => {
            if bytes.len() >= 4 {
                // Apple Silicon SMC returns little-endian floats
                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            } else {
                Err(anyhow!("flt: insufficient bytes"))
            }
        }
        SMC_KEY_TYPE_SP78 => {
            if bytes.len() >= 2 {
                // sp78: signed 7.8 fixed-point, big-endian
                let raw = i16::from_be_bytes([bytes[0], bytes[1]]);
                Ok(raw as f32 / 256.0)
            } else {
                Err(anyhow!("sp78: insufficient bytes"))
            }
        }
        SMC_KEY_TYPE_UI8 => {
            if !bytes.is_empty() {
                Ok(bytes[0] as f32)
            } else {
                Err(anyhow!("ui8: insufficient bytes"))
            }
        }
        SMC_KEY_TYPE_UI16 => {
            if bytes.len() >= 2 {
                Ok(u16::from_le_bytes([bytes[0], bytes[1]]) as f32)
            } else {
                Err(anyhow!("ui16: insufficient bytes"))
            }
        }
        SMC_KEY_TYPE_UI32 => {
            if bytes.len() >= 4 {
                Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32)
            } else {
                Err(anyhow!("ui32: insufficient bytes"))
            }
        }
        _ => Err(anyhow!("Unknown SMC data type: 0x{:08x}", data_type)),
    }
}

// IOKit FFI
extern "C" {
    fn IOServiceClose(connect: u32) -> i32;
    fn IOConnectCallStructMethod(
        connection: u32,
        selector: u32,
        input: *const u8,
        input_size: usize,
        output: *mut u8,
        output_size: *mut usize,
    ) -> i32;
    fn IOServiceGetMatchingService(
        main_port: u32,
        matching: core_foundation_sys::dictionary::CFDictionaryRef,
    ) -> u32;
    fn IOServiceOpen(service: u32, owning_task: u32, conn_type: u32, connect: *mut u32) -> i32;
    fn IOServiceMatching(name: *const i8) -> core_foundation_sys::dictionary::CFDictionaryRef;
    fn IOObjectRelease(object: u32) -> i32;
}

extern "C" {
    fn mach_task_self() -> u32;
}

unsafe fn open_smc_connection() -> Result<u32> {
    let name = CString::new("AppleSMC").unwrap();
    let matching = IOServiceMatching(name.as_ptr());
    if matching.is_null() {
        return Err(anyhow!("IOServiceMatching(AppleSMC) returned null"));
    }

    let service = IOServiceGetMatchingService(0, matching);
    if service == 0 {
        return Err(anyhow!("Could not find AppleSMC service"));
    }

    let mut conn: u32 = 0;
    let result = IOServiceOpen(service, mach_task_self(), 0, &mut conn);
    IOObjectRelease(service);

    if result != 0 {
        return Err(anyhow!("IOServiceOpen failed: {}", result));
    }

    Ok(conn)
}
