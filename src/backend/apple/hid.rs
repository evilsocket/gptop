use anyhow::{anyhow, Result};
use core_foundation::base::TCFType;
use core_foundation_sys::base::CFTypeRef;
use std::ffi::c_void;

// IOHIDEventSystemClient opaque types
type IOHIDEventSystemClientRef = *const c_void;
type IOHIDServiceClientRef = *const c_void;
type IOHIDEventRef = *const c_void;

// IOHIDEventSystemClient FFI
#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOHIDEventSystemClientCreate(allocator: CFTypeRef) -> IOHIDEventSystemClientRef;
    fn IOHIDEventSystemClientSetMatching(
        client: IOHIDEventSystemClientRef,
        matching: core_foundation_sys::dictionary::CFDictionaryRef,
    );
    fn IOHIDEventSystemClientCopyServices(
        client: IOHIDEventSystemClientRef,
    ) -> core_foundation_sys::array::CFArrayRef;
    fn IOHIDServiceClientCopyProperty(
        service: IOHIDServiceClientRef,
        key: core_foundation_sys::string::CFStringRef,
    ) -> CFTypeRef;
    fn IOHIDServiceClientCopyEvent(
        service: IOHIDServiceClientRef,
        event_type: i64,
        matching: core_foundation_sys::dictionary::CFDictionaryRef,
        options: i64,
    ) -> IOHIDEventRef;
    fn IOHIDEventGetFloatValue(event: IOHIDEventRef, field: i64) -> f64;
}

const K_IOHID_EVENT_TYPE_TEMPERATURE: i64 = 15;
const K_IOHID_EVENT_FIELD_TEMPERATURE: i64 = 0xF_0000_0001;

use super::coreutils::{cfstr, from_cfstring};

/// IOHIDSensors-based temperature reader (fallback).
pub struct HidTempReader {
    client: IOHIDEventSystemClientRef,
}

unsafe impl Send for HidTempReader {}

impl HidTempReader {
    pub fn new() -> Result<Self> {
        let client = unsafe { IOHIDEventSystemClientCreate(std::ptr::null()) };
        if client.is_null() {
            return Err(anyhow!("IOHIDEventSystemClientCreate failed"));
        }

        // Set matching for temperature sensors
        unsafe {
            let key = cfstr("PrimaryUsagePage");
            let val = core_foundation::number::CFNumber::from(0xFF00i32);
            let dict = core_foundation::dictionary::CFDictionary::from_CFType_pairs(&[
                (key.clone(), val.as_CFType()),
            ]);
            IOHIDEventSystemClientSetMatching(client, dict.as_concrete_TypeRef());
        }

        Ok(Self { client })
    }

    /// Read GPU temperature from HID sensors.
    pub fn read_gpu_temp(&self) -> Option<f32> {
        self.read_sensor_temp("GPU MTR Temp Sensor")
            .or_else(|| self.read_sensor_temp("gpu"))
    }

    /// Read a specific sensor temperature by name prefix.
    fn read_sensor_temp(&self, name_prefix: &str) -> Option<f32> {
        unsafe {
            let services = IOHIDEventSystemClientCopyServices(self.client);
            if services.is_null() {
                return None;
            }

            let count = core_foundation_sys::array::CFArrayGetCount(services);
            let product_key = cfstr("Product");

            for i in 0..count {
                let service = core_foundation_sys::array::CFArrayGetValueAtIndex(services, i)
                    as IOHIDServiceClientRef;

                let name_ref = IOHIDServiceClientCopyProperty(
                    service,
                    product_key.as_concrete_TypeRef(),
                );
                if name_ref.is_null() {
                    continue;
                }

                let name = from_cfstring(name_ref as core_foundation_sys::string::CFStringRef);
                core_foundation_sys::base::CFRelease(name_ref);

                if let Some(name) = name {
                    if !name.contains(name_prefix) {
                        continue;
                    }

                    let event = IOHIDServiceClientCopyEvent(
                        service,
                        K_IOHID_EVENT_TYPE_TEMPERATURE,
                        std::ptr::null(),
                        0,
                    );
                    if event.is_null() {
                        continue;
                    }

                    let temp = IOHIDEventGetFloatValue(event, K_IOHID_EVENT_FIELD_TEMPERATURE);
                    core_foundation_sys::base::CFRelease(event as CFTypeRef);
                    core_foundation_sys::base::CFRelease(services as CFTypeRef);

                    if temp > 0.0 && temp < 150.0 {
                        return Some(temp as f32);
                    }
                }
            }

            core_foundation_sys::base::CFRelease(services as CFTypeRef);
            None
        }
    }
}
