use anyhow::{anyhow, Result};
use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::string::CFString;
use core_foundation_sys::base::{kCFAllocatorDefault, CFRelease, CFTypeRef};
use core_foundation_sys::dictionary::{
    CFDictionaryGetCount, CFDictionaryRef, CFMutableDictionaryRef,
};
use core_foundation_sys::string::CFStringRef;
use std::ffi::c_void;
use std::mem::MaybeUninit;

use super::coreutils::{cfstr, from_cfstring, safe_cfrelease};

// IOReport C function declarations (private API from IOReport.framework)
#[link(name = "IOReport", kind = "dylib")]
extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CFStringRef,
        subgroup: CFStringRef,
        a: u64,
        b: u64,
        c: u64,
    ) -> CFDictionaryRef;

    // Merges b into a in-place. Returns void effectively (we ignore return).
    fn IOReportMergeChannels(
        a: CFDictionaryRef,
        b: CFDictionaryRef,
        nil: CFTypeRef,
    );

    fn IOReportCreateSubscription(
        a: *const c_void,
        channels: CFMutableDictionaryRef,
        b: *mut CFMutableDictionaryRef,
        c: u64,
        d: *const c_void,
    ) -> CFTypeRef; // IOReportSubscriptionRef

    fn IOReportCreateSamples(
        subscription: CFTypeRef,
        a: CFMutableDictionaryRef,
        b: CFTypeRef,
    ) -> CFDictionaryRef;

    fn IOReportCreateSamplesDelta(
        a: CFDictionaryRef,
        b: CFDictionaryRef,
        c: CFTypeRef,
    ) -> CFDictionaryRef;

    fn IOReportChannelGetGroup(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetSubGroup(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetChannelName(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportSimpleGetIntegerValue(sample: CFDictionaryRef, a: i32) -> i64;
    fn IOReportStateGetCount(sample: CFDictionaryRef) -> i32;
    fn IOReportStateGetNameForIndex(sample: CFDictionaryRef, index: i32) -> CFStringRef;
    fn IOReportStateGetResidency(sample: CFDictionaryRef, index: i32) -> i64;
    fn IOReportStateGetInTransitions(sample: CFDictionaryRef, index: i32) -> i64;
}

extern "C" {
    fn CFDictionaryCreateMutableCopy(
        allocator: core_foundation_sys::base::CFAllocatorRef,
        capacity: isize,
        theDict: CFDictionaryRef,
    ) -> CFMutableDictionaryRef;
}

/// IOReport channel groups we subscribe to
const GROUPS: &[&str] = &[
    "Energy Model",
    "CPU Stats",
    "CPU Core Performance States",
    "GPU Stats",
    "GPU Performance States",
];

/// Holds an IOReport subscription and provides sampling.
pub struct IOReportSubscription {
    subscription: CFTypeRef,
    channels: CFMutableDictionaryRef,
}

unsafe impl Send for IOReportSubscription {}

impl IOReportSubscription {
    pub fn new() -> Result<Self> {
        // Collect channels from each group
        let mut channel_dicts: Vec<CFDictionaryRef> = Vec::new();

        for group in GROUPS {
            let group_cf = cfstr(group);
            let ch = unsafe {
                IOReportCopyChannelsInGroup(
                    group_cf.as_concrete_TypeRef(),
                    std::ptr::null(),
                    0,
                    0,
                    0,
                )
            };
            if !ch.is_null() {
                channel_dicts.push(ch);
            }
        }

        if channel_dicts.is_empty() {
            return Err(anyhow!("No IOReport channels found"));
        }

        // Merge all into the first dict (in-place mutation)
        let base = channel_dicts[0];
        for ch in channel_dicts.iter().skip(1) {
            unsafe {
                IOReportMergeChannels(base, *ch, std::ptr::null());
            }
        }

        // Create a mutable copy (required by IOReportCreateSubscription)
        let size = unsafe { CFDictionaryGetCount(base) };
        let mutable_chan = unsafe {
            CFDictionaryCreateMutableCopy(kCFAllocatorDefault, size, base)
        };

        // Release all original channel dicts
        for ch in &channel_dicts {
            unsafe { CFRelease(*ch as CFTypeRef) };
        }

        if mutable_chan.is_null() {
            return Err(anyhow!("CFDictionaryCreateMutableCopy failed"));
        }

        // Create subscription
        let mut sub_out: MaybeUninit<CFMutableDictionaryRef> = MaybeUninit::uninit();
        let subscription = unsafe {
            IOReportCreateSubscription(
                std::ptr::null(),
                mutable_chan,
                sub_out.as_mut_ptr(),
                0,
                std::ptr::null(),
            )
        };

        if subscription.is_null() {
            unsafe { CFRelease(mutable_chan as CFTypeRef) };
            return Err(anyhow!("IOReportCreateSubscription failed"));
        }

        Ok(Self {
            subscription,
            channels: mutable_chan,
        })
    }

    /// Take a snapshot (returns opaque sample ref).
    fn take_sample(&self) -> Result<CFDictionaryRef> {
        let sample = unsafe {
            IOReportCreateSamples(self.subscription, self.channels, std::ptr::null())
        };
        if sample.is_null() {
            return Err(anyhow!("IOReportCreateSamples returned null"));
        }
        Ok(sample)
    }

    /// Sample over a duration, returning the delta.
    pub fn sample_delta(&self, duration_ms: u64) -> Result<Vec<ChannelSample>> {
        let sample1 = self.take_sample()?;
        std::thread::sleep(std::time::Duration::from_millis(duration_ms));
        let sample2 = self.take_sample()?;

        let delta = unsafe {
            IOReportCreateSamplesDelta(sample1, sample2, std::ptr::null())
        };
        unsafe {
            CFRelease(sample1 as CFTypeRef);
            CFRelease(sample2 as CFTypeRef);
        }

        if delta.is_null() {
            return Err(anyhow!("IOReportCreateSamplesDelta returned null"));
        }

        let results = parse_delta_samples(delta);
        unsafe { CFRelease(delta as CFTypeRef) };
        Ok(results)
    }
}

impl Drop for IOReportSubscription {
    fn drop(&mut self) {
        safe_cfrelease(self.subscription);
        safe_cfrelease(self.channels as CFTypeRef);
    }
}

/// Parsed channel sample data.
#[derive(Debug, Clone)]
pub struct ChannelSample {
    pub group: String,
    pub subgroup: String,
    pub channel_name: String,
    pub data: ChannelData,
}

#[derive(Debug, Clone)]
pub enum ChannelData {
    Integer(i64),
    State(Vec<StateResidency>),
}

#[derive(Debug, Clone)]
pub struct StateResidency {
    pub name: String,
    pub residency: i64,
    pub transitions: i64,
}

/// Parse the delta CFDictionary into structured channel samples.
fn parse_delta_samples(delta: CFDictionaryRef) -> Vec<ChannelSample> {
    let mut results = Vec::new();

    // The delta dict has an "IOReportChannels" key containing a CFArray of channel dicts
    unsafe {
        let dict: CFDictionary<CFString, CFType> = CFDictionary::wrap_under_get_rule(delta);
        let channels_key = CFString::new("IOReportChannels");

        if !dict.contains_key(&channels_key) {
            return results;
        }

        let channels_array = dict.get(channels_key);
        let arr_ref = channels_array.as_CFTypeRef() as core_foundation_sys::array::CFArrayRef;
        let count = core_foundation_sys::array::CFArrayGetCount(arr_ref);

        for i in 0..count {
            let entry =
                core_foundation_sys::array::CFArrayGetValueAtIndex(arr_ref, i) as CFDictionaryRef;
            if entry.is_null() {
                continue;
            }

            let group = from_cfstring(IOReportChannelGetGroup(entry))
                .unwrap_or_default();
            let subgroup = from_cfstring(IOReportChannelGetSubGroup(entry))
                .unwrap_or_default();
            let channel_name = from_cfstring(IOReportChannelGetChannelName(entry))
                .unwrap_or_default();

            // Try state-based first
            let state_count = IOReportStateGetCount(entry);
            let data = if state_count > 0 {
                let mut states = Vec::with_capacity(state_count as usize);
                for s in 0..state_count {
                    let name = from_cfstring(IOReportStateGetNameForIndex(entry, s))
                        .unwrap_or_default();
                    let residency = IOReportStateGetResidency(entry, s);
                    let transitions = IOReportStateGetInTransitions(entry, s);
                    states.push(StateResidency {
                        name,
                        residency,
                        transitions,
                    });
                }
                ChannelData::State(states)
            } else {
                let value = IOReportSimpleGetIntegerValue(entry, 0);
                ChannelData::Integer(value)
            };

            results.push(ChannelSample {
                group,
                subgroup,
                channel_name,
                data,
            });
        }
    }

    results
}
