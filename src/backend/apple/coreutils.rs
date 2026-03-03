use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use core_foundation_sys::base::{CFRelease, CFRetain, CFTypeRef};
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::number::CFNumberRef;
use core_foundation_sys::string::CFStringRef;

/// Create a CFString from a Rust string.
pub fn cfstr(s: &str) -> CFString {
    CFString::new(s)
}

/// Convert a CFStringRef to a Rust String. Returns None if null or conversion fails.
pub fn from_cfstring(cf: CFStringRef) -> Option<String> {
    if cf.is_null() {
        return None;
    }
    unsafe {
        let s: CFString = TCFType::wrap_under_get_rule(cf);
        Some(s.to_string())
    }
}

/// Get a value from a CFDictionary by string key, returning it as a CFType.
pub fn cfdict_get_value(dict: CFDictionaryRef, key: &str) -> Option<CFTypeRef> {
    if dict.is_null() {
        return None;
    }
    let cf_key = cfstr(key);
    unsafe {
        let dict = CFDictionary::<CFString, CFType>::wrap_under_get_rule(dict);
        if dict.contains_key(&cf_key) {
            let val = dict.get(cf_key);
            // Retain since wrap_under_get_rule doesn't transfer ownership
            CFRetain(val.as_CFTypeRef());
            Some(val.as_CFTypeRef())
        } else {
            None
        }
    }
}

/// Extract an i64 from a CFNumberRef.
pub fn cfnum_to_i64(num: CFNumberRef) -> Option<i64> {
    if num.is_null() {
        return None;
    }
    unsafe {
        let n: CFNumber = TCFType::wrap_under_get_rule(num);
        n.to_i64()
    }
}

/// Extract an f64 from a CFNumberRef.
pub fn cfnum_to_f64(num: CFNumberRef) -> Option<f64> {
    if num.is_null() {
        return None;
    }
    unsafe {
        let n: CFNumber = TCFType::wrap_under_get_rule(num);
        n.to_f64()
    }
}

/// Safe CFRelease wrapper that checks for null.
pub fn safe_cfrelease(ptr: CFTypeRef) {
    if !ptr.is_null() {
        unsafe {
            CFRelease(ptr);
        }
    }
}
