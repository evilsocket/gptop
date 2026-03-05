use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    
    if target.contains("apple") {
        // Link MetalPerformanceShaders framework on macOS
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
