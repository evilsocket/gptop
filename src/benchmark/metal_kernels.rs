//! Simplified Metal compute kernels for testing
use anyhow::Result;
#[cfg(target_os = "macos")]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::sync::OnceLock;

// Simplified shader without threadgroup memory for testing
const MATMUL_SHADER_SIMPLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= size || col >= size) return;
    
    float sum = 0.0f;
    for (uint k = 0u; k < size; k++) {
        sum = fma(A[row * size + k], B[k * size + col], sum);
    }
    C[row * size + col] = sum;
}
"#;

#[cfg(target_os = "macos")]
static METAL_DEVICE: OnceLock<Device> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_metal_device() -> Option<&'static Device> {
    Some(METAL_DEVICE.get_or_init(|| Device::system_default().expect("No Metal device found")))
}

#[cfg(target_os = "macos")]
pub struct MetalContext {
    device: &'static Device,
    queue: CommandQueue,
    matmul_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = get_metal_device().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        let queue = device.new_command_queue();

        eprintln!("[Metal] Compiling simplified matmul shader...");
        let matmul_pipeline = compile_shader(device, MATMUL_SHADER_SIMPLE, "matmul")?;
        eprintln!("[Metal] Shader compiled successfully!");

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
        })
    }

    pub fn gpu_info(&self) -> (String, u64) {
        (
            self.device.name().to_string(),
            self.device.recommended_max_working_set_size(),
        )
    }

    pub fn run_matmul(&self, size: usize, duration_ms: u64) -> Result<KernelStats> {
        let matrix_size = size * size;
        let buffer_size = (matrix_size * std::mem::size_of::<f32>()) as u64;

        eprintln!("[Metal] Preparing buffers for {}x{} matmul...", size, size);
        let a_data: Vec<f32> = vec![1.0; matrix_size];
        let b_data: Vec<f32> = vec![1.0; matrix_size];

        let buffer_a = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer = create_constant_buffer(self.device, size as u32)?;

        let threadgroups = MTLSize::new(
            ((size as u64 + 31) / 32) as u64,
            ((size as u64 + 31) / 32) as u64,
            1,
        );
        let threads_per_group = MTLSize::new(32, 32, 1);
        let operations_per_dispatch = 2 * (size as u64).pow(3);

        eprintln!("[Metal] Starting benchmark loop ({} ms)...", duration_ms);
        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;

        // Simple two-buffer ping-pong with explicit synchronization
        let mut prev_cmd_buf: Option<metal::CommandBuffer> = None;

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            // Create new command buffer
            let cmd_buf = self.queue.new_command_buffer();

            // Encode dispatch
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.matmul_pipeline);
            encoder.set_buffer(0, Some(&buffer_a), 0);
            encoder.set_buffer(1, Some(&buffer_b), 0);
            encoder.set_buffer(2, Some(&buffer_c), 0);
            encoder.set_buffer(3, Some(&size_buffer), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_group);
            encoder.end_encoding();

            cmd_buf.commit();

            // Wait for previous command buffer before dropping it
            if let Some(prev) = prev_cmd_buf.take() {
                prev.wait_until_completed();
            }

            // Store current as previous for next iteration
            prev_cmd_buf = Some(cmd_buf);

            total_operations += operations_per_dispatch;
            dispatches += 1;

            if dispatches % 100 == 0 {
                eprintln!("[Metal] {} dispatches...", dispatches);
            }
        }

        // Wait for final command buffer
        if let Some(prev) = prev_cmd_buf {
            prev.wait_until_completed();
        }

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);

        eprintln!(
            "[Metal] Benchmark complete: {} dispatches, {:.2} TFLOPS",
            dispatches, tflops
        );

        Ok(KernelStats {
            duration_ms: elapsed.as_millis() as u64,
            operations: total_operations,
            tflops,
            bandwidth_gbps: None,
            dispatches,
        })
    }
}

#[cfg(target_os = "macos")]
fn compile_shader(
    device: &Device,
    source: &str,
    entry_point: &str,
) -> Result<ComputePipelineState> {
    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("Failed to compile shader: {:?}", e))?;
    let kernel = library
        .get_function(entry_point, None)
        .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;
    Ok(pipeline)
}

#[cfg(target_os = "macos")]
fn create_buffer(device: &Device, data: &[f32], size: u64) -> Result<Buffer> {
    Ok(device.new_buffer_with_data(
        data.as_ptr() as *const _,
        size,
        MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(target_os = "macos")]
fn create_constant_buffer(device: &Device, value: u32) -> Result<Buffer> {
    Ok(device.new_buffer_with_data(
        &value as *const _ as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(not(target_os = "macos"))]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub fn new() -> Result<Self> {
        Err(anyhow::anyhow!("Metal backend only available on macOS"))
    }
}

#[derive(Debug, Clone)]
pub struct KernelStats {
    pub duration_ms: u64,
    pub operations: u64,
    pub tflops: f64,
    pub bandwidth_gbps: Option<f64>,
    pub dispatches: u64,
}
