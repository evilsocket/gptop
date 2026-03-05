//! Native Metal compute kernels for maximum performance on Apple Silicon
use anyhow::Result;
#[cfg(target_os = "macos")]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::sync::atomic::AtomicUsize;
use std::sync::OnceLock;

const MATMUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    threadgroup float* sharedA [[threadgroup(0)]],
    threadgroup float* sharedB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]
) {
    const uint BLOCK_SIZE = 32u;
    uint row = bid.x * BLOCK_SIZE + tid.x;
    uint col = bid.y * BLOCK_SIZE + tid.y;
    float sum = 0.0f;
    for (uint block = 0u; block < size; block += BLOCK_SIZE) {
        if (row < size && (block + tid.y) < size) {
            sharedA[tid.x * BLOCK_SIZE + tid.y] = A[row * size + (block + tid.y)];
        } else {
            sharedA[tid.x * BLOCK_SIZE + tid.y] = 0.0f;
        }
        if ((block + tid.x) < size && col < size) {
            sharedB[tid.x * BLOCK_SIZE + tid.y] = B[(block + tid.x) * size + col];
        } else {
            sharedB[tid.x * BLOCK_SIZE + tid.y] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint k = 0u; k < BLOCK_SIZE; k++) {
            sum = fma(sharedA[tid.x * BLOCK_SIZE + k], sharedB[k * BLOCK_SIZE + tid.y], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}
"#;

const ELEMENT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void element_wise(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float result = c[gid];
    float a_val = a[gid];
    float b_val = b[gid];
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    result = fma(a_val, b_val, result);
    c[gid] = result;
}
"#;

const BANDWIDTH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bandwidth(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    dst[gid] = src[gid];
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
    element_pipeline: ComputePipelineState,
    bandwidth_pipeline: ComputePipelineState,
    current_buffer: AtomicUsize,
    has_unified_memory: bool,
    matmul_threads: MTLSize,
    element_threads: MTLSize,
    bandwidth_threads: MTLSize,
}

#[cfg(target_os = "macos")]
impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = get_metal_device().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        let queue = device.new_command_queue();
        let has_unified_memory = device.has_unified_memory();
        let matmul_pipeline = compile_shader(device, MATMUL_SHADER, "matmul")?;
        let element_pipeline = compile_shader(device, ELEMENT_SHADER, "element_wise")?;
        let bandwidth_pipeline = compile_shader(device, BANDWIDTH_SHADER, "bandwidth")?;
        let matmul_threads = MTLSize::new(32, 32, 1);
        let element_threads = MTLSize::new(1024, 1, 1);
        let bandwidth_threads = MTLSize::new(1024, 1, 1);
        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            element_pipeline,
            bandwidth_pipeline,
            current_buffer: AtomicUsize::new(0),
            has_unified_memory,
            matmul_threads,
            element_threads,
            bandwidth_threads,
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
        let a_data: Vec<f32> = vec![1.0; matrix_size];
        let b_data: Vec<f32> = vec![1.0; matrix_size];
        let buffer_a0 = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b0 = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c0 = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer0 = create_constant_buffer(self.device, size as u32)?;
        let buffer_a1 = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b1 = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c1 = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer1 = create_constant_buffer(self.device, size as u32)?;
        let threadgroups = MTLSize::new(
            ((size as u64 + 31) / 32) as u64,
            ((size as u64 + 31) / 32) as u64,
            1,
        );
        let shared_mem_size = (2 * 32 * 32 * std::mem::size_of::<f32>()) as u64;
        let operations_per_dispatch = 2 * (size as u64).pow(3);

        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;
        let mut ping_pong = 0u8;

        // Create initial command buffers - once committed, they cannot be reused
        let mut cmd_buf_0 = self.queue.new_command_buffer();
        let mut cmd_buf_1 = self.queue.new_command_buffer();

        // Initial dispatch on buffer 0
        let encoder0 = cmd_buf_0.new_compute_command_encoder();
        encoder0.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder0.set_buffer(0, Some(&buffer_a0), 0);
        encoder0.set_buffer(1, Some(&buffer_b0), 0);
        encoder0.set_buffer(2, Some(&buffer_c0), 0);
        encoder0.set_buffer(3, Some(&size_buffer0), 0);
        encoder0.set_threadgroup_memory_length(0, shared_mem_size / 2);
        encoder0.set_threadgroup_memory_length(1, shared_mem_size / 2);
        encoder0.dispatch_thread_groups(threadgroups, self.matmul_threads);
        encoder0.end_encoding();

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            if ping_pong == 0 {
                // Commit current buffer 0, then create new encoder on fresh buffer 1
                cmd_buf_0.commit();

                // Create a fresh command buffer for the next iteration
                cmd_buf_1 = self.queue.new_command_buffer();

                let encoder1 = cmd_buf_1.new_compute_command_encoder();
                encoder1.set_compute_pipeline_state(&self.matmul_pipeline);
                encoder1.set_buffer(0, Some(&buffer_a1), 0);
                encoder1.set_buffer(1, Some(&buffer_b1), 0);
                encoder1.set_buffer(2, Some(&buffer_c1), 0);
                encoder1.set_buffer(3, Some(&size_buffer1), 0);
                encoder1.set_threadgroup_memory_length(0, shared_mem_size / 2);
                encoder1.set_threadgroup_memory_length(1, shared_mem_size / 2);
                encoder1.dispatch_thread_groups(threadgroups, self.matmul_threads);
                encoder1.end_encoding();
            } else {
                // Commit current buffer 1, then create new encoder on fresh buffer 0
                cmd_buf_1.commit();

                // Create a fresh command buffer for the next iteration
                cmd_buf_0 = self.queue.new_command_buffer();

                let encoder0 = cmd_buf_0.new_compute_command_encoder();
                encoder0.set_compute_pipeline_state(&self.matmul_pipeline);
                encoder0.set_buffer(0, Some(&buffer_a0), 0);
                encoder0.set_buffer(1, Some(&buffer_b0), 0);
                encoder0.set_buffer(2, Some(&buffer_c0), 0);
                encoder0.set_buffer(3, Some(&size_buffer0), 0);
                encoder0.set_threadgroup_memory_length(0, shared_mem_size / 2);
                encoder0.set_threadgroup_memory_length(1, shared_mem_size / 2);
                encoder0.dispatch_thread_groups(threadgroups, self.matmul_threads);
                encoder0.end_encoding();
            }
            total_operations += operations_per_dispatch;
            dispatches += 1;
            ping_pong = 1 - ping_pong;
        }

        // Wait for all work to complete
        self.queue.new_command_buffer().wait_until_completed();

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);
        let memory_gb = (buffer_size * 3) as f64 / 1e9;
        let bandwidth_gbps = (memory_gb * dispatches as f64) / elapsed_seconds;
        Ok(KernelStats {
            duration_ms: elapsed.as_millis() as u64,
            operations: total_operations,
            tflops,
            bandwidth_gbps: Some(bandwidth_gbps),
            dispatches,
        })
    }

    pub fn run_element_wise(&self, duration_ms: u64) -> Result<KernelStats> {
        let vector_size = 1 << 22;
        let buffer_size = (vector_size * std::mem::size_of::<f32>()) as u64;
        let a_data: Vec<f32> = vec![1.0; vector_size];
        let b_data: Vec<f32> = vec![1.0; vector_size];
        let buffer_a0 = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b0 = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c0 = self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer0 = create_constant_buffer(self.device, vector_size as u32)?;
        let buffer_a1 = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b1 = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c1 = self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer1 = create_constant_buffer(self.device, vector_size as u32)?;
        let threadgroups = MTLSize::new(((vector_size as u64 + 1023) / 1024) as u64, 1, 1);
        let operations_per_dispatch = (vector_size as u64) * 8;

        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;
        let mut ping_pong = 0u8;

        // Create initial command buffers - once committed, they cannot be reused
        let mut cmd_buf_0 = self.queue.new_command_buffer();
        let mut cmd_buf_1 = self.queue.new_command_buffer();

        // Initial dispatch on buffer 0
        let encoder0 = cmd_buf_0.new_compute_command_encoder();
        encoder0.set_compute_pipeline_state(&self.element_pipeline);
        encoder0.set_buffer(0, Some(&buffer_a0), 0);
        encoder0.set_buffer(1, Some(&buffer_b0), 0);
        encoder0.set_buffer(2, Some(&buffer_c0), 0);
        encoder0.set_buffer(3, Some(&size_buffer0), 0);
        encoder0.dispatch_thread_groups(threadgroups, self.element_threads);
        encoder0.end_encoding();

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            if ping_pong == 0 {
                cmd_buf_0.commit();
                cmd_buf_1 = self.queue.new_command_buffer();
                let encoder1 = cmd_buf_1.new_compute_command_encoder();
                encoder1.set_compute_pipeline_state(&self.element_pipeline);
                encoder1.set_buffer(0, Some(&buffer_a1), 0);
                encoder1.set_buffer(1, Some(&buffer_b1), 0);
                encoder1.set_buffer(2, Some(&buffer_c1), 0);
                encoder1.set_buffer(3, Some(&size_buffer1), 0);
                encoder1.dispatch_thread_groups(threadgroups, self.element_threads);
                encoder1.end_encoding();
            } else {
                cmd_buf_1.commit();
                cmd_buf_0 = self.queue.new_command_buffer();
                let encoder0 = cmd_buf_0.new_compute_command_encoder();
                encoder0.set_compute_pipeline_state(&self.element_pipeline);
                encoder0.set_buffer(0, Some(&buffer_a0), 0);
                encoder0.set_buffer(1, Some(&buffer_b0), 0);
                encoder0.set_buffer(2, Some(&buffer_c0), 0);
                encoder0.set_buffer(3, Some(&size_buffer0), 0);
                encoder0.dispatch_thread_groups(threadgroups, self.element_threads);
                encoder0.end_encoding();
            }
            total_operations += operations_per_dispatch;
            dispatches += 1;
            ping_pong = 1 - ping_pong;
        }

        self.queue.new_command_buffer().wait_until_completed();

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);
        Ok(KernelStats { duration_ms: elapsed.as_millis() as u64, operations: total_operations, tflops, bandwidth_gbps: None, dispatches })
    }

    pub fn run_bandwidth(&self, duration_ms: u64) -> Result<KernelStats> {
        let data_size = 1 << 25;
        let buffer_size = (data_size * std::mem::size_of::<f32>()) as u64;
        let src_data: Vec<f32> = vec![1.0; data_size];
        let buffer_src0 = create_buffer(self.device, &src_data, buffer_size)?;
        let buffer_dst0 = self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer0 = create_constant_buffer(self.device, data_size as u32)?;
        let buffer_src1 = create_buffer(self.device, &src_data, buffer_size)?;
        let buffer_dst1 = self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        let size_buffer1 = create_constant_buffer(self.device, data_size as u32)?;
        let threadgroups = MTLSize::new(((data_size as u64 + 1023) / 1024) as u64, 1, 1);
        let bytes_per_dispatch = buffer_size * 2;

        let start_time = std::time::Instant::now();
        let mut total_bytes: u64 = 0;
        let mut dispatches: u64 = 0;
        let mut ping_pong = 0u8;

        // Create initial command buffers - once committed, they cannot be reused
        let mut cmd_buf_0 = self.queue.new_command_buffer();
        let mut cmd_buf_1 = self.queue.new_command_buffer();

        // Initial dispatch on buffer 0
        let encoder0 = cmd_buf_0.new_compute_command_encoder();
        encoder0.set_compute_pipeline_state(&self.bandwidth_pipeline);
        encoder0.set_buffer(0, Some(&buffer_src0), 0);
        encoder0.set_buffer(1, Some(&buffer_dst0), 0);
        encoder0.set_buffer(2, Some(&size_buffer0), 0);
        encoder0.dispatch_thread_groups(threadgroups, self.bandwidth_threads);
        encoder0.end_encoding();

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            if ping_pong == 0 {
                cmd_buf_0.commit();
                cmd_buf_1 = self.queue.new_command_buffer();
                let encoder1 = cmd_buf_1.new_compute_command_encoder();
                encoder1.set_compute_pipeline_state(&self.bandwidth_pipeline);
                encoder1.set_buffer(0, Some(&buffer_src1), 0);
                encoder1.set_buffer(1, Some(&buffer_dst1), 0);
                encoder1.set_buffer(2, Some(&size_buffer1), 0);
                encoder1.dispatch_thread_groups(threadgroups, self.bandwidth_threads);
                encoder1.end_encoding();
            } else {
                cmd_buf_1.commit();
                cmd_buf_0 = self.queue.new_command_buffer();
                let encoder0 = cmd_buf_0.new_compute_command_encoder();
                encoder0.set_compute_pipeline_state(&self.bandwidth_pipeline);
                encoder0.set_buffer(0, Some(&buffer_src0), 0);
                encoder0.set_buffer(1, Some(&buffer_dst0), 0);
                encoder0.set_buffer(2, Some(&size_buffer0), 0);
                encoder0.dispatch_thread_groups(threadgroups, self.bandwidth_threads);
                encoder0.end_encoding();
            }
            total_bytes += bytes_per_dispatch;
            dispatches += 1;
            ping_pong = 1 - ping_pong;
        }

        self.queue.new_command_buffer().wait_until_completed();

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let bandwidth_gbps = (total_bytes as f64 / 1e9) / elapsed_seconds;
        Ok(KernelStats { duration_ms: elapsed.as_millis() as u64, operations: total_bytes, tflops: 0.0, bandwidth_gbps: Some(bandwidth_gbps), dispatches })
    }
}

#[cfg(target_os = "macos")]
fn compile_shader(device: &Device, source: &str, entry_point: &str) -> Result<ComputePipelineState> {
    let library = device.new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("Failed to compile shader: {:?}", e))?;
    let kernel = library.get_function(entry_point, None)
        .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;
    Ok(pipeline)
}

#[cfg(target_os = "macos")]
fn create_buffer(device: &Device, data: &[f32], size: u64) -> Result<Buffer> {
    Ok(device.new_buffer_with_data(data.as_ptr() as *const _, size, MTLResourceOptions::StorageModeShared))
}

#[cfg(target_os = "macos")]
fn create_constant_buffer(device: &Device, value: u32) -> Result<Buffer> {
    Ok(device.new_buffer_with_data(&value as *const _ as *const _, std::mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared))
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
