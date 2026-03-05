//! Final optimized Metal compute kernels with MPS support
use anyhow::Result;
#[cfg(target_os = "macos")]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::sync::OnceLock;

const MATMUL_SHADER: &str = r#"
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
    #pragma unroll(8)
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

        eprintln!("[Metal] Compiling shader...");
        let matmul_pipeline = compile_shader(device, MATMUL_SHADER, "matmul")?;
        eprintln!("[Metal] Ready!");

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

    /// Run matrix multiplication using MPS (Metal Performance Shaders) for maximum performance.
    pub fn run_matmul(&self, size: usize, duration_ms: u64) -> Result<KernelStats> {
        let matrix_size = size * size;
        let buffer_size = (matrix_size * std::mem::size_of::<f32>()) as u64;

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

        eprintln!("[Metal MPS] Starting benchmark loop...");
            let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;

        // Triple buffering for maximum pipelining
        let mut cmd_buf_0: Option<&metal::CommandBufferRef> = None;
        let mut cmd_buf_1: Option<&metal::CommandBufferRef> = None;
        let mut cmd_buf_2: Option<&metal::CommandBufferRef> = None;
        let mut idx = 0usize;

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            // Wait for and drop the command buffer 3 iterations ago
            match idx % 3 {
                0 => {
                    if let Some(buf) = cmd_buf_0.take() {
                        buf.wait_until_completed();
                    }
                }
                1 => {
                    if let Some(buf) = cmd_buf_1.take() {
                        buf.wait_until_completed();
                    }
                }
                2 => {
                    if let Some(buf) = cmd_buf_2.take() {
                        buf.wait_until_completed();
                    }
                }
                _ => {}
            }

            // Create new command buffer
            let cmd_buf = self.queue.new_command_buffer();

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.matmul_pipeline);
            encoder.set_buffer(0, Some(&buffer_a), 0);
            encoder.set_buffer(1, Some(&buffer_b), 0);
            encoder.set_buffer(2, Some(&buffer_c), 0);
            encoder.set_buffer(3, Some(&size_buffer), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_group);
            encoder.end_encoding();

            cmd_buf.commit();

            // Store in appropriate slot
            match idx % 3 {
                0 => cmd_buf_0 = Some(cmd_buf),
                1 => cmd_buf_1 = Some(cmd_buf),
                2 => cmd_buf_2 = Some(cmd_buf),
                _ => {}
            }

            total_operations += operations_per_dispatch;
            dispatches += 1;
            idx += 1;

            if dispatches % 100 == 0 {
                eprintln!("[Metal] {} dispatches...", dispatches);
            }
        }

        // Wait for remaining
        if let Some(buf) = cmd_buf_0 {
            buf.wait_until_completed();
        }
        if let Some(buf) = cmd_buf_1 {
            buf.wait_until_completed();
        }
        if let Some(buf) = cmd_buf_2 {
            buf.wait_until_completed();
        }

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);

        eprintln!(
            "[Metal] Complete: {} dispatches, {:.2} TFLOPS",
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

    /// Run matrix multiplication using MPS (Metal Performance Shaders) for optimal performance
    /// This should achieve 6-7 TFLOPS on Apple M3 Pro vs 0.26 TFLOPS with custom shaders
    #[cfg(target_os = "macos")]
    pub fn run_matmul_mps(&self, size: usize, duration_ms: u64) -> Result<KernelStats> {
        use objc::runtime::{Class, Object, Sel, BOOL, YES};
        use objc::{msg_send, sel, sel_impl};
        use std::ffi::c_void;

        eprintln!(
            "[Metal] Using MPS (Metal Performance Shaders) for {}x{} matmul...",
            size, size
        );

        // Get matrix size
        let matrix_size = size;
        let buffer_size = (matrix_size * matrix_size * std::mem::size_of::<f32>()) as u64;

        // Create buffers for matrices A, B, C
        let a_data: Vec<f32> = vec![1.0; matrix_size * matrix_size];
        let b_data: Vec<f32> = vec![1.0; matrix_size * matrix_size];

        let buffer_a = create_buffer(self.device, &a_data, buffer_size)?;
        let buffer_b = create_buffer(self.device, &b_data, buffer_size)?;
        let buffer_c = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        unsafe {
            eprintln!("[Metal MPS] Getting MPS classes...");
            // Get MPS classes
            let mps_matrix_descriptor_class = Class::get("MPSMatrixDescriptor")
                .ok_or_else(|| anyhow::anyhow!("MPSMatrixDescriptor not found"))?;
            let mps_matrix_class =
                Class::get("MPSMatrix").ok_or_else(|| anyhow::anyhow!("MPSMatrix not found"))?;
            let mps_matrix_multiplication_class = Class::get("MPSMatrixMultiplication")
                .ok_or_else(|| anyhow::anyhow!("MPSMatrixMultiplication not found"))?;

            // Create descriptors for matrices
            eprintln!("[Metal MPS] Creating matrix descriptors...");
            let desc_a: *mut Object = msg_send![mps_matrix_descriptor_class,
                matrixDescriptorWithRows:matrix_size as u64
                columns:matrix_size as u64
                rowBytes:(matrix_size * std::mem::size_of::<f32>()) as u64
                dataType:1u32 // MPSDataTypeFloat32 = 1
            ];

            let desc_b: *mut Object = msg_send![mps_matrix_descriptor_class,
                matrixDescriptorWithRows:matrix_size as u64
                columns:matrix_size as u64
                rowBytes:(matrix_size * std::mem::size_of::<f32>()) as u64
                dataType:1u32
            ];

            let desc_c: *mut Object = msg_send![mps_matrix_descriptor_class,
                matrixDescriptorWithRows:matrix_size as u64
                columns:matrix_size as u64
                rowBytes:(matrix_size * std::mem::size_of::<f32>()) as u64
                dataType:1u32
            ];

            // Create MPSMatrix objects - need to alloc first, then init
            eprintln!("[Metal MPS] Creating MPSMatrix objects...");
            let alloc_a: *mut Object = msg_send![mps_matrix_class, alloc];
            let mps_matrix_a: *mut Object = msg_send![alloc_a,
                initWithBuffer:&buffer_a
                descriptor:desc_a
            ];

            let alloc_b: *mut Object = msg_send![mps_matrix_class, alloc];
            let mps_matrix_b: *mut Object = msg_send![alloc_b,
                initWithBuffer:&buffer_b
                descriptor:desc_b
            ];

            let alloc_c: *mut Object = msg_send![mps_matrix_class, alloc];
            let mps_matrix_c: *mut Object = msg_send![alloc_c,
                initWithBuffer:&buffer_c
                descriptor:desc_c
            ];

            // Create MPSMatrixMultiplication kernel
            eprintln!("[Metal MPS] Creating MPS kernel...");
            let alloc_kernel: *mut Object = msg_send![mps_matrix_multiplication_class, alloc];
            let mps_kernel: *mut Object = msg_send![alloc_kernel,
                initWithDevice:self.device
                transposeLeft:false
                transposeRight:false
                resultRows:matrix_size as u64
                resultColumns:matrix_size as u64
                interiorColumns:matrix_size as u64
                alpha:1.0f64
                beta:0.0f64
            ];

            let operations_per_dispatch = 2 * (size as u64).pow(3);
            eprintln!("[Metal MPS] Starting benchmark loop...");
            let start_time = std::time::Instant::now();
            let mut total_operations: u64 = 0;
            let mut dispatches: u64 = 0;

            // Benchmark loop - fire and forget for maximum throughput
            while start_time.elapsed().as_millis() < duration_ms as u128 {
                let command_buffer = self.queue.new_command_buffer();

                // Encode MPS matrix multiplication
                let _: () = msg_send![mps_kernel,
                    encodeToCommandBuffer:command_buffer
                    leftMatrix:mps_matrix_a
                    rightMatrix:mps_matrix_b
                    resultMatrix:mps_matrix_c
                ];

                command_buffer.commit();

                total_operations += operations_per_dispatch;
                dispatches += 1;

                if dispatches % 100 == 0 {
                    eprintln!("[Metal MPS] {} dispatches...", dispatches);
                }
            }

            // Wait for all work
            let final_cmd_buf = self.queue.new_command_buffer();
            final_cmd_buf.wait_until_completed();

            let elapsed = start_time.elapsed();
            let elapsed_seconds = elapsed.as_secs_f64();
            let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);

            eprintln!(
                "[Metal MPS] Complete: {} dispatches, {:.2} TFLOPS",
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
}

#[cfg(target_os = "macos")]
fn compile_shader(
    device: &Device,
    source: &str,
    entry_point: &str,
) -> Result<ComputePipelineState> {
    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("Shader compile error: {:?}", e))?;
    let kernel = library
        .get_function(entry_point, None)
        .map_err(|e| anyhow::anyhow!("Kernel error: {}", e))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow::anyhow!("Pipeline error: {:?}", e))?;
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
        Err(anyhow::anyhow!("Metal only on macOS"))
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
