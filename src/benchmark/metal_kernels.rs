//! Native Metal compute kernels for maximum performance on Apple Silicon
use crate::benchmark::types::KernelType;
use anyhow::Result;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

pub struct MetalContext {
    #[cfg(target_os = "macos")]
    device: Device,
    #[cfg(target_os = "macos")]
    queue: CommandQueue,
    #[cfg(target_os = "macos")]
    has_unified_memory: bool,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let device =
                Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
            let queue = device.new_command_queue();

            // Check if device has unified memory (Apple Silicon)
            let has_unified_memory = device.has_unified_memory();

            Ok(Self {
                device,
                queue,
                has_unified_memory,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(anyhow::anyhow!("Metal backend only available on macOS"))
        }
    }

    #[cfg(target_os = "macos")]
    pub fn gpu_info(&self) -> (String, u64) {
        let name = self.device.name().to_string();
        // Get recommended max working set size as an approximation of GPU memory
        let memory = self.device.recommended_max_working_set_size();
        (name, memory)
    }

    #[cfg(target_os = "macos")]
    pub fn run_matmul(&self, size: usize, duration_ms: u64) -> Result<KernelStats> {
        // Matrix multiplication: C = A * B
        let operations_per_dispatch = 2 * (size as u64).pow(3);

        // Create buffers
        let matrix_size = size * size;
        let buffer_size = (matrix_size * std::mem::size_of::<f32>()) as u64;

        let a_data: Vec<f32> = vec![1.0; matrix_size];
        let b_data: Vec<f32> = vec![1.0; matrix_size];

        let buffer_a = self.device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_b = self.device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_c = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        // Metal shader source - highly optimized for Apple Silicon
        let shader_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]
            ) {{
                uint row = gid.x;
                uint col = gid.y;
                uint size = {}u;
                
                if (row >= size || col >= size) return;
                
                float sum = 0.0;
                #pragma unroll(4)
                for (uint k = 0; k < size; k++) {{
                    sum = fma(a[row * size + k], b[k * size + col], sum);
                }}
                c[row * size + col] = sum;
            }}
        "#,
            size
        );

        // Compile shader
        let library = self
            .device
            .new_library_with_source(&shader_source, &metal::CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile shader: {:?}", e))?;

        let kernel = library
            .get_function("matmul", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;

        // Calculate grid size
        let threads_per_threadgroup = MTLSize::new(16, 16, 1);
        let threadgroups = MTLSize::new(
            ((size as u64 + 15) / 16) as u64,
            ((size as u64 + 15) / 16) as u64,
            1,
        );

        // Run benchmark - simpler approach without batching
        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&buffer_a), 0);
            encoder.set_buffer(1, Some(&buffer_b), 0);
            encoder.set_buffer(2, Some(&buffer_c), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();

            total_operations += operations_per_dispatch;
            dispatches += 1;
        }

        // Wait for all commands to complete
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

    #[cfg(target_os = "macos")]
    pub fn run_element_wise(&self, duration_ms: u64) -> Result<KernelStats> {
        let vector_size = 1 << 22; // 4M elements
        let buffer_size = (vector_size * std::mem::size_of::<f32>()) as u64;

        let a_data: Vec<f32> = vec![1.0; vector_size];
        let b_data: Vec<f32> = vec![1.0; vector_size];

        let buffer_a = self.device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_b = self.device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_c = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void element_wise(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= 4194304u) return;
                
                // Multiple FMA operations to increase ALU utilization
                float result = c[gid];
                result = fma(a[gid], b[gid], result);
                result = fma(a[gid], b[gid], result);
                result = fma(a[gid], b[gid], result);
                result = fma(a[gid], b[gid], result);
                c[gid] = result;
            }
        "#;

        let library = self
            .device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile shader: {:?}", e))?;

        let kernel = library
            .get_function("element_wise", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(((vector_size as u64 + 255) / 256) as u64, 1, 1);

        let operations_per_dispatch = (vector_size as u64) * 8; // 8 FMA per element

        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;

        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&buffer_a), 0);
            encoder.set_buffer(1, Some(&buffer_b), 0);
            encoder.set_buffer(2, Some(&buffer_c), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();

            total_operations += operations_per_dispatch;
            dispatches += 1;
        }

        self.queue.new_command_buffer().wait_until_completed();

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);

        Ok(KernelStats {
            duration_ms: elapsed.as_millis() as u64,
            operations: total_operations,
            tflops,
            bandwidth_gbps: None,
            dispatches,
        })
    }

    #[cfg(target_os = "macos")]
    pub fn run_bandwidth(&self, duration_ms: u64) -> Result<KernelStats> {
        let data_size = 1 << 25; // 32M floats = 128MB
        let buffer_size = (data_size * std::mem::size_of::<f32>()) as u64;

        let src_data: Vec<f32> = vec![1.0; data_size];

        let buffer_src = self.device.new_buffer_with_data(
            src_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_dst = self
            .device
            .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void bandwidth(
                device const float* src [[buffer(0)]],
                device float* dst [[buffer(1)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= 33554432u) return;
                dst[gid] = src[gid];
            }
        "#;

        let library = self
            .device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile shader: {:?}", e))?;

        let kernel = library
            .get_function("bandwidth", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(((data_size as u64 + 255) / 256) as u64, 1, 1);

        let bytes_per_dispatch = buffer_size * 2; // Read + Write

        let start_time = std::time::Instant::now();
        let mut total_bytes: u64 = 0;
        let mut dispatches: u64 = 0;
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&buffer_src), 0);
            encoder.set_buffer(1, Some(&buffer_dst), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();

            total_bytes += bytes_per_dispatch;
            dispatches += 1;
        }

        self.queue.new_command_buffer().wait_until_completed();

        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let bandwidth_gbps = (total_bytes as f64 / 1e9) / elapsed_seconds;

        Ok(KernelStats {
            duration_ms: elapsed.as_millis() as u64,
            operations: total_bytes,
            tflops: 0.0,
            bandwidth_gbps: Some(bandwidth_gbps),
            dispatches,
        })
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
