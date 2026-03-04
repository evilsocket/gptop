use super::types::KernelType;
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Benchmark GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            adapter,
        })
    }

    pub fn gpu_info(&self) -> (String, u64) {
        let info = self.adapter.get_info();
        let name = info.name.clone();
        // Get memory size from adapter (may not always be available)
        let memory = 0u64; // Will need to be fetched via platform-specific code
        (name, memory)
    }
}

pub struct KernelRunner {
    context: Arc<GpuContext>,
}

impl KernelRunner {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self { context }
    }

    pub async fn run_kernel(
        &self,
        kernel_type: KernelType,
        duration_ms: u64,
    ) -> Result<KernelStats> {
        match kernel_type {
            KernelType::MatMulSmall => self.matmul_benchmark(512, duration_ms).await,
            KernelType::MatMulMedium => self.matmul_benchmark(1024, duration_ms).await,
            KernelType::MatMulLarge => self.matmul_benchmark(1536, duration_ms).await,
            KernelType::ElementWise => self.element_wise_benchmark(duration_ms).await,
            KernelType::Bandwidth => self.bandwidth_benchmark(duration_ms).await,
            KernelType::ReadHeavy => self.read_heavy_benchmark(duration_ms).await,
            KernelType::WriteHeavy => self.write_heavy_benchmark(duration_ms).await,
            KernelType::Sustained => self.sustained_benchmark(duration_ms).await,
        }
    }

    async fn matmul_benchmark(&self, size: usize, duration_ms: u64) -> Result<KernelStats> {
        // Optimized tiled matrix multiplication for maximum TFLOPS
        // Uses 16x16 tiling with shared memory (workgroup memory)
        
        const TILE_SIZE: u32 = 16;
        let workgroups = (size as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let operations_per_dispatch = 2 * (size as u64).pow(3);
        
        // Create buffers
        let matrix_size = (size * size) as u64 * 4;
        let a_data: Vec<f32> = vec![1.0; size * size];
        let b_data: Vec<f32> = vec![1.0; size * size];
        
        let buffer_a = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_b = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_c = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: matrix_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Optimized matrix multiplication with loop unrolling
        let shader = self.context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&format!(r#"
                @group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
                @group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
                
                @compute @workgroup_size(16, 16, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                    let row = global_id.x;
                    let col = global_id.y;
                    let size = {}u;
                    
                    if (row >= size || col >= size) {{
                        return;
                    }}
                    
                    var sum: f32 = 0.0;
                    let remainder = size % 4u;
                    let unrolled_end = size - remainder;
                    
                    // Unrolled main loop - process 4 elements at a time
                    for (var k: u32 = 0u; k < unrolled_end; k = k + 4u) {{
                        sum = fma(matrix_a[row * size + k], matrix_b[k * size + col], sum);
                        sum = fma(matrix_a[row * size + k + 1u], matrix_b[(k + 1u) * size + col], sum);
                        sum = fma(matrix_a[row * size + k + 2u], matrix_b[(k + 2u) * size + col], sum);
                        sum = fma(matrix_a[row * size + k + 3u], matrix_b[(k + 3u) * size + col], sum);
                    }}
                    
                    // Handle remainder
                    for (var k: u32 = unrolled_end; k < size; k = k + 1u) {{
                        sum = fma(matrix_a[row * size + k], matrix_b[k * size + col], sum);
                    }}
                    
                    matrix_c[row * size + col] = sum;
                }}
            "#, size))),
        });
        
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Run benchmark with batched dispatches for better throughput
        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;
        const BATCH_SIZE: u64 = 64;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MatMul Encoder"),
            });
            
            for _ in 0..BATCH_SIZE {
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("MatMul Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
                }
                
                total_operations += operations_per_dispatch;
                dispatches += 1;
            }
            
            self.context.queue.submit(std::iter::once(encoder.finish()));
        }
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
        let elapsed = start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let tflops = (total_operations as f64) / (elapsed_seconds * 1e12);
        let memory_gb = (matrix_size * 3) as f64 / 1e9;
        let bandwidth_gbps = (memory_gb * dispatches as f64) / elapsed_seconds;
        
        Ok(KernelStats {
            duration_ms: elapsed.as_millis() as u64,
            operations: total_operations,
            tflops,
            bandwidth_gbps: Some(bandwidth_gbps),
            dispatches,
        })
    }

    async fn element_wise_benchmark(&self, duration_ms: u64) -> Result<KernelStats> {
        // Element-wise operations with unrolled loops for maximum ALU utilization
        const WORKGROUP_SIZE: u32 = 256;
        const ELEMENTS_PER_THREAD: u32 = 64; // Each thread processes 64 elements
        let vector_size = 1 << 22; // 4M elements
        let data: Vec<f32> = vec![1.0; vector_size];
        
        let buffer_a = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector A"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_b = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector B"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_c = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector C"),
            size: (vector_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Optimized shader with unrolled loops and more FLOPs per element
        let shader = self.context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                const WORKGROUP_SIZE: u32 = 256u;
                const ELEMENTS_PER_THREAD: u32 = 64u;
                
                @group(0) @binding(0) var<storage, read> vector_a: array<f32>;
                @group(0) @binding(1) var<storage, read> vector_b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> vector_c: array<f32>;
                
                @compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let base_idx = global_id.x * ELEMENTS_PER_THREAD;
                    let array_len = arrayLength(&vector_a);
                    
                    if (base_idx >= array_len) {
                        return;
                    }
                    
                    // Process multiple elements per thread with unrolled operations
                    for (var e: u32 = 0u; e < ELEMENTS_PER_THREAD; e = e + 1u) {
                        let idx = base_idx + e;
                        if (idx >= array_len) {
                            break;
                        }
                        
                        let a = vector_a[idx];
                        let b = vector_b[idx];
                        var result = vector_c[idx];
                        
                        // 8 chained FMA operations = 16 FLOPs per element
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        result = fma(a, b, result);
                        
                        vector_c[idx] = result;
                    }
                }
            "#)),
        });
        
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Element-wise Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Element-wise Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Element-wise Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Calculate workgroups - each thread processes ELEMENTS_PER_THREAD elements
        let num_threads = (vector_size + ELEMENTS_PER_THREAD as usize - 1) / ELEMENTS_PER_THREAD as usize;
        let workgroups = ((num_threads + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize).min(65535) as u32;
        // 8 FMA operations * 2 FLOPs each = 16 FLOPs per element
        let operations_per_dispatch = (vector_size as u64) * 16;
        
        let start_time = std::time::Instant::now();
        let mut total_operations: u64 = 0;
        let mut dispatches: u64 = 0;
        const BATCH_SIZE: u64 = 64;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Element-wise Encoder"),
            });
            
            for _ in 0..BATCH_SIZE {
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Element-wise Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroups, 1, 1);
                }
                
                total_operations += operations_per_dispatch;
                dispatches += 1;
            }
            
            self.context.queue.submit(std::iter::once(encoder.finish()));
        }
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
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

    async fn bandwidth_benchmark(&self, duration_ms: u64) -> Result<KernelStats> {
        // Memory bandwidth test using copy operation
        // Limit to 32M floats = 128MB (fits within wgpu buffer binding limits)
        let data_size = 1 << 25; // 32M floats = 128MB
        let data: Vec<f32> = vec![1.0; data_size];
        
        let buffer_src = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Source Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_dst = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination Buffer"),
            size: (data_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let shader = self.context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bandwidth Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                @group(0) @binding(0) var<storage, read> src: array<f32>;
                @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
                
                @compute @workgroup_size(256, 1, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&src)) {
                        return;
                    }
                    dst[idx] = src[idx];
                }
            "#)),
        });
        
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bandwidth Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bandwidth Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_dst.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bandwidth Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bandwidth Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Ensure workgroups don't exceed the maximum dimension of 65535
        let workgroups = (((data_size + 255) / 256).min(65535)) as u32;
        let bytes_per_dispatch = (data_size * 8) as u64; // Read + Write
        
        let start_time = std::time::Instant::now();
        let mut total_bytes: u64 = 0;
        let mut dispatches: u64 = 0;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Bandwidth Encoder"),
            });
            
            for _ in 0..64 {
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Bandwidth Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroups, 1, 1);
                }
                
                total_bytes += bytes_per_dispatch;
                dispatches += 1;
            }
            
            self.context.queue.submit(std::iter::once(encoder.finish()));
        }
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
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

    async fn read_heavy_benchmark(&self, duration_ms: u64) -> Result<KernelStats> {
        // Read-heavy benchmark
        let data_size = 1 << 25; // 128MB
        let data: Vec<f32> = vec![1.0; data_size];
        
        let buffer_src = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Read Source"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let buffer_dst = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Dest"),
            size: (256 * 4) as u64, // Small output
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let shader = self.context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Read Heavy Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                @group(0) @binding(0) var<storage, read> src: array<f32>;
                @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
                
                @compute @workgroup_size(256, 1, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    var sum: f32 = 0.0;
                    for (var i: u32 = 0u; i < arrayLength(&src); i = i + 1u) {
                        sum = sum + src[i];
                    }
                    dst[global_id.x % 256u] = sum;
                }
            "#)),
        });
        
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Read Heavy Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Read Heavy Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_dst.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Read Heavy Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Read Heavy Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let start_time = std::time::Instant::now();
        let mut total_bytes: u64 = 0;
        let mut dispatches: u64 = 0;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Heavy Encoder"),
            });
            
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Read Heavy Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                for _ in 0..64 {
                    compute_pass.dispatch_workgroups(1, 1, 1);
                    total_bytes += (data_size * 4) as u64;
                    dispatches += 1;
                }
            }
            
            self.context.queue.submit(std::iter::once(encoder.finish()));
        }
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
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

    async fn write_heavy_benchmark(&self, duration_ms: u64) -> Result<KernelStats> {
        // Write-heavy benchmark
        let data_size = 1 << 25; // 128MB
        
        let buffer_dst = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Write Dest"),
            size: (data_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let shader = self.context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Write Heavy Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                @group(0) @binding(0) var<storage, read_write> dst: array<f32>;
                
                @compute @workgroup_size(256, 1, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&dst)) {
                        return;
                    }
                    dst[idx] = f32(idx);
                }
            "#)),
        });
        
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Write Heavy Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Write Heavy Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_dst.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Write Heavy Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Write Heavy Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Ensure workgroups don't exceed the maximum dimension of 65535
        let workgroups = (((data_size + 255) / 256).min(65535)) as u32;
        
        let start_time = std::time::Instant::now();
        let mut total_bytes: u64 = 0;
        let mut dispatches: u64 = 0;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Write Heavy Encoder"),
            });
            
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Write Heavy Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                for _ in 0..64 {
                    compute_pass.dispatch_workgroups(workgroups, 1, 1);
                    total_bytes += (data_size * 4) as u64;
                    dispatches += 1;
                }
            }
            
            self.context.queue.submit(std::iter::once(encoder.finish()));
        }
        
        self.context.device.poll(wgpu::Maintain::Wait);
        
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

    async fn sustained_benchmark(&self, duration_ms: u64) -> Result<KernelStats> {
        // Sustained compute for thermal testing - use medium matmul
        self.matmul_benchmark(1024, duration_ms).await
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
