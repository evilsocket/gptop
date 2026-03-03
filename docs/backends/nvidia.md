# NVIDIA Backend

GPU monitoring for NVIDIA GPUs on Linux via NVML (NVIDIA Management Library).

## Requirements

- Linux with NVIDIA proprietary drivers installed
- NVML shared library (`libnvidia-ml.so`), included with the driver package

## Data Sources

| Metric | Source |
|--------|--------|
| GPU utilization | `nvmlDeviceGetUtilizationRates` |
| Memory used/total | `nvmlDeviceGetMemoryInfo` |
| Clock speed | `nvmlDeviceGetClockInfo(Graphics)` |
| Max clock speed | `nvmlDeviceGetMaxClockInfo(Graphics)` |
| Temperature | `nvmlDeviceGetTemperature(GPU)` |
| Power draw | `nvmlDeviceGetPowerUsage` (mW, converted to W) |
| Power limit | `nvmlDeviceGetPowerManagementLimit` |
| Encoder utilization | `nvmlDeviceGetEncoderUtilization` |
| Decoder utilization | `nvmlDeviceGetDecoderUtilization` |
| CUDA core count | `nvmlDeviceGetNumCores` |
| FP32 TFLOPS | Computed: `cuda_cores * freq_MHz * 2 / 1e6` |
| Processes | `nvmlDeviceGetRunningComputeProcesses` + `nvmlDeviceGetRunningGraphicsProcesses` |
| Per-process GPU % | `nvmlDeviceGetProcessUtilizationStats` |
| Process info | `/proc/<pid>/comm`, `/proc/<pid>/status`, `/proc/<pid>/stat` |
| System memory | `/proc/meminfo` (MemTotal, MemAvailable, SwapTotal, SwapFree) |

## Multi-GPU

All NVIDIA devices are automatically enumerated. Each GPU gets its own utilization chart, memory chart and info bar displayed side by side. The process table shows a `DEV` column when more than one device is present.

## Notes

- NVML queries are instantaneous (unlike Apple's IOReport which blocks for the sample duration), so gptop sleeps for the configured interval before each sample.
- Metrics that return `NVML_ERROR_NOT_SUPPORTED` on certain GPU models are gracefully skipped and reported as zero.
- The FP32 TFLOPS figure assumes 2 FP32 operations per clock per CUDA core (FMA), computed at the current clock speed.
- Process GPU memory uses the `UsedGpuMemory` field from NVML, which may report as unavailable under WDDM (Windows), but is always available on Linux.
