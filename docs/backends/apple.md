# Apple Silicon Backend

GPU monitoring for Apple Silicon Macs (M1, M2, M3, M4 and their Pro/Max/Ultra variants).

## Requirements

- macOS on Apple Silicon (ARM64)
- No additional drivers or libraries required

## Data Sources

| Metric | Source |
|--------|--------|
| GPU utilization | IOReport (`GPU Performance States` channel) |
| GPU frequency | IOReport P-state residency + known DVFS tables |
| Temperature | SMC (primary), HID sensor (fallback) |
| Power draw | IOReport Energy Model (`GPU` channel) |
| CPU / ANE / package power | IOReport Energy Model + SMC |
| Memory | `host_statistics64` (used), `sysctl hw.memsize` (total) |
| Swap | `sysctl vm.swapusage` |
| FP32 TFLOPS | Computed: `gpu_cores * 128 ALUs * freq_MHz * 2 / 1e6` |
| Processes | IOKit `AGXDeviceUserClient` entries + `proc_pid_rusage` for memory footprint |

## Notes

- Apple Silicon uses unified memory shared between CPU and GPU. The memory chart reflects total system RAM usage.
- GPU utilization is derived from IOReport P-state residency (active vs idle), matching what Activity Monitor reports.
- Process GPU memory is reported via `phys_footprint` from `proc_pid_rusage`, which includes IOAccelerator-mapped pages.
- Each Apple GPU core contains 128 ALUs. The FP32 TFLOPS figure is theoretical peak at the current clock speed.
