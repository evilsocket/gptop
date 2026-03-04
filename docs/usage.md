# Usage

## TUI Mode

Launch the interactive terminal UI:

```sh
gptop
```

The TUI displays:

- **GPU utilization chart** per device, with clock speed, temperature, power and FP32 TFLOPS in the info bar.
- **Memory utilization chart** per device, with used/total in the title.
- **Process table** showing all GPU-attached processes with PID, user, command, device, type, GPU%, GPU memory, CPU% and host memory.

Multi-GPU systems show side-by-side charts, one column per device.

## Keybindings

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `Up` / `Down` | Select process in table |
| `Enter` | Open process detail modal |
| `Esc` | Close detail modal |
| `k` | Kill selected process (with confirmation) |
| `Left` / `Right` | Change sort column |
| `+` / `-` | Toggle sort order (ascending / descending) |
| `c` | Cycle accent color |
| `e` | Increase update interval (+250ms, max 5000ms) |
| `d` | Decrease update interval (-250ms, min 250ms) |
| `a` | Toggle Advanced view |

## Advanced View

Press `a` to toggle the Advanced view, which shows additional system and GPU metrics:

- **Host**: Hostname
- **Uptime**: System uptime (days, hours, minutes)
- **IP**: External/public IP address
- **RAM**: System memory used/total and percentage
- **Swap**: Swap used/total (if active)
- **CPU**: CPU power draw (Apple Silicon only)
- **ANE**: Neural Engine power draw (Apple Silicon only)
- **GPU Power**: Current/max power with percentage of limit
- **Enc**: Hardware encoder utilization (NVIDIA only)
- **Dec**: Hardware decoder utilization (NVIDIA only)
- **Fan**: Fan speed percentage (NVIDIA only)
- **Throttled**: Thermal throttling warning (if temperature > 85°C)
- **Efficiency**: Performance-per-watt chart and metrics (GFLOPS/Watt)

## Process Detail Modal

Press `Enter` on a selected process to see:

- PID, PPID, UID/GID, user
- Process state, nice value, start time
- CPU time, CPU%, MEM%
- RSS, VSZ, GPU memory
- Executable path, working directory
- Full command line

## Process Table Columns

| Column | Description |
|--------|-------------|
| PID | Process ID |
| USER | Owner |
| COMMAND | Executable name |
| DEV | GPU device index (hidden with single GPU) |
| TYPE | `Compute` or `Graphics` |
| GPU% | GPU utilization percentage |
| GPU MEM | GPU memory usage |
| CPU% | CPU utilization percentage |
| HOST MEM | Host RAM usage (RSS) |

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | off | Output a single JSON sample and exit |
| `--interval <ms>` | `1000` | Sample duration in milliseconds |
