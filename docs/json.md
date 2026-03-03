# JSON Mode

gptop can output a single metrics snapshot as JSON and exit, useful for scripting, monitoring pipelines and integration with other tools.

## Usage

```sh
# Default 1-second sample
gptop --json

# Custom sample interval
gptop --json --interval 2000
```

## Output Schema

```json
{
  "devices": [
    {
      "id": 0,
      "name": "Apple M4 Pro GPU",
      "vendor": "Apple",
      "total_memory": 18253611008,
      "core_count": 16
    }
  ],
  "gpu_metrics": [
    {
      "device_id": 0,
      "utilization_pct": 42.0,
      "memory_used": 8523112448,
      "memory_total": 18253611008,
      "freq_mhz": 1398,
      "freq_max_mhz": 1398,
      "power_watts": 8.2,
      "temp_celsius": 58.0
    }
  ],
  "processes": [
    {
      "pid": 1234,
      "user": "evilsocket",
      "name": "python3",
      "gpu_usage_pct": 15.0,
      "gpu_memory": 524288000,
      "cpu_usage_pct": 85.0,
      "host_memory": 1073741824,
      "process_type": "Compute"
    }
  ],
  "system": {
    "cpu_power": 3.5,
    "ane_power": 0.0,
    "package_power": 12.1,
    "ram_used": 12884901888,
    "ram_total": 18253611008,
    "swap_used": 0,
    "swap_total": 0
  }
}
```

## Examples

Extract GPU utilization with `jq`:

```sh
gptop --json | jq '.gpu_metrics[].utilization_pct'
```

List GPU processes sorted by memory:

```sh
gptop --json | jq '.processes | sort_by(-.gpu_memory) | .[] | {name, gpu_memory}'
```

Monitor in a loop:

```sh
while true; do gptop --json --interval 500 | jq -c '{util: .gpu_metrics[0].utilization_pct, temp: .gpu_metrics[0].temp_celsius}'; done
```
