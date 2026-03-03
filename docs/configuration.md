# Configuration

gptop persists user preferences to a JSON config file at:

```
~/.config/gptop.json
```

Settings are saved automatically when changed via keybindings.

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `update_interval_ms` | integer | `1000` | Sampling interval in milliseconds (250 - 5000) |
| `accent_color_idx` | integer | `0` | Active accent color index (cycled with `c`) |
| `sort_column` | integer | `0` | Process table sort column index |
| `sort_ascending` | boolean | `true` | Sort order |

## Example

```json
{
  "update_interval_ms": 1000,
  "accent_color_idx": 2,
  "sort_column": 5,
  "sort_ascending": false
}
```

## Accent Colors

The available accent colors, cycled with `c`:

| Index | Color |
|-------|-------|
| 0 | Green |
| 1 | Cyan |
| 2 | Blue |
| 3 | Magenta |
| 4 | Yellow |
| 5 | Red |
