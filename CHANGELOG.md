## Version 0.2.0 (2026-03-04)

🚀 New Features
- Advanced view with power limits, system stats, fan speed, throttling status, hostname, and uptime
- Per-process GPU usage chart in process detail modal

🐛 Fixes
- Fixed NVML fan speed API usage and removed unavailable thermal throttling status
- Fixed GPU process utilization distribution when per-process stats unavailable
- Fixed Homebrew release URL and checksum for tagged releases

📚 Documentation
- Added Advanced view documentation to README and usage guide

🔧 Miscellaneous
- Replaced shell commands with native macOS APIs for improved process detection reliability
- Improved CI/CD reliability with platform-specific test gating and headless environment detection

## Version 0.1.1 (2026-03-03)

🚀 New Features:
- Add --version flag
- Add Windows platform stubs for process inspection

🐛 Fixes:
- Fix per-process GPU% accuracy on macOS
- Clean up redundant string operations

📚 Documentation:
- Update installation instructions and remove precompiled binary references
- Update Homebrew formula to build from source instead of prebuilt binaries

🔧 Miscellaneous:
- General code refactoring and CI improvements

## Version 0.1.0 (2026-03-03)

🐛 Fixes
- Resolved all clippy warnings for improved code quality
- Adaptive chart heights to ensure process table remains visible
- Optimized screenshot thumbnail rendering

📚 Documentation
- Added screenshot gallery to README with sample images

🔧 Miscellaneous
- Various minor code improvements and refactoring
