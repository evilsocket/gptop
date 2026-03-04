use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap};
use ratatui::Frame;

use std::collections::VecDeque;

use crate::backend::{DeviceInfo, GpuMetrics, SystemMetrics};

const CHART_COLORS: &[Color] = &[
    Color::Green,
    Color::Cyan,
    Color::Magenta,
    Color::Yellow,
    Color::Red,
    Color::Blue,
];

/// Pick the color for a given device index. Single-GPU uses accent, multi-GPU
/// uses distinct colors from the palette.
pub fn device_color(index: usize, device_count: usize, accent: Color) -> Color {
    if device_count == 1 {
        accent
    } else {
        CHART_COLORS[index % CHART_COLORS.len()]
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1}G", b / GIB)
    } else {
        format!("{:.0}M", b / MIB)
    }
}

pub fn render_gpu_chart(
    f: &mut Frame,
    area: Rect,
    history: &VecDeque<(f64, f64)>,
    device_name: &str,
    color: Color,
    current_pct: f32,
) {
    let title = format!(" {} {:.0}% ", device_name, current_pct);
    render_chart(f, area, history, &title, color);
}

pub fn render_mem_chart(
    f: &mut Frame,
    area: Rect,
    history: &VecDeque<(f64, f64)>,
    device_name: &str,
    color: Color,
    mem_used: u64,
    mem_total: u64,
) {
    let mem_pct = if mem_total > 0 {
        mem_used as f64 / mem_total as f64 * 100.0
    } else {
        0.0
    };
    let title = format!(
        " {} MEM {:.0}% {}/{} ",
        device_name,
        mem_pct,
        format_bytes(mem_used),
        format_bytes(mem_total)
    );
    render_chart(f, area, history, &title, color);
}

pub fn render_efficiency_chart(
    f: &mut Frame,
    area: Rect,
    history: &VecDeque<(f64, f64)>,
    device_name: &str,
    color: Color,
    current_efficiency: f32,
    gflops_per_watt: Option<f32>,
) {
    let metric_text = gflops_per_watt
        .map(|g| format!("{:.1} GF/W", g))
        .unwrap_or_else(|| "N/A".to_string());

    let title = format!(
        " {} EFF {:.0}% {} ",
        device_name, current_efficiency, metric_text
    );
    render_chart(f, area, history, &title, color);
}

pub fn render_info_bar(f: &mut Frame, area: Rect, metrics: Option<&GpuMetrics>, accent: Color) {
    let Some(m) = metrics else {
        return;
    };

    let sep = Span::styled(" | ", Style::default().fg(Color::DarkGray));
    let label = Style::default().fg(accent).add_modifier(Modifier::BOLD);
    let value = Style::default().fg(Color::White);

    let mut spans = vec![
        Span::styled(" MHz ", label),
        Span::styled(format!("{}", m.freq_mhz), value),
        sep.clone(),
        Span::styled("Temp ", label),
        Span::styled(
            format!("{:.0}°C", m.temp_celsius),
            Style::default().fg(if m.temp_celsius > 80.0 {
                Color::Red
            } else if m.temp_celsius > 60.0 {
                Color::Yellow
            } else {
                Color::Green
            }),
        ),
        sep.clone(),
        Span::styled("Power ", label),
        Span::styled(format!("{:.1}W", m.power_watts), value),
    ];

    if let Some(tflops) = m.fp32_tflops {
        spans.push(sep.clone());
        spans.push(Span::styled("FP32 ", label));
        spans.push(Span::styled(format!("{:.1} TFLOPS", tflops), value));
    }

    if let Some(eff) = m.efficiency_score {
        spans.push(sep.clone());
        spans.push(Span::styled("Eff ", label));
        spans.push(Span::styled(
            format!("{:.0}%", eff),
            Style::default().fg(if eff > 70.0 {
                Color::Green
            } else if eff > 40.0 {
                Color::Yellow
            } else {
                Color::Red
            }),
        ));
    }

    if let Some(eff) = m.efficiency_score {
        spans.push(sep.clone());
        spans.push(Span::styled("Eff ", label));
        spans.push(Span::styled(
            format!("{:.0}%", eff),
            Style::default().fg(if eff > 70.0 {
                Color::Green
            } else if eff > 40.0 {
                Color::Yellow
            } else {
                Color::Red
            }),
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

pub(crate) fn render_chart(
    f: &mut Frame,
    area: Rect,
    history: &VecDeque<(f64, f64)>,
    title: &str,
    color: Color,
) {
    if history.is_empty() {
        return;
    }

    let x_min = history.front().map(|p| p.0).unwrap_or(0.0);
    let mut x_max = history.back().map(|p| p.0).unwrap_or(1.0);
    if x_min >= x_max {
        x_max = x_min + 1.0;
    }

    let data_vec: Vec<(f64, f64)> = history.iter().copied().collect();

    let datasets = vec![Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(&data_vec)];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(title, Style::default().fg(Color::White))),
        )
        .x_axis(
            Axis::default()
                .bounds([x_min, x_max])
                .style(Style::default().fg(Color::DarkGray)),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(vec![Span::raw("0%"), Span::raw("50%"), Span::raw("100%")])
                .style(Style::default().fg(Color::DarkGray)),
        );

    f.render_widget(chart, area);
}

pub fn render_advanced_view(
    f: &mut Frame,
    area: Rect,
    system: &SystemMetrics,
    devices: &[DeviceInfo],
    gpu_metrics: &[GpuMetrics],
    accent: Color,
) {
    let label_style = Style::default().fg(accent).add_modifier(Modifier::BOLD);
    let value_style = Style::default().fg(Color::White);
    let dim_style = Style::default().fg(Color::DarkGray);

    let mut spans: Vec<Span> = Vec::new();
    let sep = Span::styled(" | ", Style::default().fg(Color::DarkGray));

    // Hostname
    spans.push(Span::styled(" Host ", label_style));
    spans.push(Span::styled(&system.hostname, value_style));
    spans.push(sep.clone());

    // Uptime
    let uptime_days = system.uptime_secs / 86400;
    let uptime_hours = (system.uptime_secs % 86400) / 3600;
    let uptime_mins = (system.uptime_secs % 3600) / 60;
    let uptime_str = if uptime_days > 0 {
        format!("{}d {}h {}m", uptime_days, uptime_hours, uptime_mins)
    } else if uptime_hours > 0 {
        format!("{}h {}m", uptime_hours, uptime_mins)
    } else {
        format!("{}m", uptime_mins)
    };
    spans.push(Span::styled(" Uptime ", label_style));
    spans.push(Span::styled(uptime_str, value_style));
    spans.push(sep.clone());

    // External IP
    if let Some(ref ip) = system.external_ip {
        spans.push(Span::styled(" IP ", label_style));
        spans.push(Span::styled(ip, value_style));
        spans.push(sep.clone());
    }

    // System RAM
    let ram_total_gb = system.ram_total as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_used_gb = system.ram_used as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_pct = if system.ram_total > 0 {
        system.ram_used as f64 / system.ram_total as f64 * 100.0
    } else {
        0.0
    };
    spans.push(Span::styled(" RAM ", label_style));
    spans.push(Span::styled(
        format!("{:.1}/{:.1}G ({:.0}%)", ram_used_gb, ram_total_gb, ram_pct),
        value_style,
    ));
    spans.push(sep.clone());

    // Swap
    if system.swap_total > 0 {
        let swap_total_gb = system.swap_total as f64 / 1024.0 / 1024.0 / 1024.0;
        let swap_used_gb = system.swap_used as f64 / 1024.0 / 1024.0 / 1024.0;
        let swap_pct = system.swap_used as f64 / system.swap_total as f64 * 100.0;
        spans.push(Span::styled(" Swap ", label_style));
        spans.push(Span::styled(
            format!(
                "{:.1}/{:.1}G ({:.0}%)",
                swap_used_gb, swap_total_gb, swap_pct
            ),
            value_style,
        ));
        spans.push(sep.clone());
    }

    // CPU Power (Apple Silicon)
    if let Some(cpu_pow) = system.cpu_power {
        spans.push(Span::styled(" CPU ", label_style));
        spans.push(Span::styled(format!("{:.1}W", cpu_pow), value_style));
        spans.push(sep.clone());
    }

    // ANE Power (Apple Silicon)
    if let Some(ane_pow) = system.ane_power {
        spans.push(Span::styled(" ANE ", label_style));
        spans.push(Span::styled(format!("{:.1}W", ane_pow), value_style));
        spans.push(sep.clone());
    }

    // Per-GPU advanced metrics
    for (i, metrics) in gpu_metrics.iter().enumerate() {
        let device_name = devices.get(i).map(|d| d.name.as_str()).unwrap_or("GPU");

        // Power limit
        if let Some(limit) = metrics.power_limit_watts {
            let pct = if limit > 0.0 {
                metrics.power_watts as f64 / limit as f64 * 100.0
            } else {
                0.0
            };
            spans.push(Span::styled(format!(" {}Pwr ", device_name), label_style));
            spans.push(Span::styled(
                format!("{:.1}/{:.0}W ({:.0}%)", metrics.power_watts, limit, pct),
                value_style,
            ));
            spans.push(sep.clone());
        }

        // Encoder
        if let Some(enc) = metrics.encoder_pct {
            spans.push(Span::styled(" Enc ", label_style));
            spans.push(Span::styled(format!("{:.0}%", enc), value_style));
            spans.push(sep.clone());
        }

        // Decoder
        if let Some(dec) = metrics.decoder_pct {
            spans.push(Span::styled(" Dec ", label_style));
            spans.push(Span::styled(format!("{:.0}%", dec), value_style));
            spans.push(sep.clone());
        }

        // Fan speed
        if let Some(fan) = metrics.fan_speed_pct {
            spans.push(Span::styled(" Fan ", label_style));
            spans.push(Span::styled(format!("{}%", fan), value_style));
            spans.push(sep.clone());
        }

        // Throttling
        if let Some(ref reason) = metrics.throttling_reason {
            spans.push(Span::styled(
                " Throttled ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(reason.to_string(), dim_style));
            spans.push(sep.clone());
        }

        // Efficiency
        if let Some(eff_score) = metrics.efficiency_score {
            spans.push(Span::styled(format!(" {}Eff ", device_name), label_style));
            spans.push(Span::styled(
                format!(
                    "{:.0}% ({:.1} GF/W)",
                    eff_score,
                    metrics.efficiency_gflops_per_watt.unwrap_or(0.0)
                ),
                value_style,
            ));
            spans.push(sep.clone());
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Advanced ",
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(
        Paragraph::new(Line::from(spans)).wrap(Wrap { trim: false }),
        inner,
    );
}
