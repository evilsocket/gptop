use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph};
use ratatui::Frame;

use std::collections::VecDeque;

use crate::backend::GpuMetrics;

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

pub fn render_info_bar(
    f: &mut Frame,
    area: Rect,
    metrics: Option<&GpuMetrics>,
    accent: Color,
) {
    let Some(m) = metrics else {
        return;
    };

    let sep = Span::styled(" | ", Style::default().fg(Color::DarkGray));
    let label = Style::default()
        .fg(accent)
        .add_modifier(Modifier::BOLD);
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
        spans.push(sep);
        spans.push(Span::styled("FP32 ", label));
        spans.push(Span::styled(format!("{:.1} TFLOPS", tflops), value));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_chart(
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
                .labels(vec![
                    Span::raw("0%"),
                    Span::raw("50%"),
                    Span::raw("100%"),
                ])
                .style(Style::default().fg(Color::DarkGray)),
        );

    f.render_widget(chart, area);
}
