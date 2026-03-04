pub mod charts;
pub mod processes;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;

/// Top-level render function, composes all UI sections.
pub fn render(f: &mut Frame, app: &mut App) {
    let size = f.area();

    let device_count = app.devices.len();

    // Adaptive chart heights: reserve at least 8 lines for the process table + footer,
    // then split remaining space between GPU chart, memory chart and info bar.
    let available = size.height.saturating_sub(2); // footer + info bar
    let min_procs = 8u16;
    let chart_budget = available.saturating_sub(min_procs);
    // GPU chart gets ~60% of chart budget, memory chart gets ~40%
    let gpu_chart_h = (chart_budget * 6 / 10).max(5);
    let mem_chart_h = (chart_budget.saturating_sub(gpu_chart_h)).max(4);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(gpu_chart_h),
            Constraint::Length(mem_chart_h),
            Constraint::Length(1), // info bar
            Constraint::Min(min_procs),
            Constraint::Length(1), // footer
        ])
        .split(size);

    let accent = app.accent_color();

    // Split GPU, memory and info bar areas into columns (one per GPU)
    let per_gpu = |area| {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                (0..device_count)
                    .map(|_| Constraint::Ratio(1, device_count as u32))
                    .collect::<Vec<_>>(),
            )
            .split(area)
    };
    let gpu_columns = per_gpu(chunks[0]);
    let mem_columns = per_gpu(chunks[1]);
    let info_columns = per_gpu(chunks[2]);

    for (i, device) in app.devices.iter().enumerate() {
        let metrics = app.latest.as_ref().and_then(|s| s.gpu_metrics.get(i));

        let (gpu_pct, mem_used, mem_total) = metrics
            .map(|m| (m.utilization_pct, m.memory_used, m.memory_total))
            .unwrap_or((0.0, 0, 0));

        let color = charts::device_color(i, device_count, accent);

        charts::render_gpu_chart(
            f,
            gpu_columns[i],
            &app.gpu_util_history[i],
            &device.name,
            color,
            gpu_pct,
        );

        charts::render_mem_chart(
            f,
            mem_columns[i],
            &app.mem_util_history[i],
            &device.name,
            color,
            mem_used,
            mem_total,
        );

        charts::render_info_bar(f, info_columns[i], metrics, accent);
    }

    // Render process table
    let sort_col = app.process_sort_col;
    let sort_asc = app.process_sort_asc;
    if let Some(ref latest) = app.latest {
        let mut sorted_procs = latest.processes.clone();
        processes::sort_processes(&mut sorted_procs, sort_col, sort_asc);
        app.sorted_processes = sorted_procs.clone();
        processes::render_processes(
            f,
            chunks[3],
            &sorted_procs,
            sort_col,
            sort_asc,
            &mut app.table_state,
            accent,
        );
    }

    // Render footer with keybindings
    render_footer(f, chunks[4], accent);

    // Render process detail modal if open
    if let Some(ref detail) = app.process_detail {
        render_process_modal(f, size, detail, accent);
    }

    // Render kill confirmation dialog if open
    if let Some(ref confirm) = app.kill_confirm {
        render_kill_confirm(f, size, confirm, accent);
    }
}

fn render_footer(f: &mut Frame, area: Rect, accent: Color) {
    let key_style = Style::default()
        .fg(Color::Black)
        .bg(accent)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(Color::DarkGray);
    let sep = Span::styled(" ", desc_style);

    let line = Line::from(vec![
        Span::styled(" q ", key_style),
        Span::styled(" Quit ", desc_style),
        sep.clone(),
        Span::styled(" ↑↓ ", key_style),
        Span::styled(" Select ", desc_style),
        sep.clone(),
        Span::styled(" ←→ ", key_style),
        Span::styled(" Sort Column ", desc_style),
        sep.clone(),
        Span::styled(" +/- ", key_style),
        Span::styled(" Sort Order ", desc_style),
        sep.clone(),
        Span::styled(" ⏎ ", key_style),
        Span::styled(" Details ", desc_style),
        sep.clone(),
        Span::styled(" k ", key_style),
        Span::styled(" Kill ", desc_style),
        sep.clone(),
        Span::styled(" c ", key_style),
        Span::styled(" Color ", desc_style),
        sep.clone(),
        Span::styled(" e/d ", key_style),
        Span::styled(" Interval ", desc_style),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

fn render_process_modal(
    f: &mut Frame,
    area: Rect,
    detail: &crate::app::ProcessDetail,
    accent: Color,
) {
    // Center modal, 70% width, up to 20 lines tall
    let modal_w = (area.width as f32 * 0.70).max(50.0).min(area.width as f32) as u16;
    let modal_h = 20u16.min(area.height.saturating_sub(4));
    let x = (area.width.saturating_sub(modal_w)) / 2;
    let y = (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w, modal_h);

    // Clear background
    f.render_widget(Clear, modal_area);

    let label_style = Style::default().fg(accent).add_modifier(Modifier::BOLD);
    let val_style = Style::default().fg(Color::White);
    let dim_style = Style::default().fg(Color::DarkGray);

    let mut lines: Vec<Line> = Vec::new();

    let add_field = |lines: &mut Vec<Line>, label: &str, value: &str| {
        lines.push(Line::from(vec![
            Span::styled(format!(" {:>12}: ", label), label_style),
            Span::styled(value.to_string(), val_style),
        ]));
    };

    add_field(&mut lines, "PID", &format!("{}", detail.pid));
    add_field(&mut lines, "PPID", &format!("{}", detail.ppid));
    add_field(&mut lines, "User", &detail.user);
    add_field(
        &mut lines,
        "UID/GID",
        &format!("{}/{}", detail.uid, detail.gid),
    );
    add_field(&mut lines, "State", &detail.state);
    add_field(&mut lines, "Nice", &format!("{}", detail.nice));
    add_field(&mut lines, "Started", &detail.started);
    add_field(&mut lines, "CPU Time", &detail.cpu_time);
    add_field(&mut lines, "CPU%", &format!("{:.1}%", detail.cpu_pct));
    add_field(&mut lines, "MEM%", &format!("{:.1}%", detail.mem_pct));
    add_field(&mut lines, "RSS", &format_bytes_long(detail.rss));
    add_field(&mut lines, "VSZ", &format_bytes_long(detail.vsz));
    add_field(
        &mut lines,
        "GPU Memory",
        &format_bytes_long(detail.gpu_memory),
    );
    add_field(&mut lines, "Path", &detail.path);
    add_field(&mut lines, "CWD", &detail.cwd);

    // Command line can be long, show it wrapped
    lines.push(Line::from(vec![Span::styled(
        " Command Line: ",
        label_style,
    )]));
    // Wrap command into available width
    let cmd_width = modal_w.saturating_sub(4) as usize;
    if !detail.command.is_empty() {
        for chunk in detail.command.as_bytes().chunks(cmd_width) {
            let s = String::from_utf8_lossy(chunk);
            lines.push(Line::from(vec![
                Span::styled("  ", dim_style),
                Span::styled(s.to_string(), val_style),
            ]));
        }
    }

    let title = format!(" Process {} — {} ", detail.pid, detail.name);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent))
        .title(Span::styled(
            title,
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, modal_area);
}

fn render_kill_confirm(
    f: &mut Frame,
    area: Rect,
    confirm: &crate::app::KillConfirm,
    accent: Color,
) {
    let modal_w = 50u16.min(area.width.saturating_sub(4));
    let modal_h = 5u16;
    let x = (area.width.saturating_sub(modal_w)) / 2;
    let y = (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w, modal_h);

    f.render_widget(Clear, modal_area);

    let text = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            format!(" Kill process {} ({})? ", confirm.pid, confirm.name),
            Style::default().fg(Color::White),
        )]),
        Line::from(vec![
            Span::styled(
                " y ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Yes  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " n ",
                Style::default()
                    .fg(Color::Black)
                    .bg(accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" No ", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(Span::styled(
            " Confirm Kill ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, modal_area);
}

fn format_bytes_long(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.0} KiB", b / KIB)
    } else {
        format!("{} B", bytes)
    }
}
