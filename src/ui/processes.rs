use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Text;
use ratatui::widgets::{Block, Borders, Cell, Row, Table, TableState};
use ratatui::Frame;

use crate::backend::GpuProcess;

/// All possible columns in logical order.
const ALL_COLUMNS: &[(&str, usize)] = &[
    // (name, sort_idx)
    ("PID", 0),
    ("USER", 1),
    ("COMMAND", 2),
    ("DEV", 3),
    ("TYPE", 4),
    ("GPU%", 5),
    ("GPU MEM", 6),
    ("CPU%", 7),
    ("HOST MEM", 8),
];

/// Number of logical sort columns.
pub const SORT_COLUMN_COUNT: usize = 9;

fn format_bytes_short(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.0} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.0} KiB", b / KIB)
    } else {
        format!("{} B", bytes)
    }
}

fn cell_value(p: &GpuProcess, sort_idx: usize) -> String {
    match sort_idx {
        0 => format!("{}", p.pid),
        1 => p.user.clone(),
        2 => p.name.clone(),
        3 => format!("{}", p.device_id),
        4 => p.process_type.clone(),
        5 => format!("{:.0}%", p.gpu_usage_pct),
        6 => format_bytes_short(p.gpu_memory),
        7 => format!("{:.0}%", p.cpu_usage_pct),
        8 => format_bytes_short(p.host_memory),
        _ => String::new(),
    }
}

fn column_width(sort_idx: usize) -> Constraint {
    match sort_idx {
        0 => Constraint::Length(8),  // PID
        1 => Constraint::Length(12), // USER
        2 => Constraint::Min(10),    // COMMAND (fills remaining)
        3 => Constraint::Length(4),  // DEV
        4 => Constraint::Length(10), // TYPE
        5 => Constraint::Length(6),  // GPU%
        6 => Constraint::Length(10), // GPU MEM
        7 => Constraint::Length(6),  // CPU%
        8 => Constraint::Length(10), // HOST MEM
        _ => Constraint::Length(1),
    }
}

pub fn sort_processes(processes: &mut [GpuProcess], sort_col: usize, ascending: bool) {
    processes.sort_by(|a, b| {
        let cmp = match sort_col {
            0 => a.pid.cmp(&b.pid),
            1 => a.user.cmp(&b.user),
            2 => a.name.cmp(&b.name),
            3 => a.device_id.cmp(&b.device_id),
            4 => a.process_type.cmp(&b.process_type),
            5 => a
                .gpu_usage_pct
                .partial_cmp(&b.gpu_usage_pct)
                .unwrap_or(std::cmp::Ordering::Equal),
            6 => a.gpu_memory.cmp(&b.gpu_memory),
            7 => a
                .cpu_usage_pct
                .partial_cmp(&b.cpu_usage_pct)
                .unwrap_or(std::cmp::Ordering::Equal),
            8 => a.host_memory.cmp(&b.host_memory),
            _ => std::cmp::Ordering::Equal,
        };
        if ascending {
            cmp
        } else {
            cmp.reverse()
        }
    });
}

pub fn render_processes(
    f: &mut Frame,
    area: Rect,
    processes: &[GpuProcess],
    sort_col: usize,
    sort_asc: bool,
    table_state: &mut TableState,
    accent_color: Color,
) {
    // Determine which columns are visible
    let show_dev = processes.iter().any(|p| p.device_id != 0);

    let visible_cols: Vec<(usize, &str)> = ALL_COLUMNS
        .iter()
        .filter(|(name, _)| {
            if *name == "DEV" && !show_dev {
                return false;
            }
            true
        })
        .map(|(name, sort_idx)| (*sort_idx, *name))
        .collect();

    // Build header
    let header_cells: Vec<Cell> = visible_cols
        .iter()
        .map(|&(si, name)| {
            let label = if si == sort_col {
                let arrow = if sort_asc { "▲" } else { "▼" };
                format!("{}{}", name, arrow)
            } else {
                name.to_string()
            };
            let style = if si == sort_col {
                Style::default()
                    .fg(accent_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            };
            Cell::from(Text::from(label)).style(style)
        })
        .collect();

    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    // Build rows
    let rows: Vec<Row> = processes
        .iter()
        .map(|p| {
            let cells: Vec<Cell> = visible_cols
                .iter()
                .map(|&(si, _)| Cell::from(Text::from(cell_value(p, si))))
                .collect();
            Row::new(cells)
        })
        .collect();

    // Build widths
    let widths: Vec<Constraint> = visible_cols
        .iter()
        .map(|&(si, _)| column_width(si))
        .collect();

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Processes "),
        )
        .row_highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        );

    f.render_stateful_widget(table, area, table_state);
}
