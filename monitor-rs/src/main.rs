//! xdna-top — terminal monitor for AMD XDNA NPUs (amdxdna driver).
//!
//!   xdna-top              interactive TUI (q quit, w force-wake query, +/- interval)
//!   xdna-top --once       print one sample and exit

mod xdna;

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table},
};

const HISTORY: usize = 300;
const CTX_COLORS: [Color; 6] = [Color::Cyan, Color::Magenta, Color::Yellow, Color::Green, Color::Blue, Color::Red];

fn human(b: u64) -> String {
    let mut v = b as f64;
    for u in ["B", "KiB", "MiB", "GiB"] {
        if v < 1024.0 || u == "GiB" {
            return if u == "B" { format!("{v:.0} {u}") } else { format!("{v:.1} {u}") };
        }
        v /= 1024.0;
    }
    unreachable!()
}

fn source_label(s: xdna::Source) -> &'static str {
    match s {
        xdna::Source::Suspended => "suspended",
        xdna::Source::Cached => "idle back-off",
        xdna::Source::Sensors => "column sensors",
        xdna::Source::Occupancy => "column occupancy",
    }
}

struct App {
    dev: xdna::Device,
    sample: xdna::Sample,
    history: VecDeque<f64>,
    interval: Duration,
    force_wake: bool,
}

impl App {
    fn tick(&mut self) {
        self.sample = self.dev.sample(std::mem::take(&mut self.force_wake));
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(self.sample.util_pct);
    }
}

fn main() -> io::Result<()> {
    let once = std::env::args().any(|a| a == "--once");
    let Some(info) = xdna::discover().into_iter().next() else {
        eprintln!("No amdxdna device under /sys/class/accel (NPU disabled in BIOS or driver not loaded?)");
        std::process::exit(1);
    };
    let node = info.node.clone();
    let dev = xdna::Device::open(info).unwrap_or_else(|e| {
        eprintln!("{}: {e} (is your user in the 'render' group?)", node.display());
        std::process::exit(1);
    });
    let mut app = App {
        dev,
        sample: Default::default(),
        history: VecDeque::with_capacity(HISTORY),
        interval: Duration::from_millis(1000),
        force_wake: true,
    };
    app.tick();

    if once {
        print_once(&app);
        return Ok(());
    }

    let mut term = ratatui::init();
    let res = run(&mut term, &mut app);
    ratatui::restore();
    res
}

fn print_once(app: &App) {
    let i = &app.dev.info;
    let s = &app.sample;
    println!("{}  {}  {}  fw {}", i.name, i.pci, i.vbnv, i.fw);
    println!(
        "array {}x{} (core {}, mem {}, shim {})  AIE v{}.{}",
        i.cols, i.rows, i.core_rows, i.mem_rows, i.shim_rows, i.aie_version.0, i.aie_version.1
    );
    println!("runtime pm {}  MP-NPU {} MHz  H-clock {} MHz", s.runtime_status, s.mp_npu_mhz, s.h_mhz);
    println!(
        "util {:.0}% ({})  power {}  bo memory {}  contexts {}",
        s.util_pct,
        source_label(s.source),
        s.power_mw.map(|p| format!("{p:.0} mW")).unwrap_or_else(|| "n/a".into()),
        human(s.mem_used),
        s.contexts.len()
    );
}

fn run(term: &mut DefaultTerminal, app: &mut App) -> io::Result<()> {
    let mut next = Instant::now() + app.interval;
    loop {
        term.draw(|f| draw(f, app))?;
        let timeout = next.saturating_duration_since(Instant::now());
        if event::poll(timeout)? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('w') => {
                        app.force_wake = true;
                        next = Instant::now();
                    }
                    KeyCode::Char('+') => app.interval = (app.interval + Duration::from_millis(250)).min(Duration::from_secs(10)),
                    KeyCode::Char('-') => app.interval = app.interval.saturating_sub(Duration::from_millis(250)).max(Duration::from_millis(250)),
                    _ => {}
                }
            }
        }
        if Instant::now() >= next {
            app.tick();
            next = Instant::now() + app.interval;
        }
    }
}

fn draw(f: &mut Frame, app: &App) {
    let [header, middle, bottom, help] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(12),
        Constraint::Length(9),
        Constraint::Length(1),
    ])
    .areas(f.area());
    draw_header(f, header, app);
    let [grid, chart] = Layout::horizontal([Constraint::Length(44), Constraint::Min(30)]).areas(middle);
    draw_grid(f, grid, app);
    draw_chart(f, chart, app);
    draw_tables(f, bottom, app);
    f.render_widget(
        Paragraph::new(format!(
            " q quit · w wake+query · +/- interval ({} ms) · idle NPUs are never woken by polling",
            app.interval.as_millis()
        ))
        .dim(),
        help,
    );
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let i = &app.dev.info;
    let s = &app.sample;
    let pm_style = if s.runtime_status == "suspended" { Style::new().fg(Color::DarkGray) } else { Style::new().fg(Color::Green) };
    let modes = ["default", "low", "medium", "high", "turbo"];
    let lines = vec![
        Line::from(vec![
            Span::styled(i.name.clone(), Style::new().bold()),
            Span::raw(format!("  {}  {}  fw {}  AIE v{}.{}", i.pci, i.vbnv, i.fw, i.aie_version.0, i.aie_version.1)),
        ]),
        Line::from(vec![
            Span::raw("pm "),
            Span::styled(s.runtime_status.clone(), pm_style),
            Span::raw(format!(
                "   MP-NPU {} MHz   H-clock {} MHz{}   mode {}   power {}",
                s.mp_npu_mhz,
                s.h_mhz,
                if i.clk_max_mhz > 0 { format!(" (max {})", i.clk_max_mhz) } else { String::new() },
                s.power_mode.and_then(|m| modes.get(m as usize).copied()).unwrap_or("?"),
                s.power_mw.map(|p| format!("{p:.0} mW")).unwrap_or_else(|| "n/a".into()),
            )),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).block(Block::new().borders(Borders::BOTTOM)), area);
}

/// Draw the AIE array: shim row at the bottom, then mem, then core rows.
fn draw_grid(f: &mut Frame, area: Rect, app: &App) {
    let i = &app.dev.info;
    let s = &app.sample;
    let cols = i.cols.max(1) as usize;

    // Column → (context index) owner map.
    let mut owner: Vec<Option<usize>> = vec![None; cols];
    for (ci, c) in s.contexts.iter().enumerate() {
        for col in c.start_col..c.start_col + c.num_col {
            if let Some(o) = owner.get_mut(col as usize) {
                *o = Some(ci);
            }
        }
    }

    let mut lines = Vec::new();
    let mut col_hdr = vec![Span::raw("       ")];
    for c in 0..cols {
        col_hdr.push(Span::raw(format!(" col{c} ")));
    }
    lines.push(Line::from(col_hdr));

    let rows: Vec<(&str, u16)> = std::iter::repeat_n("core", i.core_rows as usize)
        .enumerate()
        .map(|(k, n)| (n, i.core_row_start + i.core_rows - 1 - k as u16))
        .chain(std::iter::repeat_n(("mem ", i.mem_row_start), i.mem_rows as usize))
        .chain(std::iter::repeat_n(("shim", 0u16), i.shim_rows as usize))
        .collect();

    for (kind, row) in rows {
        let mut spans = vec![Span::raw(format!("{kind} r{row} "))];
        for c in 0..cols {
            let (glyph, style) = match owner[c] {
                Some(ci) => {
                    let color = CTX_COLORS[ci % CTX_COLORS.len()];
                    if s.busy_cols.get(c).copied().unwrap_or(false) {
                        (" ████ ", Style::new().fg(color).add_modifier(Modifier::BOLD))
                    } else {
                        (" ▒▒▒▒ ", Style::new().fg(color))
                    }
                }
                None => (" ░░░░ ", Style::new().fg(Color::DarkGray)),
            };
            spans.push(Span::styled(glyph, style));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::raw(""));
    let mut busy = vec![Span::raw("busy    ")];
    for c in 0..cols {
        let b = s.busy_cols.get(c).copied().unwrap_or(false);
        let txt = match s.column_util.get(c) {
            Some(u) => format!("{u:>4.0}% "),
            None => (if b { "  ●   " } else { "  ·   " }).to_string(),
        };
        busy.push(Span::styled(txt, if b { Style::new().fg(Color::Green) } else { Style::new().dim() }));
    }
    lines.push(Line::from(busy));
    lines.push(Line::raw(format!(
        "{} cols × {} rows · {} AIE-ML core tiles",
        i.cols,
        i.rows,
        i.cols as u32 * i.core_rows as u32
    )).dim());

    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" AIE array (░ free ▒ owned █ busy) ")),
        area,
    );
}

fn draw_chart(f: &mut Frame, area: Rect, app: &App) {
    let s = &app.sample;
    let n = app.history.len();
    let data: Vec<(f64, f64)> = app.history.iter().enumerate().map(|(k, v)| ((HISTORY - n + k) as f64, *v)).collect();
    let ds = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(Color::Cyan))
        .data(&data);
    let title = format!(
        " utilisation {:.0}% ({}) · BO memory {} · tasks {}/{} ",
        s.util_pct,
        source_label(s.source),
        human(s.mem_used),
        s.tasks,
        app.dev.info.task_max
    );
    let chart = Chart::new(vec![ds])
        .block(Block::bordered().title(title))
        .x_axis(Axis::default().bounds([0.0, HISTORY as f64]))
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(["0", "50", "100"].map(|l| Span::raw(l).dim())),
        );
    f.render_widget(chart, area);
}

fn draw_tables(f: &mut Frame, area: Rect, app: &App) {
    let s = &app.sample;
    let [left, right] = Layout::new(Direction::Horizontal, [Constraint::Percentage(60), Constraint::Percentage(40)]).areas(area);

    let ctx_rows = s.contexts.iter().enumerate().map(|(ci, c)| {
        let color = CTX_COLORS[ci % CTX_COLORS.len()];
        Row::new(vec![
            Cell::from(format!("■ {}", c.id)).style(Style::new().fg(color)),
            Cell::from(c.pid.to_string()),
            Cell::from(c.comm.clone()),
            Cell::from(format!("{}-{}", c.start_col, c.start_col + c.num_col.saturating_sub(1))),
            Cell::from(if c.busy { "BUSY" } else { "idle" }).style(if c.busy { Style::new().fg(Color::Green) } else { Style::new().dim() }),
            Cell::from(format!("{:.0}", c.rate)),
            Cell::from(c.submissions.saturating_sub(c.completions).to_string()),
            Cell::from(c.completions.to_string()),
            Cell::from(c.errors.to_string()).style(if c.errors > 0 { Style::new().fg(Color::Red) } else { Style::new() }),
        ])
    });
    let ctx = Table::new(
        ctx_rows,
        [
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(16),
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Length(5),
        ],
    )
    .header(Row::new(["ctx", "pid", "comm", "cols", "state", "cmd/s", "infl", "done", "err"]).bold())
    .block(Block::bordered().title(" hardware contexts "));
    f.render_widget(ctx, left);

    let proc_rows = s.processes.iter().map(|p| {
        Row::new(vec![
            p.pid.to_string(),
            p.comm.clone(),
            human(p.total),
            human(p.internal),
            human(p.heap),
        ])
    });
    let procs = Table::new(
        proc_rows,
        [Constraint::Length(8), Constraint::Length(14), Constraint::Length(10), Constraint::Length(10), Constraint::Length(10)],
    )
    .header(Row::new(["pid", "comm", "total", "internal", "heap"]).bold())
    .block(Block::bordered().title(" BO memory per process "));
    f.render_widget(procs, right);
}
