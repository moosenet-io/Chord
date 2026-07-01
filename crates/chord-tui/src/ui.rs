//! ratatui rendering (S91 CTUI-01 shell + per-mode panes).
//!
//! Layout: a top bar (mode indicator + switch key), a main pane (per current
//! mode/panel), and a status/toast line. Resize is handled by ratatui reflow —
//! we only compute a constraint layout each frame, which never panics on tiny
//! terminals. Stubbed S85 panels render a "pending S85 integration" banner.
//!
//! Secret VALUES are never rendered — only names/status.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use crate::app::{App, Mode, ToastLevel};
use crate::connection::InstanceStatus;
use crate::modes::chord::chord_client::ChordSnapshot;
use crate::modes::chord::serving::PENDING_S85_BANNER;
use crate::modes::chord::ChordPanel;

/// Everything the renderer needs for one frame (read-only borrows).
pub struct Framedata<'a> {
    pub app: &'a App,
    pub instances: &'a [InstanceStatus],
    pub chord: Option<&'a ChordSnapshot>,
}

/// Draw the whole UI for one frame.
pub fn draw(f: &mut Frame, data: &Framedata) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // top bar
            Constraint::Min(3),    // main pane
            Constraint::Length(1), // status/toast
        ])
        .split(area);

    draw_top_bar(f, chunks[0], data.app);
    draw_main(f, chunks[1], data);
    draw_status(f, chunks[2], data.app);

    // Confirm overlay draws on top of the main area if pending.
    if data.app.confirm.is_some() {
        draw_confirm_overlay(f, chunks[1], data.app);
    }
}

fn draw_top_bar(f: &mut Frame, area: Rect, app: &App) {
    let mode = app.mode.label();
    let line = Line::from(vec![
        Span::styled(
            format!(" chord-tui  [{mode}] "),
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  Tab: switch mode  ·  q: quit"),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let (text, color) = match &app.toast {
        Some(t) => {
            let c = match t.level {
                ToastLevel::Info => Color::Green,
                ToastLevel::Warn => Color::Yellow,
                ToastLevel::Error => Color::Red,
            };
            (t.text.clone(), c)
        }
        None => ("ready".to_string(), Color::DarkGray),
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(text, Style::default().fg(color)))),
        area,
    );
}

fn draw_main(f: &mut Frame, area: Rect, data: &Framedata) {
    match data.app.mode {
        Mode::Chord => draw_chord_mode(f, area, data),
        Mode::TerminusFleet => draw_fleet_mode(f, area, data),
    }
}

fn draw_chord_mode(f: &mut Frame, area: Rect, data: &Framedata) {
    let panel = data.app.chord_panel;
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!("Chord · {}", panel.title()));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if data.app.show_add_instance_prompt {
        f.render_widget(
            Paragraph::new("No instances configured. Press 'a' to add a Chord control endpoint."),
            inner,
        );
        return;
    }

    match panel {
        ChordPanel::Models => draw_models(f, inner, data.chord),
        ChordPanel::Backends => draw_backends(f, inner, data.chord),
        // CTUI-03 stubbed panels: render the pending-S85 banner.
        ChordPanel::Serving | ChordPanel::Coordinator | ChordPanel::CleanSwap => {
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        PENDING_S85_BANNER,
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    )),
                    Line::from(""),
                    Line::from("Panels render + navigate; mutations are typed-confirm + INERT until S85."),
                ]),
                inner,
            );
        }
    }
}

fn draw_models(f: &mut Frame, area: Rect, chord: Option<&ChordSnapshot>) {
    use crate::modes::chord::models::display_row;
    let Some(snap) = chord else {
        f.render_widget(Paragraph::new("connecting to Chord…"), area);
        return;
    };
    let rows: Vec<Row> = snap
        .models
        .iter()
        .map(|m| {
            let d = display_row(m);
            Row::new(vec![d.name, d.tier, d.loaded, d.backend, d.size])
        })
        .collect();
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(40),
            Constraint::Percentage(12),
            Constraint::Percentage(14),
            Constraint::Percentage(20),
            Constraint::Percentage(14),
        ],
    )
    .header(Row::new(vec!["model", "tier", "state", "backend", "size"])
        .style(Style::default().add_modifier(Modifier::BOLD)));
    f.render_widget(table, area);
}

fn draw_backends(f: &mut Frame, area: Rect, chord: Option<&ChordSnapshot>) {
    use crate::modes::chord::backends::display_backend;
    let Some(snap) = chord else {
        f.render_widget(Paragraph::new("connecting to Chord…"), area);
        return;
    };
    let rows: Vec<Row> = snap
        .backends
        .iter()
        .map(|b| {
            let d = display_backend(b);
            Row::new(vec![d.name, d.loaded])
        })
        .collect();
    let table = Table::new(rows, [Constraint::Percentage(60), Constraint::Percentage(40)])
        .header(Row::new(vec!["backend", "loaded"]).style(Style::default().add_modifier(Modifier::BOLD)));
    f.render_widget(table, area);
}

fn draw_fleet_mode(f: &mut Frame, area: Rect, data: &Framedata) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Terminus-Fleet · Instances");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = vec![Line::from(
        crate::modes::terminus_fleet::FLEET_SCAFFOLD_NOTE,
    )];
    for s in data.instances {
        let color = match s.health {
            crate::connection::Health::Connected => Color::Green,
            crate::connection::Health::Degraded => Color::Yellow,
            crate::connection::Health::Unreachable => Color::Red,
            crate::connection::Health::Unknown => Color::DarkGray,
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" ● {} ", s.name), Style::default().fg(color)),
            Span::raw(format!("{} ({})", s.base_url, s.health.label())),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_confirm_overlay(f: &mut Frame, area: Rect, app: &App) {
    let Some(cs) = &app.confirm else { return };
    use crate::confirm::Severity;
    let (prompt, hint) = match &cs.mutation.severity {
        Severity::Simple => (
            format!("Confirm: {}", cs.mutation.description),
            "press 'y' to confirm · Esc to cancel".to_string(),
        ),
        Severity::Destructive { challenge } => (
            format!("DESTRUCTIVE: {}", cs.mutation.description),
            format!("type '{challenge}' then Enter · Esc to cancel   [{}]", cs.typed),
        ),
    };
    let stub_note = if cs.mutation.is_stub {
        "  (stub: INERT unless enabled — no real op)"
    } else {
        ""
    };
    let block = Block::default().borders(Borders::ALL).title("Confirm");
    let inner = block.inner(centered(area, 70, 30));
    f.render_widget(ratatui::widgets::Clear, centered(area, 70, 30));
    f.render_widget(block, centered(area, 70, 30));
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(prompt, Style::default().add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from(format!("{hint}{stub_note}")),
        ]),
        inner,
    );
}

/// Compute a centered rect (percent of `area`). Clamped so it never exceeds the
/// parent — safe on tiny terminals.
fn centered(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let w = (area.width.saturating_mul(pct_w) / 100).max(1).min(area.width);
    let h = (area.height.saturating_mul(pct_h) / 100).max(1).min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_rect_never_exceeds_parent_even_when_tiny() {
        let parent = Rect { x: 0, y: 0, width: 1, height: 1 };
        let c = centered(parent, 70, 30);
        assert!(c.width <= parent.width);
        assert!(c.height <= parent.height);
        assert!(c.width >= 1 && c.height >= 1);
    }

    #[test]
    fn centered_rect_is_within_bounds() {
        let parent = Rect { x: 0, y: 0, width: 100, height: 40 };
        let c = centered(parent, 70, 30);
        assert!(c.x + c.width <= parent.x + parent.width);
        assert!(c.y + c.height <= parent.y + parent.height);
    }
}
