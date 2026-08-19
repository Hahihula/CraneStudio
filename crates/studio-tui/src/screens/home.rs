//! The dashboard (§4.1) — first thing a user sees. Answers "what's running"
//! and "what can I do next" at a glance, with no menu-diving required.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph};

use crate::app::App;
use crate::daemon_client::ChildState;

#[derive(Default)]
pub struct State;

pub fn render(app: &mut App, frame: &mut Frame) {
    let [header, hardware, running, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(
            "CraneStudio",
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )])),
        header,
    );

    frame.render_widget(Paragraph::new(hardware_summary(app)), hardware);

    let items: Vec<ListItem> = if app.last_running.is_empty() {
        vec![ListItem::new(
            "No models running — press [b] to browse and launch one.",
        )]
    } else {
        app.last_running
            .iter()
            .map(|child| {
                let (glyph, color) = match &child.state {
                    ChildState::Healthy => ("\u{25cf} healthy", Color::Green),
                    ChildState::Starting => ("\u{25cb} starting", Color::Yellow),
                    ChildState::Exited { classification, .. } => {
                        return ListItem::new(Line::from(vec![Span::styled(
                            format!("\u{2715} {} — exited ({classification})", child.info.label),
                            Style::new().fg(Color::Red),
                        )]));
                    }
                    ChildState::Unknown => ("? unknown", Color::Gray),
                };
                let port = app.known_ports.get(&child.info.id).copied();
                let port_note = port
                    .map(|p| format!(" — 127.0.0.1:{p}"))
                    .unwrap_or_default();
                ListItem::new(Line::from(vec![
                    Span::styled(glyph, Style::new().fg(color)),
                    Span::raw(format!("  {}{port_note}", child.info.label)),
                ]))
            })
            .collect()
    };
    frame.render_widget(
        List::new(items).block(Block::bordered().title("Running")),
        running,
    );

    let status = app
        .status_line
        .as_deref()
        .unwrap_or("[b] browse models   [d] hardware   [r] connect/chat (if running)   [q] quit");
    frame.render_widget(Paragraph::new(status), footer);
}

fn hardware_summary(app: &App) -> String {
    let hw = &app.hardware;
    if let Some(gpu) = hw.gpus.first() {
        format!(
            "{} — {} free / {} total VRAM",
            gpu.name,
            crate::fmt::bytes(gpu.vram_free),
            crate::fmt::bytes(gpu.vram_total)
        )
    } else {
        format!(
            "CPU only — {} available / {} total RAM",
            crate::fmt::bytes(hw.ram_available),
            crate::fmt::bytes(hw.ram_total)
        )
    }
}
