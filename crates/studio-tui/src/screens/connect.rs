//! The "server running" / connect screen (§4.5): the payoff screen — a
//! stable base URL any OpenAI-compatible client can point at, regardless
//! of which model is actually running behind it (§3.2).

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

use crate::app::App;
use crate::daemon_client::ChildState;

#[derive(Default)]
pub struct State {
    pub id: Option<u64>,
    pub name: String,
    pub port: u16,
    pub gateway_port: u16,
}

impl State {
    pub fn set_active(&mut self, id: u64, name: String, port: u16, gateway_port: u16) {
        self.id = Some(id);
        self.name = name;
        self.port = port;
        self.gateway_port = gateway_port;
    }
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Server running",
            Style::new().fg(Color::Green).add_modifier(Modifier::BOLD),
        ))),
        header,
    );

    let state = app
        .connect
        .id
        .and_then(|id| app.last_running.iter().find(|c| c.info.id == id))
        .map(|c| &c.state);

    let status_line = match state {
        Some(ChildState::Healthy) => "\u{25cf} healthy — ready for requests",
        Some(ChildState::Starting) => "\u{25cb} starting — loading the model onto the GPU…",
        Some(ChildState::Exited { .. }) => "\u{2715} exited — check the daemon log",
        _ => "? status unknown",
    };

    let base_url = format!("http://127.0.0.1:{}/v1", app.connect.gateway_port);
    let lines = vec![
        Line::from(status_line),
        Line::raw(""),
        Line::from(format!("Model: {}", app.connect.name)),
        Line::from(format!("Base URL (never changes when you switch models): {base_url}")),
        Line::from(format!("Model name for API calls: {}", app.connect.name)),
        Line::raw(""),
        Line::from("Point any OpenAI-compatible client (coding agent, curl, SDK) at the base URL above."),
        Line::from("Example: curl http://127.0.0.1:PORT/v1/chat/completions -d '{\"model\":\"NAME\",\"messages\":[...]}'".replace("PORT", &app.connect.gateway_port.to_string()).replace("NAME", &app.connect.name)),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::bordered()),
        body,
    );

    frame.render_widget(
        Paragraph::new("[r] open chat playground   [h] home   [q] quit"),
        footer,
    );
}
