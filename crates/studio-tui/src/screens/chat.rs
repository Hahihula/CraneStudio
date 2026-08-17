//! The chat playground (§4.6): talk to whatever's running behind the
//! gateway's stable base URL, streamed token-by-token — the same `/v1/*`
//! path any external client would use (§3.2), so this screen is also a
//! live smoke test of the connect instructions on the previous screen.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use futures_util::StreamExt as _;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

use crate::app::{App, BackgroundEvent, Screen};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Default)]
pub struct State {
    pub messages: Vec<(Role, String)>,
    pub input: String,
    pub streaming: bool,
    pub error: Option<String>,
}

impl State {
    pub fn apply_delta(&mut self, role_started: bool, text: &str) {
        if role_started || !matches!(self.messages.last(), Some((Role::Assistant, _))) {
            self.messages.push((Role::Assistant, text.to_string()));
        } else if let Some((_, content)) = self.messages.last_mut() {
            content.push_str(text);
        }
    }

    pub fn finish_turn(&mut self) {
        self.streaming = false;
    }

    pub fn fail_turn(&mut self, err: &str) {
        self.streaming = false;
        self.error = Some(err.to_string());
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => {
            app.screen = Screen::Connect;
            true
        }
        KeyCode::Enter if !app.chat.streaming => {
            submit(app);
            true
        }
        KeyCode::Backspace => {
            app.chat.input.pop();
            true
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.chat.input.push(c);
            true
        }
        _ => false,
    }
}

fn submit(app: &mut App) {
    let text = app.chat.input.trim().to_string();
    if text.is_empty() {
        return;
    }
    app.chat.input.clear();
    app.chat.error = None;
    app.chat.messages.push((Role::User, text));
    app.chat.streaming = true;

    let history: Vec<(&'static str, String)> = app
        .chat
        .messages
        .iter()
        .map(|(role, content)| (if *role == Role::User { "user" } else { "assistant" }, content.clone()))
        .collect();
    let model = app.connect.name.clone();
    let gateway_base = app.gateway_base();
    let tx = app.sender();

    tokio::spawn(async move {
        if let Err(e) = stream_chat(&gateway_base, &model, &history, &tx).await {
            let _ = tx.send(BackgroundEvent::ChatError(e));
        } else {
            let _ = tx.send(BackgroundEvent::ChatDone);
        }
    });
}

async fn stream_chat(
    gateway_base: &str,
    model: &str,
    history: &[(&'static str, String)],
    tx: &tokio::sync::mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), String> {
    let messages: Vec<serde_json::Value> = history.iter().map(|(role, content)| serde_json::json!({"role": role, "content": content})).collect();
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{gateway_base}/v1/chat/completions"))
        .json(&serde_json::json!({"model": model, "messages": messages, "stream": true}))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err(format!("chat request failed: {}", response.text().await.unwrap_or_default()));
    }

    let mut stream = response.bytes_stream();
    let mut buf = String::new();
    let mut role_started = true;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim_end_matches('\r').to_string();
            buf.drain(..=pos);
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data == "[DONE]" {
                return Ok(());
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            if let Some(text) = value["choices"][0]["delta"]["content"].as_str() {
                let _ = tx.send(BackgroundEvent::ChatDelta { role_started, text: text.to_string() });
                role_started = false;
            }
        }
    }
    Ok(())
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let [header, body, input_area, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    frame.render_widget(Paragraph::new(format!("Chat — {}", app.connect.name)), header);

    let mut lines: Vec<Line> = Vec::new();
    for (role, content) in &app.chat.messages {
        let (label, color) = match role {
            Role::User => ("you", Color::Cyan),
            Role::Assistant => ("model", Color::Green),
        };
        lines.push(Line::from(Span::styled(format!("{label}:"), Style::new().fg(color).add_modifier(Modifier::BOLD))));
        for line in content.lines() {
            lines.push(Line::raw(line.to_string()));
        }
        lines.push(Line::raw(""));
    }
    if let Some(err) = &app.chat.error {
        lines.push(Line::from(Span::styled(format!("error: {err}"), Style::new().fg(Color::Red))));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(Block::bordered()), body);

    let input_title = if app.chat.streaming { "generating…" } else { "message" };
    frame.render_widget(Paragraph::new(format!("{}_", app.chat.input)).block(Block::bordered().title(input_title)), input_area);

    // Unlike every other screen, plain characters here are message text, not
    // shortcuts — 'q' would just get typed — so quitting needs Ctrl-C.
    frame.render_widget(Paragraph::new("[Enter] send   [Esc] back to connect   [Ctrl-C] quit"), footer);
}
