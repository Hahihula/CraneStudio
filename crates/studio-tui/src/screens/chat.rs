//! The chat playground (§4.6): talk to whatever's running behind the
//! gateway's stable base URL, streamed token-by-token — the same `/v1/*`
//! path any external client would use (§3.2), so this screen is also a
//! live smoke test of the connect instructions on the previous screen.
//!
//! Vision models (MiniCPM-V 4.6 et al.) reject *every* request with no
//! image, not just the first (verified live: the requirement is checked
//! per-request, not once per conversation) — so `Ctrl-A` attaches a local
//! image file that stays active and gets resent on every turn until
//! replaced or cleared (Ctrl-A with an empty path), the one real thing a
//! VL model needs that a plain chat box doesn't have. For a plain-text
//! turn with no image attached, a failed request is automatically retried
//! once with a blank placeholder image (`PLACEHOLDER_IMAGE_DATA_URL`) —
//! satisfies the model's requirement without making every text-only
//! conversation start with "attach a picture first."

use std::path::PathBuf;

use base64::Engine as _;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Message,
    ImagePath,
    SystemPrompt,
    MaxTokens,
    Temperature,
}

/// Which field a streamed chunk of text came from — see `apply_delta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKind {
    Content,
    Reasoning,
}

pub struct State {
    pub messages: Vec<(Role, String)>,
    pub input: String,
    pub mode: InputMode,
    /// Set by `Ctrl-A`, resent with *every* subsequent message until
    /// replaced or cleared — vision models need one in every request, not
    /// just the first (see module docs).
    pub active_image: Option<PathBuf>,
    /// Sent as a `"system"` message with every request (§4.6). Editable
    /// with `Ctrl-P`; remembered across sessions via `config.ron` — see
    /// `set_system_prompt`.
    pub system_prompt: String,
    /// Sent as `max_tokens` with every request — crane-serve's own default
    /// (512) is too low for anything beyond a short reply (verified live: a
    /// multi-file code answer got cut off mid sentence). Editable with
    /// `Ctrl-L`; remembered via `config.ron`.
    pub max_tokens: usize,
    /// Sent as `temperature` with every request. Editable with `Ctrl-T`;
    /// remembered via `config.ron`.
    pub temperature: f64,
    pub streaming: bool,
    pub error: Option<String>,
    last_delta_kind: Option<DeltaKind>,
}

impl Default for State {
    fn default() -> Self {
        State::new(
            studio_core::config::DEFAULT_SYSTEM_PROMPT.to_string(),
            studio_core::config::DEFAULT_MAX_TOKENS,
            studio_core::config::DEFAULT_TEMPERATURE,
        )
    }
}

impl State {
    #[must_use]
    pub fn new(system_prompt: String, max_tokens: usize, temperature: f64) -> Self {
        State {
            messages: Vec::new(),
            input: String::new(),
            mode: InputMode::Message,
            active_image: None,
            system_prompt,
            max_tokens,
            temperature,
            streaming: false,
            error: None,
            last_delta_kind: None,
        }
    }

    /// Reasoning models (`MiniCPM5` et al. — verified live) stream their
    /// whole answer through `reasoning_content`, sometimes never touching
    /// `content` at all. Both are shown — hiding reasoning left the chat
    /// window blank for an entire real response — but labelled apart, so
    /// a rambling chain-of-thought doesn't read as a broken/garbled reply.
    pub fn apply_delta(&mut self, role_started: bool, kind: DeltaKind, text: &str) {
        if role_started || !matches!(self.messages.last(), Some((Role::Assistant, _))) {
            let marker = if kind == DeltaKind::Reasoning {
                "(thinking) "
            } else {
                ""
            };
            self.messages
                .push((Role::Assistant, format!("{marker}{text}")));
        } else if let Some((_, content)) = self.messages.last_mut() {
            if self.last_delta_kind != Some(kind) {
                let marker = match kind {
                    DeltaKind::Reasoning => "\n(thinking) ",
                    DeltaKind::Content => "\n(answer) ",
                };
                content.push_str(marker);
            }
            content.push_str(text);
        }
        self.last_delta_kind = Some(kind);
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
            if app.chat.mode == InputMode::Message {
                app.screen = Screen::Connect;
            } else {
                app.chat.mode = InputMode::Message;
                app.chat.input.clear();
            }
            true
        }
        KeyCode::Char('a')
            if key.modifiers.contains(KeyModifiers::CONTROL) && !app.chat.streaming =>
        {
            app.chat.mode = InputMode::ImagePath;
            app.chat.input.clear();
            app.chat.error = None;
            true
        }
        KeyCode::Char('p')
            if key.modifiers.contains(KeyModifiers::CONTROL) && !app.chat.streaming =>
        {
            app.chat.mode = InputMode::SystemPrompt;
            app.chat.input = app.chat.system_prompt.clone();
            app.chat.error = None;
            true
        }
        KeyCode::Char('l')
            if key.modifiers.contains(KeyModifiers::CONTROL) && !app.chat.streaming =>
        {
            app.chat.mode = InputMode::MaxTokens;
            app.chat.input = app.chat.max_tokens.to_string();
            app.chat.error = None;
            true
        }
        KeyCode::Char('t')
            if key.modifiers.contains(KeyModifiers::CONTROL) && !app.chat.streaming =>
        {
            app.chat.mode = InputMode::Temperature;
            app.chat.input = app.chat.temperature.to_string();
            app.chat.error = None;
            true
        }
        KeyCode::Enter if !app.chat.streaming => {
            match app.chat.mode {
                InputMode::Message => submit(app),
                InputMode::ImagePath => attach_image(app),
                InputMode::SystemPrompt => set_system_prompt(app),
                InputMode::MaxTokens => set_max_tokens(app),
                InputMode::Temperature => set_temperature(app),
            }
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

fn attach_image(app: &mut App) {
    let trimmed = app.chat.input.trim();
    if trimmed.is_empty() {
        app.chat.active_image = None;
    } else {
        let path = PathBuf::from(trimmed);
        if path.is_file() {
            app.chat.active_image = Some(path);
            app.chat.error = None;
        } else {
            app.chat.error = Some(format!("{}: not a file", path.display()));
        }
    }
    app.chat.input.clear();
    app.chat.mode = InputMode::Message;
}

/// Loads `config.ron`, applies `mutate`, and saves it back — the shared
/// "remember as last used" tail of every chat-setting editor below.
fn save_config(mutate: impl FnOnce(&mut studio_core::config::Config)) {
    let path = studio_core::paths::config_dir().join("config.ron");
    let mut config = studio_core::config::Config::load(&path);
    mutate(&mut config);
    let _ = config.save(&path);
}

/// Saves the edited system prompt both to live state and to `config.ron`,
/// so it's remembered as "last used" next time the chat screen opens.
fn set_system_prompt(app: &mut App) {
    let text = app.chat.input.trim().to_string();
    app.chat.system_prompt.clone_from(&text);
    app.chat.input.clear();
    app.chat.mode = InputMode::Message;
    save_config(|c| c.system_prompt = text);
}

/// Saves the edited `max_tokens` both to live state and to `config.ron`.
/// An invalid entry is reported and left unchanged, matching `attach_image`'s
/// "report and fall through to Message mode" behaviour.
fn set_max_tokens(app: &mut App) {
    let trimmed = app.chat.input.trim();
    match trimmed.parse::<usize>() {
        Ok(value) if value > 0 => {
            app.chat.max_tokens = value;
            app.chat.error = None;
            save_config(|c| c.max_tokens = value);
        }
        _ => app.chat.error = Some(format!("{trimmed:?}: not a positive integer")),
    }
    app.chat.input.clear();
    app.chat.mode = InputMode::Message;
}

/// Saves the edited `temperature` both to live state and to `config.ron`.
fn set_temperature(app: &mut App) {
    let trimmed = app.chat.input.trim();
    match trimmed.parse::<f64>() {
        Ok(value) if value.is_finite() && value >= 0.0 => {
            app.chat.temperature = value;
            app.chat.error = None;
            save_config(|c| c.temperature = value);
        }
        _ => app.chat.error = Some(format!("{trimmed:?}: not a non-negative number")),
    }
    app.chat.input.clear();
    app.chat.mode = InputMode::Message;
}

/// A 1×1 PNG, used as a placeholder `image_url` when a vision model
/// demands one but the user just wants a plain-text reply — see the
/// module docs' automatic-retry note.
const PLACEHOLDER_IMAGE_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

/// `data:` URL for a local image file — the one thing `image_url` content
/// blocks need, since a local chat playground has nothing to host the file
/// at a real URL for the model to fetch.
fn image_data_url(path: &std::path::Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/png",
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{encoded}"))
}

fn submit(app: &mut App) {
    let text = app.chat.input.trim().to_string();
    if text.is_empty() {
        return;
    }
    app.chat.input.clear();
    app.chat.error = None;

    let image_url = match app
        .chat
        .active_image
        .as_deref()
        .map(image_data_url)
        .transpose()
    {
        Ok(url) => url,
        Err(e) => {
            app.chat.error = Some(e);
            return;
        }
    };

    app.chat.messages.push((Role::User, text.clone()));
    app.chat.streaming = true;

    let system_prompt = app.chat.system_prompt.trim().to_string();
    let mut history: Vec<(&'static str, String, Option<String>)> = Vec::new();
    if !system_prompt.is_empty() {
        history.push(("system", system_prompt, None));
    }
    history.extend(app.chat.messages.iter().map(|(role, content)| {
        (
            if *role == Role::User {
                "user"
            } else {
                "assistant"
            },
            content.clone(),
            None,
        )
    }));
    // The active image is resent with *every* turn while it's set (see
    // `active_image`'s docs) — attaching it only to earlier history would
    // still fail a model that checks every request for one.
    if let Some(last) = history.last_mut() {
        last.2 = image_url;
    }

    let model = app.connect.name.clone();
    let gateway_base = app.gateway_base();
    let tx = app.sender();
    let had_image = history.last().is_some_and(|(_, _, img)| img.is_some());
    let max_tokens = app.chat.max_tokens;
    let temperature = app.chat.temperature;

    tokio::spawn(async move {
        let outcome = match stream_chat(
            &gateway_base,
            &model,
            &history,
            max_tokens,
            temperature,
            &tx,
        )
        .await
        {
            // A vision model rejected a plain-text turn — retry once with
            // a placeholder image instead of bothering the user, unless
            // they'd already attached a real one (then there's nothing a
            // retry would change).
            Err(SendError::NeedsImage) if !had_image => {
                if let Some(last) = history.last_mut() {
                    last.2 = Some(PLACEHOLDER_IMAGE_DATA_URL.to_string());
                }
                stream_chat(
                    &gateway_base,
                    &model,
                    &history,
                    max_tokens,
                    temperature,
                    &tx,
                )
                .await
            }
            other => other,
        };
        match outcome {
            Ok(()) => {
                let _ = tx.send(BackgroundEvent::ChatDone);
            }
            Err(SendError::NeedsImage) => {
                let _ = tx.send(BackgroundEvent::ChatError(
                    "this model requires an image — attach one with Ctrl-A, then send your message"
                        .to_string(),
                ));
            }
            Err(SendError::Other(e)) => {
                let _ = tx.send(BackgroundEvent::ChatError(e));
            }
        }
    });
}

enum SendError {
    /// The backend rejected the request specifically because a vision
    /// model found no `image_url` anywhere in it — distinguished from a
    /// generic failure so the caller can retry with a placeholder.
    NeedsImage,
    Other(String),
}

impl From<String> for SendError {
    fn from(e: String) -> Self {
        SendError::Other(e)
    }
}

async fn stream_chat(
    gateway_base: &str,
    model: &str,
    history: &[(&'static str, String, Option<String>)],
    max_tokens: usize,
    temperature: f64,
    tx: &tokio::sync::mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), SendError> {
    let messages: Vec<serde_json::Value> = history
        .iter()
        .map(|(role, content, image_url)| match image_url {
            Some(url) => serde_json::json!({
                "role": role,
                "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }),
            None => serde_json::json!({"role": role, "content": content}),
        })
        .collect();
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{gateway_base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(if body.contains("No image_url found") {
            SendError::NeedsImage
        } else {
            SendError::Other(format!("chat request failed: {body}"))
        });
    }

    let mut stream = response.bytes_stream();
    let mut buf = String::new();
    let mut role_started = true;
    let mut saw_sse = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim_end_matches('\r').to_string();
            buf.drain(..=pos);
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            saw_sse = true;
            if data == "[DONE]" {
                return Ok(());
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let delta = &value["choices"][0]["delta"];
            let (kind, text) = match delta["content"].as_str() {
                Some(text) => (DeltaKind::Content, Some(text)),
                None => (DeltaKind::Reasoning, delta["reasoning_content"].as_str()),
            };
            if let Some(text) = text {
                let _ = tx.send(BackgroundEvent::ChatDelta {
                    role_started,
                    kind,
                    text: text.to_string(),
                });
                role_started = false;
            }
        }
    }

    // Some backends (crane-serve's vision path, at least today) ignore
    // `"stream": true` outright and send one plain completion object
    // instead of SSE framing — never hit the "data: " branch above at all.
    // Treat that the same as a one-chunk stream rather than silently
    // showing nothing.
    if !saw_sse {
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            let value: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("could not parse response: {e}"))?;
            if let Some(message) = value["error"]["message"].as_str() {
                return Err(if message.contains("No image_url found") {
                    SendError::NeedsImage
                } else {
                    SendError::Other(format!("chat request failed: {message}"))
                });
            }
            let message = &value["choices"][0]["message"];
            let (kind, text) = match message["content"].as_str() {
                Some(text) => (DeltaKind::Content, text),
                None => (
                    DeltaKind::Reasoning,
                    message["reasoning_content"].as_str().unwrap_or(""),
                ),
            };
            let _ = tx.send(BackgroundEvent::ChatDelta {
                role_started: true,
                kind,
                text: text.to_string(),
            });
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

    frame.render_widget(
        Paragraph::new(format!(
            "Chat — {}  (temp {}, max_tokens {})",
            app.connect.name, app.chat.temperature, app.chat.max_tokens
        )),
        header,
    );

    let mut lines: Vec<Line> = Vec::new();
    for (role, content) in &app.chat.messages {
        let (label, color) = match role {
            Role::User => ("you", Color::Cyan),
            Role::Assistant => ("model", Color::Green),
        };
        lines.push(Line::from(Span::styled(
            format!("{label}:"),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        )));
        for line in content.lines() {
            lines.push(Line::raw(line.to_string()));
        }
        lines.push(Line::raw(""));
    }
    if let Some(err) = &app.chat.error {
        lines.push(Line::from(Span::styled(
            format!("error: {err}"),
            Style::new().fg(Color::Red),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::bordered()),
        body,
    );

    let input_title = match (app.chat.streaming, app.chat.mode) {
        (true, _) => "generating…".to_string(),
        (false, InputMode::ImagePath) => "image path".to_string(),
        (false, InputMode::SystemPrompt) => "system prompt (empty clears it)".to_string(),
        (false, InputMode::MaxTokens) => "max_tokens".to_string(),
        (false, InputMode::Temperature) => "temperature".to_string(),
        (false, InputMode::Message) => {
            let image = app
                .chat
                .active_image
                .as_ref()
                .map(|path| format!("image active: {}", path.display()));
            let prompt = (!app.chat.system_prompt.is_empty())
                .then(|| format!("system: {}", app.chat.system_prompt));
            match (image, prompt) {
                (Some(i), Some(p)) => format!("message — {i} — {p}"),
                (Some(i), None) => format!("message — {i}"),
                (None, Some(p)) => format!("message — {p}"),
                (None, None) => "message".to_string(),
            }
        }
    };
    frame.render_widget(
        Paragraph::new(format!("{}_", app.chat.input)).block(Block::bordered().title(input_title)),
        input_area,
    );

    // Unlike every other screen, plain characters here are message text, not
    // shortcuts — 'q' would just get typed — so quitting needs Ctrl-C.
    let footer_text = match app.chat.mode {
        InputMode::ImagePath => "[Enter] attach (empty path clears it)   [Esc] cancel",
        InputMode::SystemPrompt => "[Enter] save (empty clears it)   [Esc] cancel",
        InputMode::MaxTokens | InputMode::Temperature => "[Enter] save   [Esc] cancel",
        InputMode::Message => {
            "[Enter] send   [Ctrl-A] image   [Ctrl-P] system prompt   [Ctrl-L] max_tokens   [Ctrl-T] temperature   [Esc] back   [Ctrl-C] quit"
        }
    };
    frame.render_widget(Paragraph::new(footer_text), footer);
}

#[cfg(test)]
mod tests {
    use super::{DeltaKind, Role, State};

    #[test]
    fn content_only_deltas_are_shown_unlabelled() {
        let mut state = State::default();
        state.apply_delta(true, DeltaKind::Content, "Hello");
        state.apply_delta(false, DeltaKind::Content, ", world");
        assert_eq!(
            state.messages,
            vec![(Role::Assistant, "Hello, world".to_string())]
        );
    }

    #[test]
    fn reasoning_only_deltas_are_labelled_thinking() {
        let mut state = State::default();
        state.apply_delta(true, DeltaKind::Reasoning, "let me consider");
        state.apply_delta(false, DeltaKind::Reasoning, " this further");
        assert_eq!(
            state.messages,
            vec![(
                Role::Assistant,
                "(thinking) let me consider this further".to_string()
            )]
        );
    }

    #[test]
    fn switching_from_reasoning_to_content_inserts_an_answer_marker() {
        let mut state = State::default();
        state.apply_delta(true, DeltaKind::Reasoning, "thinking aloud");
        state.apply_delta(false, DeltaKind::Content, "the real answer");
        assert_eq!(
            state.messages,
            vec![(
                Role::Assistant,
                "(thinking) thinking aloud\n(answer) the real answer".to_string()
            )]
        );
    }

    #[test]
    fn default_state_carries_the_built_in_chat_settings() {
        let state = State::default();
        assert_eq!(
            state.system_prompt,
            studio_core::config::DEFAULT_SYSTEM_PROMPT
        );
        assert_eq!(state.max_tokens, studio_core::config::DEFAULT_MAX_TOKENS);
        assert!(
            (state.temperature - studio_core::config::DEFAULT_TEMPERATURE).abs() < f64::EPSILON
        );
    }

    #[test]
    fn a_new_role_started_flag_starts_a_fresh_message_even_mid_kind() {
        let mut state = State::default();
        state.apply_delta(true, DeltaKind::Content, "first turn");
        state.apply_delta(true, DeltaKind::Content, "second turn");
        assert_eq!(
            state.messages,
            vec![
                (Role::Assistant, "first turn".to_string()),
                (Role::Assistant, "second turn".to_string()),
            ]
        );
    }
}
