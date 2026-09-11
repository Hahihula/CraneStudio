//! The payoff screen (§4.5) *and* the studio's app launcher. A model finishes
//! loading here, and from here you pick what to do with it.
//!
//! Chat is only the first app — the gateway's stable `/v1` base URL is what
//! every future app (bench, vision, agent runner) will talk to as well, so the
//! list is built from an `App` table rather than hard-wiring a single "open
//! chat" key.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, Paragraph};

use crate::app::{App as TuiApp, Screen};
use crate::daemon_client::ChildState;
use crate::theme::{Theme, glyph};
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[
    ("↑↓", "select app"),
    ("⏎", "open"),
    ("l", "logs"),
    ("p", "public"),
    ("esc", "models"),
    ("f2", "theme"),
    ("q", "quit"),
];

/// How many of the most recent captured stderr lines to show — the log
/// panel is a fixed-height strip on this screen (like `Endpoint`), not a
/// scrollable pager, so only the tail end (where a crash's actual error
/// usually is) needs to fit.
const LOG_PANEL_LINES: usize = 12;

/// One entry in the app launcher. `available: false` entries are deliberately
/// visible: they're the roadmap, and hiding them would make the launcher look
/// like it only ever had one thing in it.
struct StudioApp {
    key: &'static str,
    name: &'static str,
    blurb: &'static str,
    available: bool,
}

const CHAT_APPS: [StudioApp; 4] = [
    StudioApp {
        key: "chat",
        name: "Chat",
        blurb: "streaming conversation, images, system prompt",
        available: true,
    },
    StudioApp {
        key: "endpoint",
        name: "Endpoint",
        blurb: "connect an external client to this model",
        available: true,
    },
    StudioApp {
        key: "bench",
        name: "Benchmark",
        blurb: "prompt and decode throughput at your context",
        available: false,
    },
    StudioApp {
        key: "agent",
        name: "Agent tools",
        blurb: "exercise tool calling against the running model",
        available: false,
    },
];

/// Ready-screen apps for a speech model (no Chat).
const TTS_APPS: [StudioApp; 2] = [
    StudioApp {
        key: "tts",
        name: "TTS Playground",
        blurb: "type text, generate speech, play it back",
        available: true,
    },
    StudioApp {
        key: "endpoint",
        name: "Endpoint",
        blurb: "connect an external client to this model",
        available: true,
    },
];

fn apps(is_tts: bool) -> &'static [StudioApp] {
    if is_tts { &TTS_APPS } else { &CHAT_APPS }
}

#[derive(Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct State {
    pub id: Option<u64>,
    pub name: String,
    pub port: u16,
    pub gateway_port: u16,
    /// Context the model was launched with, for the summary card.
    pub context: Option<usize>,
    /// True between "launch requested" and the daemon reporting an id.
    pub starting: bool,
    /// Speech model; selects the app list and endpoint example.
    pub is_tts: bool,
    pub selected: usize,
    /// The endpoint app is a panel on this screen rather than a screen of its
    /// own — it's four lines of copy-paste, not a place to navigate to.
    pub show_endpoint: bool,
    /// §4.5: "Log tail pane, toggleable" — the captured stderr tail for this
    /// child, most useful for seeing *why* a model crashed beyond the
    /// one-line exit classification. `None` while a fetch is in flight or
    /// none has been requested yet; `Some(Err(_))` if the control API call
    /// failed (e.g. daemon restarted and forgot this child).
    pub show_logs: bool,
    pub log_tail: Option<Result<String, String>>,
}

impl State {
    /// Called the moment a launch is requested, so the screen can show what's
    /// starting before the daemon has assigned it an id.
    pub fn begin(&mut self, name: String, port: u16, gateway_port: u16, is_tts: bool) {
        self.id = None;
        self.name = name;
        self.port = port;
        self.gateway_port = gateway_port;
        self.starting = true;
        self.is_tts = is_tts;
        self.context = None;
        self.selected = 0;
        self.show_logs = false;
        self.log_tail = None;
    }

    pub fn set_active(
        &mut self,
        id: u64,
        name: String,
        port: u16,
        gateway_port: u16,
        is_tts: bool,
    ) {
        self.id = Some(id);
        self.name = name;
        self.port = port;
        self.gateway_port = gateway_port;
        self.starting = false;
        self.is_tts = is_tts;
        self.selected = self.selected.min(apps(is_tts).len() - 1);
    }
}

pub fn handle_key(app: &mut TuiApp, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => {
            app.screen = Screen::Launchpad;
            app.message = None;
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.ready.selected = (app.ready.selected + 1).min(apps(app.ready.is_tts).len() - 1);
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.ready.selected = app.ready.selected.saturating_sub(1);
            true
        }
        KeyCode::Enter => {
            open_selected(app);
            true
        }
        KeyCode::Char('l') => {
            toggle_logs(app);
            true
        }
        KeyCode::Char('p') => {
            toggle_public(app);
            true
        }
        _ => false,
    }
}

/// PLAN.md §10.1's "explicit opt-in" for LAN exposure — this is gateway-wide
/// (every model multiplexed through it, not just this one), toggled live via
/// `/control/gateway/public` with no restart and no other model disturbed.
fn toggle_public(app: &mut TuiApp) {
    let currently_public = app.gateway_status.as_ref().is_some_and(|s| s.public);
    app.message = Some(Message::info(if currently_public {
        "disabling public access…".to_string()
    } else {
        "enabling public access…".to_string()
    }));
    app.spawn_gateway_public_toggle(!currently_public);
}

fn toggle_logs(app: &mut TuiApp) {
    app.ready.show_logs = !app.ready.show_logs;
    if !app.ready.show_logs {
        return;
    }
    let Some(id) = app.ready.id else {
        app.ready.log_tail = Some(Err("no child to fetch logs for yet".to_string()));
        return;
    };
    app.ready.log_tail = None;
    app.spawn_log_tail_fetch(id);
}

fn open_selected(app: &mut TuiApp) {
    let Some(entry) = apps(app.ready.is_tts).get(app.ready.selected) else {
        return;
    };
    if !entry.available {
        app.message = Some(Message::info(format!("{} isn't built yet", entry.name)));
        return;
    }
    app.message = None;
    match entry.key {
        "chat" => app.screen = Screen::Chat,
        "tts" => app.screen = Screen::TtsPlayground,
        "endpoint" => app.ready.show_endpoint = !app.ready.show_endpoint,
        _ => {}
    }
}

pub fn render(app: &mut TuiApp, frame: &mut Frame) {
    let chrome = Chrome::new(HINTS)
        .crumb(crate::ui::text::truncate(&app.ready.name, 28))
        .status(crate::screens::hardware::status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    // Every panel is exactly as tall as what's in it, and the leftover space is
    // left blank — a half-empty bordered box reads as something missing.
    let endpoint_height = if app.ready.show_endpoint { 8 } else { 0 };
    let is_public = app.gateway_status.as_ref().is_some_and(|s| s.public);
    let public_height = if is_public { 8 } else { 0 };
    let logs_height = if app.ready.show_logs {
        u16::try_from(LOG_PANEL_LINES).unwrap_or(u16::MAX) + 2
    } else {
        0
    };
    #[allow(clippy::cast_possible_truncation)]
    let apps_height = (apps(app.ready.is_tts).len() * 2) as u16 + 2;
    let [status_area, endpoint_area, public_area, logs_area, apps_area, _] = Layout::vertical([
        Constraint::Length(7),
        Constraint::Length(endpoint_height),
        Constraint::Length(public_height),
        Constraint::Length(logs_height),
        Constraint::Length(apps_height),
        Constraint::Min(0),
    ])
    .areas(body);

    render_status(app, frame, status_area);
    if app.ready.show_endpoint {
        render_endpoint(app, frame, endpoint_area);
    }
    if is_public {
        render_public(app, frame, public_area);
    }
    if app.ready.show_logs {
        render_logs(app, frame, logs_area);
    }
    render_apps(app, frame, apps_area);
}

fn render_status(app: &TuiApp, frame: &mut Frame, area: Rect) {
    let state = app
        .ready
        .id
        .and_then(|id| app.last_running.iter().find(|c| c.info.id == id))
        .map(|c| &c.state);

    let starting = (
        glyph::IDLE,
        format!("{} loading weights", crate::theme::spinner(app.tick)),
        app.theme.warning,
    );
    let unknown = (
        glyph::UNKNOWN,
        "status unknown".to_string(),
        app.theme.muted,
    );
    let (glyph_text, label, color) = match state {
        Some(ChildState::Healthy) => (glyph::HEALTHY, "ready".to_string(), app.theme.success),
        Some(ChildState::Starting) => starting,
        Some(ChildState::Exited { classification, .. }) => (
            glyph::FAILED,
            format!("exited — {classification}"),
            app.theme.error,
        ),
        // No child to look up yet: the launch was accepted moments ago and the
        // first status refresh hasn't come back.
        None if app.ready.starting => starting,
        Some(ChildState::Unknown) | None => unknown,
    };

    let block = app.theme.block("Model");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!("{glyph_text} "), Style::new().fg(color)),
            Span::styled(
                crate::ui::text::truncate(&app.ready.name, 48),
                Style::new().fg(app.theme.text).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("   {label}"), Style::new().fg(color)),
        ]),
        Line::raw(""),
        ui::field(
            &app.theme,
            "base url",
            format!("http://127.0.0.1:{}/v1", app.ready.gateway_port),
        ),
        ui::field(&app.theme, "model name", app.ready.name.clone()),
    ];
    if let Some(context) = app.ready.context {
        lines.push(ui::field(&app.theme, "context", context.to_string()));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The base URL never changes when the model behind it does (§3.2) — that's the
/// whole point of the gateway, so it's what gets shown, not the child's port.
fn render_endpoint(app: &TuiApp, frame: &mut Frame, area: Rect) {
    let block = app.theme.block("Endpoint");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let base = format!("http://127.0.0.1:{}/v1", app.ready.gateway_port);
    let example = if app.ready.is_tts {
        format!(
            "curl {base}/audio/speech -d '{{\"model\":\"{}\",\"input\":\"hello\"}}' --output speech.wav",
            app.ready.name
        )
    } else {
        format!(
            "curl {base}/chat/completions -d '{{\"model\":\"{}\",\"messages\":[…]}}'",
            app.ready.name
        )
    };
    let lines = vec![
        Line::from(Span::styled(
            "point any OpenAI-compatible client at this — it survives switching models",
            app.theme.muted_style(),
        )),
        Line::raw(""),
        ui::field(&app.theme, "OPENAI_BASE_URL", base.clone()),
        ui::field(&app.theme, "OPENAI_API_KEY", "not required"),
        Line::raw(""),
        Line::from(Span::styled(
            crate::ui::text::truncate(&example, inner.width as usize),
            Style::new().fg(app.theme.accent_alt),
        )),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// PLAN.md §10.1: shown only while the gateway is actually LAN-exposed —
/// this is gateway-wide (every model behind the same base URL, not just the
/// one on screen), toggled with `p` and requiring no daemon restart.
fn render_public(app: &TuiApp, frame: &mut Frame, area: Rect) {
    let block = app.theme.block("Public Access");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(status) = &app.gateway_status else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled("checking…", app.theme.muted_style()))),
            inner,
        );
        return;
    };

    let host = app
        .local_ip
        .map_or_else(|| status.host.clone(), |ip| ip.to_string());
    let base = format!("http://{host}:{}/v1", status.port);
    let token = status.auth_token.as_deref().unwrap_or("");

    let lines = vec![
        Line::from(Span::styled(
            "every model behind this gateway is now reachable from your network — anyone with the token below can use your GPU and read your prompts",
            Style::new().fg(app.theme.warning),
        )),
        Line::raw(""),
        ui::field(&app.theme, "OPENAI_BASE_URL", base),
        ui::field(&app.theme, "OPENAI_API_KEY", token.to_string()),
        Line::raw(""),
        Line::from(Span::styled(
            "press p again to stop sharing",
            app.theme.muted_style(),
        )),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// §4.5's "log tail pane" — the captured stderr tail for this child, most
/// useful right after `Model` above reads "exited": the one-line
/// classification says *what kind* of crash, this shows the actual output
/// that produced it (OOM messages, stack traces, config errors, ...).
fn render_logs(app: &TuiApp, frame: &mut Frame, area: Rect) {
    let block = app.theme.block("Logs");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines: Vec<Line> = match &app.ready.log_tail {
        None => vec![Line::from(Span::styled(
            "fetching…",
            app.theme.muted_style(),
        ))],
        Some(Err(err)) => vec![Line::from(Span::styled(
            format!("couldn't fetch logs: {err}"),
            Style::new().fg(app.theme.error),
        ))],
        Some(Ok(tail)) if tail.is_empty() => vec![Line::from(Span::styled(
            "(no output captured yet)",
            app.theme.muted_style(),
        ))],
        Some(Ok(tail)) => tail
            .lines()
            .rev()
            .take(LOG_PANEL_LINES)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|line| {
                Line::from(Span::styled(
                    crate::ui::text::truncate(line, inner.width as usize),
                    Style::new().fg(app.theme.text),
                ))
            })
            .collect(),
    };
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_apps(app: &TuiApp, frame: &mut Frame, area: Rect) {
    let block = app.theme.block_focused("Apps", true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items: Vec<ListItem> = apps(app.ready.is_tts)
        .iter()
        .enumerate()
        .map(|(i, entry)| app_item(&app.theme, entry, i == app.ready.selected, inner.width))
        .collect();
    frame.render_widget(List::new(items), inner);
}

fn app_item(theme: &Theme, entry: &StudioApp, selected: bool, width: u16) -> ListItem<'static> {
    let marker = if selected {
        Span::styled(
            format!("{} ", glyph::BAR_HALF),
            Style::new().fg(theme.accent),
        )
    } else {
        Span::raw("  ")
    };
    let name_style = match (selected, entry.available) {
        (_, false) => Style::new().fg(theme.muted),
        (true, true) => Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        (false, true) => Style::new().fg(theme.text).add_modifier(Modifier::BOLD),
    };
    let right = if entry.available {
        Vec::new()
    } else {
        vec![ui::badge(theme, "soon", theme.muted)]
    };

    ListItem::new(vec![
        ui::split_row(
            width,
            vec![marker, Span::styled(entry.name, name_style)],
            right,
        ),
        Line::from(vec![
            Span::raw("    "),
            Span::styled(entry.blurb, theme.muted_style()),
        ]),
    ])
    .style(if selected {
        Style::new().bg(theme.surface)
    } else {
        Style::new()
    })
}
