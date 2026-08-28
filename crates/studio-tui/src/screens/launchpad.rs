//! The launchpad — what you land on after the splash, and the only screen the
//! common case needs (§4.1). Live hardware on top, everything runnable
//! underneath: models already serving first, then models on disk, then the way
//! to get more. `Enter` does the obvious thing for whatever is selected, so
//! "open the app, pick a model, run it" involves no menus at all.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use crate::app::{App, Screen};
use crate::daemon_client::ChildState;
use crate::models::LocalModel;
use crate::theme::{Theme, glyph};
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[
    ("↑↓", "select"),
    ("⏎", "run"),
    ("c", "configure"),
    ("g", "get models"),
    ("d", "hardware"),
    ("f2", "theme"),
    ("q", "quit"),
];

/// What a row in the list stands for. Rebuilt every frame from the app's own
/// state (running children, scanned models) so the list can never drift out of
/// sync with what it's describing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    /// Index into `App::last_running`.
    Running(usize),
    /// Index into `App::local_models`.
    Local(usize),
    /// The catalog / `HuggingFace` browser.
    Browse,
}

#[derive(Default)]
pub struct State {
    pub selected: usize,
    list: ListState,
}

#[must_use]
pub fn rows(app: &App) -> Vec<Row> {
    // Exited children stay listed rather than silently vanishing — a model that
    // died is the single thing a user most needs told about, and `Enter` on it
    // opens the ready screen where the exit classification is spelled out.
    let mut rows: Vec<Row> = (0..app.last_running.len()).map(Row::Running).collect();
    rows.extend((0..app.local_models.len()).map(Row::Local));
    rows.push(Row::Browse);
    rows
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    let count = rows(app).len();
    match key.code {
        KeyCode::Down | KeyCode::Char('j') => {
            if count > 0 {
                app.launchpad.selected = (app.launchpad.selected + 1).min(count - 1);
            }
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.launchpad.selected = app.launchpad.selected.saturating_sub(1);
            true
        }
        KeyCode::Home => {
            app.launchpad.selected = 0;
            true
        }
        KeyCode::End => {
            app.launchpad.selected = count.saturating_sub(1);
            true
        }
        KeyCode::Enter => {
            activate(app);
            true
        }
        KeyCode::Char('c') => {
            configure(app);
            true
        }
        KeyCode::Char('g' | 'b' | 'm') => {
            app.screen = Screen::Browser;
            app.message = None;
            true
        }
        KeyCode::Char('R') => {
            app.rescan_models();
            app.message = Some(Message::info("rescanning the models directory…"));
            true
        }
        _ => false,
    }
}

/// `Enter`: open a running model's apps, launch a model on disk with the
/// solver's own best answer, or open the browser.
fn activate(app: &mut App) {
    app.message = None;
    match selected_row(app) {
        Some(Row::Running(index)) => {
            if let Some(child) = app.last_running.get(index) {
                let port = app.known_ports.get(&child.info.id).copied().unwrap_or(0);
                let (id, label) = (child.info.id, child.info.label.clone());
                app.ready.set_active(id, label, port, app.gateway_port);
                app.screen = Screen::Ready;
            }
        }
        Some(Row::Local(index)) => {
            if let Some(model) = app.local_models.get(index).cloned() {
                crate::screens::wizard::quick_launch(app, &model);
            }
        }
        Some(Row::Browse) | None => app.screen = Screen::Browser,
    }
}

/// `c`: the same launch, but stopping at the wizard so the solver's
/// alternatives can be inspected and overridden first (§4.4).
fn configure(app: &mut App) {
    app.message = None;
    match selected_row(app) {
        Some(Row::Local(index)) => {
            if let Some(model) = app.local_models.get(index).cloned() {
                crate::screens::wizard::load_local(app, &model.candidate);
                app.screen = Screen::Wizard;
            }
        }
        _ => {
            app.message = Some(Message::info(
                "select a model on disk to open its launch options",
            ));
        }
    }
}

#[must_use]
pub fn selected_row(app: &App) -> Option<Row> {
    rows(app).get(app.launchpad.selected).copied()
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let chrome = Chrome::new(HINTS)
        .status(crate::screens::hardware::status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    let hardware_height = crate::screens::hardware::meter_height(&app.hardware, app.live.as_ref())
        .saturating_add(2)
        .min(body.height.saturating_sub(6));
    let [hardware_area, models_area] =
        Layout::vertical([Constraint::Length(hardware_height), Constraint::Min(4)]).areas(body);

    crate::screens::hardware::panel(app, frame, hardware_area, "Hardware");
    render_models(app, frame, models_area);
}

fn render_models(app: &mut App, frame: &mut Frame, area: Rect) {
    let rows = rows(app);
    let count = rows.len();
    app.launchpad.selected = app.launchpad.selected.min(count.saturating_sub(1));

    let running = app
        .last_running
        .iter()
        .filter(|child| child.state.is_running())
        .count();
    let title = if app.local_scan_done {
        format!("Models  ·  {} on disk", app.local_models.len())
    } else {
        format!("Models  ·  {} scanning…", crate::theme::spinner(app.tick))
    };

    let block = app.theme.block_focused(title, true);
    let inner = block.inner(area);
    let footer = if running > 0 {
        Line::from(vec![Span::styled(
            format!(" {running} serving "),
            Style::new().fg(app.theme.success),
        )])
        .right_aligned()
    } else {
        Line::from(Vec::new())
    };
    frame.render_widget(block.title_bottom(footer), area);

    let items: Vec<ListItem> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| item(app, *row, i == app.launchpad.selected, inner.width))
        .collect();

    // A first run has nothing on disk to list, and a bare "+ Get more models"
    // row on its own doesn't explain itself — say it in words above the list.
    let list_area = if app.local_scan_done && app.local_models.is_empty() {
        let [hint, list] =
            Layout::vertical([Constraint::Length(4), Constraint::Min(2)]).areas(inner);
        frame.render_widget(empty_hint(&app.theme), hint);
        list
    } else {
        inner
    };

    app.launchpad.list.select(Some(app.launchpad.selected));
    frame.render_stateful_widget(
        List::new(items).highlight_style(Style::new().bg(app.theme.surface)),
        list_area,
        &mut app.launchpad.list,
    );
}

fn item(app: &App, row: Row, selected: bool, width: u16) -> ListItem<'static> {
    match row {
        Row::Running(index) => app.last_running.get(index).map_or_else(
            || ListItem::new(Line::raw("")),
            |child| {
                let (glyph_text, color) = match &child.state {
                    ChildState::Healthy => (glyph::HEALTHY, app.theme.success),
                    ChildState::Starting => (glyph::IDLE, app.theme.warning),
                    ChildState::Exited { .. } => (glyph::FAILED, app.theme.error),
                    ChildState::Unknown => (glyph::UNKNOWN, app.theme.muted),
                };
                let note = match &child.state {
                    ChildState::Healthy => "ready — enter to open apps".to_string(),
                    ChildState::Starting => "starting — loading weights".to_string(),
                    ChildState::Exited { classification, .. } => format!("exited ({classification})"),
                    ChildState::Unknown => "status unknown".to_string(),
                };
                let port = app.known_ports.get(&child.info.id).copied();
                two_line(
                    app,
                    selected,
                    width,
                    vec![
                        Span::styled(format!("{glyph_text} "), Style::new().fg(color)),
                        Span::styled(
                            crate::ui::text::truncate(&child.info.label, 52),
                            title_style(&app.theme, selected, true),
                        ),
                    ],
                    vec![Span::styled(
                        port.map_or_else(String::new, |p| format!("127.0.0.1:{p}")),
                        app.theme.muted_style(),
                    )],
                    vec![Span::styled(note, app.theme.muted_style())],
                )
            },
        ),
        Row::Local(index) => app.local_models.get(index).map_or_else(
            || ListItem::new(Line::raw("")),
            |model| local_item(app, model, selected, width),
        ),
        Row::Browse => two_line(
            app,
            selected,
            width,
            vec![
                Span::styled("+ ", Style::new().fg(app.theme.accent_alt)),
                Span::styled(
                    "Get more models",
                    title_style(&app.theme, selected, true),
                ),
            ],
            Vec::new(),
            vec![Span::styled(
                "curated catalog · HuggingFace search · resumable downloads",
                app.theme.muted_style(),
            )],
        ),
    }
}

fn local_item(app: &App, model: &LocalModel, selected: bool, width: u16) -> ListItem<'static> {
    let mut detail: Vec<String> = Vec::new();
    if let Some(model_type) = &model.model_type {
        detail.push(model_type.clone());
    }
    detail.push(model.format_label().to_string());
    if let Some(quant) = &model.quant {
        detail.push(quant.clone());
    }
    if let Some(repo) = &model.repo {
        detail.push(repo.clone());
    }
    if let Some(reason) = &model.reason {
        detail.push(reason.clone());
    }

    let name_color = if model.supported {
        title_style(&app.theme, selected, true)
    } else {
        Style::new().fg(app.theme.muted)
    };
    let mark = if model.supported {
        Span::styled(format!("{} ", glyph::IDLE), app.theme.muted_style())
    } else {
        Span::styled(
            format!("{} ", glyph::FAILED),
            Style::new().fg(app.theme.error),
        )
    };

    two_line(
        app,
        selected,
        width,
        vec![mark, Span::styled(crate::ui::text::truncate(&model.name, 52), name_color)],
        vec![Span::styled(
            crate::fmt::bytes(model.size),
            Style::new().fg(if model.supported {
                app.theme.text
            } else {
                app.theme.muted
            }),
        )],
        vec![Span::styled(
            crate::ui::text::truncate(&detail.join("  ·  "), width as usize),
            app.theme.muted_style(),
        )],
    )
}

/// Every row is the same two lines — a title row pinned to both edges, and a
/// muted detail row indented under it — so the list reads as a column of cards
/// rather than a table the eye has to parse.
fn two_line(
    app: &App,
    selected: bool,
    width: u16,
    mut title: Vec<Span<'static>>,
    right: Vec<Span<'static>>,
    detail: Vec<Span<'static>>,
) -> ListItem<'static> {
    let marker = if selected {
        Span::styled(
            format!("{} ", glyph::BAR_HALF),
            Style::new().fg(app.theme.accent),
        )
    } else {
        Span::raw("  ")
    };
    title.insert(0, marker);
    let mut detail_spans = vec![Span::raw("    ")];
    detail_spans.extend(detail);
    ListItem::new(vec![
        ui::split_row(width, title, right),
        Line::from(detail_spans),
    ])
}

fn title_style(theme: &Theme, selected: bool, strong: bool) -> Style {
    let mut style = Style::new().fg(if selected { theme.accent } else { theme.text });
    if strong {
        style = style.add_modifier(Modifier::BOLD);
    }
    style
}

/// Shown in place of the list when the scan found nothing — the one moment a
/// first-run user has no obvious next step unless we spell it out.
#[must_use]
pub fn empty_hint(theme: &Theme) -> Paragraph<'static> {
    Paragraph::new(vec![
        Line::raw(""),
        Line::from(Span::styled(
            "No models on disk yet.",
            Style::new().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Pick “Get more models” below to download one — the catalog knows which ones fit this machine.",
            theme.muted_style(),
        )),
    ])
}
