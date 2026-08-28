//! Where models come from (§4.3): the curated catalog, a scan of the local
//! filesystem (§8.3), and filtered `HuggingFace` search (§8.2). `Enter` always
//! does the fastest thing that ends with a running server — a local file goes
//! straight to the launch options, a catalog or search entry starts a real
//! download and lands on the launchpad with the model ready to run.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};
use studio_core::catalog::hf::HfCandidate;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::{Catalog, Classification, Source};

use crate::app::App;
use crate::theme::{Theme, glyph};
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[
    ("tab", "catalog / local / search"),
    ("↑↓", "select"),
    ("⏎", "get it"),
    ("/", "search"),
    ("esc", "back"),
];
const SEARCH_HINTS: &[(&str, &str)] = &[("⏎", "search"), ("esc", "cancel")];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Catalog,
    Local,
    Search,
}

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::Catalog => "Catalog",
            Tab::Local => "On disk",
            Tab::Search => "HuggingFace",
        }
    }

    fn next(self) -> Self {
        match self {
            Tab::Catalog => Tab::Local,
            Tab::Local => Tab::Search,
            Tab::Search => Tab::Catalog,
        }
    }

    fn previous(self) -> Self {
        self.next().next()
    }
}

pub struct State {
    pub tab: Tab,
    pub catalog: Option<Catalog>,
    pub catalog_source: Option<Source>,
    pub local: Vec<LocalCandidate>,
    pub search_results: Vec<HfCandidate>,
    pub search_query: String,
    pub editing_query: bool,
    pub searching: bool,
    pub selected: usize,
    list: ListState,
}

impl Default for State {
    fn default() -> Self {
        State {
            tab: Tab::Catalog,
            catalog: None,
            catalog_source: None,
            local: Vec::new(),
            search_results: Vec::new(),
            search_query: String::new(),
            editing_query: false,
            searching: false,
            selected: 0,
            list: ListState::default(),
        }
    }
}

impl State {
    pub fn set_catalog(&mut self, catalog: Catalog, source: Source) {
        self.catalog = Some(catalog);
        self.catalog_source = Some(source);
    }

    pub fn set_local(&mut self, candidates: Vec<LocalCandidate>) {
        self.local = candidates;
    }

    pub fn set_search_results(&mut self, results: Vec<HfCandidate>) {
        self.search_results = results;
        self.searching = false;
        self.selected = 0;
    }

    fn item_count(&self) -> usize {
        match self.tab {
            Tab::Catalog => self.catalog.as_ref().map_or(0, |c| c.models.len()),
            Tab::Local => self.local.len(),
            Tab::Search => self.search_results.len(),
        }
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if app.browser.editing_query {
        return handle_query_input(app, key);
    }

    match key.code {
        KeyCode::Esc => {
            app.screen = crate::app::Screen::Launchpad;
            true
        }
        KeyCode::Tab | KeyCode::Right => {
            app.browser.tab = app.browser.tab.next();
            app.browser.selected = 0;
            app.message = None;
            true
        }
        KeyCode::BackTab | KeyCode::Left => {
            app.browser.tab = app.browser.tab.previous();
            app.browser.selected = 0;
            app.message = None;
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let count = app.browser.item_count();
            if count > 0 {
                app.browser.selected = (app.browser.selected + 1).min(count - 1);
            }
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.browser.selected = app.browser.selected.saturating_sub(1);
            true
        }
        KeyCode::Char('/') => {
            app.browser.tab = Tab::Search;
            app.browser.editing_query = true;
            true
        }
        KeyCode::Enter => {
            select_current(app);
            true
        }
        _ => false,
    }
}

fn handle_query_input(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => app.browser.editing_query = false,
        KeyCode::Enter => {
            app.browser.editing_query = false;
            if !app.browser.search_query.trim().is_empty() {
                app.browser.searching = true;
                spawn_search(app);
            }
        }
        KeyCode::Backspace => {
            app.browser.search_query.pop();
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return false,
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.browser.search_query.push(c);
        }
        _ => {}
    }
    true
}

fn spawn_search(app: &App) {
    let tx = app.sender();
    let query = app.browser.search_query.clone();
    tokio::spawn(async move {
        let client = studio_core::catalog::hf::reqwest::Client::new();
        match studio_core::catalog::hf::search(&client, &query, 20).await {
            Ok(results) => {
                let _ = tx.send(crate::app::BackgroundEvent::SearchDone(results));
            }
            Err(e) => {
                let _ = tx.send(crate::app::BackgroundEvent::SearchFailed(e.to_string()));
            }
        }
    });
}

fn select_current(app: &mut App) {
    app.message = None;
    match app.browser.tab {
        Tab::Local => {
            if let Some(candidate) = app.browser.local.get(app.browser.selected).cloned() {
                if matches!(candidate.classification, Classification::Supported { .. }) {
                    crate::screens::wizard::load_local(app, &candidate);
                    app.screen = crate::app::Screen::Wizard;
                } else {
                    app.message = Some(Message::error(
                        "that file isn't a Crane-supported architecture",
                    ));
                }
            }
        }
        Tab::Search => {
            let Some(candidate) = app
                .browser
                .search_results
                .get(app.browser.selected)
                .cloned()
            else {
                return;
            };
            if !matches!(candidate.classification, Classification::Supported { .. }) {
                app.message = Some(Message::error(
                    "that repo isn't a Crane-supported architecture",
                ));
            } else if let Some(gguf_file) = candidate.gguf_files.first() {
                crate::screens::download::start_hf(app, &candidate.repo_id, gguf_file);
            } else {
                app.message = Some(Message::error(format!(
                    "{}: no .gguf file in this repo to download",
                    candidate.repo_id
                )));
            }
        }
        Tab::Catalog => {
            let Some(catalog) = &app.browser.catalog else {
                return;
            };
            let Some(model) = catalog.models.get(app.browser.selected).cloned() else {
                return;
            };
            crate::screens::download::start_catalog(app, &model);
        }
    }
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let hints = if app.browser.editing_query {
        SEARCH_HINTS
    } else {
        HINTS
    };
    let chrome = Chrome::new(hints)
        .crumb("Get models")
        .status(crate::screens::hardware::status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    let [tabs_area, list_area] =
        Layout::vertical([Constraint::Length(2), Constraint::Min(4)]).areas(body);
    frame.render_widget(Paragraph::new(tabs_line(app)), tabs_area);

    match app.browser.tab {
        Tab::Catalog => render_catalog(app, frame, list_area),
        Tab::Local => render_local(app, frame, list_area),
        Tab::Search => render_search(app, frame, list_area),
    }
}

/// Tabs as pills rather than ratatui's underlined `Tabs` — the selected one
/// needs to be obvious at a glance next to a dense list.
fn tabs_line(app: &App) -> Line<'static> {
    let mut spans = Vec::new();
    for tab in [Tab::Catalog, Tab::Local, Tab::Search] {
        let selected = tab == app.browser.tab;
        spans.push(Span::styled(
            format!(" {} ", tab.label()),
            if selected {
                Style::new()
                    .fg(app.theme.highlight_fg)
                    .bg(app.theme.highlight_bg)
                    .add_modifier(Modifier::BOLD)
            } else {
                app.theme.muted_style()
            },
        ));
        spans.push(Span::raw(" "));
    }
    Line::from(spans)
}

fn render_catalog(app: &mut App, frame: &mut Frame, area: Rect) {
    let Some(catalog) = app.browser.catalog.clone() else {
        waiting(app, frame, area, "fetching the curated catalog");
        return;
    };

    let source = match app.browser.catalog_source {
        Some(Source::Remote) => "live",
        Some(Source::Cached) => "cached",
        Some(Source::Bundled) => "bundled",
        None => "",
    };
    let block = app
        .theme
        .block_focused(format!("Catalog  ·  {} models", catalog.models.len()), true);
    let inner = block.inner(area);
    frame.render_widget(
        block.title_bottom(
            Line::from(Span::styled(format!(" {source} "), app.theme.muted_style()))
                .right_aligned(),
        ),
        area,
    );

    let items: Vec<ListItem> = catalog
        .models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let smallest = model
                .variants
                .iter()
                .map(|v| v.download_bytes)
                .min()
                .unwrap_or(0);
            row(
                &app.theme,
                i == app.browser.selected,
                inner.width,
                Span::styled(glyph::IDLE.to_string(), app.theme.muted_style()),
                &model.display_name,
                vec![Span::styled(
                    format!("from {}", crate::fmt::bytes(smallest)),
                    Style::new().fg(app.theme.text),
                )],
                &catalog_detail(model),
            )
        })
        .collect();
    render_list(app, frame, inner, items);
}

/// What a catalog row says about a model underneath its name. The catalog
/// carries far more than a row can hold, so this is the subset that actually
/// decides "is this the one I want": the family Crane will load it as, how much
/// context it can reach, what it can do, and how many ways it ships.
fn catalog_detail(model: &studio_core::catalog::schema::ModelEntry) -> String {
    use studio_core::catalog::schema::Capability;

    let mut parts = vec![model.model_type.clone(), context_label(model.native_context)];

    let mut abilities = Vec::new();
    if model.capabilities.contains(&Capability::Tools) {
        abilities.push("tools");
    }
    if model.capabilities.contains(&Capability::Vision) {
        abilities.push("vision");
    }
    if !abilities.is_empty() {
        parts.push(abilities.join(" + "));
    }

    // "quants" only reads right for GGUF; unquantized safetensors variants are
    // just variants.
    let quantized = model.variants.iter().all(|v| v.quant.is_some());
    parts.push(match (model.variants.len(), quantized) {
        (1, _) => "1 variant".to_string(),
        (n, true) => format!("{n} quants"),
        (n, false) => format!("{n} variants"),
    });
    parts.push(model.license.clone());
    parts.join("  ·  ")
}

/// `262144` → `256k ctx`, so a row can carry the number that decides whether a
/// model is worth downloading without spending 6 columns on digits.
fn context_label(context: usize) -> String {
    if context >= 1024 && context % 1024 == 0 {
        format!("{}k ctx", context / 1024)
    } else {
        format!("{context} ctx")
    }
}

fn render_local(app: &mut App, frame: &mut Frame, area: Rect) {
    if app.browser.local.is_empty() {
        waiting(app, frame, area, "scanning the models directory");
        return;
    }
    let block = app.theme.block_focused(
        format!("On disk  ·  {} files", app.browser.local.len()),
        true,
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let local = app.browser.local.clone();
    let items: Vec<ListItem> = local
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let (mark, color, note) = classification_note(&app.theme, &candidate.classification);
            let name = candidate.path.file_name().map_or_else(
                || candidate.path.display().to_string(),
                |n| n.to_string_lossy().to_string(),
            );
            row(
                &app.theme,
                i == app.browser.selected,
                inner.width,
                Span::styled(mark.to_string(), Style::new().fg(color)),
                &name,
                Vec::new(),
                &format!(
                    "{}  ·  {}",
                    note.trim_start_matches(' '),
                    crate::ui::text::truncate_start(
                        &candidate.path.display().to_string(),
                        (inner.width as usize).saturating_sub(24)
                    )
                ),
            )
        })
        .collect();
    render_list(app, frame, inner, items);
}

fn render_search(app: &mut App, frame: &mut Frame, area: Rect) {
    let title = if app.browser.search_query.is_empty() {
        "HuggingFace".to_string()
    } else {
        format!("HuggingFace  ·  “{}”", app.browser.search_query)
    };
    let block = app.theme.block_focused(title, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.browser.editing_query {
        #[allow(clippy::cast_possible_truncation)]
        let cursor_x = inner.x + 2 + app.browser.search_query.chars().count() as u16;
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("{} ", glyph::ARROW),
                    Style::new().fg(app.theme.accent),
                ),
                Span::styled(
                    app.browser.search_query.clone(),
                    Style::new().fg(app.theme.text),
                ),
            ])),
            inner,
        );
        frame.set_cursor_position((cursor_x.min(inner.right().saturating_sub(1)), inner.y));
        return;
    }

    if app.browser.searching {
        frame.render_widget(spinner_line(app, "searching HuggingFace"), inner);
        return;
    }
    if app.browser.search_results.is_empty() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "press / to search HuggingFace",
                    Style::new().fg(app.theme.text),
                )),
                Line::from(Span::styled(
                    "results are filtered to architectures Crane can actually run",
                    app.theme.muted_style(),
                )),
            ]),
            inner,
        );
        return;
    }

    let results = app.browser.search_results.clone();
    let items: Vec<ListItem> = results
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let (mark, color, note) = classification_note(&app.theme, &candidate.classification);
            let right = if candidate.gated {
                vec![ui::badge(&app.theme, "gated", app.theme.warning)]
            } else {
                Vec::new()
            };
            row(
                &app.theme,
                i == app.browser.selected,
                inner.width,
                Span::styled(mark.to_string(), Style::new().fg(color)),
                &candidate.repo_id,
                right,
                note.trim_start_matches(' '),
            )
        })
        .collect();
    render_list(app, frame, inner, items);
}

fn render_list(app: &mut App, frame: &mut Frame, area: Rect, items: Vec<ListItem<'static>>) {
    app.browser.list.select(Some(app.browser.selected));
    frame.render_stateful_widget(
        List::new(items).highlight_style(Style::new().bg(app.theme.surface)),
        area,
        &mut app.browser.list,
    );
}

/// Every list in this screen uses the same two-line row as the launchpad, so
/// moving between them doesn't feel like moving between two different apps.
fn row(
    theme: &Theme,
    selected: bool,
    width: u16,
    mark: Span<'static>,
    name: &str,
    right: Vec<Span<'static>>,
    detail: &str,
) -> ListItem<'static> {
    let marker = if selected {
        Span::styled(
            format!("{} ", glyph::BAR_HALF),
            Style::new().fg(theme.accent),
        )
    } else {
        Span::raw("  ")
    };
    let name_style = if selected {
        Style::new().fg(theme.accent).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.text).add_modifier(Modifier::BOLD)
    };
    ListItem::new(vec![
        ui::split_row(
            width,
            vec![
                marker,
                mark,
                Span::raw(" "),
                Span::styled(crate::ui::text::truncate(name, 52), name_style),
            ],
            right,
        ),
        Line::from(vec![
            Span::raw("    "),
            Span::styled(
                crate::ui::text::truncate(detail, (width as usize).saturating_sub(4)),
                theme.muted_style(),
            ),
        ]),
    ])
}

fn waiting(app: &App, frame: &mut Frame, area: Rect, what: &str) {
    let block = app.theme.block("");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(spinner_line(app, what), inner);
}

fn spinner_line(app: &App, what: &str) -> Paragraph<'static> {
    Paragraph::new(Line::from(vec![
        Span::styled(
            format!("{} ", crate::theme::spinner(app.tick)),
            Style::new().fg(app.theme.accent),
        ),
        Span::styled(format!("{what}…"), app.theme.muted_style()),
    ]))
}

fn classification_note(
    theme: &Theme,
    classification: &Classification,
) -> (&'static str, ratatui::style::Color, String) {
    match classification {
        Classification::Supported { model_type, .. } => {
            (glyph::IDLE, theme.success, (*model_type).to_string())
        }
        Classification::Unsupported { reason, .. } => (glyph::FAILED, theme.error, reason.clone()),
        Classification::Unknown { reason } => (glyph::UNKNOWN, theme.muted, reason.clone()),
    }
}
