//! The model browser (§4.3): curated catalog, a local-filesystem scan
//! (§8.3 needs a UI home somewhere; this is it), and `HuggingFace` search
//! (§8.2). Selecting a supported local file and pressing Enter hands it to
//! the wizard — that's the fastest, most reliable path from "never used
//! this" to a running server, since it needs no download step.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph, Tabs};
use studio_core::catalog::hf::HfCandidate;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::{Catalog, Classification, Source};

use crate::app::App;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Catalog,
    Local,
    Search,
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
            app.screen = crate::app::Screen::Home;
            true
        }
        KeyCode::Tab => {
            app.browser.tab = match app.browser.tab {
                Tab::Catalog => Tab::Local,
                Tab::Local => Tab::Search,
                Tab::Search => Tab::Catalog,
            };
            app.browser.selected = 0;
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
        KeyCode::Char('/') if app.browser.tab == Tab::Search => {
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
                spawn_search(app);
            }
        }
        KeyCode::Backspace => {
            app.browser.search_query.pop();
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => app.browser.search_query.push(c),
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return false,
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
    match app.browser.tab {
        Tab::Local => {
            if let Some(candidate) = app.browser.local.get(app.browser.selected).cloned() {
                if matches!(candidate.classification, Classification::Supported { .. }) {
                    crate::screens::wizard::load_local(app, candidate);
                    app.screen = crate::app::Screen::Wizard;
                } else {
                    app.status_line = Some("that file isn't a Crane-supported architecture".to_string());
                }
            }
        }
        Tab::Search => {
            app.browser.searching = true;
            app.browser.editing_query = true;
        }
        Tab::Catalog => {
            app.status_line = Some(
                "download this with `cranestudio download`, then launch it from the Local tab".to_string(),
            );
        }
    }
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let [tabs_area, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(3), Constraint::Length(1)]).areas(frame.area());

    let titles = ["Catalog", "Local", "Search"];
    let selected_tab = match app.browser.tab {
        Tab::Catalog => 0,
        Tab::Local => 1,
        Tab::Search => 2,
    };
    frame.render_widget(Tabs::new(titles).select(selected_tab).highlight_style(Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)), tabs_area);

    match app.browser.tab {
        Tab::Catalog => render_catalog(app, frame, body),
        Tab::Local => render_local(app, frame, body),
        Tab::Search => render_search(app, frame, body),
    }

    let footer_text = if app.browser.editing_query {
        format!("search: {}_   [Enter] run   [Esc] cancel", app.browser.search_query)
    } else {
        "[Tab] switch tab   [\u{2191}\u{2193}] move   [Enter] select   [/] search   [Esc] back".to_string()
    };
    frame.render_widget(Paragraph::new(footer_text), footer);
}

fn render_catalog(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let Some(catalog) = &app.browser.catalog else {
        frame.render_widget(Paragraph::new("loading catalog…").block(Block::bordered()), area);
        return;
    };
    let items: Vec<ListItem> = catalog
        .models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let variants = model.variants.len();
            styled_item(
                i == app.browser.selected,
                format!("{} — {} ({variants} variant(s))", model.id, model.display_name),
            )
        })
        .collect();
    let title = format!("Catalog ({} models)", catalog.models.len());
    frame.render_widget(List::new(items).block(Block::bordered().title(title)), area);
}

fn render_local(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    if app.browser.local.is_empty() {
        frame.render_widget(Paragraph::new("no local models found (scanning…)").block(Block::bordered()), area);
        return;
    }
    let items: Vec<ListItem> = app
        .browser
        .local
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let (glyph, note) = classification_note(&candidate.classification);
            styled_item(i == app.browser.selected, format!("{glyph} {}{note}", candidate.path.display()))
        })
        .collect();
    frame.render_widget(List::new(items).block(Block::bordered().title("Local models")), area);
}

fn render_search(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    if app.browser.searching {
        frame.render_widget(Paragraph::new("searching…").block(Block::bordered()), area);
        return;
    }
    if app.browser.search_results.is_empty() {
        frame.render_widget(Paragraph::new("press [/] to search HuggingFace").block(Block::bordered()), area);
        return;
    }
    let items: Vec<ListItem> = app
        .browser
        .search_results
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let (glyph, note) = classification_note(&candidate.classification);
            let gated = if candidate.gated { " [gated]" } else { "" };
            styled_item(i == app.browser.selected, format!("{glyph} {}{gated}{note}", candidate.repo_id))
        })
        .collect();
    frame.render_widget(List::new(items).block(Block::bordered().title("HuggingFace search")), area);
}

fn classification_note(classification: &Classification) -> (&'static str, String) {
    match classification {
        Classification::Supported { model_type, .. } => ("\u{25cf}", format!(" — {model_type}")),
        Classification::Unsupported { reason, .. } => ("\u{2715}", format!(" — {reason}")),
        Classification::Unknown { reason } => ("?", format!(" — {reason}")),
    }
}

fn styled_item(selected: bool, text: String) -> ListItem<'static> {
    let style = if selected {
        Style::new().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::new()
    };
    ListItem::new(Line::from(vec![Span::styled(text, style)]))
}
