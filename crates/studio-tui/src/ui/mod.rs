//! The shared chrome every screen sits inside: a centered canvas with a max
//! width (the terminal is never fully "filled" — content stays in a readable
//! column, the way a well-behaved TUI app does), a title row, and a hint row.
//!
//! Screens don't lay out the terminal themselves; they ask for `shell(...)`
//! and draw into the body `Rect` it returns.

pub mod art;
pub mod bars;
pub mod text;

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};

use crate::theme::{Theme, glyph};

/// Widest the content column ever gets. Beyond this, long list rows and
/// wrapped paragraphs stop being comfortable to read and the eye loses the
/// left edge — the same reason web layouts cap their measure.
pub const MAX_WIDTH: u16 = 104;

/// The centered content column: at most `MAX_WIDTH` wide, with a blank row
/// above and below so nothing is ever glued to the terminal edge.
#[must_use]
pub fn canvas(area: Rect) -> Rect {
    let width = area.width.min(MAX_WIDTH);
    let [row] = Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .areas(area);
    let vertical_pad = u16::from(area.height > 10);
    Rect {
        x: row.x,
        y: row.y + vertical_pad,
        width: row.width,
        height: row.height.saturating_sub(vertical_pad * 2),
    }
}

/// What the shell draws around a screen's body.
pub struct Chrome<'a> {
    /// Breadcrumb after the wordmark, e.g. `["Download", "Qwen3.5 9B"]`.
    pub crumbs: Vec<String>,
    /// Right-hand side of the title row — backend, gateway port, etc.
    pub status: Vec<Span<'static>>,
    /// `(key, what it does)` pairs for the bottom row.
    pub hints: &'a [(&'a str, &'a str)],
    /// Transient message (an error, a "theme: Dracula" ack) shown just above
    /// the hints.
    pub message: Option<Message>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub text: String,
    pub kind: MessageKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    Info,
    Warn,
    Error,
}

impl Message {
    #[must_use]
    pub fn info(text: impl Into<String>) -> Self {
        Message {
            text: text.into(),
            kind: MessageKind::Info,
        }
    }

    #[must_use]
    pub fn error(text: impl Into<String>) -> Self {
        Message {
            text: text.into(),
            kind: MessageKind::Error,
        }
    }

    #[must_use]
    pub fn warn(text: impl Into<String>) -> Self {
        Message {
            text: text.into(),
            kind: MessageKind::Warn,
        }
    }
}

impl<'a> Chrome<'a> {
    #[must_use]
    pub fn new(hints: &'a [(&'a str, &'a str)]) -> Self {
        Chrome {
            crumbs: Vec::new(),
            status: Vec::new(),
            hints,
            message: None,
        }
    }

    #[must_use]
    pub fn crumb(mut self, crumb: impl Into<String>) -> Self {
        self.crumbs.push(crumb.into());
        self
    }

    #[must_use]
    pub fn status(mut self, status: Vec<Span<'static>>) -> Self {
        self.status = status;
        self
    }

    #[must_use]
    pub fn message(mut self, message: Option<Message>) -> Self {
        self.message = message;
        self
    }
}

/// Draws the title row and hint row, returning the body area between them.
pub fn shell(frame: &mut Frame, theme: &Theme, chrome: &Chrome) -> Rect {
    let area = canvas(frame.area());
    let message_height = u16::from(chrome.message.is_some());

    let [title, _, body, _, message, hints] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(1),
        Constraint::Length(message_height),
        Constraint::Length(1),
    ])
    .areas(area);

    frame.render_widget(Paragraph::new(wordmark_line(theme, &chrome.crumbs)), title);
    if !chrome.status.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(chrome.status.clone()).right_aligned()),
            title,
        );
    }

    if let Some(msg) = &chrome.message {
        let color = match msg.kind {
            MessageKind::Info => theme.accent_alt,
            MessageKind::Warn => theme.warning,
            MessageKind::Error => theme.error,
        };
        let glyph = match msg.kind {
            MessageKind::Info => glyph::ARROW,
            MessageKind::Warn => glyph::WARN,
            MessageKind::Error => glyph::FAILED,
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(format!("{glyph} "), Style::new().fg(color)),
                Span::styled(msg.text.clone(), Style::new().fg(color)),
            ])),
            message,
        );
    }

    frame.render_widget(
        Paragraph::new(hint_line_fitted(theme, chrome.hints, hints.width)),
        hints,
    );
    body
}

/// `◆ CraneStudio › Download › Qwen3.5 9B`
fn wordmark_line(theme: &Theme, crumbs: &[String]) -> Line<'static> {
    let mut spans = vec![
        Span::styled("◆ ", Style::new().fg(theme.accent)),
        Span::styled(
            "CraneStudio",
            Style::new().fg(theme.text).add_modifier(Modifier::BOLD),
        ),
    ];
    for crumb in crumbs {
        spans.push(Span::styled(
            format!("  {}  ", glyph::ARROW),
            theme.muted_style(),
        ));
        spans.push(Span::styled(
            crumb.clone(),
            Style::new().fg(theme.accent_alt),
        ));
    }
    Line::from(spans)
}

/// `↑↓ select   ⏎ run   q quit`
#[must_use]
pub fn hint_line(theme: &Theme, hints: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (key, label)) in hints.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("   ", theme.muted_style()));
        }
        spans.extend(theme.hint(key, label));
    }
    Line::from(spans)
}

/// As `hint_line`, but drops whole hints from the end rather than letting the
/// row spill past the canvas — a half-written key name is worse than a missing
/// one, and the important keys come first.
#[must_use]
pub fn hint_line_fitted(theme: &Theme, hints: &[(&str, &str)], width: u16) -> Line<'static> {
    let mut fitted: Vec<(&str, &str)> = Vec::new();
    let mut used = 0usize;
    for (key, label) in hints {
        let cost =
            key.chars().count() + label.chars().count() + 1 + usize::from(!fitted.is_empty()) * 3;
        if used + cost > width as usize {
            break;
        }
        used += cost;
        fitted.push((key, label));
    }
    hint_line(theme, &fitted)
}

/// A small pill: ` CUDA `, ` gated `, ` measured `.
#[must_use]
pub fn badge(theme: &Theme, text: &str, color: ratatui::style::Color) -> Span<'static> {
    Span::styled(
        format!(" {text} "),
        Style::new().fg(color).bg(theme.surface),
    )
}

/// A `label   value` row where labels line up in a fixed gutter — the shape
/// every detail panel in the app uses.
#[must_use]
pub fn field(theme: &Theme, label: &str, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        // Two spaces *after* the padding, so a label longer than the gutter
        // still reads as a label rather than running into its own value.
        Span::styled(format!("{label:<13}  "), theme.muted_style()),
        Span::styled(value.into(), Style::new().fg(theme.text)),
    ])
}

/// A section heading inside a panel: small, muted, letter-spaced.
#[must_use]
pub fn section(theme: &Theme, label: &str) -> Line<'static> {
    Line::from(Span::styled(
        label.to_uppercase(),
        Style::new()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD | Modifier::DIM),
    ))
}

/// One row with content pinned to both edges — `name … 5.4 GiB` — since
/// `Line`'s own alignment applies to the whole line, not to halves of it.
#[must_use]
pub fn split_row(width: u16, left: Vec<Span<'static>>, right: Vec<Span<'static>>) -> Line<'static> {
    let used: usize = left
        .iter()
        .chain(right.iter())
        .map(|s| s.content.chars().count())
        .sum();
    let gap = (width as usize).saturating_sub(used).max(1);
    let mut spans = left;
    spans.push(Span::raw(" ".repeat(gap)));
    spans.extend(right);
    Line::from(spans)
}

/// A centered box of `width`×`height`, clipped to `area` — for modals.
#[must_use]
pub fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [cell] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    cell
}

/// Clears the region a modal is about to draw over, so the screen behind it
/// doesn't bleed through.
pub fn modal(frame: &mut Frame, area: Rect) {
    frame.render_widget(Clear, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_caps_width_and_centers_it() {
        let wide = canvas(Rect::new(0, 0, 200, 50));
        assert_eq!(wide.width, MAX_WIDTH);
        assert_eq!(wide.x, (200 - MAX_WIDTH) / 2);
    }

    #[test]
    fn canvas_uses_every_column_of_a_narrow_terminal() {
        let narrow = canvas(Rect::new(0, 0, 60, 20));
        assert_eq!(narrow.width, 60);
        assert_eq!(narrow.x, 0);
    }

    #[test]
    fn a_short_terminal_keeps_all_its_rows() {
        let short = canvas(Rect::new(0, 0, 80, 8));
        assert_eq!(short.height, 8);
        assert_eq!(short.y, 0);
    }
}
