//! Meters and sparklines, drawn as styled text rather than with `Gauge` —
//! `btop`'s look comes from per-cell coloring across the bar (calm at the
//! left, hot at the right) and from bars that share a row with their label,
//! neither of which a single-style `Gauge` widget can do.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::Theme;

/// Eighth-blocks, for sub-cell precision in sparklines.
const LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// The block that stands for `ratio` (0–1) in a one-row graph.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn level_char(ratio: f64) -> char {
    let top = LEVELS.len() - 1;
    let level = ((ratio.clamp(0.0, 1.0) * top as f64).round() as usize).min(top);
    LEVELS[level]
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64).clamp(0.0, 1.0)
    }
}

/// A gradient bar: cell *i* takes the color that utilization would have at
/// that position, so a bar that's 90% full is visibly red at its tip while a
/// 20% one is entirely calm green.
#[must_use]
pub fn load_bar(theme: &Theme, width: u16, filled_ratio: f64) -> Vec<Span<'static>> {
    bar_spans(theme, width, filled_ratio, None)
}

/// As `load_bar`, but every filled cell takes one flat color — for progress
/// (a download, a launch) where position along the bar carries no meaning.
#[must_use]
pub fn progress_bar(
    theme: &Theme,
    width: u16,
    filled_ratio: f64,
    color: Color,
) -> Vec<Span<'static>> {
    bar_spans(theme, width, filled_ratio, Some(color))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn bar_spans(
    theme: &Theme,
    width: u16,
    filled_ratio: f64,
    flat: Option<Color>,
) -> Vec<Span<'static>> {
    let width = width as usize;
    if width == 0 {
        return Vec::new();
    }
    let filled = (filled_ratio.clamp(0.0, 1.0) * width as f64).round() as usize;
    let mut spans = Vec::with_capacity(width);
    for cell in 0..width {
        if cell < filled {
            let color =
                flat.unwrap_or_else(|| theme.load_color((cell as f64 + 0.5) / width as f64));
            spans.push(Span::styled("█", Style::new().fg(color)));
        } else {
            spans.push(Span::styled(
                "░",
                Style::new().fg(theme.border).add_modifier(Modifier::DIM),
            ));
        }
    }
    spans
}

/// One full meter row: `CPU    ████████░░░░░░░░  42%  ·  8 cores`.
///
/// `bar_width` is the bar itself; the label gutter and the value are drawn
/// around it, so callers pass the width they want the bar to occupy, not the
/// row.
#[must_use]
pub fn meter(
    theme: &Theme,
    label: &str,
    bar_width: u16,
    filled_ratio: f64,
    value: &str,
    note: Option<&str>,
) -> Line<'static> {
    let mut spans = vec![Span::styled(
        format!("{label:<7}"),
        Style::new().fg(theme.text),
    )];
    spans.extend(load_bar(theme, bar_width, filled_ratio));
    spans.push(Span::styled(
        format!("  {value}"),
        Style::new()
            .fg(theme.load_color(filled_ratio))
            .add_modifier(Modifier::BOLD),
    ));
    if let Some(note) = note {
        spans.push(Span::styled(format!("  {note}"), theme.muted_style()));
    }
    Line::from(spans)
}

/// A compact per-core strip: one eighth-block per logical core, colored by
/// that core's own load. Reads as a single texture at a glance — busy cores
/// stand out without needing one row each.
#[must_use]
pub fn core_strip(theme: &Theme, per_core: &[f32]) -> Vec<Span<'static>> {
    per_core
        .iter()
        .map(|&usage| {
            let ratio = f64::from(usage.clamp(0.0, 100.0)) / 100.0;
            Span::styled(
                level_char(ratio).to_string(),
                Style::new().fg(theme.load_color(ratio)),
            )
        })
        .collect()
}

/// A single-row sparkline of `history` (oldest first), right-aligned so the
/// newest sample is always at the same place: the right edge.
#[must_use]
pub fn sparkline(theme: &Theme, history: &[f64], width: u16) -> Vec<Span<'static>> {
    let width = width as usize;
    if width == 0 {
        return Vec::new();
    }
    let start = history.len().saturating_sub(width);
    let recent = &history[start..];
    let mut spans = Vec::with_capacity(width);
    for _ in 0..width.saturating_sub(recent.len()) {
        spans.push(Span::styled(
            " ",
            Style::new().fg(theme.border).add_modifier(Modifier::DIM),
        ));
    }
    for &value in recent {
        let ratio = value.clamp(0.0, 1.0);
        spans.push(Span::styled(
            level_char(ratio).to_string(),
            Style::new().fg(theme.load_color(ratio)),
        ));
    }
    spans
}

#[cfg(test)]
mod tests {
    use studio_core::config::ThemeName;

    use super::*;

    fn theme() -> Theme {
        Theme::from_name(ThemeName::Crane)
    }

    #[test]
    fn a_bar_is_exactly_as_wide_as_asked() {
        assert_eq!(load_bar(&theme(), 20, 0.5).len(), 20);
        assert_eq!(load_bar(&theme(), 0, 0.5).len(), 0);
    }

    #[test]
    fn an_empty_bar_has_no_filled_cells_and_a_full_one_is_all_filled() {
        let empty = load_bar(&theme(), 10, 0.0);
        assert!(empty.iter().all(|s| s.content == "░"));
        let full = load_bar(&theme(), 10, 1.0);
        assert!(full.iter().all(|s| s.content == "█"));
    }

    #[test]
    fn a_sparkline_is_right_aligned_and_padded_to_width() {
        let spans = sparkline(&theme(), &[1.0, 1.0], 5);
        assert_eq!(spans.len(), 5);
        assert_eq!(spans[0].content, " ");
        assert_eq!(spans[4].content, "█");
    }

    #[test]
    fn a_sparkline_longer_than_the_width_keeps_its_newest_samples() {
        let spans = sparkline(&theme(), &[1.0, 1.0, 1.0, 0.0], 2);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].content, "█");
        assert_eq!(spans[1].content, "▁");
    }
}
