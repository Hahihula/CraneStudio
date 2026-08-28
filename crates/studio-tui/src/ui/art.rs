//! The splash art. Kept as plain `&str` rows so the shapes stay readable in
//! source, and colored on the way out — the wordmark takes a vertical accent
//! gradient, the crane a single calm hue.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::Theme;

/// A crane in flight: neck and beak reaching forward on the left, wings on the
/// upstroke, long legs trailing behind. Block shading rather than an outline —
/// at this size a silhouette reads as a bird and an outline reads as noise.
const CRANE: [&str; 4] = [
    "     ▀▚▄▄▖                      ▗▄▄▞▀     ",
    "          ▀▀▚▄▄▖          ▗▄▄▞▀▀          ",
    "▂▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▟█▙▄▞                   ",
    "                       ▀▀▚▄▄▄▄▄▄▂         ",
];

/// `CRANE`, ANSI-shadow style — 42 columns.
const CRANE_WORD: [&str; 6] = [
    " ██████╗██████╗  █████╗ ███╗   ██╗███████╗",
    "██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝",
    "██║     ██████╔╝███████║██╔██╗ ██║█████╗  ",
    "██║     ██╔══██╗██╔══██║██║╚██╗██║██╔══╝  ",
    "╚██████╗██║  ██║██║  ██║██║ ╚████║███████╗",
    " ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝",
];

/// `STUDIO`, same font — 46 columns.
const STUDIO_WORD: [&str; 6] = [
    "███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗ ",
    "██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗",
    "███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║",
    "╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║",
    "███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝",
    "╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝ ",
];

/// Widest row of the big wordmark, in columns.
pub const WORDMARK_WIDTH: u16 = 46;
/// Rows the big wordmark plus its gap occupy.
pub const WORDMARK_HEIGHT: u16 = 12;
/// Rows the crane occupies.
#[allow(clippy::cast_possible_truncation)]
pub const CRANE_HEIGHT: u16 = CRANE.len() as u16;

/// The stacked `CRANE` / `STUDIO` wordmark, gradient-shaded from the accent
/// down to its cooler sibling.
#[must_use]
pub fn wordmark(theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::with_capacity(WORDMARK_HEIGHT as usize);
    for row in CRANE_WORD {
        lines.push(Line::from(Span::styled(
            row.to_string(),
            Style::new().fg(theme.accent),
        )));
    }
    for row in STUDIO_WORD {
        lines.push(Line::from(Span::styled(
            row.to_string(),
            Style::new().fg(theme.accent_alt),
        )));
    }
    lines
}

/// A one-row wordmark for terminals too small for the big one.
#[must_use]
pub fn wordmark_small(theme: &Theme) -> Vec<Line<'static>> {
    vec![Line::from(vec![
        Span::styled(
            "CRANE",
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "STUDIO",
            Style::new()
                .fg(theme.accent_alt)
                .add_modifier(Modifier::BOLD),
        ),
    ])]
}

#[must_use]
pub fn crane(theme: &Theme) -> Vec<Line<'static>> {
    CRANE
        .iter()
        .map(|row| {
            Line::from(Span::styled(
                (*row).to_string(),
                Style::new().fg(theme.accent),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wordmark_row_is_the_same_width() {
        for row in CRANE_WORD {
            assert_eq!(row.chars().count(), 42, "{row}");
        }
        for row in STUDIO_WORD {
            assert_eq!(row.chars().count(), WORDMARK_WIDTH as usize, "{row}");
        }
    }

    #[test]
    fn every_crane_row_is_the_same_width() {
        for row in CRANE {
            assert_eq!(row.chars().count(), 42, "{row}");
        }
    }

    #[test]
    fn the_declared_heights_match_the_art() {
        assert_eq!(
            WORDMARK_HEIGHT as usize,
            CRANE_WORD.len() + STUDIO_WORD.len()
        );
        assert_eq!(CRANE_HEIGHT as usize, CRANE.len());
    }
}
