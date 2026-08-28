//! One palette, shared by every screen and cycled with `F2` (see `app.rs`).
//! Screens never name a raw `Color` — they ask the theme for a *semantic*
//! one, so a new palette is a single entry in `Theme::from_name` rather than
//! a hunt through every widget.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, BorderType, Padding};
use studio_core::config::ThemeName;

/// Animation frames for a braille spinner, shown during any async wait
/// (downloading, launching, generating…) that has no real percentage to
/// report. Advanced once per `on_tick` (250ms).
const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

#[must_use]
pub fn spinner(tick: u64) -> char {
    #[allow(clippy::cast_possible_truncation)]
    let index = (tick % SPINNER_FRAMES.len() as u64) as usize;
    SPINNER_FRAMES[index]
}

/// Glyphs used for state, kept in one place so "healthy" looks identical on
/// the launchpad, the ready screen and the running list.
pub mod glyph {
    pub const HEALTHY: &str = "●";
    pub const IDLE: &str = "○";
    pub const FAILED: &str = "✕";
    pub const UNKNOWN: &str = "◌";
    pub const DONE: &str = "✓";
    pub const WARN: &str = "▲";
    pub const ARROW: &str = "›";
    pub const BAR_FULL: &str = "█";
    pub const BAR_HALF: &str = "▌";
    pub const BAR_EMPTY: &str = "░";
}

type Rgb = (u8, u8, u8);

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    /// Ordinary body text.
    pub text: Color,
    /// Brand color: headers, panel titles, selected tabs.
    pub accent: Color,
    /// Second brand hue, for gradients and secondary emphasis.
    pub accent_alt: Color,
    /// Fits / healthy / done / measured-good-news.
    pub success: Color,
    /// Starting / short-of-target / in-progress.
    pub warning: Color,
    /// Unusable / exited / failed.
    pub error: Color,
    /// Footers, secondary text, hints.
    pub muted: Color,
    pub border: Color,
    /// Border of the panel that currently has focus.
    pub border_focus: Color,
    pub highlight_fg: Color,
    pub highlight_bg: Color,
    /// Backdrop for cards and selected rows — a hair off the terminal
    /// background, never a full-contrast block.
    pub surface: Color,
    /// Low → mid → high stops for utilization meters (`load_color`).
    load: [Rgb; 3],
}

impl Theme {
    #[must_use]
    pub fn from_name(name: ThemeName) -> Self {
        match name {
            // Warm origami gold against cool slate — CraneStudio's own look.
            ThemeName::Crane => Theme {
                text: Color::Rgb(226, 232, 240),
                accent: Color::Rgb(240, 180, 90),
                accent_alt: Color::Rgb(94, 200, 199),
                success: Color::Rgb(126, 213, 148),
                warning: Color::Rgb(233, 184, 114),
                error: Color::Rgb(240, 113, 113),
                muted: Color::Rgb(122, 134, 154),
                border: Color::Rgb(58, 66, 82),
                border_focus: Color::Rgb(240, 180, 90),
                highlight_fg: Color::Rgb(20, 24, 32),
                highlight_bg: Color::Rgb(240, 180, 90),
                surface: Color::Rgb(32, 38, 50),
                load: [(126, 213, 148), (233, 184, 114), (240, 113, 113)],
            },
            ThemeName::Monokai => Theme {
                text: Color::Rgb(248, 248, 242),
                accent: Color::Rgb(102, 217, 239),
                accent_alt: Color::Rgb(174, 129, 255),
                success: Color::Rgb(166, 226, 46),
                warning: Color::Rgb(230, 219, 116),
                error: Color::Rgb(249, 38, 114),
                muted: Color::Rgb(117, 113, 94),
                border: Color::Rgb(73, 72, 62),
                border_focus: Color::Rgb(102, 217, 239),
                highlight_fg: Color::Rgb(39, 40, 34),
                highlight_bg: Color::Rgb(102, 217, 239),
                surface: Color::Rgb(49, 50, 44),
                load: [(166, 226, 46), (230, 219, 116), (249, 38, 114)],
            },
            ThemeName::Dracula => Theme {
                text: Color::Rgb(248, 248, 242),
                accent: Color::Rgb(189, 147, 249),
                accent_alt: Color::Rgb(139, 233, 253),
                success: Color::Rgb(80, 250, 123),
                warning: Color::Rgb(255, 184, 108),
                error: Color::Rgb(255, 85, 85),
                muted: Color::Rgb(98, 114, 164),
                border: Color::Rgb(68, 71, 90),
                border_focus: Color::Rgb(189, 147, 249),
                highlight_fg: Color::Rgb(40, 42, 54),
                highlight_bg: Color::Rgb(189, 147, 249),
                surface: Color::Rgb(52, 55, 70),
                load: [(80, 250, 123), (255, 184, 108), (255, 85, 85)],
            },
            // Named ANSI colors only — for terminals (and screenshots) where
            // truecolor either isn't available or isn't wanted.
            ThemeName::Plain => Theme {
                text: Color::Reset,
                accent: Color::Cyan,
                accent_alt: Color::Blue,
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                muted: Color::DarkGray,
                border: Color::DarkGray,
                border_focus: Color::Cyan,
                highlight_fg: Color::Black,
                highlight_bg: Color::Cyan,
                surface: Color::Black,
                load: [(0, 187, 0), (187, 187, 0), (187, 0, 0)],
            },
        }
    }

    #[must_use]
    pub fn text_style(&self) -> Style {
        Style::new().fg(self.text)
    }

    #[must_use]
    pub fn header_style(&self) -> Style {
        Style::new().fg(self.accent).add_modifier(Modifier::BOLD)
    }

    #[must_use]
    pub fn muted_style(&self) -> Style {
        Style::new().fg(self.muted)
    }

    #[must_use]
    pub fn highlight_style(&self) -> Style {
        Style::new()
            .fg(self.highlight_fg)
            .bg(self.highlight_bg)
            .add_modifier(Modifier::BOLD)
    }

    /// Selected row inside a list: a soft surface wash plus a bold accent
    /// foreground, rather than the full inverse block `highlight_style` gives
    /// — a whole row of inverse video in a dense screen reads as an error.
    #[must_use]
    pub fn selected_style(&self) -> Style {
        Style::new()
            .fg(self.accent)
            .bg(self.surface)
            .add_modifier(Modifier::BOLD)
    }

    /// A rounded, theme-bordered panel with one column of breathing room on
    /// each side. Pass `""` for an untitled panel.
    #[must_use]
    pub fn block(&self, title: impl Into<String>) -> Block<'static> {
        self.block_focused(title, false)
    }

    /// As `block`, but `focused` panels take the accent border — the one
    /// cheap signal that says "your keys go here".
    #[must_use]
    pub fn block_focused(&self, title: impl Into<String>, focused: bool) -> Block<'static> {
        let border = if focused {
            self.border_focus
        } else {
            self.border
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(border))
            .padding(Padding::horizontal(1));
        let title = title.into();
        if title.is_empty() {
            block
        } else {
            block.title(self.panel_title(title))
        }
    }

    /// ` Title ` — padded so the border doesn't touch the letters.
    #[must_use]
    pub fn panel_title(&self, title: impl Into<String>) -> Span<'static> {
        Span::styled(format!(" {} ", title.into()), self.header_style())
    }

    /// `[key] label` as a footer hint — the key in accent, the words muted.
    #[must_use]
    pub fn hint(&self, key: &str, label: &str) -> Vec<Span<'static>> {
        vec![
            Span::styled(
                key.to_string(),
                Style::new().fg(self.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {label}"), self.muted_style()),
        ]
    }

    /// Utilization color, `btop`-style: calm below half, warm through
    /// three-quarters, hot above it.
    #[must_use]
    pub fn load_color(&self, ratio: f64) -> Color {
        let ratio = ratio.clamp(0.0, 1.0);
        if ratio <= 0.5 {
            lerp(self.load[0], self.load[1], ratio / 0.5)
        } else {
            lerp(self.load[1], self.load[2], (ratio - 0.5) / 0.5)
        }
    }

    /// Progress color for things that are *good* when they finish (a
    /// download, a launch): accent → success as it completes.
    #[must_use]
    pub fn progress_color(&self, ratio: f64) -> Color {
        match (self.accent, self.success) {
            (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
                lerp((r1, g1, b1), (r2, g2, b2), ratio.clamp(0.0, 1.0))
            }
            _ if ratio >= 0.999 => self.success,
            _ => self.accent,
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn lerp(a: Rgb, b: Rgb, t: f64) -> Color {
    let mix = |x: u8, y: u8| (f64::from(x) + (f64::from(y) - f64::from(x)) * t).round() as u8;
    Color::Rgb(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_color_walks_from_calm_to_hot() {
        let theme = Theme::from_name(ThemeName::Crane);
        assert_eq!(theme.load_color(0.0), theme.success);
        assert_eq!(theme.load_color(0.5), theme.warning);
        assert_eq!(theme.load_color(1.0), theme.error);
    }

    #[test]
    fn out_of_range_ratios_are_clamped_not_wrapped() {
        let theme = Theme::from_name(ThemeName::Crane);
        assert_eq!(theme.load_color(-1.0), theme.success);
        assert_eq!(theme.load_color(9.0), theme.error);
    }
}
