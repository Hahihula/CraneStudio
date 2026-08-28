//! Shared color palette, cycled with `F2` from any screen (see `app.rs`) —
//! every screen draws from one consistent set of semantic colors instead of
//! sprinkling `Color::Cyan`/`Color::Green` ad hoc, which is the state this
//! crate was in before this module existed.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, BorderType};
use studio_core::config::ThemeName;

/// Animation frames for a braille spinner, shown during any async wait
/// (downloading, launching, generating…) that has no real percentage to
/// report. Advanced once per `on_tick` (500ms), so a full rotation is 5s.
const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

#[must_use]
pub fn spinner(tick: u64) -> char {
    #[allow(clippy::cast_possible_truncation)]
    let index = (tick % SPINNER_FRAMES.len() as u64) as usize;
    SPINNER_FRAMES[index]
}

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    /// Headers, tab highlights, panel titles — the primary brand color.
    pub accent: Color,
    /// Fits / healthy / done / measured-good-news.
    pub success: Color,
    /// Starting / short-of-target / in-progress.
    pub warning: Color,
    /// Unusable / exited / failed.
    pub error: Color,
    /// Footers, secondary text, borders.
    pub muted: Color,
    pub border: Color,
    pub highlight_fg: Color,
    pub highlight_bg: Color,
}

impl Theme {
    #[must_use]
    pub fn from_name(name: ThemeName) -> Self {
        match name {
            ThemeName::Monokai => Theme {
                accent: Color::Rgb(102, 217, 239),
                success: Color::Rgb(166, 226, 46),
                warning: Color::Rgb(230, 219, 116),
                error: Color::Rgb(249, 38, 114),
                muted: Color::Rgb(117, 113, 94),
                border: Color::Rgb(117, 113, 94),
                highlight_fg: Color::Rgb(39, 40, 34),
                highlight_bg: Color::Rgb(102, 217, 239),
            },
            ThemeName::Dracula => Theme {
                accent: Color::Rgb(189, 147, 249),
                success: Color::Rgb(80, 250, 123),
                warning: Color::Rgb(255, 184, 108),
                error: Color::Rgb(255, 85, 85),
                muted: Color::Rgb(98, 114, 164),
                border: Color::Rgb(98, 114, 164),
                highlight_fg: Color::Rgb(40, 42, 54),
                highlight_bg: Color::Rgb(189, 147, 249),
            },
            ThemeName::Plain => Theme {
                accent: Color::Cyan,
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                muted: Color::DarkGray,
                border: Color::DarkGray,
                highlight_fg: Color::Black,
                highlight_bg: Color::Cyan,
            },
        }
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

    /// A rounded, theme-bordered panel. Pass `""` for an untitled block —
    /// `Block::title("")` renders nothing, so this covers both cases.
    #[must_use]
    pub fn block(&self, title: impl Into<String>) -> Block<'static> {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(self.border));
        let title = title.into();
        if title.is_empty() {
            block
        } else {
            block.title(Span::styled(title, self.header_style()))
        }
    }
}
