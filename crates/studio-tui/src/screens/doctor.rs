//! The hardware report (§4.2) — reuses M1's plain-text `doctor::render`
//! output verbatim, just wrapped in a scrollable `Paragraph`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::App;

pub fn render(app: &mut App, frame: &mut Frame) {
    let [body, footer] =
        Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(frame.area());

    let text = crate::doctor::render(&app.hardware);
    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .block(app.theme.block("Hardware")),
        body,
    );
    frame.render_widget(
        Paragraph::new("[h] home   [F2] theme   [q] quit").style(app.theme.muted_style()),
        footer,
    );
}
