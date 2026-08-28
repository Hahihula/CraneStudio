//! The quit-lease popup, per PLAN.md §3.1a: "Quitting the TUI never leaves
//! a model resident without an explicit 'keep serving'." Rendered as an
//! overlay on top of whatever screen is behind it.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::QuitChoice;
use crate::daemon_client::ChildSummary;
use crate::theme::Theme;

pub fn render(frame: &mut Frame, theme: Theme, choice: QuitChoice, running: &[ChildSummary]) {
    let area = centered_rect(60, 40, frame.area());
    frame.render_widget(Clear, area);

    let running_names: Vec<String> = running
        .iter()
        .filter(|c| c.state.is_running())
        .map(|c| c.info.label.clone())
        .collect();
    let running_line = if running_names.is_empty() {
        "(nothing is running)".to_string()
    } else {
        format!("Still running: {}", running_names.join(", "))
    };

    let lines = vec![
        Line::raw(running_line),
        Line::raw(""),
        option_line(
            theme,
            "[K]eep serving — leave models running in the background",
            choice == QuitChoice::Keep,
        ),
        option_line(
            theme,
            "[S]top everything — stop all models and the daemon now",
            choice == QuitChoice::Stop,
        ),
        option_line(theme, "[C]ancel — go back", choice == QuitChoice::Cancel),
        Line::raw(""),
        Line::raw("Enter to confirm, Esc to cancel"),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(theme.block("Quit CraneStudio?")),
        area,
    );
}

fn option_line(theme: Theme, text: &str, selected: bool) -> Line<'static> {
    let style = if selected {
        theme.highlight_style()
    } else {
        Style::new()
    };
    Line::from(ratatui::text::Span::styled(text.to_string(), style))
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [area] = Layout::vertical([Constraint::Percentage(percent_y)])
        .flex(Flex::Center)
        .areas(area);
    let [area] = Layout::horizontal([Constraint::Percentage(percent_x)])
        .flex(Flex::Center)
        .areas(area);
    area
}
