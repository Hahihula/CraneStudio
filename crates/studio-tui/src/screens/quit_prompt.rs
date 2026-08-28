//! The quit-lease modal, per PLAN.md §3.1a: "Quitting the TUI never leaves a
//! model resident without an explicit 'keep serving'." Drawn over whatever
//! screen is behind it, with the choices as selectable rows — the arrow keys
//! work here exactly like they do everywhere else, so there's nothing new to
//! learn at the one moment a wrong key costs you a loaded model.

use ratatui::Frame;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::QuitChoice;
use crate::daemon_client::ChildSummary;
use crate::theme::{Theme, glyph};
use crate::ui;

pub fn render(frame: &mut Frame, theme: &Theme, choice: QuitChoice, running: &[ChildSummary]) {
    let area = ui::centered(ui::canvas(frame.area()), 64, 9);
    ui::modal(frame, area);

    let block = theme.block_focused("Quit CraneStudio", true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let running_names: Vec<String> = running
        .iter()
        .filter(|c| c.state.is_running())
        .map(|c| c.info.label.clone())
        .collect();

    let mut lines = vec![if running_names.is_empty() {
        Line::from(Span::styled("nothing is running", theme.muted_style()))
    } else {
        Line::from(vec![
            Span::styled(
                format!("{} ", glyph::HEALTHY),
                Style::new().fg(theme.success),
            ),
            Span::styled(
                crate::ui::text::truncate(&running_names.join(", "), 52),
                Style::new().fg(theme.text),
            ),
        ])
    }];
    lines.push(Line::raw(""));
    lines.push(option(
        theme,
        "k",
        "Keep serving",
        "leave models loaded in the background",
        choice == QuitChoice::Keep,
    ));
    lines.push(option(
        theme,
        "s",
        "Stop everything",
        "unload every model and stop the daemon",
        choice == QuitChoice::Stop,
    ));
    lines.push(option(
        theme,
        "c",
        "Cancel",
        "stay in CraneStudio",
        choice == QuitChoice::Cancel,
    ));
    lines.push(Line::raw(""));
    lines.push(ui::hint_line(
        theme,
        &[("↑↓", "choose"), ("⏎", "confirm"), ("esc", "cancel")],
    ));

    frame.render_widget(Paragraph::new(lines), inner);
}

fn option(
    theme: &Theme,
    key: &str,
    title: &str,
    blurb: &str,
    selected: bool,
) -> Line<'static> {
    let (marker, title_style) = if selected {
        (
            Span::styled(
                format!("{} ", glyph::BAR_HALF),
                Style::new().fg(theme.accent),
            ),
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        )
    } else {
        (
            Span::raw("  "),
            Style::new().fg(theme.text),
        )
    };
    Line::from(vec![
        marker,
        Span::styled(format!("{key}  "), theme.muted_style()),
        Span::styled(format!("{title:<17}"), title_style),
        Span::styled(blurb.to_string(), theme.muted_style()),
    ])
}
