//! The splash (the first ~second of the app): wordmark, crane, and the boot
//! work that's genuinely happening behind it — the hardware probe, the catalog
//! fetch, the local model scan. It dismisses itself the moment that work is
//! done, or immediately on any keypress, so it's never in the way.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::theme::{Theme, glyph};
use crate::ui::art;

/// A boot step and whether it's finished — shown as a checklist so a slow
/// network (the catalog fetch) is visibly *the* thing being waited on, rather
/// than an unexplained pause.
struct Step {
    label: &'static str,
    done: bool,
}

pub fn render(app: &App, frame: &mut Frame) {
    let area = crate::ui::canvas(frame.area());
    let big = area.width >= art::WORDMARK_WIDTH && area.height >= 24;

    let logo: Vec<Line> = if big {
        art::wordmark(&app.theme)
    } else {
        art::wordmark_small(&app.theme)
    };
    let crane = art::crane(&app.theme);
    let crane_height = if big { art::CRANE_HEIGHT } else { 0 };
    #[allow(clippy::cast_possible_truncation)]
    let logo_height = logo.len() as u16;

    let steps = [
        Step {
            label: "hardware",
            done: true,
        },
        Step {
            label: "models",
            done: app.local_scan_done,
        },
        Step {
            label: "catalog",
            done: app.browser.catalog.is_some(),
        },
    ];

    let [block] = Layout::vertical([Constraint::Length(crane_height + logo_height + 4)])
        .flex(Flex::Center)
        .areas(area);
    let [crane_area, logo_area, _, tagline_area, steps_area] = Layout::vertical([
        Constraint::Length(crane_height),
        Constraint::Length(logo_height),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(2),
    ])
    .areas(block);

    if crane_height > 0 {
        render_centered(frame, crane_area, crane);
    }
    render_centered(frame, logo_area, logo);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("run local models ", app.theme.muted_style()),
            Span::styled(glyph::ARROW, Style::new().fg(app.theme.accent)),
            Span::styled(" pick one and press enter", app.theme.muted_style()),
        ]))
        .centered(),
        tagline_area,
    );

    frame.render_widget(
        Paragraph::new(vec![
            steps_line(&app.theme, app.tick, &steps),
            Line::from(Span::styled(
                if steps.iter().all(|s| s.done) {
                    "press any key"
                } else {
                    ""
                },
                app.theme
                    .muted_style()
                    .add_modifier(Modifier::DIM | Modifier::ITALIC),
            ))
            .centered(),
        ])
        .centered(),
        steps_area,
    );
}

fn steps_line(theme: &Theme, tick: u64, steps: &[Step]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("   ", theme.muted_style()));
        }
        let (mark, style) = if step.done {
            (glyph::DONE.to_string(), Style::new().fg(theme.success))
        } else {
            (
                crate::theme::spinner(tick).to_string(),
                Style::new().fg(theme.accent),
            )
        };
        spans.push(Span::styled(format!("{mark} "), style));
        spans.push(Span::styled(
            step.label.to_string(),
            if step.done {
                theme.muted_style()
            } else {
                Style::new().fg(theme.text)
            },
        ));
    }
    Line::from(spans).centered()
}

/// Art is a fixed-width block, so it's centered as a unit — centering each row
/// on its own would ragged-edge the wordmark's trailing spaces.
fn render_centered(frame: &mut Frame, area: Rect, lines: Vec<Line<'static>>) {
    #[allow(clippy::cast_possible_truncation)]
    let width = lines
        .iter()
        .map(ratatui::text::Line::width)
        .max()
        .unwrap_or(0) as u16;
    let [inner] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(area);
    frame.render_widget(Paragraph::new(lines), inner);
}
