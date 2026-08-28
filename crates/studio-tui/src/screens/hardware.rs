//! Live hardware meters — the `btop`-flavored block that sits at the top of
//! the launchpad, plus the full report behind `[d]`. The compact meters are
//! built here (not in `launchpad`) so both screens show the same numbers in
//! the same shape.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use studio_core::hardware::{Backend, HardwareReport, Sample};

use crate::app::{App, History};
use crate::theme::Theme;
use crate::ui::bars::{self, ratio};
use crate::ui::{self, Chrome};

const HINTS: &[(&str, &str)] = &[("esc", "back"), ("f2", "theme"), ("q", "quit")];

/// Rows the compact meter block occupies, so callers can size a panel for it
/// without guessing: CPU, its history strip, RAM, one row per GPU, disk.
#[must_use]
pub fn meter_height(hardware: &HardwareReport, live: Option<&Sample>) -> u16 {
    let gpus = live
        .map_or(hardware.gpus.len(), |sample| sample.gpus.len())
        .min(2);
    #[allow(clippy::cast_possible_truncation)]
    let gpu_rows = gpus as u16;
    3 + gpu_rows + u16::from(!hardware.disk.is_empty())
}

/// The meter block. `width` is the inner width of the panel it goes into.
#[must_use]
pub fn meters(
    theme: &Theme,
    hardware: &HardwareReport,
    live: Option<&Sample>,
    history: &History,
    width: u16,
) -> Vec<Line<'static>> {
    let bar = width.saturating_sub(44).clamp(8, 30);
    // Whatever's left after the label gutter, the bar and the percentage — the
    // notes are the one part that can be shortened without losing a number.
    let note = (width as usize).saturating_sub(bar as usize + 15);
    let mut lines = Vec::new();

    let cpu_ratio = live.map_or(0.0, |s| f64::from(s.cpu_total.clamp(0.0, 100.0)) / 100.0);
    lines.push(bars::meter(
        theme,
        "cpu",
        bar,
        cpu_ratio,
        &percent(cpu_ratio),
        Some(&crate::ui::text::truncate(
            &format!(
                "{} · {}c/{}t",
                short_cpu_model(&hardware.cpu.model),
                hardware.cpu.physical_cores,
                hardware.cpu.logical_cores
            ),
            note,
        )),
    ));

    // History under the bar it belongs to, per-core texture to its right —
    // the two things `btop` shows about a CPU that a single percentage can't.
    let mut trend = vec![Span::styled("       ", theme.muted_style())];
    trend.extend(bars::sparkline(theme, &history.cpu, bar));
    if let Some(sample) = live {
        trend.push(Span::styled("  cores ", theme.muted_style()));
        trend.extend(bars::core_strip(theme, &fold_cores(&sample.per_core, 24)));
    }
    lines.push(Line::from(trend));

    let (ram_total, ram_available) = live
        .map_or((hardware.ram_total, hardware.ram_available), |s| {
            (s.ram_total, s.ram_available)
        });
    let ram_used = ram_total.saturating_sub(ram_available);
    let ram_ratio = ratio(ram_used, ram_total);
    lines.push(bars::meter(
        theme,
        "ram",
        bar,
        ram_ratio,
        &percent(ram_ratio),
        Some(&crate::ui::text::truncate(
            &format!(
                "{} / {}",
                crate::fmt::bytes(ram_used),
                crate::fmt::bytes(ram_total)
            ),
            note,
        )),
    ));

    let gpus = live.map_or(&hardware.gpus, |s| &s.gpus);
    for gpu in gpus.iter().take(2) {
        let used = gpu.vram_total.saturating_sub(gpu.vram_free);
        let vram_ratio = ratio(used, gpu.vram_total);
        lines.push(bars::meter(
            theme,
            "vram",
            bar,
            vram_ratio,
            &percent(vram_ratio),
            Some(&crate::ui::text::truncate(
                &format!(
                    "{} / {} · {}",
                    crate::fmt::bytes(used),
                    crate::fmt::bytes(gpu.vram_total),
                    gpu.name
                ),
                note,
            )),
        ));
    }

    if let Some(disk) = hardware.disk.first() {
        let used = disk.total.saturating_sub(disk.available);
        let disk_ratio = ratio(used, disk.total);
        lines.push(bars::meter(
            theme,
            "disk",
            bar,
            disk_ratio,
            &percent(disk_ratio),
            Some(&crate::ui::text::truncate(
                &format!("{} free for models", crate::fmt::bytes(disk.available)),
                note,
            )),
        ));
    }

    lines
}

/// CPU brand strings pad themselves with marketing ("12-Core Processor", "CPU @
/// 3.70GHz") that a meter row can't spare the columns for — and the core count
/// is already shown right next to it.
fn short_cpu_model(model: &str) -> String {
    let mut short = model;
    if let Some(at) = short.find(" @ ") {
        short = &short[..at];
    }
    for suffix in [" Processor", " CPU"] {
        if let Some(stripped) = short.strip_suffix(suffix) {
            short = stripped;
        }
    }
    let words: Vec<&str> = short
        .split_whitespace()
        .filter(|word| !word.ends_with("-Core"))
        .collect();
    words.join(" ")
}

/// A machine with more logical cores than columns to draw them in gets its
/// cores folded into groups, each shown at the group's busiest — better than
/// silently dropping the tail of the CPU.
fn fold_cores(per_core: &[f32], max: usize) -> Vec<f32> {
    if per_core.len() <= max || max == 0 {
        return per_core.to_vec();
    }
    let group = per_core.len().div_ceil(max);
    per_core
        .chunks(group)
        .map(|chunk| chunk.iter().copied().fold(0.0_f32, f32::max))
        .collect()
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn percent(value: f64) -> String {
    format!("{:>3}%", (value.clamp(0.0, 1.0) * 100.0).round() as u32)
}

#[must_use]
pub fn backend_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "CUDA",
        Backend::Metal => "Metal",
        Backend::Rocm => "ROCm",
        Backend::Cpu => "CPU",
    }
}

/// The full report (§4.2): live meters on top, then everything the probe knows
/// — the detail a bug report needs.
pub fn render(app: &App, frame: &mut Frame) {
    let chrome = Chrome::new(HINTS)
        .crumb("Hardware")
        .status(status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    let live_height = meter_height(&app.hardware, app.live.as_ref()) + 2;
    let [meters_area, detail_area] =
        Layout::vertical([Constraint::Length(live_height), Constraint::Min(4)]).areas(body);

    panel(app, frame, meters_area, "Live");
    let block = app.theme.block("Detail");
    let inner = block.inner(detail_area);
    frame.render_widget(block, detail_area);
    frame.render_widget(
        Paragraph::new(detail_lines(app, inner.width)).scroll((app.hardware_scroll, 0)),
        inner,
    );
}

/// The meter block wrapped in its own titled panel — what the launchpad puts
/// above its model list.
pub fn panel(app: &App, frame: &mut Frame, area: Rect, title: &str) {
    let block = app.theme.block(title);
    let inner = block.inner(area);
    frame.render_widget(
        block.title_bottom(
            Line::from(vec![Span::styled(
                format!(" {} ", backend_label(app.hardware.backend)),
                Style::new()
                    .fg(app.theme.accent_alt)
                    .add_modifier(Modifier::BOLD),
            )])
            .right_aligned(),
        ),
        area,
    );
    frame.render_widget(
        Paragraph::new(meters(
            &app.theme,
            &app.hardware,
            app.live.as_ref(),
            &app.history,
            inner.width,
        )),
        inner,
    );
}

fn detail_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let hw = &app.hardware;
    let mut lines = vec![
        ui::field(&app.theme, "backend", backend_label(hw.backend)),
        ui::field(&app.theme, "cpu", hw.cpu.model.clone()),
        ui::field(
            &app.theme,
            "cores",
            format!(
                "{} physical / {} logical",
                hw.cpu.physical_cores, hw.cpu.logical_cores
            ),
        ),
        ui::field(
            &app.theme,
            "ram",
            format!(
                "{} available of {}",
                crate::fmt::bytes(hw.ram_available),
                crate::fmt::bytes(hw.ram_total)
            ),
        ),
        Line::raw(""),
        ui::section(&app.theme, "gpu"),
    ];
    lines.extend(gpu_lines(app, width));
    lines.push(Line::raw(""));
    lines.push(ui::section(&app.theme, "disk"));
    lines.extend(disk_lines(app));
    lines.push(Line::raw(""));
    lines.push(ui::section(&app.theme, "paths"));
    lines.push(ui::field(
        &app.theme,
        "models",
        studio_core::paths::models_dir().display().to_string(),
    ));
    lines.push(ui::field(
        &app.theme,
        "config",
        studio_core::paths::config_dir()
            .join("config.ron")
            .display()
            .to_string(),
    ));
    lines
}

/// §4.2's explicit warnings live here: a GPU build that found no GPU is the
/// single most common "why is this so slow" report, so it says so in words.
fn gpu_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let hw = &app.hardware;
    let mut lines = Vec::new();
    if hw.gpus.is_empty() {
        lines.push(ui::field(&app.theme, "gpu", "none detected"));
        if matches!(hw.backend, Backend::Cuda | Backend::Metal | Backend::Rocm) {
            for row in crate::ui::text::wrap(
                &format!(
                    "this build expects a {} GPU but none was found — inference falls back to the CPU and will be very slow",
                    backend_label(hw.backend)
                ),
                width,
            ) {
                lines.push(Line::from(Span::styled(
                    row,
                    Style::new().fg(app.theme.warning),
                )));
            }
        }
    }
    for gpu in &hw.gpus {
        lines.push(ui::field(
            &app.theme,
            &format!("gpu {}", gpu.index),
            gpu.name.clone(),
        ));
        lines.push(ui::field(
            &app.theme,
            "vram",
            format!(
                "{} free of {}",
                crate::fmt::bytes(gpu.vram_free),
                crate::fmt::bytes(gpu.vram_total)
            ),
        ));
        if let Some((major, minor)) = gpu.compute_capability {
            lines.push(ui::field(&app.theme, "compute", format!("{major}.{minor}")));
        }
        if let Some(driver) = &gpu.driver_version {
            lines.push(ui::field(&app.theme, "driver", driver.clone()));
        }
        if gpu.unified_memory {
            lines.push(ui::field(
                &app.theme,
                "memory",
                "unified — shared with system RAM and other apps",
            ));
        }
    }
    lines
}

fn disk_lines(app: &App) -> Vec<Line<'static>> {
    if app.hardware.disk.is_empty() {
        return vec![ui::field(
            &app.theme,
            "disk",
            "could not determine free space for the models directory",
        )];
    }
    app.hardware
        .disk
        .iter()
        .map(|disk| {
            ui::field(
                &app.theme,
                &disk.mount_point,
                format!(
                    "{} available of {}",
                    crate::fmt::bytes(disk.available),
                    crate::fmt::bytes(disk.total)
                ),
            )
        })
        .collect()
}

/// `CUDA · RTX 3090 · gateway :1234` — the title-row status every screen
/// shares, so "what am I running on" never scrolls away.
#[must_use]
pub fn status_spans(app: &App) -> Vec<Span<'static>> {
    let mut spans = vec![Span::styled(
        backend_label(app.hardware.backend).to_string(),
        Style::new().fg(app.theme.accent_alt),
    )];
    if let Some(gpu) = app.hardware.gpus.first() {
        spans.push(Span::styled(" · ", app.theme.muted_style()));
        spans.push(Span::styled(
            crate::ui::text::truncate(&gpu.name, 22),
            app.theme.muted_style(),
        ));
    }
    spans.push(Span::styled(" · ", app.theme.muted_style()));
    spans.push(Span::styled(
        format!(":{}", app.gateway_port),
        app.theme.muted_style(),
    ));
    spans
}

#[cfg(test)]
mod tests {
    use super::short_cpu_model;

    #[test]
    fn cpu_names_lose_their_marketing_padding() {
        assert_eq!(
            short_cpu_model("AMD Ryzen 9 5900X 12-Core Processor"),
            "AMD Ryzen 9 5900X"
        );
        assert_eq!(
            short_cpu_model("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"),
            "Intel(R) Core(TM) i7-9750H"
        );
        assert_eq!(short_cpu_model("Apple M4 Pro"), "Apple M4 Pro");
    }
}
