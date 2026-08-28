//! The download screen (§9) — and, for most first-time users, the longest
//! single wait in the app, so it earns real progress reporting: one headline
//! bar for the whole job, per-file bars underneath, live throughput, and an
//! ETA that comes from measured bytes rather than a guess.
//!
//! Wires the browser's Catalog and Search tabs to the same downloader
//! `cranestudio download` uses.

use std::path::Path;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::{Format, ModelEntry};
use studio_core::download::{CancellationToken, Event, RepoDownload, download_repo};

use crate::app::{App, BackgroundEvent, Screen};
use crate::theme::glyph;
use crate::ui::bars::{self, ratio};
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[("esc", "cancel"), ("f2", "theme")];
const DONE_HINTS: &[(&str, &str)] = &[("esc", "back")];

/// How many throughput samples the sparkline keeps — at one sample per tick
/// (250ms) that's the last half minute of the transfer.
const SPEED_HISTORY: usize = 120;

/// A single file's download state, kept structured (not pre-formatted text) so
/// `render` can drive a real progress bar instead of a literal percent string.
#[derive(Debug, Clone, Copy)]
pub enum FileStatus {
    Starting { resumed_from: u64, total: u64 },
    Progress { downloaded: u64, total: u64 },
    Verifying,
    Completed,
    Cancelled,
}

impl FileStatus {
    fn bytes(self) -> (u64, u64) {
        match self {
            FileStatus::Starting {
                resumed_from,
                total,
            } => (resumed_from, total),
            FileStatus::Progress { downloaded, total } => (downloaded, total),
            // Verifying happens after the last byte lands, so it counts as
            // whole — otherwise the headline bar would visibly go backwards.
            FileStatus::Verifying | FileStatus::Completed => (1, 1),
            FileStatus::Cancelled => (0, 1),
        }
    }
}

#[derive(Default)]
pub struct State {
    pub label: String,
    pub repo: String,
    /// One entry per file — `Progress` events update a file's own entry in
    /// place rather than appending, so a multi-file download doesn't scroll
    /// past its own progress.
    pub lines: Vec<(String, FileStatus)>,
    pub error: Option<String>,
    pub cancel: Option<CancellationToken>,
    /// Bytes/second, and the samples behind the sparkline.
    pub speed: f64,
    pub speed_history: Vec<f64>,
    last_sample: Option<(Instant, u64)>,
    peak_speed: f64,
}

impl State {
    fn set_status(&mut self, file: &str, status: FileStatus) {
        if let Some(pos) = self.lines.iter().position(|(f, _)| f == file) {
            self.lines[pos].1 = status;
        } else {
            self.lines.push((file.to_string(), status));
        }
    }

    pub fn apply_event(&mut self, event: &Event) {
        match event {
            Event::Started {
                file,
                resume_from,
                total,
            } => self.set_status(
                file,
                FileStatus::Starting {
                    resumed_from: *resume_from,
                    total: *total,
                },
            ),
            Event::Progress {
                file,
                downloaded,
                total,
            } => self.set_status(
                file,
                FileStatus::Progress {
                    downloaded: *downloaded,
                    total: *total,
                },
            ),
            Event::Verifying { file } => self.set_status(file, FileStatus::Verifying),
            Event::Completed { file } => self.set_status(file, FileStatus::Completed),
            Event::Cancelled { file } => self.set_status(file, FileStatus::Cancelled),
        }
    }

    /// Total bytes in flight across every file, and the total expected.
    #[must_use]
    pub fn totals(&self) -> (u64, u64) {
        self.lines.iter().fold((0, 0), |(done, total), (_, status)| {
            match status {
                FileStatus::Starting {
                    resumed_from,
                    total: file_total,
                } => (done + resumed_from, total + file_total),
                FileStatus::Progress {
                    downloaded,
                    total: file_total,
                } => (done + downloaded, total + file_total),
                // A completed file's own total is no longer reported by the
                // event stream, so it's held at whatever it last was.
                FileStatus::Verifying | FileStatus::Completed | FileStatus::Cancelled => {
                    (done, total)
                }
            }
        })
    }

    /// Called once per app tick: turns the byte counter into a throughput
    /// number, smoothed so the readout doesn't flicker between chunk arrivals.
    pub fn sample_speed(&mut self) {
        let (downloaded, _) = self.totals();
        let now = Instant::now();
        if let Some((then, previous)) = self.last_sample {
            let elapsed = now.duration_since(then).as_secs_f64();
            if elapsed > 0.0 {
                #[allow(clippy::cast_precision_loss)]
                let instant = downloaded.saturating_sub(previous) as f64 / elapsed;
                self.speed = if self.speed == 0.0 {
                    instant
                } else {
                    self.speed * 0.6 + instant * 0.4
                };
                self.peak_speed = self.peak_speed.max(self.speed);
                self.speed_history.push(self.speed);
                if self.speed_history.len() > SPEED_HISTORY {
                    self.speed_history.remove(0);
                }
            }
        }
        self.last_sample = Some((now, downloaded));
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        !self.lines.is_empty()
            && self
                .lines
                .iter()
                .all(|(_, s)| matches!(s, FileStatus::Completed))
    }
}

/// Picks the smallest variant (fastest path to something running) and downloads
/// it — the wizard's own "not a form" philosophy (§4.4) applied one step
/// earlier, to the download itself.
pub fn start_catalog(app: &mut App, model: &ModelEntry) {
    let Some(variant) = model.variants.iter().min_by_key(|v| v.download_bytes) else {
        app.message = Some(Message::error(format!(
            "{} has no downloadable variants",
            model.display_name
        )));
        return;
    };
    start(
        app,
        format!("{} · {}", model.display_name, variant.id),
        variant.repo.clone(),
        variant.revision.clone(),
        variant.files.clone(),
        // Trust the catalog's own verified model_type directly rather than
        // re-classifying the download afterward — some families (MiniCPM5) are
        // deliberately never auto-detected at all (their config/GGUF header is
        // indistinguishable from real Llama; see `catalog::architecture`'s
        // `minicpm5` entry), so re-scanning would wrongly reject a model the
        // catalog already knows is supported.
        Some((model.model_type.clone(), variant.format)),
    );
}

/// `main` rather than a pinned commit sha — acceptable for an ad-hoc search
/// download (§5's "pin to a commit sha" concern is about a *saved profile*
/// silently changing meaning later, which doesn't apply here since nothing is
/// saved). No known `model_type` here — the search result's own classification
/// already came from re-scannable config/GGUF-header detection, so re-scanning
/// after download is consistent, unlike the catalog case above.
pub fn start_hf(app: &mut App, repo_id: &str, gguf_file: &str) {
    start(
        app,
        repo_id.to_string(),
        repo_id.to_string(),
        "main".to_string(),
        vec![gguf_file.to_string()],
        None,
    );
}

/// Builds a `LocalCandidate` straight from the catalog's own verified
/// `model_type`, without touching the downloaded file's contents — the point of
/// this path (see `start_catalog`'s doc comment).
fn known_candidate(
    dest_dir: &Path,
    first_file: &str,
    model_type: &str,
    format: Format,
) -> Option<LocalCandidate> {
    let family = studio_core::catalog::FAMILIES
        .iter()
        .find(|f| f.model_type == model_type)?;
    let path = match format {
        Format::Gguf => dest_dir.join(first_file),
        Format::Safetensors => dest_dir.to_path_buf(),
    };
    Some(LocalCandidate {
        path,
        format,
        classification: Classification::Supported {
            model_type: family.model_type,
            vision: family.vision,
            gated: family.gated,
        },
    })
}

fn start(
    app: &mut App,
    label: String,
    repo: String,
    revision: String,
    files: Vec<String>,
    known: Option<(String, Format)>,
) {
    let cancel = CancellationToken::new();
    app.download = State {
        label,
        repo: repo.clone(),
        cancel: Some(cancel.clone()),
        ..State::default()
    };
    app.screen = Screen::Download;
    app.message = None;

    let dest_dir = studio_core::paths::models_dir().join(&repo).join(&revision);
    let hf_token =
        studio_core::config::Config::load(&studio_core::paths::config_dir().join("config.ron"))
            .hf_token;
    let tx = app.sender();

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut request = RepoDownload::new(repo, revision, dest_dir.clone());
        request.token = hf_token;

        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let forward_tx = tx.clone();
        let forwarder = tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                let _ = forward_tx.send(BackgroundEvent::DownloadProgress(event));
            }
        });

        let result = download_repo(&client, &request, &files, &event_tx, &cancel).await;
        drop(event_tx);
        let _ = forwarder.await;

        match result {
            Ok(()) => {
                let candidate = known
                    .as_ref()
                    .and_then(|(model_type, format)| {
                        known_candidate(&dest_dir, &files[0], model_type, *format)
                    })
                    .or_else(|| {
                        studio_core::catalog::local::scan(&dest_dir)
                            .into_iter()
                            .next()
                    });
                match candidate {
                    Some(candidate) => {
                        let _ = tx.send(BackgroundEvent::DownloadDone(candidate));
                    }
                    None => {
                        let _ = tx.send(BackgroundEvent::DownloadFailed(
                            "downloaded, but couldn't classify the result — check the Local tab"
                                .to_string(),
                        ));
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(BackgroundEvent::DownloadFailed(e.to_string()));
            }
        }
    });
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if key.code == KeyCode::Esc {
        if app.download.error.is_none()
            && let Some(cancel) = &app.download.cancel
        {
            cancel.cancel();
        }
        app.screen = Screen::Browser;
        return true;
    }
    // Swallow everything else while a download is actually in flight; once it's
    // failed, let global navigation (h/b/q/…) through again.
    app.download.error.is_none()
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let hints = if app.download.error.is_some() {
        DONE_HINTS
    } else {
        HINTS
    };
    let chrome = Chrome::new(hints)
        .crumb("Download")
        .crumb(crate::ui::text::truncate(&app.download.label, 30))
        .status(crate::screens::hardware::status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    if let Some(err) = app.download.error.clone() {
        let [panel, _] =
            Layout::vertical([Constraint::Length(9), Constraint::Min(0)]).areas(body);
        render_failure(app, &err, frame, panel);
        return;
    }

    #[allow(clippy::cast_possible_truncation)]
    let files_height = (app.download.lines.len().max(1) as u16).saturating_add(2);
    let [headline, files, _] = Layout::vertical([
        Constraint::Length(8),
        Constraint::Length(files_height),
        Constraint::Min(0),
    ])
    .areas(body);
    render_headline(app, frame, headline);
    render_files(app, frame, files);
}

/// The one number a waiting user actually wants: how far along the whole job
/// is, how fast it's moving, and when it will be done.
fn render_headline(app: &App, frame: &mut Frame, area: Rect) {
    let block = app.theme.block_focused("Downloading", true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let (downloaded, total) = app.download.totals();
    let done = ratio(downloaded, total);
    let bar_width = inner.width.saturating_sub(16).clamp(10, 60);

    let mut headline = vec![Span::styled(
        crate::ui::text::truncate(&app.download.label, (inner.width as usize).saturating_sub(10)),
        Style::new().fg(app.theme.text).add_modifier(Modifier::BOLD),
    )];
    if app.download.finished() {
        headline.push(Span::styled(
            format!("   {} verified", glyph::DONE),
            Style::new().fg(app.theme.success),
        ));
    }

    let mut bar = bars::progress_bar(
        &app.theme,
        bar_width,
        done,
        app.theme.progress_color(done),
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let percent = (done * 100.0).round() as u32;
    bar.push(Span::styled(
        format!("  {percent:>3}%"),
        Style::new()
            .fg(app.theme.progress_color(done))
            .add_modifier(Modifier::BOLD),
    ));

    let mut stats = vec![
        Span::styled(
            format!(
                "{} of {}",
                crate::fmt::bytes(downloaded),
                crate::fmt::bytes(total)
            ),
            Style::new().fg(app.theme.text),
        ),
        Span::styled("   ", app.theme.muted_style()),
        Span::styled(
            format!("{}/s", crate::fmt::bytes(speed_bytes(app.download.speed))),
            Style::new().fg(app.theme.accent_alt),
        ),
        Span::styled("   ", app.theme.muted_style()),
        Span::styled(eta(app), app.theme.muted_style()),
    ];
    if !app.download.speed_history.is_empty() {
        stats.push(Span::styled("   ", app.theme.muted_style()));
        stats.extend(bars::sparkline(
            &app.theme,
            &normalized_speeds(&app.download.speed_history),
            24,
        ));
    }

    let lines = vec![
        Line::from(headline),
        Line::from(Span::styled(
            crate::ui::text::truncate(&app.download.repo, inner.width as usize),
            app.theme.muted_style(),
        )),
        Line::raw(""),
        Line::from(bar),
        Line::raw(""),
        Line::from(stats),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_files(app: &App, frame: &mut Frame, area: Rect) {
    let block = app
        .theme
        .block(format!("Files  ·  {}", app.download.lines.len()));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.download.lines.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("{} ", crate::theme::spinner(app.tick)),
                    Style::new().fg(app.theme.accent),
                ),
                Span::styled(
                    "asking HuggingFace for file sizes…",
                    app.theme.muted_style(),
                ),
            ])),
            inner,
        );
        return;
    }

    let name_width = 28.min(inner.width / 3) as usize;
    let bar_width = inner
        .width
        .saturating_sub(u16::try_from(name_width).unwrap_or(28) + 26)
        .clamp(8, 40);

    let lines: Vec<Line> = app
        .download
        .lines
        .iter()
        .map(|(file, status)| file_row(app, file, *status, name_width, bar_width))
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn file_row(
    app: &App,
    file: &str,
    status: FileStatus,
    name_width: usize,
    bar_width: u16,
) -> Line<'static> {
    let name = crate::ui::text::truncate_start(file, name_width);
    let mut spans = vec![Span::styled(
        format!("{name:<name_width$} "),
        Style::new().fg(app.theme.text),
    )];

    let (done, total) = status.bytes();
    let filled = ratio(done, total);
    let (color, note) = match status {
        FileStatus::Starting {
            resumed_from,
            total,
        } if resumed_from > 0 => (
            app.theme.accent,
            format!(
                "resuming at {} of {}",
                crate::fmt::bytes(resumed_from),
                crate::fmt::bytes(total)
            ),
        ),
        FileStatus::Starting { total, .. } => (
            app.theme.accent,
            format!("queued · {}", crate::fmt::bytes(total)),
        ),
        FileStatus::Progress { downloaded, total } => (
            app.theme.progress_color(filled),
            format!(
                "{} / {}",
                crate::fmt::bytes(downloaded),
                crate::fmt::bytes(total)
            ),
        ),
        FileStatus::Verifying => (
            app.theme.warning,
            format!("{} verifying checksum", crate::theme::spinner(app.tick)),
        ),
        FileStatus::Completed => (app.theme.success, format!("{} done", glyph::DONE)),
        FileStatus::Cancelled => (app.theme.muted, "cancelled".to_string()),
    };

    spans.extend(bars::progress_bar(&app.theme, bar_width, filled, color));
    spans.push(Span::styled(format!("  {note}"), Style::new().fg(color)));
    Line::from(spans)
}

fn render_failure(app: &App, err: &str, frame: &mut Frame, area: Rect) {
    let block = app.theme.block("Download failed");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", glyph::FAILED),
            Style::new().fg(app.theme.error),
        ),
        Span::styled(
            crate::ui::text::truncate(&app.download.label, 48),
            Style::new().fg(app.theme.text).add_modifier(Modifier::BOLD),
        ),
    ])];
    lines.push(Line::raw(""));
    for row in crate::ui::text::wrap(err, inner.width) {
        lines.push(Line::from(Span::styled(
            row,
            Style::new().fg(app.theme.error),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "partly downloaded files are kept — starting the same download again resumes it",
        app.theme.muted_style(),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn speed_bytes(speed: f64) -> u64 {
    speed.max(0.0).round() as u64
}

/// Speeds as a 0–1 series against the fastest sample seen, so the sparkline
/// shows the *shape* of the transfer (stalls, ramp-ups) rather than needing an
/// axis nobody would read.
fn normalized_speeds(history: &[f64]) -> Vec<f64> {
    let peak = history.iter().copied().fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return vec![0.0; history.len()];
    }
    history.iter().map(|s| s / peak).collect()
}

#[allow(clippy::cast_precision_loss)]
fn eta(app: &App) -> String {
    let (downloaded, total) = app.download.totals();
    let remaining = total.saturating_sub(downloaded);
    if remaining == 0 {
        return "finishing".to_string();
    }
    if app.download.speed <= 1.0 {
        return "estimating…".to_string();
    }
    let seconds = (remaining as f64 / app.download.speed).round();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let seconds = seconds as u64;
    match seconds {
        0..=59 => format!("{seconds}s left"),
        60..=3599 => format!("{}m {}s left", seconds / 60, seconds % 60),
        _ => format!("{}h {}m left", seconds / 3600, (seconds % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(file: &str, downloaded: u64, total: u64) -> Event {
        Event::Progress {
            file: file.to_string(),
            downloaded,
            total,
        }
    }

    #[test]
    fn totals_add_up_across_files() {
        let mut state = State::default();
        state.apply_event(&progress("a", 5, 10));
        state.apply_event(&progress("b", 1, 10));
        assert_eq!(state.totals(), (6, 20));
    }

    #[test]
    fn a_files_progress_updates_in_place() {
        let mut state = State::default();
        state.apply_event(&progress("a", 5, 10));
        state.apply_event(&progress("a", 9, 10));
        assert_eq!(state.lines.len(), 1);
        assert_eq!(state.totals(), (9, 10));
    }

    #[test]
    fn verifying_counts_as_complete_so_the_bar_never_goes_backwards() {
        let mut state = State::default();
        state.apply_event(&progress("a", 10, 10));
        let before = state.totals();
        state.apply_event(&Event::Verifying {
            file: "a".to_string(),
        });
        assert!(matches!(state.lines[0].1, FileStatus::Verifying));
        assert_eq!(before, (10, 10));
    }

    #[test]
    fn finished_only_once_every_file_completed() {
        let mut state = State::default();
        state.apply_event(&progress("a", 10, 10));
        state.apply_event(&progress("b", 1, 10));
        assert!(!state.finished());
        state.apply_event(&Event::Completed {
            file: "a".to_string(),
        });
        assert!(!state.finished());
        state.apply_event(&Event::Completed {
            file: "b".to_string(),
        });
        assert!(state.finished());
    }
}
