//! Wires the browser's Catalog and Search tabs to the same real downloader
//! `cranestudio download` already uses (§9) — the piece those tabs were
//! missing in M7: browsing without any way to actually get a model onto
//! disk before the wizard can do anything with it.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::text::Span;
use ratatui::widgets::{Gauge, Paragraph, Wrap};
use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::{Format, ModelEntry};
use studio_core::download::{CancellationToken, Event, RepoDownload, download_repo};

use crate::app::{App, BackgroundEvent, Screen};

/// A single file's download state, kept structured (not pre-formatted text)
/// so `render` can drive a real progress bar instead of a literal percent
/// string.
#[derive(Debug, Clone)]
pub enum FileStatus {
    Starting { resumed_from: u64, total: u64 },
    Progress { downloaded: u64, total: u64 },
    Verifying,
    Completed,
    Cancelled,
}

#[derive(Default)]
pub struct State {
    pub label: String,
    /// One entry per file — `Progress` events update a file's own entry in
    /// place rather than appending, so a multi-file download doesn't
    /// scroll past its own progress.
    pub lines: Vec<(String, FileStatus)>,
    pub error: Option<String>,
    pub cancel: Option<CancellationToken>,
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
}

#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64).clamp(0.0, 1.0)
    }
}

/// Picks the smallest variant (fastest path to something running) and
/// downloads it — the wizard's own "not a form" philosophy (§4.4) applied
/// one step earlier, to the download itself.
pub fn start_catalog(app: &mut App, model: &ModelEntry) {
    let Some(variant) = model.variants.iter().min_by_key(|v| v.download_bytes) else {
        app.status_line = Some(format!(
            "{} has no downloadable variants",
            model.display_name
        ));
        return;
    };
    start(
        app,
        format!("{} ({})", model.display_name, variant.id),
        variant.repo.clone(),
        variant.revision.clone(),
        variant.files.clone(),
        // Trust the catalog's own verified model_type directly rather than
        // re-classifying the download afterward — some families (MiniCPM5)
        // are deliberately never auto-detected at all (their config/GGUF
        // header is indistinguishable from real Llama; see
        // `catalog::architecture`'s `minicpm5` entry), so re-scanning would
        // wrongly reject a model the catalog already knows is supported.
        Some((model.model_type.clone(), variant.format)),
    );
}

/// `main` rather than a pinned commit sha — acceptable for an ad-hoc search
/// download (§5's "pin to a commit sha" concern is about a *saved profile*
/// silently changing meaning later, which doesn't apply here since nothing
/// is saved). No known `model_type` here — the search result's own
/// classification already came from re-scannable config/GGUF-header
/// detection, so re-scanning after download is consistent, unlike the
/// catalog case above.
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
/// `model_type`, without touching the downloaded file's contents — the
/// point of this path (see `start_catalog`'s doc comment).
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
        cancel: Some(cancel.clone()),
        ..State::default()
    };
    app.screen = Screen::Download;
    app.status_line = None;

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
    // Swallow everything else while a download is actually in flight; once
    // it's failed, let global navigation (h/b/q/…) through again.
    app.download.error.is_none()
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(Span::styled(
            format!("Downloading — {}", app.download.label),
            app.theme.header_style(),
        )),
        header,
    );

    if let Some(err) = &app.download.error {
        frame.render_widget(
            Paragraph::new(format!("download failed: {err}"))
                .wrap(Wrap { trim: false })
                .style(Style::new().fg(app.theme.error))
                .block(app.theme.block("Failed")),
            body,
        );
    } else {
        let block = app.theme.block("Progress");
        let inner = block.inner(body);
        frame.render_widget(block, body);
        if app.download.lines.is_empty() {
            frame.render_widget(
                Paragraph::new(format!("{} starting…", crate::theme::spinner(app.tick))),
                inner,
            );
        } else {
            let rows = Layout::vertical(vec![Constraint::Length(1); app.download.lines.len()])
                .split(inner);
            for ((file, status), row) in app.download.lines.iter().zip(rows.iter()) {
                render_file_row(&app.theme, app.tick, file, status, frame, *row);
            }
        }
    }

    let footer_text = if app.download.error.is_some() {
        "[Esc] back"
    } else {
        "[Esc] cancel"
    };
    frame.render_widget(
        Paragraph::new(footer_text).style(app.theme.muted_style()),
        footer,
    );
}

fn render_file_row(
    theme: &crate::theme::Theme,
    tick: u64,
    file: &str,
    status: &FileStatus,
    frame: &mut Frame,
    area: ratatui::layout::Rect,
) {
    match status {
        FileStatus::Starting {
            resumed_from,
            total,
        } => {
            let label = if *resumed_from > 0 {
                format!(
                    "{file}: resuming from {} of {}",
                    crate::fmt::bytes(*resumed_from),
                    crate::fmt::bytes(*total)
                )
            } else {
                format!("{file}: starting ({})", crate::fmt::bytes(*total))
            };
            frame.render_widget(
                Gauge::default()
                    .ratio(ratio(*resumed_from, *total))
                    .label(label)
                    .gauge_style(Style::new().fg(theme.accent)),
                area,
            );
        }
        FileStatus::Progress { downloaded, total } => {
            let percent = ratio(*downloaded, *total) * 100.0;
            let label = format!(
                "{file}: {} / {} ({percent:.1}%)",
                crate::fmt::bytes(*downloaded),
                crate::fmt::bytes(*total)
            );
            frame.render_widget(
                Gauge::default()
                    .ratio(ratio(*downloaded, *total))
                    .label(label)
                    .gauge_style(Style::new().fg(theme.accent)),
                area,
            );
        }
        FileStatus::Verifying => {
            frame.render_widget(
                Paragraph::new(format!(
                    "{} {file}: verifying checksum…",
                    crate::theme::spinner(tick)
                ))
                .style(Style::new().fg(theme.warning)),
                area,
            );
        }
        FileStatus::Completed => {
            frame.render_widget(
                Gauge::default()
                    .ratio(1.0)
                    .label(format!("{file}: done"))
                    .gauge_style(Style::new().fg(theme.success)),
                area,
            );
        }
        FileStatus::Cancelled => {
            frame.render_widget(
                Paragraph::new(format!("{file}: cancelled")).style(Style::new().fg(theme.muted)),
                area,
            );
        }
    }
}
