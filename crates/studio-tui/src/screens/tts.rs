//! The TTS Playground app (§4.7): type text, POST it to the gateway's
//! `/v1/audio/speech` (a single non-streaming response), and play the returned
//! WAV back in-process. Text-only — `VoxCPM2` ignores sampling params here.

use std::path::PathBuf;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use tokio::task::AbortHandle;

use crate::app::{App, BackgroundEvent, Screen};
use crate::theme::glyph;
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[
    ("⏎", "generate"),
    ("space", "play / stop"),
    ("↑↓", "select clip"),
    ("^r", "restart voice"),
    ("^n", "clear"),
    ("esc", "back"),
];
const GENERATING_HINTS: &[(&str, &str)] = &[("esc", "cancel"), ("^c", "quit")];

/// Length cap sent with every request so generation always terminates.
const DEFAULT_MAX_LEN: usize = 2000;

/// One generated clip, kept for replay and as a visible transcript.
#[derive(Debug, Clone)]
pub struct Clip {
    pub text: String,
    pub wav_path: PathBuf,
    pub sample_rate: u32,
    pub bytes: u64,
    pub duration_secs: f64,
    /// Wall-clock generation time.
    pub gen_secs: f64,
}

#[derive(Default)]
pub struct State {
    pub input: String,
    pub clips: Vec<Clip>,
    pub selected: usize,
    pub generating: bool,
    pub gen_started: Option<Instant>,
    pub error: Option<String>,
    gen_task: Option<AbortHandle>,
    /// Playback thread, created lazily on the first `play`.
    player: Option<crate::audio::Player>,
    pub now_playing: Option<usize>,
}

impl State {
    /// Records a finished clip and selects it.
    pub fn on_generated(&mut self, clip: Clip) {
        self.clips.push(clip);
        self.selected = self.clips.len() - 1;
        self.generating = false;
        self.gen_task = None;
        self.gen_started = None;
        self.error = None;
    }

    /// Plays the most recent clip.
    pub fn play_latest(&mut self) {
        if !self.clips.is_empty() {
            self.play(self.clips.len() - 1);
        }
    }

    pub fn fail(&mut self, err: &str) {
        self.generating = false;
        self.gen_task = None;
        self.gen_started = None;
        self.error = Some(err.to_string());
    }

    /// Seconds elapsed since generation started, if generating.
    #[must_use]
    pub fn elapsed_secs(&self) -> Option<u64> {
        self.gen_started.map(|t| t.elapsed().as_secs())
    }

    fn play(&mut self, index: usize) {
        let Some(clip) = self.clips.get(index) else {
            return;
        };
        let handle = self.player.get_or_insert_with(crate::audio::Player::new);
        if handle.play(&clip.wav_path) {
            self.now_playing = Some(index);
        } else {
            self.error = Some(format!(
                "no audio output device — saved to {}",
                clip.wav_path.display()
            ));
        }
    }

    fn stop(&mut self) {
        if let Some(handle) = &self.player {
            handle.stop();
        }
        self.now_playing = None;
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => {
            if app.tts.generating {
                if let Some(task) = app.tts.gen_task.take() {
                    task.abort();
                }
                app.tts.generating = false;
                app.tts.gen_started = None;
                app.message = Some(Message::info("cancelled"));
            } else {
                app.tts.stop();
                app.screen = Screen::Ready;
            }
            true
        }
        KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.tts.stop();
            app.tts.clips.clear();
            app.tts.selected = 0;
            app.tts.error = None;
            true
        }
        KeyCode::Char('r')
            if key.modifiers.contains(KeyModifiers::CONTROL) && !app.tts.generating =>
        {
            restart_voice(app);
            true
        }
        _ if app.tts.generating => true,
        KeyCode::Enter => {
            generate(app);
            true
        }
        // `space` selects/plays only when the input box is empty.
        KeyCode::Up => {
            app.tts.selected = app.tts.selected.saturating_sub(1);
            true
        }
        KeyCode::Down => {
            let last = app.tts.clips.len().saturating_sub(1);
            app.tts.selected = (app.tts.selected + 1).min(last);
            true
        }
        KeyCode::Char(' ') if app.tts.input.is_empty() => {
            toggle_play(app);
            true
        }
        KeyCode::Backspace => {
            app.tts.input.pop();
            true
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.tts.input.push(c);
            true
        }
        _ => false,
    }
}

fn toggle_play(app: &mut App) {
    if app.tts.clips.is_empty() {
        return;
    }
    let selected = app.tts.selected;
    if app.tts.now_playing == Some(selected) {
        app.tts.stop();
    } else {
        app.tts.play(selected);
    }
}

/// Asks the gateway to replace this model's crane-serve process; the next
/// generate spawns the fresh child.
fn restart_voice(app: &mut App) {
    let model = app.ready.name.clone();
    let gateway_base = app.gateway_base();
    app.tts.stop();
    app.ready.id = None;
    app.message = Some(Message::info(
        "restarting the voice — your next take starts from a fresh process",
    ));
    tokio::spawn(async move {
        let _ = reqwest::Client::new()
            .post(format!("{gateway_base}/restart"))
            .json(&serde_json::json!({ "name": model }))
            .send()
            .await;
    });
}

fn generate(app: &mut App) {
    let text = app.tts.input.trim().to_string();
    if text.is_empty() {
        return;
    }
    app.tts.input.clear();
    app.tts.error = None;
    app.tts.generating = true;
    app.tts.gen_started = Some(Instant::now());

    let model = app.ready.name.clone();
    let gateway_base = app.gateway_base();
    let tx = app.sender();

    let task = tokio::spawn(async move {
        let started = Instant::now();
        match synth(&gateway_base, &model, &text).await {
            Ok(bytes) => match save_clip(&text, &bytes, started.elapsed().as_secs_f64()) {
                Ok(clip) => {
                    let _ = tx.send(BackgroundEvent::TtsGenerated(Box::new(clip)));
                }
                Err(e) => {
                    let _ = tx.send(BackgroundEvent::TtsError(e));
                }
            },
            Err(e) => {
                let _ = tx.send(BackgroundEvent::TtsError(e));
            }
        }
    });
    app.tts.gen_task = Some(task.abort_handle());
}

async fn synth(gateway_base: &str, model: &str, text: &str) -> Result<Vec<u8>, String> {
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{gateway_base}/v1/audio/speech"))
        .json(&serde_json::json!({
            "model": model,
            "input": text,
            "response_format": "wav",
            "max_tokens": DEFAULT_MAX_LEN,
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or(body);
        return Err(format!("speech request failed: {message}"));
    }

    Ok(response.bytes().await.map_err(|e| e.to_string())?.to_vec())
}

/// Writes the WAV under `<data_dir>/tts/` and reads back its header.
fn save_clip(text: &str, bytes: &[u8], gen_secs: f64) -> Result<Clip, String> {
    let dir = studio_core::paths::data_dir().join("tts");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let stem: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .take(40)
        .collect();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{}-{ts}.wav", stem.trim_matches('-')));
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;

    let (sample_rate, duration_secs) = match hound::WavReader::open(&path) {
        Ok(reader) => {
            let sr = reader.spec().sample_rate;
            let frames = f64::from(reader.duration());
            (sr, if sr > 0 { frames / f64::from(sr) } else { 0.0 })
        }
        Err(_) => (0, 0.0),
    };

    Ok(Clip {
        text: text.to_string(),
        wav_path: path,
        sample_rate,
        bytes: bytes.len() as u64,
        duration_secs,
        gen_secs,
    })
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let hints = if app.tts.generating {
        GENERATING_HINTS
    } else {
        HINTS
    };
    let chrome = Chrome::new(hints)
        .crumb("TTS Playground")
        .crumb(crate::ui::text::truncate(&app.ready.name, 24))
        .status(status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    let [transcript, input] =
        Layout::vertical([Constraint::Min(4), Constraint::Length(3)]).areas(body);
    render_clips(app, frame, transcript);
    render_input(app, frame, input);
}

fn status_spans(app: &App) -> Vec<Span<'static>> {
    if app.tts.generating {
        let secs = app.tts.elapsed_secs().unwrap_or(0);
        return vec![Span::styled(
            format!("{} generating {secs}s", crate::theme::spinner(app.tick)),
            Style::new()
                .fg(app.theme.warning)
                .add_modifier(Modifier::BOLD),
        )];
    }
    let n = app.tts.clips.len();
    vec![Span::styled(
        format!("{n} clip{}", if n == 1 { "" } else { "s" }),
        app.theme.muted_style(),
    )]
}

fn render_clips(app: &App, frame: &mut Frame, area: Rect) {
    let block = app.theme.block("Clips");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.tts.clips.is_empty() && app.tts.error.is_none() {
        let lines = vec![
            Line::raw(""),
            Line::from(Span::styled(
                format!(
                    "{} type something for {} to say",
                    glyph::ARROW,
                    app.ready.name
                ),
                app.theme.muted_style(),
            )),
            Line::from(Span::styled(
                "  ⏎ generate · space plays the result",
                app.theme
                    .muted_style()
                    .add_modifier(Modifier::DIM | Modifier::ITALIC),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        return;
    }

    let text_width = inner.width.saturating_sub(4).max(8);
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, clip) in app.tts.clips.iter().enumerate() {
        let selected = i == app.tts.selected;
        let playing = app.tts.now_playing == Some(i);
        let (marker, color) = if playing {
            ("▶ ", app.theme.accent)
        } else if selected {
            ("▌ ", app.theme.accent)
        } else {
            ("• ", app.theme.muted)
        };
        lines.push(Line::from(vec![
            Span::styled(marker.to_string(), Style::new().fg(color)),
            Span::styled(
                format!("clip {}", i + 1),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "   {:.1}s audio · generated in {:.1}s",
                    clip.duration_secs, clip.gen_secs
                ),
                app.theme.muted_style(),
            ),
        ]));
        for row in crate::ui::text::wrap(&clip.text, text_width) {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::new()),
                Span::styled(format!(" {row}"), Style::new().fg(app.theme.text)),
            ]));
        }
        lines.push(Line::from(Span::styled(
            format!(
                "  {}",
                crate::ui::text::truncate_start(
                    &clip.wav_path.display().to_string(),
                    text_width as usize
                )
            ),
            app.theme.muted_style().add_modifier(Modifier::DIM),
        )));
        lines.push(Line::raw(""));
    }

    if let Some(err) = &app.tts.error {
        for row in crate::ui::text::wrap(err, text_width) {
            lines.push(Line::from(Span::styled(
                format!("{} {row}", glyph::FAILED),
                Style::new().fg(app.theme.error),
            )));
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let total = lines.len() as u16;
    let scroll = total.saturating_sub(inner.height);
    frame.render_widget(Paragraph::new(lines).scroll((scroll, 0)), inner);
}

fn render_input(app: &App, frame: &mut Frame, area: Rect) {
    let (title, focused) = if app.tts.generating {
        (
            format!("{} generating", crate::theme::spinner(app.tick)),
            false,
        )
    } else {
        ("text to speak".to_string(), true)
    };
    let block = app.theme.block_focused(title, focused);
    let inner = block.inner(area);

    let mut footnote: Vec<Span<'static>> = Vec::new();
    if let Some(i) = app.tts.now_playing {
        footnote.push(Span::styled(
            format!(" ▶ playing clip {} ", i + 1),
            Style::new().fg(app.theme.accent),
        ));
    }
    frame.render_widget(
        block.title_bottom(Line::from(footnote).right_aligned()),
        area,
    );

    let prompt = Span::styled(
        format!("{} ", glyph::ARROW),
        Style::new().fg(app.theme.accent),
    );
    let visible = inner.width.saturating_sub(2) as usize;
    let text: String = if app.tts.input.chars().count() > visible {
        app.tts
            .input
            .chars()
            .skip(app.tts.input.chars().count() - visible)
            .collect()
    } else {
        app.tts.input.clone()
    };
    #[allow(clippy::cast_possible_truncation)]
    let cursor_x = inner.x + 2 + text.chars().count() as u16;
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            prompt,
            Span::styled(text, Style::new().fg(app.theme.text)),
        ])),
        inner,
    );
    if !app.tts.generating {
        frame.set_cursor_position((cursor_x.min(inner.right().saturating_sub(1)), inner.y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(text: &str) -> Clip {
        Clip {
            text: text.to_string(),
            wav_path: PathBuf::from("/tmp/x.wav"),
            sample_rate: 16_000,
            bytes: 1024,
            duration_secs: 1.5,
            gen_secs: 2.0,
        }
    }

    #[test]
    fn a_generated_clip_becomes_selected_and_ends_generation() {
        let mut state = State {
            generating: true,
            ..State::default()
        };
        state.on_generated(clip("hello"));
        assert!(!state.generating);
        assert_eq!(state.selected, 0);
        assert_eq!(state.clips.len(), 1);
    }

    #[test]
    fn a_failure_surfaces_and_clears_the_generating_flag() {
        let mut state = State {
            generating: true,
            ..State::default()
        };
        state.fail("boom");
        assert!(!state.generating);
        assert_eq!(state.error.as_deref(), Some("boom"));
    }
}
