//! The solver-led launch options screen (§4.4). Two ways in:
//!
//! * `quick_launch` — the launchpad's `Enter`. Solves, takes the solver's own
//!   best answer, and starts the model. No screen of its own; the user goes
//!   straight to `ready`.
//! * `load_local` + this screen — the launchpad's `c`. Same solve, but shows
//!   the answer and its alternatives first, because "which trade-off do I
//!   want" is a real question once you know the defaults exist.
//!
//! Either way the screen leads with an answer, never a blank form of knobs.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::fs::File;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, Paragraph};
use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::{Format, KvQuant};
use studio_core::estimator::{
    Config as SolvedConfig, ModelConfig, SolveRequest, SolveResult, Suggestion, Variant,
    read_model_config_from_gguf, safetensors_dir_bytes, solve,
};
use studio_core::hardware::Backend;
use studio_core::launch::LaunchSpec;
use studio_core::measurement::MeasurementDb;

use crate::app::{App, BackgroundEvent, Screen};
use crate::models::LocalModel;
use crate::theme::glyph;
use crate::ui::bars::{self, ratio};
use crate::ui::{self, Chrome, Message};

const HINTS: &[(&str, &str)] = &[
    ("↑↓", "choose"),
    ("⏎", "start"),
    ("esc", "back"),
    ("f2", "theme"),
];

pub struct State {
    pub candidate: Option<LocalCandidate>,
    pub model_type: String,
    pub variant_label: String,
    pub backend: Backend,
    pub device: usize,
    pub result: Option<SolveResult>,
    pub error: Option<String>,
    pub selected_config: usize,
}

impl Default for State {
    fn default() -> Self {
        State {
            candidate: None,
            model_type: String::new(),
            variant_label: String::new(),
            backend: Backend::Cpu,
            device: 0,
            result: None,
            error: None,
            selected_config: 0,
        }
    }
}

/// Reads the model's own dimensions and runs the solver, entirely
/// synchronously — a GGUF header read or a `config.json` parse plus the
/// solver's arithmetic both complete in well under a frame's worth of time, so
/// there's no need to thread this through a background task.
pub fn load_local(app: &mut App, candidate: &LocalCandidate) {
    app.wizard = State::default();
    app.message = None;

    let Classification::Supported { model_type, .. } = &candidate.classification else {
        app.message = Some(Message::error(
            "that file isn't a Crane-supported architecture",
        ));
        return;
    };

    let variant_label = candidate.path.file_name().map_or_else(
        || candidate.path.display().to_string(),
        |n| n.to_string_lossy().to_string(),
    );

    app.wizard.candidate = Some(candidate.clone());
    app.wizard.model_type = (*model_type).to_string();
    app.wizard.variant_label.clone_from(&variant_label);
    app.wizard.backend = app.hardware.backend;

    let (usable_vram, device) = usable_memory(app);
    app.wizard.device = device;

    match plan(candidate, &variant_label, app.hardware.backend, usable_vram) {
        Ok(result) => app.wizard.result = Some(result),
        Err(e) => app.wizard.error = Some(e),
    }
}

/// The launchpad's `Enter`: solve and start the best configuration in one step.
/// A model that doesn't fit is the one case that stops and explains itself.
pub fn quick_launch(app: &mut App, model: &LocalModel) {
    if !model.supported {
        app.message = Some(Message::error(model.reason.clone().unwrap_or_else(|| {
            "this model isn't a Crane-supported architecture".to_string()
        })));
        return;
    }

    load_local(app, &model.candidate);
    if let Some(err) = app.wizard.error.clone() {
        app.message = Some(Message::error(format!(
            "could not evaluate this model: {err}"
        )));
        return;
    }
    if matches!(app.wizard.result, Some(SolveResult::Unusable { .. })) {
        app.message = Some(Message::warn(
            "this model doesn't fit usably on this hardware — press [c] to see why",
        ));
        return;
    }
    attempt_launch(app);
}

/// Solves for the best reachable context, given what this machine can spare.
fn plan(
    candidate: &LocalCandidate,
    variant_label: &str,
    backend: Backend,
    usable_vram: u64,
) -> Result<SolveResult, String> {
    let cfg = read_config(candidate)?;
    let weight_bytes = match candidate.format {
        Format::Gguf => std::fs::metadata(&candidate.path).map_or(0, |m| m.len()),
        Format::Safetensors => safetensors_dir_bytes(&candidate.path).unwrap_or(0),
    };
    if weight_bytes == 0 {
        return Err("could not read this model's weight size on disk".to_string());
    }

    let variants = vec![Variant {
        label: variant_label.to_string(),
        weight_bytes,
        is_isq: false,
    }];
    let request = SolveRequest {
        cfg: &cfg,
        variants: &variants,
        supports_kv_quant: cfg.hybrid.is_some(),
        // §2.11b: no family CraneStudio v1 targets supports kv-swap, so
        // concurrency is always pinned to 1.
        supports_kv_swap: false,
        native_context: cfg.max_position_embeddings,
        usable_vram,
        backend,
        compute_dtype_bytes: 2.0,
        vision: false,
        max_concurrent: 1,
    };
    Ok(solve(&request, 262_144))
}

/// The memory the solver is allowed to plan inside, and which GPU it belongs
/// to. §7.3: once there's at least one local measurement, the global
/// measured÷predicted correction factor shrinks the budget — equivalent to
/// scaling every prediction up by the same factor, without touching the
/// estimator's own math.
fn usable_memory(app: &App) -> (u64, usize) {
    let (usable, device) = app
        .hardware
        .gpus
        .iter()
        .enumerate()
        .max_by_key(|(_, g)| g.vram_free)
        .map_or((app.hardware.ram_available, 0), |(idx, g)| {
            (g.vram_free, idx)
        });

    let db = MeasurementDb::load(&studio_core::paths::measurements_file());
    let usable = match db.correction_factor() {
        Some(factor) if factor > 0.0 => (usable as f64 / factor) as u64,
        _ => usable,
    };
    (usable, device)
}

/// The §7.3 measurement-DB key for one solved config — must match exactly
/// between the wizard's measured/predicted display and what gets sent along
/// with the actual launch, or a launch's own outcome would never be found by
/// the very config that produced it.
fn measurement_key_for(app: &App, config: &SolvedConfig) -> String {
    let backend_class = studio_core::measurement::backend_class(
        app.hardware.backend,
        app.hardware.gpus.get(app.wizard.device),
    );
    studio_core::measurement::build_key(
        &app.wizard.model_type,
        &app.wizard.variant_label,
        config.kv_quant,
        config.context,
        config.concurrency,
        &backend_class,
    )
}

fn read_config(candidate: &LocalCandidate) -> Result<ModelConfig, String> {
    match candidate.format {
        Format::Gguf => {
            let mut file = File::open(&candidate.path).map_err(|e| e.to_string())?;
            read_model_config_from_gguf(&mut file)
        }
        Format::Safetensors => {
            let text = std::fs::read_to_string(candidate.path.join("config.json"))
                .map_err(|e| e.to_string())?;
            ModelConfig::parse(&text)
        }
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => {
            app.screen = Screen::Launchpad;
            app.message = None;
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if let Some(SolveResult::Reaches(configs)) = &app.wizard.result
                && !configs.is_empty()
            {
                app.wizard.selected_config =
                    (app.wizard.selected_config + 1).min(configs.len() - 1);
            }
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.wizard.selected_config = app.wizard.selected_config.saturating_sub(1);
            true
        }
        KeyCode::Enter => {
            attempt_launch(app);
            true
        }
        _ => false,
    }
}

fn attempt_launch(app: &mut App) {
    let Some(result) = app.wizard.result.clone() else {
        return;
    };
    let chosen = match &result {
        SolveResult::Reaches(configs) => configs.get(app.wizard.selected_config).cloned(),
        SolveResult::Short { best, .. } => Some(best.clone()),
        SolveResult::Unusable { .. } => None,
    };
    let Some(config) = chosen else {
        app.message = Some(Message::error(
            "no viable configuration to launch on this hardware",
        ));
        return;
    };
    let Some(candidate) = app.wizard.candidate.clone() else {
        app.message = Some(Message::error("no model selected"));
        return;
    };

    let preferred_port = 41100 + u16::try_from(app.last_running.len()).unwrap_or(0);
    let port = studio_core::launch::pick_free_port(preferred_port, 50);
    let measurement_key = measurement_key_for(app, &config);
    let predicted_bytes = config.predicted.total();
    let spec = LaunchSpec {
        model_path: candidate.path.to_string_lossy().to_string(),
        model_type: app.wizard.model_type.clone(),
        model_name: Some(app.wizard.variant_label.clone()),
        port,
        cpu: matches!(app.wizard.backend, Backend::Cpu),
        max_concurrent: config.concurrency,
        decode_tokens_per_seq: 16,
        format: None,
        quant: None,
        dtype: None,
        max_seq_len: config.context,
        gpu_memory_limit: None,
        text_only: false,
        kv_quant: config.kv_quant,
        prefill_chunk: None,
        device: app.wizard.device,
    };

    // Straight to the ready screen with a "starting" state, rather than
    // freezing whichever screen asked for the launch: loading weights onto a
    // GPU takes tens of seconds, and that wait is exactly what that screen is
    // for.
    let label = app.wizard.variant_label.clone();
    app.ready
        .begin(label.clone(), port, app.gateway_port, false);
    app.ready.context = Some(config.context);
    app.screen = Screen::Ready;
    app.message = None;
    spawn_launch(
        app,
        spec,
        label,
        false,
        Some((measurement_key, predicted_bytes)),
    );
}

/// Launches a speech model directly, skipping the VRAM solver and the wizard.
pub fn launch_tts(app: &mut App, model: &LocalModel) {
    let Some(model_type) = model.model_type.clone() else {
        app.message = Some(Message::error(
            "this model isn't a Crane-supported architecture",
        ));
        return;
    };

    let name = model
        .path()
        .file_name()
        .map_or_else(|| model.name.clone(), |n| n.to_string_lossy().to_string());

    let (_, device) = usable_memory(app);
    let preferred_port = 41100 + u16::try_from(app.last_running.len()).unwrap_or(0);
    let port = studio_core::launch::pick_free_port(preferred_port, 50);

    let spec = LaunchSpec {
        model_path: model.path().to_string_lossy().to_string(),
        model_type,
        model_name: Some(name.clone()),
        port,
        cpu: matches!(app.hardware.backend, Backend::Cpu),
        max_concurrent: 1,
        decode_tokens_per_seq: 16,
        format: None,
        quant: None,
        dtype: None,
        max_seq_len: 0,
        gpu_memory_limit: None,
        text_only: false,
        kv_quant: None,
        prefill_chunk: None,
        device,
    };

    app.ready.begin(name.clone(), port, app.gateway_port, true);
    app.ready.context = None;
    app.screen = Screen::Ready;
    app.message = None;
    spawn_launch(app, spec, name, true, None);
}

fn spawn_launch(
    app: &App,
    spec: LaunchSpec,
    name: String,
    is_tts: bool,
    measured: Option<(String, u64)>,
) {
    let tx = app.sender();
    let control_base = app.control_base();
    let gateway_base = app.gateway_base();
    tokio::spawn(async move {
        let outcome = do_launch(&control_base, &gateway_base, &spec, &name, measured).await;
        match outcome {
            Ok(id) => {
                let _ = tx.send(BackgroundEvent::Launched {
                    id,
                    name,
                    port: spec.port,
                    is_tts,
                });
            }
            Err(e) => {
                let _ = tx.send(BackgroundEvent::LaunchFailed(e));
            }
        }
    });
}

async fn do_launch(
    control_base: &str,
    gateway_base: &str,
    spec: &LaunchSpec,
    name: &str,
    measured: Option<(String, u64)>,
) -> Result<u64, String> {
    let client = reqwest::Client::new();

    let register = client
        .post(format!("{gateway_base}/register"))
        .json(&serde_json::json!({"name": name, "spec": spec}))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !register.status().is_success() {
        return Err(format!(
            "register failed: {}",
            register.text().await.unwrap_or_default()
        ));
    }

    // Only a solver-led launch carries a measurement key + prediction.
    let mut launch_body = serde_json::json!({"spec": spec, "label": name});
    if let Some((key, predicted_bytes)) = measured {
        launch_body["measurement_key"] = serde_json::json!(key);
        launch_body["predicted_bytes"] = serde_json::json!(predicted_bytes);
    }
    let launch = client
        .post(format!("{control_base}/control/launch"))
        .json(&launch_body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !launch.status().is_success() {
        return Err(format!(
            "launch failed: {}",
            launch.text().await.unwrap_or_default()
        ));
    }
    let body: serde_json::Value = launch.json().await.map_err(|e| e.to_string())?;
    Ok(body["id"].as_u64().unwrap_or(0))
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let chrome = Chrome::new(HINTS)
        .crumb("Launch options")
        .crumb(crate::ui::text::truncate(&app.wizard.variant_label, 28))
        .status(crate::screens::hardware::status_spans(app))
        .message(app.message.clone());
    let body = ui::shell(frame, &app.theme, &chrome);

    if let Some(err) = app.wizard.error.clone() {
        let block = app.theme.block("Can't evaluate this model");
        let inner = block.inner(body);
        frame.render_widget(block, body);
        frame.render_widget(
            Paragraph::new(
                crate::ui::text::wrap(&err, inner.width)
                    .into_iter()
                    .map(|row| Line::from(Span::styled(row, Style::new().fg(app.theme.error))))
                    .collect::<Vec<_>>(),
            ),
            inner,
        );
        return;
    }

    let Some(result) = app.wizard.result.clone() else {
        let block = app.theme.block("");
        frame.render_widget(Paragraph::new("no model selected").block(block), body);
        return;
    };

    match &result {
        SolveResult::Reaches(configs) => render_reaches(app, configs, frame, body),
        SolveResult::Short {
            best,
            achieved_context,
            blockers,
        } => render_short(app, best, *achieved_context, blockers, frame, body),
        SolveResult::Unusable {
            achieved_context,
            suggestions,
        } => render_unusable(app, *achieved_context, suggestions, frame, body),
    }
}

/// Lead with the recommendation as a card, and list the alternatives under it
/// — the shape §4.4 asks for: an answer first, knobs second.
fn render_reaches(app: &App, configs: &[SolvedConfig], frame: &mut Frame, area: Rect) {
    let db = MeasurementDb::load(&studio_core::paths::measurements_file());
    let selected = app
        .wizard
        .selected_config
        .min(configs.len().saturating_sub(1));
    let Some(chosen) = configs.get(selected) else {
        return;
    };

    #[allow(clippy::cast_possible_truncation)]
    let alternatives_height = (configs.len() as u16).saturating_add(2);
    let [headline_area, alternatives_area, _] = Layout::vertical([
        Constraint::Length(10),
        Constraint::Length(alternatives_height),
        Constraint::Min(0),
    ])
    .areas(area);

    let block = app.theme.block_focused("Recommended", true);
    let inner = block.inner(headline_area);
    frame.render_widget(block, headline_area);
    frame.render_widget(
        Paragraph::new(config_card(app, &db, chosen, inner.width)),
        inner,
    );

    let items: Vec<ListItem> = configs
        .iter()
        .enumerate()
        .map(|(i, config)| {
            let marker = if i == selected {
                Span::styled(
                    format!("{} ", glyph::BAR_HALF),
                    Style::new().fg(app.theme.accent),
                )
            } else {
                Span::raw("  ")
            };
            ListItem::new(Line::from(vec![
                marker,
                Span::styled(
                    one_line_config(config),
                    if i == selected {
                        Style::new().fg(app.theme.accent)
                    } else {
                        Style::new().fg(app.theme.text)
                    },
                ),
                Span::styled(
                    format!("   {}", size_note(app, &db, config)),
                    app.theme.muted_style(),
                ),
            ]))
        })
        .collect();
    let block = app
        .theme
        .block(format!("Alternatives  ·  {}", configs.len()));
    let inner = block.inner(alternatives_area);
    frame.render_widget(block, alternatives_area);
    frame.render_widget(List::new(items), inner);
}

fn render_short(
    app: &App,
    best: &SolvedConfig,
    achieved_context: usize,
    blockers: &[studio_core::estimator::Blocker],
    frame: &mut Frame,
    area: Rect,
) {
    let db = MeasurementDb::load(&studio_core::paths::measurements_file());
    let block = app
        .theme
        .block_focused("Short of the 256k target — still worth running", true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{} ", glyph::WARN),
                Style::new().fg(app.theme.warning),
            ),
            Span::styled(
                format!("best reachable context is {achieved_context}"),
                Style::new()
                    .fg(app.theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
    ];
    lines.extend(config_card(app, &db, best, inner.width));
    lines.push(Line::raw(""));
    lines.push(ui::section(&app.theme, "what's using the memory"));
    let total: u64 = blockers.iter().map(|b| b.bytes).sum();
    for blocker in blockers {
        let mut spans = vec![Span::styled(
            format!(
                "{:<18}",
                crate::ui::text::truncate(&blocker.description, 17)
            ),
            app.theme.muted_style(),
        )];
        spans.extend(bars::progress_bar(
            &app.theme,
            18,
            ratio(blocker.bytes, total),
            app.theme.accent_alt,
        ));
        spans.push(Span::styled(
            format!("  {}", crate::fmt::bytes(blocker.bytes)),
            Style::new().fg(app.theme.text),
        ));
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_unusable(
    app: &App,
    achieved_context: usize,
    suggestions: &[Suggestion],
    frame: &mut Frame,
    area: Rect,
) {
    let block = app.theme.block("Doesn't fit on this hardware");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = Vec::new();
    for row in crate::ui::text::wrap(
        &format!(
            "The best this machine can do for this model is {achieved_context} tokens of context, below the {} floor a usable session needs.",
            studio_core::estimator::CONTEXT_FLOOR
        ),
        inner.width,
    ) {
        lines.push(Line::from(Span::styled(
            row,
            Style::new().fg(app.theme.error),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(ui::section(&app.theme, "what would help"));
    for suggestion in suggestions {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", glyph::ARROW),
                Style::new().fg(app.theme.accent),
            ),
            Span::styled(
                describe_suggestion(suggestion),
                Style::new().fg(app.theme.text),
            ),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The recommendation card: what will run, how much memory it's expected to
/// take, and — the §4.4 rule — whether that number is predicted or measured.
fn config_card(
    app: &App,
    db: &MeasurementDb,
    config: &SolvedConfig,
    width: u16,
) -> Vec<Line<'static>> {
    let key = measurement_key_for(app, config);
    let measured = db.latest_successful_for(&key);
    let bytes = measured.map_or_else(|| config.predicted.total(), |m| m.measured_peak_bytes);
    let (usable, _) = usable_memory(app);
    let fill = ratio(bytes, usable.max(1));

    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{} ", glyph::DONE),
                Style::new().fg(app.theme.success),
            ),
            Span::styled(
                crate::ui::text::truncate(&config.variant_label, 46),
                Style::new().fg(app.theme.text).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
        ui::field(&app.theme, "context", config.context.to_string()),
        ui::field(
            &app.theme,
            "kv cache",
            config.kv_quant.map_or("fp16", kv_quant_label),
        ),
        ui::field(
            &app.theme,
            "sequences",
            format!("{} concurrent", config.concurrency),
        ),
        Line::raw(""),
    ];

    let bar_width = width.saturating_sub(34).clamp(10, 30);
    let mut spans = vec![Span::styled(
        format!(
            "{:<13}  ",
            if measured.is_some() {
                "measured"
            } else {
                "predicted"
            }
        ),
        app.theme.muted_style(),
    )];
    spans.extend(bars::load_bar(&app.theme, bar_width, fill));
    spans.push(Span::styled(
        format!(
            "  {} of {}",
            crate::fmt::bytes(bytes),
            crate::fmt::bytes(usable)
        ),
        Style::new().fg(app.theme.text),
    ));
    lines.push(Line::from(spans));

    if db.prior_oom_for(&key).is_some() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", glyph::WARN),
                Style::new().fg(app.theme.warning),
            ),
            Span::styled(
                "this exact configuration ran out of memory last time it was tried",
                Style::new().fg(app.theme.warning),
            ),
        ]));
    }
    lines
}

fn one_line_config(config: &SolvedConfig) -> String {
    format!(
        "context {:<7}  kv {:<5}  {} concurrent",
        config.context,
        config.kv_quant.map_or("fp16", kv_quant_label),
        config.concurrency
    )
}

fn size_note(app: &App, db: &MeasurementDb, config: &SolvedConfig) -> String {
    let key = measurement_key_for(app, config);
    db.latest_successful_for(&key).map_or_else(
        || format!("predicted {}", crate::fmt::bytes(config.predicted.total())),
        |m| format!("measured {}", crate::fmt::bytes(m.measured_peak_bytes)),
    )
}

fn kv_quant_label(quant: KvQuant) -> &'static str {
    match quant {
        KvQuant::Int8 => "int8",
        KvQuant::Int4 => "int4",
    }
}

fn describe_suggestion(suggestion: &Suggestion) -> String {
    match suggestion {
        Suggestion::SmallerVariant {
            label,
            achievable_context,
        } => format!(
            "try a smaller or more heavily quantized variant like {label} — that reaches {achievable_context}"
        ),
        Suggestion::NeedMoreVram {
            additional_bytes_for_floor,
        } => format!(
            "about {} more free VRAM would reach the usable floor",
            crate::fmt::bytes(*additional_bytes_for_floor)
        ),
    }
}
