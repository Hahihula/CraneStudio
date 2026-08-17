//! The solver-led launch wizard (§4.4): the wizard leads with the answer —
//! the best reachable context for this model on this hardware — rather
//! than a blank form of knobs. Operates on local files directly (a `.gguf`
//! read via its own embedded metadata, or a safetensors dir via its
//! `config.json`), since that needs no network round trip and covers the
//! model the user already has on disk, which is the fastest path from
//! "never used this" to a running server.

use std::fs::File;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph, Wrap};
use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::{Format, KvQuant};
use studio_core::estimator::{
    Config as SolvedConfig, ModelConfig, SolveRequest, SolveResult, Suggestion, Variant,
    read_model_config_from_gguf, safetensors_dir_bytes, solve,
};
use studio_core::hardware::Backend;
use studio_core::launch::LaunchSpec;

use crate::app::{App, BackgroundEvent, Screen};

pub struct State {
    pub candidate: Option<LocalCandidate>,
    pub model_type: String,
    pub variant_label: String,
    pub backend: Backend,
    pub device: usize,
    pub result: Option<SolveResult>,
    pub error: Option<String>,
    pub selected_config: usize,
    pub launching: bool,
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
            launching: false,
        }
    }
}

/// Reads the model's own dimensions and runs the solver, entirely
/// synchronously — a GGUF header read or a `config.json` parse plus the
/// solver's arithmetic both complete in well under a frame's worth of
/// time, so there's no need to thread this through a background task.
pub fn load_local(app: &mut App, candidate: LocalCandidate) {
    app.wizard = State::default();

    let model_type = match &candidate.classification {
        Classification::Supported { model_type, .. } => (*model_type).to_string(),
        _ => {
            app.status_line = Some("that file isn't a Crane-supported architecture".to_string());
            return;
        }
    };

    let variant_label = candidate.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| candidate.path.display().to_string());

    app.wizard.candidate = Some(candidate.clone());
    app.wizard.model_type = model_type;
    app.wizard.variant_label = variant_label.clone();
    app.wizard.backend = app.hardware.backend;

    let cfg = match read_config(&candidate) {
        Ok(cfg) => cfg,
        Err(e) => {
            app.wizard.error = Some(e);
            return;
        }
    };

    let weight_bytes = match candidate.format {
        Format::Gguf => std::fs::metadata(&candidate.path).map(|m| m.len()).unwrap_or(0),
        Format::Safetensors => safetensors_dir_bytes(&candidate.path).unwrap_or(0),
    };
    if weight_bytes == 0 {
        app.wizard.error = Some("could not read this model's weight size on disk".to_string());
        return;
    }

    let (usable_vram, device) = app
        .hardware
        .gpus
        .iter()
        .enumerate()
        .max_by_key(|(_, g)| g.vram_free)
        .map_or((app.hardware.ram_available, 0), |(idx, g)| (g.vram_free, idx));
    app.wizard.device = device;

    let variants = vec![Variant { label: variant_label, weight_bytes, is_isq: false }];
    let request = SolveRequest {
        cfg: &cfg,
        variants: &variants,
        supports_kv_quant: cfg.hybrid.is_some(),
        // §2.11b: no family CraneStudio v1 targets supports kv-swap, so
        // concurrency is always pinned to 1.
        supports_kv_swap: false,
        native_context: cfg.max_position_embeddings,
        usable_vram,
        backend: app.hardware.backend,
        compute_dtype_bytes: 2.0,
        vision: false,
        max_concurrent: 1,
    };
    app.wizard.result = Some(solve(&request, 262_144));
}

fn read_config(candidate: &LocalCandidate) -> Result<ModelConfig, String> {
    match candidate.format {
        Format::Gguf => {
            let mut file = File::open(&candidate.path).map_err(|e| e.to_string())?;
            read_model_config_from_gguf(&mut file)
        }
        Format::Safetensors => {
            let text = std::fs::read_to_string(candidate.path.join("config.json")).map_err(|e| e.to_string())?;
            ModelConfig::parse(&text)
        }
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if app.wizard.launching {
        return true;
    }
    match key.code {
        KeyCode::Esc => {
            app.screen = Screen::Browser;
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if let Some(SolveResult::Reaches(configs)) = &app.wizard.result
                && !configs.is_empty()
            {
                app.wizard.selected_config = (app.wizard.selected_config + 1).min(configs.len() - 1);
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
        app.status_line = Some("no viable configuration to launch on this hardware".to_string());
        return;
    };
    let Some(candidate) = app.wizard.candidate.clone() else {
        return;
    };

    let port = 41100 + u16::try_from(app.last_running.len()).unwrap_or(0);
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

    app.wizard.launching = true;
    spawn_launch(app, spec, app.wizard.variant_label.clone());
}

fn spawn_launch(app: &App, spec: LaunchSpec, name: String) {
    let tx = app.sender();
    let control_base = app.control_base();
    let gateway_base = app.gateway_base();
    tokio::spawn(async move {
        let outcome = do_launch(&control_base, &gateway_base, &spec, &name).await;
        match outcome {
            Ok(id) => {
                let _ = tx.send(BackgroundEvent::Launched { id, name, port: spec.port });
            }
            Err(e) => {
                let _ = tx.send(BackgroundEvent::LaunchFailed(e));
            }
        }
    });
}

async fn do_launch(control_base: &str, gateway_base: &str, spec: &LaunchSpec, name: &str) -> Result<u64, String> {
    let client = reqwest::Client::new();

    let register = client.post(format!("{gateway_base}/register")).json(&serde_json::json!({"name": name, "spec": spec})).send().await.map_err(|e| e.to_string())?;
    if !register.status().is_success() {
        return Err(format!("register failed: {}", register.text().await.unwrap_or_default()));
    }

    let launch = client.post(format!("{control_base}/control/launch")).json(&serde_json::json!({"spec": spec, "label": name})).send().await.map_err(|e| e.to_string())?;
    if !launch.status().is_success() {
        return Err(format!("launch failed: {}", launch.text().await.unwrap_or_default()));
    }
    let body: serde_json::Value = launch.json().await.map_err(|e| e.to_string())?;
    Ok(body["id"].as_u64().unwrap_or(0))
}

pub fn render(app: &mut App, frame: &mut Frame) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(2), Constraint::Min(3), Constraint::Length(1)]).areas(frame.area());

    let title = app.wizard.candidate.as_ref().map_or("Launch wizard".to_string(), |c| format!("Launch wizard — {}", c.path.display()));
    frame.render_widget(Paragraph::new(Line::from(Span::styled(title, Style::new().add_modifier(Modifier::BOLD)))), header);

    if app.wizard.launching {
        frame.render_widget(Paragraph::new("launching…").block(Block::bordered()), body);
    } else if let Some(err) = &app.wizard.error {
        frame.render_widget(Paragraph::new(format!("could not evaluate this model: {err}")).wrap(Wrap { trim: false }).block(Block::bordered()), body);
    } else if let Some(result) = app.wizard.result.clone() {
        render_result(app, &result, frame, body);
    } else {
        frame.render_widget(Paragraph::new("no model selected").block(Block::bordered()), body);
    }

    let footer_text = if app.wizard.launching {
        "please wait…".to_string()
    } else {
        "[\u{2191}\u{2193}] choose   [Enter] launch   [Esc] back".to_string()
    };
    frame.render_widget(Paragraph::new(footer_text), footer);
}

fn render_result(app: &App, result: &SolveResult, frame: &mut Frame, area: ratatui::layout::Rect) {
    match result {
        SolveResult::Reaches(configs) => {
            let items: Vec<ListItem> = configs
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let style = if i == app.wizard.selected_config { Style::new().fg(Color::Black).bg(Color::Green) } else { Style::new() };
                    ListItem::new(Line::from(vec![Span::styled(describe_config(c), style)]))
                })
                .collect();
            frame.render_widget(List::new(items).block(Block::bordered().title("Reaches 256k — pick a configuration")), area);
        }
        SolveResult::Short { best, achieved_context, blockers } => {
            let mut lines = vec![Line::from(Span::styled(
                format!("Short of 256k — best reachable context is {achieved_context}"),
                Style::new().fg(Color::Yellow),
            ))];
            lines.push(Line::from(describe_config(best)));
            lines.push(Line::raw(""));
            lines.push(Line::from("What's using the VRAM:"));
            for b in blockers {
                lines.push(Line::from(format!("  {} — {}", b.description, crate::fmt::bytes(b.bytes))));
            }
            lines.push(Line::raw(""));
            lines.push(Line::from("[Enter] launch anyway at this context"));
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(Block::bordered().title("Short of target")), area);
        }
        SolveResult::Unusable { achieved_context, suggestions } => {
            let mut lines = vec![Line::from(Span::styled(
                format!("This model doesn't fit usably on this hardware (best: {achieved_context} context, below the {} floor).", studio_core::estimator::CONTEXT_FLOOR),
                Style::new().fg(Color::Red),
            ))];
            for s in suggestions {
                lines.push(Line::from(describe_suggestion(s)));
            }
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(Block::bordered().title("Unusable")), area);
        }
    }
}

fn describe_config(c: &SolvedConfig) -> String {
    let kv = c.kv_quant.map_or("kv fp16", kv_quant_label);
    format!(
        "{} — context {} — {kv} — {} concurrent — predicted {}",
        c.variant_label,
        c.context,
        c.concurrency,
        crate::fmt::bytes(c.predicted.total())
    )
}

fn kv_quant_label(q: KvQuant) -> &'static str {
    match q {
        KvQuant::Int8 => "kv int8",
        KvQuant::Int4 => "kv int4",
    }
}

fn describe_suggestion(s: &Suggestion) -> String {
    match s {
        Suggestion::SmallerVariant { label, achievable_context } => {
            format!("  try a smaller/more-quantized variant like {label} — achievable context: {achievable_context}")
        }
        Suggestion::NeedMoreVram { additional_bytes_for_floor } => {
            format!("  needs about {} more VRAM to reach the usable floor", crate::fmt::bytes(*additional_bytes_for_floor))
        }
    }
}

