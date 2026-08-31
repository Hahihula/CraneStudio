//! Rendering previews. Every screen is drawn into a `TestBackend` with
//! plausible fake state, which does two jobs: it's a regression test that no
//! screen panics at the sizes people actually use (including an 80×24 terminal
//! and a 40-column one), and — run with `--nocapture` — it prints the frames so
//! a layout can be *looked at* without launching a model.
//!
//! ```text
//! cargo test -p studio-tui --lib preview -- --nocapture
//! ```

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::Format;
use studio_core::estimator::{ModelConfig, SolveRequest, Variant, solve};
use studio_core::hardware::{Backend, CpuInfo, DiskInfo, GpuInfo, HardwareReport, Sample};

use crate::app::{App, History, QuitChoice, Screen};
use crate::daemon_client::{ChildInfo, ChildState, ChildSummary};
use crate::models::LocalModel;
use crate::screens;

const GIB: u64 = 1024 * 1024 * 1024;

pub(crate) fn hardware() -> HardwareReport {
    HardwareReport {
        gpus: vec![GpuInfo {
            index: 0,
            name: "NVIDIA GeForce RTX 3090".to_string(),
            vram_total: 24 * GIB,
            vram_free: 17 * GIB,
            compute_capability: Some((8, 6)),
            driver_version: Some("560.35.03".to_string()),
            unified_memory: false,
        }],
        cpu: CpuInfo {
            model: "AMD Ryzen 9 5900X 12-Core Processor".to_string(),
            physical_cores: 12,
            logical_cores: 24,
        },
        ram_total: 32 * GIB,
        ram_available: 13 * GIB,
        disk: vec![DiskInfo {
            mount_point: "/home".to_string(),
            total: 1800 * GIB,
            available: 712 * GIB,
        }],
        backend: Backend::Cuda,
    }
}

pub(crate) fn sample() -> Sample {
    #[allow(clippy::cast_precision_loss)]
    let per_core = (0..24)
        .map(|i| ((i * 17) % 100) as f32)
        .collect::<Vec<f32>>();
    Sample {
        cpu_total: 38.5,
        per_core,
        ram_total: 32 * GIB,
        ram_available: 13 * GIB,
        gpus: hardware().gpus,
    }
}

pub(crate) fn history() -> History {
    let mut history = History::default();
    for i in 0..90 {
        let phase = f64::from(i) / 7.0;
        history.cpu.push((phase.sin() * 0.4 + 0.45).clamp(0.0, 1.0));
        history.ram.push(0.55 + phase.cos() * 0.05);
        history.vram.push(0.28 + phase.sin() * 0.03);
    }
    history
}

fn local(
    name: &str,
    model_type: &str,
    size: u64,
    quant: Option<&str>,
    supported: bool,
) -> LocalModel {
    let path = std::path::PathBuf::from(format!(
        "/home/dev/.local/share/cranestudio/models/unsloth/{name}-GGUF/main/{name}.gguf"
    ));
    LocalModel {
        candidate: LocalCandidate {
            path,
            format: Format::Gguf,
            classification: if supported {
                Classification::Supported {
                    // A borrowed &'static str in the real type; the previews only
                    // need it to be *a* family name, and the row shows
                    // `model_type` below anyway.
                    model_type: "qwen3_5",
                    vision: false,
                    gated: false,
                    audio: false,
                }
            } else {
                Classification::Unsupported {
                    detected: Some("phi3".to_string()),
                    reason: "phi3 is not a Crane-supported architecture".to_string(),
                }
            },
        },
        name: name.to_string(),
        repo: Some(format!("unsloth/{name}-GGUF")),
        quant: quant.map(str::to_string),
        size,
        supported,
        audio: false,
        model_type: supported.then(|| model_type.to_string()),
        reason: (!supported).then(|| "phi3 is not a Crane-supported architecture".to_string()),
    }
}

fn running(id: u64, label: &str, state: ChildState) -> ChildSummary {
    ChildSummary {
        info: ChildInfo {
            id,
            pid: 4242,
            label: label.to_string(),
        },
        state,
    }
}

/// A populated app: models on disk, one model serving, a solved launch plan.
fn populated() -> App {
    let mut app = App::mock();
    app.local_models = vec![
        local(
            "Qwen3.5-9B-Instruct-Q4_K_M",
            "qwen3_5",
            5_800_000_000,
            Some("Q4_K_M"),
            true,
        ),
        local(
            "gemma-4-4b-it-Q6_K",
            "gemma4",
            3_400_000_000,
            Some("Q6_K"),
            true,
        ),
        local(
            "MiniCPM-V-4_6-Q4_K_M",
            "minicpmv4_6",
            2_600_000_000,
            Some("Q4_K_M"),
            true,
        ),
        local(
            "Phi-3-mini-4k-instruct-Q8_0",
            "phi3",
            4_060_000_000,
            Some("Q8_0"),
            false,
        ),
    ];
    app.last_running = vec![running(
        1,
        "Qwen3.5-9B-Instruct-Q4_K_M.gguf",
        ChildState::Healthy,
    )];
    app.known_ports.insert(1, 41100);
    app.launchpad.selected = 1;
    app
}

/// A realistic solver answer, from the real solver — a 9B hybrid model on the
/// fake 24 GiB card above.
fn solved() -> studio_core::estimator::SolveResult {
    let cfg = ModelConfig {
        hidden_size: 4096,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        num_hidden_layers: 48,
        max_position_embeddings: 262_144,
        vocab_size: 151_936,
        has_vision_config: false,
        hybrid: Some(studio_core::estimator::HybridConfig {
            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
        }),
    };
    let variants = vec![Variant {
        label: "Qwen3.5-9B-Instruct-Q4_K_M.gguf".to_string(),
        weight_bytes: 5_800_000_000,
        is_isq: false,
    }];
    solve(
        &SolveRequest {
            cfg: &cfg,
            variants: &variants,
            supports_kv_quant: true,
            supports_kv_swap: false,
            native_context: cfg.max_position_embeddings,
            usable_vram: 17 * GIB,
            backend: Backend::Cuda,
            compute_dtype_bytes: 2.0,
            vision: false,
            max_concurrent: 1,
        },
        262_144,
    )
}

/// Draws `app` at `width`×`height` and prints the frame with a caption.
fn show(caption: &str, width: u16, height: u16, app: &mut App) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.render_for_test(frame)).unwrap();
    println!(
        "\n### {caption}  ({width}×{height})\n{}",
        terminal.backend()
    );
}

#[test]
fn preview_splash() {
    let mut app = App::mock();
    app.screen = Screen::Splash;
    app.local_scan_done = false;
    show("splash — still booting", 100, 30, &mut app);
    app.local_scan_done = true;
    app.browser.set_catalog(
        studio_core::catalog::load::bundled(),
        studio_core::catalog::Source::Bundled,
    );
    show("splash — ready", 120, 32, &mut app);
    show("splash — small terminal", 80, 20, &mut app);
}

#[test]
fn preview_launchpad() {
    let mut app = populated();
    show("launchpad", 120, 34, &mut app);
    show("launchpad — 80×24", 80, 24, &mut app);

    let mut empty = App::mock();
    empty.screen = Screen::Launchpad;
    show("launchpad — nothing on disk yet", 100, 26, &mut empty);
}

#[test]
fn preview_download() {
    let mut app = populated();
    app.screen = Screen::Download;
    app.download.label = "Qwen3.5 9B Instruct · Q4_K_M".to_string();
    app.download.repo = "unsloth/Qwen3.5-9B-Instruct-GGUF · main".to_string();
    app.download
        .apply_event(&studio_core::download::Event::Progress {
            file: "Qwen3.5-9B-Instruct-Q4_K_M.gguf".to_string(),
            downloaded: 3_100_000_000,
            total: 5_800_000_000,
        });
    app.download
        .apply_event(&studio_core::download::Event::Completed {
            file: "tokenizer.json".to_string(),
        });
    app.download
        .apply_event(&studio_core::download::Event::Verifying {
            file: "params.json".to_string(),
        });
    app.download.speed = 46_000_000.0;
    app.download.speed_history = (0..40)
        .map(|i| 30_000_000.0 + f64::from(i % 9) * 3_000_000.0)
        .collect();
    show("download", 120, 30, &mut app);

    app.download.error = Some(
        "unsloth/Qwen3.5-9B-Instruct-GGUF: HTTP 401 — this repo is gated, add a token with `cranestudio config set hf-token`".to_string(),
    );
    show("download — failed", 100, 24, &mut app);
}

#[test]
fn preview_launch_options() {
    let mut app = populated();
    app.screen = Screen::Wizard;
    app.wizard.variant_label = "Qwen3.5-9B-Instruct-Q4_K_M.gguf".to_string();
    app.wizard.model_type = "qwen3_5".to_string();
    app.wizard.result = Some(solved());
    show("launch options", 120, 32, &mut app);
}

#[test]
fn preview_ready_and_apps() {
    let mut app = populated();
    app.screen = Screen::Ready;
    app.ready.set_active(
        1,
        "Qwen3.5-9B-Instruct-Q4_K_M".to_string(),
        41100,
        1234,
        false,
    );
    app.ready.context = Some(262_144);
    show("ready — apps", 120, 30, &mut app);

    app.ready.show_endpoint = true;
    app.ready.selected = 1;
    show("ready — endpoint open", 120, 34, &mut app);

    let mut tts = populated();
    tts.screen = Screen::Ready;
    tts.ready
        .set_active(2, "VoxCPM2".to_string(), 41101, 1234, true);
    tts.ready.show_endpoint = true;
    show("ready — TTS model apps", 120, 30, &mut tts);

    let mut starting = populated();
    starting.screen = Screen::Ready;
    starting
        .ready
        .begin("Qwen3.5-9B-Instruct-Q4_K_M".to_string(), 41100, 1234, false);
    show("ready — still loading", 100, 26, &mut starting);
}

#[test]
fn preview_chat() {
    let mut app = populated();
    app.screen = Screen::Chat;
    app.ready.set_active(
        1,
        "Qwen3.5-9B-Instruct-Q4_K_M".to_string(),
        41100,
        1234,
        false,
    );
    show("chat — empty", 100, 26, &mut app);

    app.chat.messages = vec![
        (
            screens::chat::Role::User,
            "Explain what a KV cache is, briefly.".to_string(),
        ),
        (
            screens::chat::Role::Assistant,
            "(thinking) the user wants a short answer, so no derivation\n(answer) A KV cache stores the key and value tensors already computed for every token in the context, so generating the next token only needs attention over the cache instead of recomputing the whole prompt each step. It's why the first token is slow and the rest are fast — and why long contexts cost memory rather than compute.".to_string(),
        ),
    ];
    app.chat.input = "and why does quantizing it to int4 help?".to_string();
    app.chat.active_image = Some(std::path::PathBuf::from("/home/dev/pictures/chart.png"));
    show("chat — mid conversation", 100, 28, &mut app);
    show("chat — 80×24", 80, 24, &mut app);
}

#[test]
fn preview_tts() {
    let mut app = populated();
    app.screen = Screen::TtsPlayground;
    app.ready
        .set_active(2, "VoxCPM2".to_string(), 41101, 1234, true);
    show("tts — empty", 100, 26, &mut app);

    app.tts.clips = vec![
        screens::tts::Clip {
            text: "Welcome to CraneStudio — local models, one binary.".to_string(),
            wav_path: std::path::PathBuf::from(
                "/home/dev/.local/share/cranestudio/tts/Welcome-to-CraneStudio-1756600000.wav",
            ),
            sample_rate: 24_000,
            bytes: 480_044,
            duration_secs: 3.4,
            gen_secs: 5.1,
        },
        screens::tts::Clip {
            text: "The second clip is a little shorter.".to_string(),
            wav_path: std::path::PathBuf::from(
                "/home/dev/.local/share/cranestudio/tts/The-second-clip-1756600042.wav",
            ),
            sample_rate: 24_000,
            bytes: 264_044,
            duration_secs: 2.1,
            gen_secs: 3.3,
        },
    ];
    app.tts.selected = 1;
    app.tts.now_playing = Some(1);
    show("tts — clips", 100, 28, &mut app);

    app.tts.now_playing = None;
    app.tts.generating = true;
    app.tts.input.clear();
    show("tts — generating", 90, 24, &mut app);
}

#[test]
fn preview_browser() {
    let mut app = populated();
    app.screen = Screen::Browser;
    app.browser.set_catalog(
        studio_core::catalog::load::bundled(),
        studio_core::catalog::Source::Bundled,
    );
    show("get models — catalog", 120, 30, &mut app);

    // The vision entries live at the end of the list, so the tail is worth a
    // look of its own: it's where the catalog's widest detail lines are.
    app.browser.selected = 17;
    show("get models — catalog tail", 120, 30, &mut app);

    app.browser.tab = screens::browser::Tab::Search;
    app.browser.search_query = "qwen3.5 gguf".to_string();
    app.browser.searching = true;
    show("get models — searching", 100, 24, &mut app);
}

#[test]
fn preview_hardware() {
    let mut app = populated();
    app.screen = Screen::Hardware;
    show("hardware", 110, 32, &mut app);
}

#[test]
fn preview_quit_modal() {
    let mut app = populated();
    app.quit_prompt = Some(QuitChoice::Stop);
    show("quit modal over the launchpad", 110, 30, &mut app);
}

/// The sizes a terminal can plausibly be, including absurd ones — a screen that
/// panics on a 20-row window is a screen that panics in someone's tmux split.
#[test]
fn every_screen_survives_extreme_sizes() {
    let screens = [
        Screen::Splash,
        Screen::Launchpad,
        Screen::Hardware,
        Screen::Browser,
        Screen::Download,
        Screen::Wizard,
        Screen::Ready,
        Screen::Chat,
        Screen::TtsPlayground,
    ];
    for screen in screens {
        for (width, height) in [(20, 5), (40, 10), (80, 24), (100, 30), (250, 80)] {
            let mut app = populated();
            app.screen = screen;
            app.quit_prompt = Some(QuitChoice::Keep);
            app.message = Some(crate::ui::Message::error("something went wrong"));
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| app.render_for_test(frame))
                .unwrap_or_else(|e| panic!("{screen:?} at {width}×{height}: {e}"));
        }
    }
}
