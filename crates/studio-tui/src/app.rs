//! The TUI's event loop and shared state, per PLAN.md §4. Async work (network
//! calls, model launches) runs as background tasks that report back over a
//! channel, so the render loop never blocks — see `BackgroundEvent`.
//!
//! Screen flow, deliberately shallow (§4.1 "no menu-diving"):
//!
//! ```text
//! splash ──▶ launchpad ──enter──▶ ready ──enter──▶ chat (and future apps)
//!              │  │
//!              │  └──c──▶ launch options
//!              └──g──▶ get models ──enter──▶ download ──▶ launchpad
//! ```

use std::collections::HashMap;
use std::io::IsTerminal as _;
use std::time::{Duration, Instant};

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt as _;
use ratatui::Frame;
use studio_core::catalog::hf::HfCandidate;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::{Catalog, Source};
use studio_core::download::Event as DownloadEvent;
use studio_core::hardware::{HardwareReport, Sample, Sampler};

use crate::daemon_client::{ChildSummary, DaemonClient};
use crate::models::LocalModel;
use crate::screens;
use crate::theme::Theme;
use crate::ui::Message;

/// How often the UI redraws its live parts (spinners, meters, throughput). Fast
/// enough for a spinner to look like motion, slow enough that sysinfo's CPU
/// deltas stay meaningful (it needs ≥200ms between refreshes).
const TICK: Duration = Duration::from_millis(250);

/// The splash never outstays this, even if the catalog fetch is still going —
/// and it leaves as soon as the boot work finishes, which is usually sooner.
const SPLASH_MAX: Duration = Duration::from_millis(2500);
/// …but it's always on screen at least this long, so it reads as an intro
/// rather than a flash of garbage on a warm cache.
const SPLASH_MIN: Duration = Duration::from_millis(650);

/// Samples kept behind the launchpad's sparklines.
const HISTORY: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Splash,
    Launchpad,
    Hardware,
    Browser,
    Download,
    Wizard,
    Ready,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuitChoice {
    Keep,
    Stop,
    Cancel,
}

impl QuitChoice {
    fn next(self) -> Self {
        match self {
            QuitChoice::Keep => QuitChoice::Stop,
            QuitChoice::Stop => QuitChoice::Cancel,
            QuitChoice::Cancel => QuitChoice::Keep,
        }
    }

    fn previous(self) -> Self {
        self.next().next()
    }
}

pub enum BackgroundEvent {
    CatalogLoaded(Box<Catalog>, Source),
    LocalScanDone(Vec<LocalModel>),
    SearchDone(Vec<HfCandidate>),
    SearchFailed(String),
    DownloadProgress(DownloadEvent),
    DownloadDone(LocalCandidate),
    DownloadFailed(String),
    Launched {
        id: u64,
        name: String,
        port: u16,
    },
    LaunchFailed(String),
    StatusRefresh(Vec<ChildSummary>),
    ChatDelta {
        role_started: bool,
        kind: screens::chat::DeltaKind,
        text: String,
    },
    ChatDone,
    ChatError(String),
}

/// Rolling utilization history, 0–1 per sample, oldest first.
#[derive(Debug, Default, Clone)]
pub struct History {
    pub cpu: Vec<f64>,
    pub ram: Vec<f64>,
    pub vram: Vec<f64>,
}

impl History {
    fn push(&mut self, sample: &Sample) {
        push_capped(
            &mut self.cpu,
            f64::from(sample.cpu_total.clamp(0.0, 100.0)) / 100.0,
        );
        push_capped(
            &mut self.ram,
            crate::ui::bars::ratio(
                sample.ram_total.saturating_sub(sample.ram_available),
                sample.ram_total,
            ),
        );
        if let Some(gpu) = sample.gpus.first() {
            push_capped(
                &mut self.vram,
                crate::ui::bars::ratio(
                    gpu.vram_total.saturating_sub(gpu.vram_free),
                    gpu.vram_total,
                ),
            );
        }
    }
}

fn push_capped(series: &mut Vec<f64>, value: f64) {
    series.push(value);
    if series.len() > HISTORY {
        series.remove(0);
    }
}

pub struct App {
    pub screen: Screen,
    pub daemon: DaemonClient,
    pub hardware: HardwareReport,
    /// Live CPU/RAM/VRAM, re-sampled every tick — `None` only for the first
    /// frame, before the first sample lands.
    pub live: Option<Sample>,
    pub history: History,
    pub control_port: u16,
    pub gateway_port: u16,

    pub launchpad: screens::launchpad::State,
    pub browser: screens::browser::State,
    pub download: screens::download::State,
    pub wizard: screens::wizard::State,
    pub ready: screens::ready::State,
    pub chat: screens::chat::State,

    /// Models found on disk, with their sizes and quantization labels.
    pub local_models: Vec<LocalModel>,
    pub local_scan_done: bool,
    pub hardware_scroll: u16,

    pub quit_prompt: Option<QuitChoice>,
    pub should_quit: bool,
    /// One transient line shown above the hints on every screen.
    pub message: Option<Message>,
    pub last_running: Vec<ChildSummary>,
    pub known_ports: HashMap<u64, u16>,
    /// Cycled with `F2`, remembered in `config.ron`.
    pub theme: Theme,
    /// Incremented every tick — drives spinners and the live meters.
    pub tick: u64,
    started: Instant,
    sampler: Sampler,

    bg_tx: tokio::sync::mpsc::UnboundedSender<BackgroundEvent>,
    bg_rx: tokio::sync::mpsc::UnboundedReceiver<BackgroundEvent>,
}

impl App {
    /// `preferred_control_port`/`preferred_gateway_port` are only a starting
    /// preference — see `DaemonClient::connect_or_spawn`.
    ///
    /// # Errors
    /// If the daemon can't be reached or spawned.
    pub async fn new(
        preferred_control_port: u16,
        preferred_gateway_port: u16,
    ) -> anyhow::Result<Self> {
        let mut daemon =
            DaemonClient::connect_or_spawn(preferred_control_port, preferred_gateway_port).await?;
        daemon.attach().await?;
        let (control_port, gateway_port) = (daemon.control_port(), daemon.gateway_port());
        let hardware = studio_core::hardware::probe(&studio_core::paths::models_dir());
        let (bg_tx, bg_rx) = tokio::sync::mpsc::unbounded_channel();
        let config =
            studio_core::config::Config::load(&studio_core::paths::config_dir().join("config.ron"));

        let app = App {
            screen: Screen::Splash,
            daemon,
            hardware,
            live: None,
            history: History::default(),
            control_port,
            gateway_port,
            launchpad: screens::launchpad::State::default(),
            browser: screens::browser::State::default(),
            download: screens::download::State::default(),
            wizard: screens::wizard::State::default(),
            ready: screens::ready::State::default(),
            chat: screens::chat::State::new(
                config.system_prompt,
                config.max_tokens,
                config.temperature,
            ),
            local_models: Vec::new(),
            local_scan_done: false,
            hardware_scroll: 0,
            quit_prompt: None,
            should_quit: false,
            message: None,
            last_running: Vec::new(),
            known_ports: HashMap::new(),
            theme: Theme::from_name(config.theme),
            tick: 0,
            started: Instant::now(),
            sampler: Sampler::new(),
            bg_tx,
            bg_rx,
        };
        app.spawn_catalog_load();
        app.spawn_local_scan();
        Ok(app)
    }

    #[must_use]
    pub fn sender(&self) -> tokio::sync::mpsc::UnboundedSender<BackgroundEvent> {
        self.bg_tx.clone()
    }

    fn spawn_catalog_load(&self) {
        let tx = self.sender();
        tokio::spawn(async move {
            let cache_path = studio_core::paths::data_dir().join("catalog-cache.ron");
            let (catalog, source) = studio_core::catalog::load(
                studio_core::catalog::load::DEFAULT_REMOTE_URL,
                &cache_path,
            )
            .await;
            let _ = tx.send(BackgroundEvent::CatalogLoaded(Box::new(catalog), source));
        });
    }

    /// Walks the models directory *and* stats every candidate, so it runs off
    /// the render thread — a cold directory of large GGUFs is not instant.
    fn spawn_local_scan(&self) {
        let tx = self.sender();
        tokio::spawn(async move {
            let models = tokio::task::spawn_blocking(|| {
                crate::models::collect(&studio_core::paths::models_dir())
            })
            .await
            .unwrap_or_default();
            let _ = tx.send(BackgroundEvent::LocalScanDone(models));
        });
    }

    /// Re-runs the scan after something changes on disk (a finished download, or
    /// the user asking with `R`).
    pub fn rescan_models(&mut self) {
        self.local_scan_done = false;
        self.spawn_local_scan();
    }

    pub fn spawn_status_refresh(&self) {
        let tx = self.sender();
        let base = self.control_base();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            if let Ok(resp) = client.get(format!("{base}/control/list")).send().await
                && let Ok(list) = resp.json::<Vec<ChildSummary>>().await
            {
                let _ = tx.send(BackgroundEvent::StatusRefresh(list));
            }
        });
    }

    /// Mirrors `DaemonClient`'s own base — kept here too since background tasks
    /// (status refresh, launch, chat) make their own short-lived requests
    /// independently of the long-held attach connection. Uses the port the
    /// daemon actually reported binding, not a re-derived guess — it may have
    /// fallen forward from what was originally requested.
    #[must_use]
    pub fn control_base(&self) -> String {
        format!("http://127.0.0.1:{}", self.control_port)
    }

    #[must_use]
    pub fn gateway_base(&self) -> String {
        format!("http://127.0.0.1:{}", self.gateway_port)
    }

    /// # Errors
    /// If the terminal can't be initialised, or an I/O error occurs.
    pub async fn run_standalone() -> anyhow::Result<()> {
        if !std::io::stdin().is_terminal() {
            anyhow::bail!("cranestudio needs an interactive terminal (stdin is not a tty)");
        }
        let control_port = std::env::var("CRANESTUDIO_CONTROL_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(studio_gateway::DEFAULT_CONTROL_PORT);
        let gateway_port = std::env::var("CRANESTUDIO_GATEWAY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(studio_gateway::DEFAULT_GATEWAY_PORT);

        let mut app = App::new(control_port, gateway_port).await?;
        let mut terminal = ratatui::init();
        let result = app.run(&mut terminal).await;
        ratatui::restore();
        app.daemon.detach_connection().await;
        result
    }

    async fn run(&mut self, terminal: &mut ratatui::DefaultTerminal) -> anyhow::Result<()> {
        let mut events = EventStream::new();
        let mut ticker = tokio::time::interval(TICK);

        loop {
            terminal.draw(|f| self.render(f))?;

            tokio::select! {
                biased;
                () = termination_signal() => {
                    // §3.1a: interaction is impossible here — always kill, no
                    // prompt.
                    let _ = self.daemon.stop_everything().await;
                    break;
                }
                maybe_event = events.next() => {
                    match maybe_event {
                        Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                            self.handle_key(key).await;
                        }
                        Some(Ok(_)) => {}
                        Some(Err(e)) => return Err(e.into()),
                        None => break,
                    }
                }
                Some(event) = self.bg_rx.recv() => {
                    self.handle_background(event);
                }
                _ = ticker.tick() => {
                    self.on_tick();
                }
            }

            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn on_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);

        let sample = self.sampler.sample();
        self.history.push(&sample);
        self.live = Some(sample);

        if self.screen == Screen::Download {
            self.download.sample_speed();
        }
        // Two ticks per second is plenty for a health poll, and the launchpad
        // and ready screens are the only ones that show it.
        if self.tick.is_multiple_of(2) && matches!(self.screen, Screen::Launchpad | Screen::Ready) {
            self.spawn_status_refresh();
        }
        if self.screen == Screen::Splash && self.splash_done() {
            self.screen = Screen::Launchpad;
        }
    }

    /// The splash leaves once the boot work is done (and it's been up long
    /// enough to be seen), or once it's simply been up too long.
    fn splash_done(&self) -> bool {
        let elapsed = self.started.elapsed();
        if elapsed >= SPLASH_MAX {
            return true;
        }
        elapsed >= SPLASH_MIN && self.local_scan_done && self.browser.catalog.is_some()
    }

    fn cycle_theme(&mut self) {
        let path = studio_core::paths::config_dir().join("config.ron");
        let mut config = studio_core::config::Config::load(&path);
        config.theme = config.theme.next();
        let _ = config.save(&path);
        self.theme = Theme::from_name(config.theme);
        self.message = Some(Message::info(format!("theme: {}", config.theme.label())));
    }

    fn handle_background(&mut self, event: BackgroundEvent) {
        match event {
            BackgroundEvent::StatusRefresh(list) => self.last_running = list,
            BackgroundEvent::CatalogLoaded(catalog, source) => {
                self.browser.set_catalog(*catalog, source);
            }
            BackgroundEvent::LocalScanDone(models) => {
                self.browser
                    .set_local(models.iter().map(|m| m.candidate.clone()).collect());
                self.local_models = models;
                self.local_scan_done = true;
            }
            BackgroundEvent::SearchDone(results) => self.browser.set_search_results(results),
            BackgroundEvent::SearchFailed(err) => {
                self.browser.searching = false;
                self.message = Some(Message::error(format!("search failed: {err}")));
            }
            BackgroundEvent::DownloadProgress(event) => self.download.apply_event(&event),
            BackgroundEvent::DownloadDone(candidate) => {
                // Straight into the launch options for what was just fetched:
                // downloading a model is only ever a step towards running it.
                self.rescan_models();
                screens::wizard::load_local(self, &candidate);
                self.screen = Screen::Wizard;
            }
            BackgroundEvent::DownloadFailed(err) => self.download.error = Some(err),
            BackgroundEvent::Launched { id, name, port } => {
                self.known_ports.insert(id, port);
                self.ready.set_active(id, name, port, self.gateway_port);
                self.screen = Screen::Ready;
            }
            BackgroundEvent::LaunchFailed(err) => {
                self.ready.starting = false;
                self.screen = Screen::Launchpad;
                self.message = Some(Message::error(format!("launch failed: {err}")));
            }
            BackgroundEvent::ChatDelta {
                role_started,
                kind,
                text,
            } => {
                self.chat.apply_delta(role_started, kind, &text);
            }
            BackgroundEvent::ChatDone => self.chat.finish_turn(),
            BackgroundEvent::ChatError(err) => self.chat.fail_turn(&err),
        }
    }

    async fn handle_key(&mut self, key: KeyEvent) {
        if self.quit_prompt.is_some() {
            self.handle_quit_prompt_key(key).await;
            return;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.begin_quit().await;
            return;
        }
        // Any key dismisses the splash — it's an intro, not a gate.
        if self.screen == Screen::Splash {
            self.screen = Screen::Launchpad;
            return;
        }

        // Screens own their keys first; only what they don't claim falls through
        // to the global shortcuts below.
        let consumed = match self.screen {
            Screen::Launchpad => screens::launchpad::handle_key(self, key),
            Screen::Browser => screens::browser::handle_key(self, key),
            Screen::Download => screens::download::handle_key(self, key),
            Screen::Wizard => screens::wizard::handle_key(self, key),
            Screen::Ready => screens::ready::handle_key(self, key),
            Screen::Chat => screens::chat::handle_key(self, key),
            Screen::Hardware => self.handle_hardware_key(key),
            Screen::Splash => false,
        };
        if consumed {
            return;
        }

        match key.code {
            KeyCode::F(2) => self.cycle_theme(),
            KeyCode::Char('q') | KeyCode::Esc => self.begin_quit().await,
            KeyCode::Char('h') => self.screen = Screen::Launchpad,
            KeyCode::Char('d') => self.screen = Screen::Hardware,
            KeyCode::Char('g' | 'b' | 'm') => self.screen = Screen::Browser,
            _ => {}
        }
    }

    fn handle_hardware_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                self.screen = Screen::Launchpad;
                true
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.hardware_scroll = self.hardware_scroll.saturating_add(1);
                true
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.hardware_scroll = self.hardware_scroll.saturating_sub(1);
                true
            }
            _ => false,
        }
    }

    async fn begin_quit(&mut self) {
        match self.daemon.any_running().await {
            Ok(true) => self.quit_prompt = Some(QuitChoice::Stop),
            _ => self.should_quit = true,
        }
    }

    async fn handle_quit_prompt_key(&mut self, key: KeyEvent) {
        let selected = self.quit_prompt.unwrap_or(QuitChoice::Cancel);
        match key.code {
            KeyCode::Char('k' | 'K') => self.quit_prompt = Some(QuitChoice::Keep),
            KeyCode::Char('s' | 'S') => self.quit_prompt = Some(QuitChoice::Stop),
            KeyCode::Char('c' | 'C') => self.quit_prompt = Some(QuitChoice::Cancel),
            KeyCode::Down => self.quit_prompt = Some(selected.next()),
            KeyCode::Up => self.quit_prompt = Some(selected.previous()),
            KeyCode::Esc => self.quit_prompt = None,
            KeyCode::Enter => match selected {
                QuitChoice::Keep => {
                    let _ = self.daemon.keep_serving().await;
                    self.should_quit = true;
                }
                QuitChoice::Stop => {
                    let _ = self.daemon.stop_everything().await;
                    self.should_quit = true;
                }
                QuitChoice::Cancel => self.quit_prompt = None,
            },
            _ => {}
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        match self.screen {
            Screen::Splash => screens::splash::render(self, frame),
            Screen::Launchpad => screens::launchpad::render(self, frame),
            Screen::Hardware => screens::hardware::render(self, frame),
            Screen::Browser => screens::browser::render(self, frame),
            Screen::Download => screens::download::render(self, frame),
            Screen::Wizard => screens::wizard::render(self, frame),
            Screen::Ready => screens::ready::render(self, frame),
            Screen::Chat => screens::chat::render(self, frame),
        }
        if let Some(choice) = self.quit_prompt {
            screens::quit_prompt::render(frame, &self.theme, choice, &self.last_running);
        }
    }
}

#[cfg(unix)]
async fn termination_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let Ok(mut term) = signal(SignalKind::terminate()) else {
        return std::future::pending().await;
    };
    let Ok(mut hup) = signal(SignalKind::hangup()) else {
        return std::future::pending().await;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = hup.recv() => {}
    }
}

#[cfg(not(unix))]
async fn termination_signal() {
    std::future::pending::<()>().await;
}

#[cfg(test)]
impl App {
    /// `render` is the event loop's own private business; the preview tests need
    /// exactly it, and nothing else, to draw a screen.
    pub(crate) fn render_for_test(&mut self, frame: &mut Frame) {
        self.render(frame);
    }

    /// An `App` wired to nothing, for the rendering previews in
    /// `crate::preview`. Screens read the whole `App`, so this is what lets
    /// them be drawn (and looked at, and regression-tested) without a daemon,
    /// a GPU, or a model on disk.
    pub(crate) fn mock() -> Self {
        let (bg_tx, bg_rx) = tokio::sync::mpsc::unbounded_channel();
        App {
            screen: Screen::Launchpad,
            daemon: DaemonClient::offline(1235, 1234),
            hardware: crate::preview::hardware(),
            live: Some(crate::preview::sample()),
            history: crate::preview::history(),
            control_port: 1235,
            gateway_port: 1234,
            launchpad: screens::launchpad::State::default(),
            browser: screens::browser::State::default(),
            download: screens::download::State::default(),
            wizard: screens::wizard::State::default(),
            ready: screens::ready::State::default(),
            chat: screens::chat::State::default(),
            local_models: Vec::new(),
            local_scan_done: true,
            hardware_scroll: 0,
            quit_prompt: None,
            should_quit: false,
            message: None,
            last_running: Vec::new(),
            known_ports: HashMap::new(),
            theme: Theme::from_name(studio_core::config::ThemeName::Crane),
            tick: 3,
            started: Instant::now(),
            sampler: Sampler::new(),
            bg_tx,
            bg_rx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quit_choices_cycle_in_both_directions() {
        assert_eq!(QuitChoice::Keep.next(), QuitChoice::Stop);
        assert_eq!(QuitChoice::Cancel.next(), QuitChoice::Keep);
        assert_eq!(QuitChoice::Keep.previous(), QuitChoice::Cancel);
    }

    #[test]
    fn history_is_capped_and_keeps_the_newest_samples() {
        let mut series = Vec::new();
        for i in 0..(HISTORY + 10) {
            #[allow(clippy::cast_precision_loss)]
            push_capped(&mut series, i as f64);
        }
        assert_eq!(series.len(), HISTORY);
        #[allow(clippy::cast_precision_loss)]
        let newest = (HISTORY + 9) as f64;
        assert_eq!(series.last().copied(), Some(newest));
    }
}
