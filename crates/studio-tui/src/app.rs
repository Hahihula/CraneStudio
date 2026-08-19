//! The TUI's event loop and shared state, per PLAN.md §4. Async work
//! (network calls, model launches) runs as background tasks that report
//! back over a channel, so the render loop never blocks — see `BackgroundEvent`.

use std::collections::HashMap;
use std::io::IsTerminal as _;
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt as _;
use ratatui::Frame;
use studio_core::catalog::hf::HfCandidate;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::{Catalog, Source};
use studio_core::download::Event as DownloadEvent;
use studio_core::hardware::HardwareReport;

use crate::daemon_client::{ChildSummary, DaemonClient};
use crate::screens;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Home,
    Doctor,
    Browser,
    Download,
    Wizard,
    Connect,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuitChoice {
    Keep,
    Stop,
    Cancel,
}

pub enum BackgroundEvent {
    CatalogLoaded(Box<Catalog>, Source),
    LocalScanDone(Vec<LocalCandidate>),
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

pub struct App {
    pub screen: Screen,
    pub daemon: DaemonClient,
    pub hardware: HardwareReport,
    pub control_port: u16,
    pub gateway_port: u16,

    pub home: screens::home::State,
    pub browser: screens::browser::State,
    pub download: screens::download::State,
    pub wizard: screens::wizard::State,
    pub connect: screens::connect::State,
    pub chat: screens::chat::State,

    pub quit_prompt: Option<QuitChoice>,
    pub should_quit: bool,
    pub status_line: Option<String>,
    pub last_running: Vec<ChildSummary>,
    pub known_ports: HashMap<u64, u16>,

    bg_tx: tokio::sync::mpsc::UnboundedSender<BackgroundEvent>,
    bg_rx: tokio::sync::mpsc::UnboundedReceiver<BackgroundEvent>,
}

impl App {
    /// `preferred_control_port`/`preferred_gateway_port` are only a
    /// starting preference — see `DaemonClient::connect_or_spawn`.
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
        let chat_config =
            studio_core::config::Config::load(&studio_core::paths::config_dir().join("config.ron"));

        let app = App {
            screen: Screen::Home,
            daemon,
            hardware,
            control_port,
            gateway_port,
            home: screens::home::State,
            browser: screens::browser::State::default(),
            download: screens::download::State::default(),
            wizard: screens::wizard::State::default(),
            connect: screens::connect::State::default(),
            chat: screens::chat::State::new(
                chat_config.system_prompt,
                chat_config.max_tokens,
                chat_config.temperature,
            ),
            quit_prompt: None,
            should_quit: false,
            status_line: None,
            last_running: Vec::new(),
            known_ports: HashMap::new(),
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

    fn spawn_local_scan(&self) {
        let tx = self.sender();
        tokio::spawn(async move {
            let candidates = studio_core::catalog::local::scan(&studio_core::paths::models_dir());
            let _ = tx.send(BackgroundEvent::LocalScanDone(candidates));
        });
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

    /// Mirrors `DaemonClient`'s own base — kept here too since background
    /// tasks (status refresh, launch, chat) make their own short-lived
    /// requests independently of the long-held attach connection. Uses the
    /// port the daemon actually reported binding, not a re-derived guess —
    /// it may have fallen forward from what was originally requested.
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
        let mut ticker = tokio::time::interval(Duration::from_millis(500));

        loop {
            terminal.draw(|f| self.render(f))?;

            tokio::select! {
                biased;
                () = termination_signal() => {
                    // §3.1a: interaction is impossible here — always kill,
                    // no prompt.
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
        if matches!(self.screen, Screen::Home | Screen::Connect) {
            self.spawn_status_refresh();
        }
    }

    fn handle_background(&mut self, event: BackgroundEvent) {
        match event {
            BackgroundEvent::StatusRefresh(list) => self.last_running = list,
            BackgroundEvent::CatalogLoaded(catalog, source) => {
                self.browser.set_catalog(*catalog, source);
            }
            BackgroundEvent::LocalScanDone(candidates) => self.browser.set_local(candidates),
            BackgroundEvent::SearchDone(results) => self.browser.set_search_results(results),
            BackgroundEvent::SearchFailed(err) => {
                self.status_line = Some(format!("search failed: {err}"));
            }
            BackgroundEvent::DownloadProgress(event) => self.download.apply_event(&event),
            BackgroundEvent::DownloadDone(candidate) => {
                screens::wizard::load_local(self, &candidate);
                self.screen = Screen::Wizard;
            }
            BackgroundEvent::DownloadFailed(err) => self.download.error = Some(err),
            BackgroundEvent::Launched { id, name, port } => {
                self.known_ports.insert(id, port);
                self.connect.set_active(id, name, port, self.gateway_port);
                self.screen = Screen::Connect;
                self.wizard.launching = false;
            }
            BackgroundEvent::LaunchFailed(err) => {
                self.wizard.launching = false;
                self.status_line = Some(format!("launch failed: {err}"));
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

        // Screens that capture free-text input get first refusal on keys.
        let consumed = match self.screen {
            Screen::Browser => screens::browser::handle_key(self, key),
            Screen::Download => screens::download::handle_key(self, key),
            Screen::Wizard => screens::wizard::handle_key(self, key),
            Screen::Chat => screens::chat::handle_key(self, key),
            Screen::Home | Screen::Doctor | Screen::Connect => false,
        };
        if consumed {
            return;
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.begin_quit().await;
            }
            KeyCode::Char('q') | KeyCode::Esc => self.begin_quit().await,
            KeyCode::Char('h') => self.screen = Screen::Home,
            KeyCode::Char('d') => self.screen = Screen::Doctor,
            KeyCode::Char('b' | 'm') => self.screen = Screen::Browser,
            KeyCode::Char('r') if matches!(self.screen, Screen::Connect) => {
                self.screen = Screen::Chat;
            }
            _ => {}
        }
    }

    async fn begin_quit(&mut self) {
        match self.daemon.any_running().await {
            Ok(true) => self.quit_prompt = Some(QuitChoice::Stop),
            _ => self.should_quit = true,
        }
    }

    async fn handle_quit_prompt_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('k' | 'K') => self.quit_prompt = Some(QuitChoice::Keep),
            KeyCode::Char('s' | 'S') => self.quit_prompt = Some(QuitChoice::Stop),
            KeyCode::Char('c' | 'C') => self.quit_prompt = Some(QuitChoice::Cancel),
            KeyCode::Esc => {
                self.quit_prompt = None;
            }
            KeyCode::Enter => match self.quit_prompt {
                Some(QuitChoice::Keep) => {
                    let _ = self.daemon.keep_serving().await;
                    self.should_quit = true;
                }
                Some(QuitChoice::Stop) => {
                    let _ = self.daemon.stop_everything().await;
                    self.should_quit = true;
                }
                Some(QuitChoice::Cancel) | None => {
                    self.quit_prompt = None;
                }
            },
            _ => {}
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        match self.screen {
            Screen::Home => screens::home::render(self, frame),
            Screen::Doctor => screens::doctor::render(self, frame),
            Screen::Browser => screens::browser::render(self, frame),
            Screen::Download => screens::download::render(self, frame),
            Screen::Wizard => screens::wizard::render(self, frame),
            Screen::Connect => screens::connect::render(self, frame),
            Screen::Chat => screens::chat::render(self, frame),
        }
        if let Some(choice) = self.quit_prompt {
            screens::quit_prompt::render(frame, choice, &self.last_running);
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
