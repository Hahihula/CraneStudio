use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// `CraneStudio` — a local model studio for the Crane inference stack.
#[derive(Parser)]
#[command(name = "cranestudio", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Launch the interactive TUI (default when no subcommand is given).
    Tui,
    /// Run the daemon in the foreground.
    Daemon,
    /// Print a hardware report and exit.
    Doctor,
    /// Browse the model catalog and local/remote models (§8). A stand-in
    /// for the real ratatui browser (M7) — plain-text output only.
    Catalog {
        #[command(subcommand)]
        action: CatalogAction,
    },
    /// Download one or more files from a `HuggingFace` repo (§9). Ctrl-C is
    /// safe to use — it just kills the process; the next run of the same
    /// command resumes from wherever the `.part` files stopped.
    Download {
        /// `org/repo`, e.g. `unsloth/Qwen3.5-9B-GGUF`.
        repo: String,
        /// A commit sha, not a floating branch/tag (§5).
        revision: String,
        files: Vec<String>,
        /// Defaults to `<models_dir>/<repo>/<revision>/` (§9).
        #[arg(long)]
        dest: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        concurrency: usize,
    },
    /// Read or write global settings (§5's `config.ron`).
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Launch a model through a running daemon (§3.1a) — starts the daemon
    /// on demand if one isn't already running.
    Launch {
        model_path: String,
        #[arg(long, default_value = "auto")]
        model_type: String,
        #[arg(long)]
        port: Option<u16>,
        #[arg(long, default_value_t = 262_144)]
        max_seq_len: usize,
    },
    /// Register a model with the gateway (§3.2) — makes it show up in
    /// `/v1/models` and eligible for on-demand start, without launching it
    /// immediately. The base URL never changes when you switch models.
    Register {
        name: String,
        model_path: String,
        #[arg(long, default_value = "auto")]
        model_type: String,
        #[arg(long)]
        port: u16,
        #[arg(long, default_value_t = 262_144)]
        max_seq_len: usize,
    },
    /// Stop every running model and shut the daemon down.
    Stop,
    /// Report what the daemon is running and whether it's detached.
    Status,
    /// Hold an interactive control-client connection open — keeps the
    /// daemon alive per the detach lease (§3.1a). Ctrl-C or killing this
    /// process releases it; if nothing else is attached and the daemon
    /// wasn't explicitly detached, it stops every model shortly after.
    Attach,
    /// Internal re-exec target: runs a single crane-serve child process.
    /// Not a supported CLI surface — do not call this directly.
    #[command(name = "__serve", hide = true)]
    Serve(Box<crane_serve::Args>),
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Store a `HuggingFace` access token (§9), used for gated repos. Written
    /// to `config.ron` with owner-only (0600) permissions.
    #[command(name = "set")]
    SetHfToken {
        #[arg(value_name = "KEY", value_parser = ["hf-token"])]
        key: String,
        value: String,
    },
}

#[derive(Subcommand)]
enum CatalogAction {
    /// List the curated catalog (remote, falling back to cached, falling
    /// back to the copy bundled in this binary).
    List,
    /// Scan a directory (default: the configured models directory) for
    /// local checkpoints and classify each one.
    Scan {
        #[arg(default_value = None)]
        path: Option<PathBuf>,
    },
    /// Search `HuggingFace`, filtered to architectures Crane supports.
    Search {
        query: String,
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Serve(args)) => {
            crane_serve::init_logging();
            crane_serve::run(*args).await
        }
        Some(Command::Doctor) => {
            let report = studio_core::hardware::probe(&studio_core::paths::models_dir());
            print!("{}", studio_tui::doctor::render(&report));
            Ok(())
        }
        Some(Command::Catalog { action }) => run_catalog(action).await,
        Some(Command::Download {
            repo,
            revision,
            files,
            dest,
            concurrency,
        }) => run_download(repo, revision, files, dest, concurrency).await,
        Some(Command::Config { action }) => run_config(action),
        Some(Command::Daemon) => run_daemon().await,
        Some(Command::Launch {
            model_path,
            model_type,
            port,
            max_seq_len,
        }) => run_launch(model_path, model_type, port, max_seq_len).await,
        Some(Command::Register {
            name,
            model_path,
            model_type,
            port,
            max_seq_len,
        }) => run_register(name, model_path, model_type, port, max_seq_len).await,
        Some(Command::Stop) => run_stop().await,
        Some(Command::Status) => run_status().await,
        Some(Command::Attach) => run_attach().await,
        Some(Command::Tui) | None => studio_tui::app::App::run_standalone().await,
    }
}

async fn run_catalog(action: CatalogAction) -> anyhow::Result<()> {
    match action {
        CatalogAction::List => {
            let cache_path = studio_core::paths::data_dir().join("catalog-cache.ron");
            let (catalog, source) = studio_core::catalog::load(
                studio_core::catalog::load::DEFAULT_REMOTE_URL,
                &cache_path,
            )
            .await;
            print!("{}", studio_tui::catalog::render_catalog(&catalog, source));
        }
        CatalogAction::Scan { path } => {
            let root = path.unwrap_or_else(studio_core::paths::models_dir);
            let candidates = studio_core::catalog::local::scan(&root);
            print!(
                "{}",
                studio_tui::catalog::render_local_candidates(&candidates)
            );
        }
        CatalogAction::Search { query, limit } => {
            let client = studio_core::catalog::hf::reqwest::Client::new();
            let candidates = studio_core::catalog::hf::search(&client, &query, limit).await?;
            print!("{}", studio_tui::catalog::render_hf_candidates(&candidates));
        }
    }
    Ok(())
}

async fn run_download(
    repo: String,
    revision: String,
    files: Vec<String>,
    dest: Option<PathBuf>,
    concurrency: usize,
) -> anyhow::Result<()> {
    let dest_dir =
        dest.unwrap_or_else(|| studio_core::paths::models_dir().join(&repo).join(&revision));
    let token =
        studio_core::config::Config::load(&studio_core::paths::config_dir().join("config.ron"))
            .hf_token;

    let mut request = studio_core::download::RepoDownload::new(repo, revision, dest_dir);
    request.token = token;
    request.max_concurrent = concurrency.max(1);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel = studio_core::download::CancellationToken::new();
    let client = studio_core::catalog::hf::reqwest::Client::new();

    let printer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            studio_tui::download::render_event(&event);
        }
    });

    let result =
        studio_core::download::download_repo(&client, &request, &files, &tx, &cancel).await;
    drop(tx);
    let _ = printer.await;
    result.map_err(|e| anyhow::anyhow!("{e}"))
}

fn run_config(action: ConfigAction) -> anyhow::Result<()> {
    let ConfigAction::SetHfToken { value, .. } = action;
    let config_path = studio_core::paths::config_dir().join("config.ron");
    let mut config = studio_core::config::Config::load(&config_path);
    config.hf_token = Some(value);
    config.save(&config_path)?;
    println!("saved (0600) to {}", config_path.display());
    Ok(())
}

// Overridable so a machine that already has something else bound to the
// default ports (e.g. LM Studio also defaults to :1234) isn't stuck — and
// even without an override, `run_daemon` falls forward to the next free
// port rather than hard-failing (§7.4), so these are starting *preferences*
// for the daemon, not guarantees for a client.
fn preferred_control_port() -> u16 {
    std::env::var("CRANESTUDIO_CONTROL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(studio_gateway::DEFAULT_CONTROL_PORT)
}

fn preferred_gateway_port() -> u16 {
    std::env::var("CRANESTUDIO_GATEWAY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(studio_gateway::DEFAULT_GATEWAY_PORT)
}

// Client commands (status/stop/attach/launch/register) need to find
// wherever the daemon actually ended up, which may not be the preferred
// port above if it had to fall forward — see `studio_core::endpoints`.
fn control_base_url() -> String {
    format!(
        "http://127.0.0.1:{}",
        studio_core::endpoints::resolve_control_port(studio_gateway::DEFAULT_CONTROL_PORT)
    )
}

fn gateway_base_url() -> String {
    format!(
        "http://127.0.0.1:{}",
        studio_core::endpoints::resolve_gateway_port(studio_gateway::DEFAULT_GATEWAY_PORT)
    )
}

const MAX_PORT_ATTEMPTS: u16 = 20;

/// Binds `preferred`, or the next free port after it (up to `max_tries`
/// attempts) if that one's already taken — PLAN.md §7.4: "port in use →
/// retry on a different port automatically," applied to the daemon's own
/// listening ports rather than hard-failing the whole daemon over a
/// collision with something unrelated (LM Studio's default is also :1234).
async fn bind_with_fallback(
    preferred: u16,
    max_tries: u16,
) -> anyhow::Result<(tokio::net::TcpListener, u16)> {
    for offset in 0..max_tries {
        let port = preferred.saturating_add(offset);
        if let Ok(listener) = tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            return Ok((listener, port));
        }
    }
    anyhow::bail!("could not find a free port starting from {preferred} after {max_tries} attempts")
}

async fn run_daemon() -> anyhow::Result<()> {
    let data_dir = studio_core::paths::data_dir();
    std::fs::create_dir_all(&data_dir)?;
    let pidfile = data_dir.join("children.pids");

    let reaped = studio_gateway::reap_stale_children(&pidfile);
    if !reaped.is_empty() {
        println!(
            "Reaped {} stale child process(es) left by a previous crashed run: {reaped:?}",
            reaped.len()
        );
    }

    let supervisor = studio_supervisor::Supervisor::new().with_pidfile(pidfile);
    let (daemon, mut shutdown_rx) = studio_gateway::Daemon::new(supervisor);
    let gateway_state = studio_gateway::GatewayState::new(daemon.clone());

    let control_app = studio_gateway::router(daemon);
    let gateway_app = studio_gateway::gateway_router().with_state(gateway_state);

    let (control_listener, control_port) =
        bind_with_fallback(preferred_control_port(), MAX_PORT_ATTEMPTS).await?;
    let (gateway_listener, gateway_port) =
        bind_with_fallback(preferred_gateway_port(), MAX_PORT_ATTEMPTS).await?;
    println!("control API listening on 127.0.0.1:{control_port}");
    println!("gateway (/v1/*) listening on 127.0.0.1:{gateway_port}");
    studio_core::endpoints::save(studio_core::endpoints::Endpoints {
        control_port,
        gateway_port,
    });

    let mut gateway_shutdown_rx = shutdown_rx.clone();
    let control_server =
        axum::serve(control_listener, control_app).with_graceful_shutdown(async move {
            let _ = shutdown_rx.changed().await;
        });
    let gateway_server =
        axum::serve(gateway_listener, gateway_app).with_graceful_shutdown(async move {
            let _ = gateway_shutdown_rx.changed().await;
        });

    let (control_result, gateway_result) = tokio::join!(control_server, gateway_server);
    studio_core::endpoints::clear();
    control_result?;
    gateway_result?;
    println!("daemon stopped.");
    Ok(())
}

fn build_launch_spec(
    model_path: String,
    model_type: String,
    port: u16,
    max_seq_len: usize,
) -> studio_core::launch::LaunchSpec {
    studio_core::launch::LaunchSpec {
        model_path,
        model_type,
        model_name: None,
        port,
        cpu: false,
        max_concurrent: 1,
        decode_tokens_per_seq: 16,
        format: None,
        quant: None,
        dtype: None,
        max_seq_len,
        gpu_memory_limit: None,
        text_only: false,
        kv_quant: None,
        prefill_chunk: None,
        device: 0,
    }
}

async fn run_launch(
    model_path: String,
    model_type: String,
    port: Option<u16>,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    // An explicit --port is a deliberate choice, honored as-is; without one,
    // pick a free port automatically rather than hard-coding 41100 (§7.4).
    let port = port.unwrap_or_else(|| studio_core::launch::pick_free_port(41100, 50));
    let spec = build_launch_spec(model_path, model_type, port, max_seq_len);

    let response = client
        .post(format!("{}/control/launch", control_base_url()))
        .json(&serde_json::json!({ "spec": spec, "label": format!("port-{port}") }))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!(
            "launch failed: {}",
            response.text().await.unwrap_or_default()
        );
    }
    let body: serde_json::Value = response.json().await?;
    println!(
        "launched — id {}, listening on 127.0.0.1:{port} once healthy",
        body["id"]
    );
    Ok(())
}

async fn run_register(
    name: String,
    model_path: String,
    model_type: String,
    port: u16,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let spec = build_launch_spec(model_path, model_type, port, max_seq_len);

    let response = client
        .post(format!("{}/register", gateway_base_url()))
        .json(&serde_json::json!({ "name": name.clone(), "spec": spec }))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!(
            "register failed: {}",
            response.text().await.unwrap_or_default()
        );
    }
    println!(
        "registered '{name}' — visible in /v1/models, starts on demand when a request names it"
    );
    Ok(())
}

async fn run_stop() -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let base = control_base_url();
    match client.post(format!("{base}/control/shutdown")).send().await {
        Ok(_) => println!("daemon stopped."),
        Err(_) => println!("no daemon running at {base}."),
    }
    Ok(())
}

async fn run_status() -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let base = control_base_url();
    let Ok(status) = client.get(format!("{base}/control/status")).send().await else {
        println!("no daemon running at {base}.");
        return Ok(());
    };
    let status: serde_json::Value = status.json().await?;
    let list: serde_json::Value = client
        .get(format!("{base}/control/list"))
        .send()
        .await?
        .json()
        .await?;

    println!("detached: {}", status["detached"]);
    println!("attached control clients: {}", status["attached_clients"]);
    println!("running children: {}", status["running_children"]);
    if let Some(children) = list.as_array() {
        for child in children {
            println!(
                "  [{}] {} — pid {} — {:?}",
                child["info"]["id"], child["info"]["label"], child["info"]["pid"], child["state"]
            );
        }
    }
    Ok(())
}

async fn run_attach() -> anyhow::Result<()> {
    use futures_util::StreamExt;

    let url = format!(
        "{}/control/attach",
        control_base_url().replacen("http://", "ws://", 1)
    );
    let (stream, _) = tokio_tungstenite::connect_async(&url).await?;
    println!(
        "attached — holding the control lease. Ctrl-C to detach and let the daemon decide whether to shut down."
    );
    let (_write, mut read) = stream.split();

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("detaching.");
        }
        () = async { while read.next().await.is_some() {} } => {
            println!("daemon closed the connection.");
        }
    }
    Ok(())
}
