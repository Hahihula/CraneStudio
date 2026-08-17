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
        Some(Command::Daemon) => {
            println!("`cranestudio daemon` is not implemented yet (see PLAN.md M5).");
            Ok(())
        }
        Some(Command::Tui) | None => {
            println!("`cranestudio` (TUI) is not implemented yet (see PLAN.md M7).");
            Ok(())
        }
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
