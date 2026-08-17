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
    /// Internal re-exec target: runs a single crane-serve child process.
    /// Not a supported CLI surface — do not call this directly.
    #[command(name = "__serve", hide = true)]
    Serve(Box<crane_serve::Args>),
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
