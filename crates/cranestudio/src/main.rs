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
    /// Internal re-exec target: runs a single crane-serve child process.
    /// Not a supported CLI surface — do not call this directly.
    #[command(name = "__serve", hide = true)]
    Serve(Box<crane_serve::Args>),
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
            println!("`cranestudio doctor` is not implemented yet (see PLAN.md M1).");
            Ok(())
        }
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
