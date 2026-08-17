//! Native Rust `HuggingFace` downloader, per PLAN.md §9. No Python anywhere
//! (§2.15) — `data/crane-model-download` in upstream Crane is a `uv run`
//! script; this replaces it, driven directly against `reqwest`.

mod disk;
mod file;
mod hf_api;
mod repo;

pub use disk::{InsufficientSpace, check as check_disk_space};
pub use file::{DownloadError, Event, download_file};
pub use hf_api::{FetchError, FileSpec, fetch_file_specs};
pub use repo::{RepoDownload, RepoDownloadError, download_repo};
pub use tokio_util::sync::CancellationToken;

#[allow(clippy::cast_precision_loss)]
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
