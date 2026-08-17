//! Orchestrates downloading several files from one repo with bounded
//! concurrency (§9: "2–4 parallel connections; configurable, defaulting low
//! enough to not saturate a home link"), after a disk-space precheck across
//! all of them together.

use std::path::PathBuf;

use futures_util::StreamExt;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use super::disk::InsufficientSpace;
use super::file::{DownloadError, Event, download_file};
use super::hf_api::{FetchError, FileSpec, fetch_file_specs};

pub struct RepoDownload {
    pub repo: String,
    pub revision: String,
    pub dest_dir: PathBuf,
    pub token: Option<String>,
    pub max_concurrent: usize,
}

impl RepoDownload {
    #[must_use]
    pub fn new(repo: impl Into<String>, revision: impl Into<String>, dest_dir: PathBuf) -> Self {
        RepoDownload {
            repo: repo.into(),
            revision: revision.into(),
            dest_dir,
            token: None,
            max_concurrent: 3,
        }
    }
}

#[derive(Debug)]
pub enum RepoDownloadError {
    Fetch(FetchError),
    /// A requested filename isn't in the repo at this revision.
    FileNotInRepo(String),
    InsufficientSpace(InsufficientSpace),
    File {
        repo: String,
        filename: String,
        error: DownloadError,
    },
}

impl std::fmt::Display for RepoDownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepoDownloadError::Fetch(e) => write!(f, "could not list repo files: {e}"),
            RepoDownloadError::FileNotInRepo(name) => {
                write!(f, "{name}: not found in this repo at this revision")
            }
            RepoDownloadError::InsufficientSpace(e) => write!(f, "{e}"),
            // §9: actionable, not a raw HTTP error — say exactly what to do.
            RepoDownloadError::File {
                repo,
                filename,
                error: DownloadError::Unauthorized | DownloadError::Forbidden,
            } => write!(
                f,
                "{filename}: this repo is gated. Visit https://huggingface.co/{repo}, accept the license, \
                 create an access token at https://huggingface.co/settings/tokens, then set it with \
                 `cranestudio config set hf-token <TOKEN>`."
            ),
            RepoDownloadError::File {
                filename, error, ..
            } => write!(f, "{filename}: {error}"),
        }
    }
}

impl std::error::Error for RepoDownloadError {}

/// # Errors
/// See [`RepoDownloadError`]. On the first file that fails, every other
/// in-flight file download in this batch is cancelled too (same repo, same
/// token — one auth failure means the rest will fail the same way).
pub async fn download_repo(
    client: &reqwest::Client,
    request: &RepoDownload,
    filenames: &[String],
    events: &UnboundedSender<Event>,
    cancel: &CancellationToken,
) -> Result<(), RepoDownloadError> {
    let specs = fetch_file_specs(
        client,
        &request.repo,
        &request.revision,
        request.token.as_deref(),
    )
    .await
    .map_err(RepoDownloadError::Fetch)?;

    let mut targets = Vec::with_capacity(filenames.len());
    for name in filenames {
        let spec = specs
            .iter()
            .find(|s| &s.filename == name)
            .ok_or_else(|| RepoDownloadError::FileNotInRepo(name.clone()))?;
        targets.push(spec.clone());
    }

    let total_bytes: u64 = targets.iter().map(|s| s.size).sum();
    super::disk::check(&request.dest_dir, total_bytes)
        .map_err(RepoDownloadError::InsufficientSpace)?;

    let max_concurrent = request.max_concurrent.max(1);
    let results: Vec<Result<(), RepoDownloadError>> = futures_util::stream::iter(targets)
        .map(|spec: FileSpec| {
            let client = client.clone();
            let dest = request.dest_dir.join(&spec.filename);
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                request.repo, request.revision, spec.filename
            );
            let token = request.token.clone();
            let events = events.clone();
            let cancel = cancel.clone();
            let repo = request.repo.clone();
            async move {
                let size = if spec.size > 0 { Some(spec.size) } else { None };
                download_file(
                    &client,
                    &url,
                    token.as_deref(),
                    &dest,
                    size,
                    spec.sha256.as_deref(),
                    &events,
                    &cancel,
                )
                .await
                .map_err(|error| {
                    cancel.cancel();
                    RepoDownloadError::File {
                        repo,
                        filename: spec.filename.clone(),
                        error,
                    }
                })
            }
        })
        .buffer_unordered(max_concurrent)
        .collect()
        .await;

    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    // Hits the real HuggingFace API — not run by default (§12). Downloads
    // two small, real, non-LFS files (fast) to prove the orchestration
    // (fetch specs, precheck, concurrent per-file download) end to end.
    #[tokio::test]
    #[ignore]
    async fn downloads_multiple_real_small_files_concurrently() {
        let client = reqwest::Client::new();
        let dir = TempDir::new().unwrap();
        let request = RepoDownload::new(
            "unsloth/Qwen3.5-0.8B-GGUF",
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            dir.path().to_path_buf(),
        );
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        download_repo(
            &client,
            &request,
            &["README.md".to_string(), ".gitattributes".to_string()],
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap();

        assert!(dir.path().join("README.md").is_file());
        assert!(dir.path().join(".gitattributes").is_file());
    }

    #[tokio::test]
    #[ignore]
    async fn unknown_filename_is_reported_before_any_download_starts() {
        let client = reqwest::Client::new();
        let dir = TempDir::new().unwrap();
        let request = RepoDownload::new(
            "unsloth/Qwen3.5-0.8B-GGUF",
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            dir.path().to_path_buf(),
        );
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let err = download_repo(
            &client,
            &request,
            &["does-not-exist.gguf".to_string()],
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, RepoDownloadError::FileNotInRepo(_)),
            "{err:?}"
        );
    }
}
