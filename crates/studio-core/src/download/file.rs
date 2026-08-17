//! Resumable single-file download, per PLAN.md §9: HTTP range resume,
//! progress events, sha256 integrity verification, atomic `.part` → final
//! rename. Cancellation (explicit, via `CancellationToken`) deletes the
//! partial file; an abrupt process kill (Ctrl-C, SIGKILL) does not run this
//! code at all, so the `.part` file is simply left on disk — the next call
//! to `download_file` finds it and resumes from its length. No signal
//! handling needed for that half of §9's "interrupt and resume" behaviour.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::StatusCode;
use sha2::{Digest, Sha256};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub enum Event {
    Started {
        file: String,
        resume_from: u64,
        total: u64,
    },
    Progress {
        file: String,
        downloaded: u64,
        total: u64,
    },
    Verifying {
        file: String,
    },
    Completed {
        file: String,
    },
    Cancelled {
        file: String,
    },
}

#[derive(Debug)]
pub enum DownloadError {
    Unauthorized,
    Forbidden,
    NotFound,
    IntegrityMismatch { expected: String, actual: String },
    Cancelled,
    Io(std::io::Error),
    Request(reqwest::Error),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Unauthorized => write!(f, "unauthorized — an access token is required"),
            DownloadError::Forbidden => write!(
                f,
                "forbidden — this repo is gated and the license hasn't been accepted for this token"
            ),
            DownloadError::NotFound => write!(f, "file not found"),
            DownloadError::IntegrityMismatch { expected, actual } => write!(
                f,
                "integrity check failed: expected sha256 {expected}, got {actual}"
            ),
            DownloadError::Cancelled => write!(f, "cancelled"),
            DownloadError::Io(e) => write!(f, "I/O error: {e}"),
            DownloadError::Request(e) => write!(f, "request failed: {e}"),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        DownloadError::Io(e)
    }
}

const PROGRESS_INTERVAL: Duration = Duration::from_millis(200);

/// # Errors
/// See [`DownloadError`]. On any error except a fresh 401/403/404, a
/// partially-written `.part` file may remain on disk — deliberately, so a
/// retry resumes rather than starting over (see module docs).
#[allow(clippy::too_many_arguments)]
pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    dest: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
    events: &UnboundedSender<Event>,
    cancel: &CancellationToken,
) -> Result<(), DownloadError> {
    // `dest`'s directory need not exist yet — a fresh `<models_dir>/<repo>/
    // <revision>/` for a repo never downloaded before never does. Every
    // existing caller (and every test, via `TempDir::new()`) had always
    // happened to pass an already-existing directory, so this was never
    // exercised until the TUI's browser started downloading straight into
    // a brand new nested path.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await?;
    }

    let part = part_path(dest);
    let existing = fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    if existing > 0 && expected_size == Some(existing) {
        // Fully written already, just never verified/renamed — e.g. the
        // process died between the last write and the rename.
        let digest = hash_file(&part).await?;
        return finish(&part, dest, expected_sha256, digest, events).await;
    }

    let mut request = client.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }

    let response = request.send().await.map_err(DownloadError::Request)?;
    match response.status() {
        StatusCode::UNAUTHORIZED => return Err(DownloadError::Unauthorized),
        StatusCode::FORBIDDEN => return Err(DownloadError::Forbidden),
        StatusCode::NOT_FOUND => return Err(DownloadError::NotFound),
        _ => {}
    }
    let response = response
        .error_for_status()
        .map_err(DownloadError::Request)?;

    // Only trust the existing bytes if the server actually honoured our
    // Range request — some servers/proxies silently ignore it and send 200
    // with the whole body instead, which must not be appended to.
    let resuming = existing > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let start_offset = if resuming { existing } else { 0 };
    let total = expected_size
        .or_else(|| response.content_length().map(|len| len + start_offset))
        .unwrap_or(0);
    let file_name = filename(dest);

    let _ = events.send(Event::Started {
        file: file_name.clone(),
        resume_from: start_offset,
        total,
    });

    let mut hasher = Sha256::new();
    let mut file = if resuming {
        prime_hasher(&mut hasher, &part).await?;
        OpenOptions::new().append(true).open(&part).await?
    } else {
        File::create(&part).await?
    };

    let mut downloaded = start_offset;
    let mut last_report = Instant::now();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            drop(file);
            let _ = fs::remove_file(&part).await;
            let _ = events.send(Event::Cancelled { file: file_name });
            return Err(DownloadError::Cancelled);
        }

        let chunk = chunk.map_err(DownloadError::Request)?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if last_report.elapsed() >= PROGRESS_INTERVAL {
            let _ = events.send(Event::Progress {
                file: file_name.clone(),
                downloaded,
                total,
            });
            last_report = Instant::now();
        }
    }
    file.flush().await?;
    drop(file);
    let _ = events.send(Event::Progress {
        file: file_name,
        downloaded,
        total,
    });

    let digest = to_hex(&hasher.finalize());
    finish(&part, dest, expected_sha256, digest, events).await
}

async fn finish(
    part: &Path,
    dest: &Path,
    expected_sha256: Option<&str>,
    digest: String,
    events: &UnboundedSender<Event>,
) -> Result<(), DownloadError> {
    if let Some(expected) = expected_sha256 {
        let _ = events.send(Event::Verifying {
            file: filename(dest),
        });
        if !digest.eq_ignore_ascii_case(expected) {
            let _ = fs::remove_file(part).await;
            return Err(DownloadError::IntegrityMismatch {
                expected: expected.to_string(),
                actual: digest,
            });
        }
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::rename(part, dest).await?;
    let _ = events.send(Event::Completed {
        file: filename(dest),
    });
    Ok(())
}

async fn prime_hasher(hasher: &mut Sha256, path: &Path) -> Result<(), DownloadError> {
    let mut file = File::open(path).await?;
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(())
}

async fn hash_file(path: &Path) -> Result<String, DownloadError> {
    let mut hasher = Sha256::new();
    prime_hasher(&mut hasher, path).await?;
    Ok(to_hex(&hasher.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

fn part_path(dest: &Path) -> PathBuf {
    let mut os = dest.as_os_str().to_os_string();
    os.push(".part");
    PathBuf::from(os)
}

fn filename(dest: &Path) -> String {
    dest.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    use super::*;

    /// A minimal range-aware HTTP/1.1 server for exactly one GET request,
    /// serving a fixed in-memory body. `cut_at` optionally closes the
    /// connection early (after N bytes of the response), simulating a
    /// dropped connection mid-transfer.
    async fn serve_one(listener: &TcpListener, body: &'static [u8], cut_at: Option<usize>) {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let n = socket.read(&mut buf).await.unwrap();
        let request = String::from_utf8_lossy(&buf[..n]);

        let range_start = request
            .lines()
            .find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name.eq_ignore_ascii_case("range").then_some(value)
            })
            .and_then(|v| v.trim().strip_prefix("bytes="))
            .and_then(|r| r.trim_end_matches(['\r', '-']).parse::<usize>().ok());

        let (status, slice, content_range) = match range_start {
            Some(start) if start < body.len() => (
                "206 Partial Content",
                &body[start..],
                format!(
                    "Content-Range: bytes {start}-{}/{}\r\n",
                    body.len() - 1,
                    body.len()
                ),
            ),
            _ => ("200 OK", body, String::new()),
        };

        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{content_range}Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
            slice.len()
        );
        let _ = socket.write_all(header.as_bytes()).await;

        let to_send = cut_at.map_or(slice.len(), |c| c.min(slice.len()));
        let _ = socket.write_all(&slice[..to_send]).await;
        let _ = socket.shutdown().await;
    }

    async fn local_server() -> (TcpListener, String) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        (listener, format!("http://{addr}/file"))
    }

    fn sha256_hex(data: &[u8]) -> String {
        to_hex(&Sha256::digest(data))
    }

    #[tokio::test]
    async fn downloads_verifies_and_renames_atomically() {
        let body: &'static [u8] = b"hello resumable world, this is the full file body";
        let (listener, url) = local_server().await;
        let server = tokio::spawn(async move { serve_one(&listener, body, None).await });

        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("model.bin");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();

        download_file(
            &client,
            &url,
            None,
            &dest,
            Some(body.len() as u64),
            Some(&sha256_hex(body)),
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap();

        server.await.unwrap();
        assert!(dest.exists());
        assert!(!part_path(&dest).exists());
        assert_eq!(tokio::fs::read(&dest).await.unwrap(), body);
    }

    #[tokio::test]
    async fn creates_a_destination_directory_that_does_not_exist_yet() {
        // The real-world shape this covers: `<models_dir>/<org>/<repo>/
        // <revision>/file.gguf` for a repo downloaded for the first time —
        // nothing upstream of `download_file` pre-creates that path.
        let body: &'static [u8] = b"nested destination directory contents";
        let (listener, url) = local_server().await;
        let server = tokio::spawn(async move { serve_one(&listener, body, None).await });

        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("org").join("repo").join("a1b2c3d").join("model.bin");
        assert!(!dest.parent().unwrap().exists());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();

        download_file(&client, &url, None, &dest, Some(body.len() as u64), None, &tx, &CancellationToken::new()).await.unwrap();

        server.await.unwrap();
        assert_eq!(tokio::fs::read(&dest).await.unwrap(), body);
    }

    #[tokio::test]
    async fn resumes_from_a_partial_part_file() {
        let body: &'static [u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("model.bin");
        let part = part_path(&dest);
        // Simulate a prior run that got 20 bytes in before being killed.
        tokio::fs::write(&part, &body[..20]).await.unwrap();

        let (listener, url) = local_server().await;
        let server = tokio::spawn(async move { serve_one(&listener, body, None).await });

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        download_file(
            &client,
            &url,
            None,
            &dest,
            Some(body.len() as u64),
            Some(&sha256_hex(body)),
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
        server.await.unwrap();

        let started = rx.recv().await.unwrap();
        assert!(
            matches!(
                started,
                Event::Started {
                    resume_from: 20,
                    ..
                }
            ),
            "{started:?}"
        );
        assert_eq!(tokio::fs::read(&dest).await.unwrap(), body);
    }

    #[tokio::test]
    async fn cancellation_deletes_the_part_file() {
        // A body big enough that the cancellation check (per streamed
        // chunk) has a real window to land mid-transfer.
        let body: &'static [u8] = vec![7u8; 4 * 1024 * 1024].leak();
        let (listener, url) = local_server().await;
        let server = tokio::spawn(async move { serve_one(&listener, body, None).await });

        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("model.bin");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = download_file(
            &client,
            &url,
            None,
            &dest,
            Some(body.len() as u64),
            None,
            &tx,
            &cancel,
        )
        .await;
        server.await.unwrap();

        assert!(
            matches!(result, Err(DownloadError::Cancelled)),
            "{result:?}"
        );
        assert!(!part_path(&dest).exists());
        assert!(!dest.exists());
    }

    #[tokio::test]
    async fn integrity_mismatch_is_rejected_and_cleaned_up() {
        let body: &'static [u8] = b"this body will not match the expected hash";
        let (listener, url) = local_server().await;
        let server = tokio::spawn(async move { serve_one(&listener, body, None).await });

        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("model.bin");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();

        let result = download_file(
            &client,
            &url,
            None,
            &dest,
            Some(body.len() as u64),
            Some("0000000000000000000000000000000000000000000000000000000000000000"),
            &tx,
            &CancellationToken::new(),
        )
        .await;
        server.await.unwrap();

        assert!(
            matches!(result, Err(DownloadError::IntegrityMismatch { .. })),
            "{result:?}"
        );
        assert!(!dest.exists());
        assert!(!part_path(&dest).exists());
    }

    #[tokio::test]
    async fn unauthorized_status_is_reported_distinctly() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/file");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let _ = socket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await;
            let _ = socket.shutdown().await;
        });

        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("model.bin");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        let result = download_file(
            &client,
            &url,
            None,
            &dest,
            None,
            None,
            &tx,
            &CancellationToken::new(),
        )
        .await;
        server.await.unwrap();

        assert!(
            matches!(result, Err(DownloadError::Unauthorized)),
            "{result:?}"
        );
    }

    // Hits the real HuggingFace API — not run by default (§12). This is
    // the real gated-repo boundary (see hf_api's module docs): the file
    // *listing* for meta-llama/Llama-3.2-1B is public, but this actual
    // file download 401s without a token.
    #[tokio::test]
    #[ignore = "hits a real network API — not run by default (§12)"]
    async fn real_gated_repo_download_401s_without_a_token() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("config.json");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        let result = download_file(
            &client,
            "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json",
            None,
            &dest,
            None,
            None,
            &tx,
            &CancellationToken::new(),
        )
        .await;
        assert!(
            matches!(result, Err(DownloadError::Unauthorized)),
            "{result:?}"
        );
    }
}
