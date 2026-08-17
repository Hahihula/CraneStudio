//! Per-file size and integrity metadata from the `HuggingFace` Hub API, per
//! PLAN.md §9's "verify against the HF-reported sha where available".
//! `?blobs=true` on the revision-scoped model-info endpoint returns each
//! LFS file's real `sha256` and `size` — verified live against
//! `unsloth/Qwen3.5-0.8B-GGUF`, whose reported sha256 matches a real
//! `sha256sum` of the file on disk.
//!
//! Gating note (verified live against `meta-llama/Llama-3.2-1B`, which is
//! gated): the model-*info* endpoint this module calls returns its file
//! listing (names, sizes, LFS hashes) with **no token needed** — only
//! `gated: "manual"` in the response marks it. Gating is enforced on the
//! actual file *download* (`…/resolve/main/<file>`, a 401 without a valid,
//! license-accepted token), which is `download_file`'s concern
//! (`DownloadError::Unauthorized`/`Forbidden`), not this module's.

use reqwest::StatusCode;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpec {
    pub filename: String,
    pub size: u64,
    /// `None` for small non-LFS files (READMEs, configs) the API doesn't
    /// hash the same way — integrity is only checked when this is `Some`.
    pub sha256: Option<String>,
}

#[derive(Debug)]
pub enum FetchError {
    /// No token, or an invalid one, on a repo whose *metadata itself*
    /// requires auth (rare — see the module docs: gated repos normally
    /// expose their file listing publicly and only gate the actual file
    /// bytes, which `download_file`'s `DownloadError::Unauthorized` /
    /// `Forbidden` catch instead).
    Unauthorized,
    Forbidden,
    NotFound,
    Request(reqwest::Error),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Unauthorized => write!(f, "unauthorized — an access token is required"),
            FetchError::Forbidden => write!(
                f,
                "forbidden — this repo is gated and the license hasn't been accepted for this token"
            ),
            FetchError::NotFound => write!(f, "repo or revision not found"),
            FetchError::Request(e) => write!(f, "request failed: {e}"),
        }
    }
}

impl std::error::Error for FetchError {}

#[derive(Deserialize, Default)]
struct ModelInfoResponse {
    #[serde(default)]
    siblings: Vec<SiblingBlob>,
}

#[derive(Deserialize)]
struct SiblingBlob {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Deserialize)]
struct LfsInfo {
    sha256: String,
    size: u64,
}

/// # Errors
/// See [`FetchError`] — distinguishes auth failures (actionable per §9)
/// from a generic request failure.
pub async fn fetch_file_specs(
    client: &reqwest::Client,
    repo: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<Vec<FileSpec>, FetchError> {
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true");
    let mut request = client.get(&url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await.map_err(FetchError::Request)?;
    match response.status() {
        StatusCode::UNAUTHORIZED => return Err(FetchError::Unauthorized),
        StatusCode::FORBIDDEN => return Err(FetchError::Forbidden),
        StatusCode::NOT_FOUND => return Err(FetchError::NotFound),
        _ => {}
    }
    let response = response.error_for_status().map_err(FetchError::Request)?;
    let info: ModelInfoResponse = response.json().await.map_err(FetchError::Request)?;

    Ok(info
        .siblings
        .into_iter()
        .map(|sibling| {
            let (size, sha256) = match sibling.lfs {
                Some(lfs) => (lfs.size, Some(lfs.sha256)),
                None => (sibling.size.unwrap_or(0), None),
            };
            FileSpec {
                filename: sibling.rfilename,
                size,
                sha256,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hits the real HuggingFace API — not run by default (§12).
    #[tokio::test]
    #[ignore]
    async fn matches_a_real_known_file() {
        let client = reqwest::Client::new();
        let specs = fetch_file_specs(
            &client,
            "unsloth/Qwen3.5-0.8B-GGUF",
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            None,
        )
        .await
        .unwrap();
        let file = specs
            .iter()
            .find(|f| f.filename == "Qwen3.5-0.8B-Q8_0.gguf")
            .unwrap();
        assert_eq!(file.size, 811_843_840);
        assert_eq!(
            file.sha256.as_deref(),
            Some("0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c")
        );
    }

    // Confirms the module-doc claim: metadata for a genuinely gated repo
    // (meta-llama/Llama-3.2-1B) is fetchable with no token at all.
    #[tokio::test]
    #[ignore]
    async fn gated_repos_still_expose_their_file_listing() {
        let client = reqwest::Client::new();
        let specs = fetch_file_specs(&client, "meta-llama/Llama-3.2-1B", "main", None)
            .await
            .unwrap();
        assert!(
            specs.iter().any(|f| f.filename == "config.json"),
            "{specs:?}"
        );
    }
}
