//! Filtered `HuggingFace` search, per PLAN.md §8.2. Queries the public HF Hub
//! API, then classifies each result the same way `local.rs` classifies a
//! local checkpoint: read `config.json` when present, otherwise range-fetch
//! a GGUF header. Unsupported architectures are returned, not dropped —
//! silently omitting them makes search look broken.

use std::io::Cursor;

pub use reqwest;
use reqwest::StatusCode;
use serde::Deserialize;

use super::classify::{self, Classification, ConfigJson};
use super::gguf;

/// How much of a GGUF file to range-fetch looking for `general.architecture`
/// — llama.cpp's writer puts `general.*` keys first, so this comfortably
/// covers real files without downloading the checkpoint (§8.2).
const GGUF_PROBE_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfCandidate {
    pub repo_id: String,
    pub gated: bool,
    pub classification: Classification,
}

#[derive(Debug)]
pub struct SearchError(reqwest::Error);

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`HuggingFace` search request failed: {}", self.0)
    }
}

impl std::error::Error for SearchError {}

#[derive(Deserialize)]
struct SearchResultItem {
    id: String,
}

#[derive(Deserialize, Default)]
struct ModelInfo {
    #[serde(default)]
    gated: GatedField,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

// The `String` variant only needs to exist for deserialization to accept
// HF's `"manual"`/`"auto"` gated-kind strings — `is_gated` below only cares
// that it matched, not which kind.
#[derive(Deserialize)]
#[serde(untagged)]
enum GatedField {
    Bool(bool),
    Kind(#[allow(dead_code)] String),
}

impl Default for GatedField {
    fn default() -> Self {
        GatedField::Bool(false)
    }
}

impl GatedField {
    fn is_gated(&self) -> bool {
        match self {
            GatedField::Bool(b) => *b,
            GatedField::Kind(_) => true,
        }
    }
}

#[derive(Deserialize, Clone)]
struct Sibling {
    rfilename: String,
}

/// # Errors
/// Returns `SearchError` only if the initial search request itself fails
/// (network error or non-2xx status) — per-repo classification failures are
/// folded into `Classification::Unknown` instead, since one unclassifiable
/// result shouldn't sink the whole search.
pub async fn search(
    client: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<HfCandidate>, SearchError> {
    let results: Vec<SearchResultItem> = client
        .get("https://huggingface.co/api/models")
        .query(&[("search", query), ("limit", &limit.to_string())])
        .send()
        .await
        .map_err(SearchError)?
        .error_for_status()
        .map_err(SearchError)?
        .json()
        .await
        .map_err(SearchError)?;

    let mut out = Vec::with_capacity(results.len());
    for result in results {
        out.push(classify_repo(client, &result.id).await);
    }
    Ok(out)
}

async fn classify_repo(client: &reqwest::Client, repo_id: &str) -> HfCandidate {
    let info = fetch_model_info(client, repo_id).await;
    let gated = info.as_ref().is_some_and(|info| info.gated.is_gated());
    let siblings = info.map(|info| info.siblings).unwrap_or_default();

    let classification = match classify_from_config(client, repo_id).await {
        Some(c) => c,
        None => match classify_from_gguf(client, repo_id, &siblings).await {
            Some(c) => c,
            None => Classification::Unknown {
                reason: "no config.json and no readable GGUF architecture header".to_string(),
            },
        },
    };

    HfCandidate {
        repo_id: repo_id.to_string(),
        gated,
        classification,
    }
}

async fn fetch_model_info(client: &reqwest::Client, repo_id: &str) -> Option<ModelInfo> {
    client
        .get(format!("https://huggingface.co/api/models/{repo_id}"))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()
}

async fn classify_from_config(client: &reqwest::Client, repo_id: &str) -> Option<Classification> {
    let text = client
        .get(format!(
            "https://huggingface.co/{repo_id}/raw/main/config.json"
        ))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .await
        .ok()?;
    let config: ConfigJson = serde_json::from_str(&text).ok()?;
    Some(config.classify())
}

async fn classify_from_gguf(
    client: &reqwest::Client,
    repo_id: &str,
    siblings: &[Sibling],
) -> Option<Classification> {
    let gguf_file = siblings
        .iter()
        .find(|s| s.rfilename.to_lowercase().ends_with(".gguf"))?;

    let response = client
        .get(format!(
            "https://huggingface.co/{repo_id}/resolve/main/{}",
            gguf_file.rfilename
        ))
        .header(
            reqwest::header::RANGE,
            format!("bytes=0-{}", GGUF_PROBE_BYTES - 1),
        )
        .send()
        .await
        .ok()?;

    // If the server ignored our Range header, don't buffer a multi-GB body.
    if response.status() != StatusCode::PARTIAL_CONTENT
        && response
            .content_length()
            .is_some_and(|len| len > GGUF_PROBE_BYTES * 2)
    {
        return None;
    }

    let bytes = response.bytes().await.ok()?;
    let arch = gguf::read_architecture(&mut Cursor::new(bytes.as_ref()))?;
    Some(classify::classify_gguf_architecture(&arch))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hits the real `HuggingFace` API — not run by default (§12: CI does not
    // attempt network tests). Run with `cargo test -p studio-core -- --ignored`.
    #[tokio::test]
    #[ignore = "hits a real network API — not run by default (§12)"]
    async fn search_finds_a_known_supported_and_a_known_unsupported_repo() {
        let client = reqwest::Client::new();

        let qwen = search(&client, "Qwen/Qwen3.5-0.8B", 3).await.unwrap();
        assert!(
            qwen.iter().any(|c| matches!(
                c.classification,
                Classification::Supported {
                    model_type: "qwen3_5",
                    ..
                }
            )),
            "{qwen:?}"
        );

        let llama = search(&client, "meta-llama/Llama-3.2-1B", 3).await.unwrap();
        assert!(
            llama
                .iter()
                .any(|c| matches!(c.classification, Classification::Unsupported { .. })),
            "{llama:?}"
        );
    }

    #[tokio::test]
    #[ignore = "hits a real network API — not run by default (§12)"]
    async fn gguf_only_repo_is_classified_from_its_header() {
        let client = reqwest::Client::new();
        let results = search(&client, "unsloth/Qwen3.5-0.8B-GGUF", 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            matches!(
                results[0].classification,
                Classification::Supported {
                    model_type: "qwen3_5",
                    ..
                }
            ),
            "{:?}",
            results[0]
        );
    }
}
