//! Catalog schema, per PLAN.md §8.1. RON on disk (§5) — not JSON, not TOML.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    Text,
    Tools,
    Vision,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    Gguf,
    Safetensors,
}

/// The `reasoning_effort` the Qwen 3.6/3.8 chat templates read — how long
/// the model thinks before answering. Crane takes it per request only (the
/// top-level request field, or `chat_template_kwargs`), and the templates
/// fall back to `xhigh` when it's absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningEffort {
    Xhigh,
    Medium,
    Low,
}

impl ReasoningEffort {
    /// The value the chat templates accept.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Xhigh => "xhigh",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::Low => "low",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvQuant {
    Int8,
    Int4,
}

/// Which knobs this model family actually exposes — the wizard (§4.4) must
/// only show what applies. `kv_quant` is Qwen 3.5-family-only (§2.8);
/// `kv_swap` is false for every family `CraneStudio` v1 targets (§2.11b), which
/// is what forces `max_concurrent` to 1 and makes 256k reachable at all.
// Four independent, orthogonal capability flags, deliberately shaped to
// match the catalog RON schema in PLAN.md §8.1 exactly.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Supports {
    pub isq: bool,
    pub kv_quant: bool,
    pub kv_swap: bool,
    pub vision: bool,
}

/// A measurement for one variant on one backend class, per §7.3. Ship-time
/// entries come from the maintainer's reference hardware; local runs
/// overwrite/extend this via the (separate, M8) measurement DB, which always
/// takes precedence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measured {
    pub max_context_achievable: usize,
    pub kv_quant: Option<KvQuant>,
    pub conc: usize,
    pub peak_bytes: u64,
    pub decode_tps: f32,
}

/// A secondary HF repo whose files are downloaded into the same directory as a
/// variant's main weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraRepo {
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
    pub format: Format,
    pub quant: Option<String>,
    pub download_bytes: u64,
    /// Secondary repos pulled into the same directory after `files`.
    #[serde(default)]
    pub extra_repos: Vec<ExtraRepo>,
    /// Keyed by backend class, e.g. `"cuda_sm86"`, `"metal_m3"`, `"cpu"`
    /// (§7.3, §13).
    #[serde(default)]
    pub measured: HashMap<String, Measured>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub display_name: String,
    pub family: String,
    /// The exact `--model-type` string to pass to crane-serve (§2.13).
    pub model_type: String,
    pub params: u64,
    pub native_context: usize,
    pub capabilities: Vec<Capability>,
    pub license: String,
    pub gated: bool,
    pub supports: Supports,
    /// Sent with every chat request that doesn't set its own — the gateway
    /// fills it in. `None` leaves the chat template's own default.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub schema_version: u32,
    pub updated: String,
    pub models: Vec<ModelEntry>,
}

impl Catalog {
    #[must_use]
    pub fn empty() -> Self {
        Catalog {
            schema_version: 1,
            updated: String::new(),
            models: Vec::new(),
        }
    }

    /// The catalog entry a local model path came from: a GGUF file by its
    /// file name (catalog file names are publisher-specific, and a file
    /// linked in from elsewhere keeps it), a safetensors directory by the
    /// `<repo>/<revision>` the downloader stores it under.
    #[must_use]
    pub fn entry_for_path(&self, path: &Path) -> Option<&ModelEntry> {
        let file_name = path.file_name()?;
        self.models.iter().find(|model| {
            model.variants.iter().any(|variant| match variant.format {
                Format::Gguf => variant
                    .files
                    .first()
                    .and_then(|f| Path::new(f).file_name())
                    .is_some_and(|f| f == file_name),
                Format::Safetensors => {
                    path.ends_with(Path::new(&variant.repo).join(&variant.revision))
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant(repo: &str, files: &[&str], format: Format) -> Variant {
        Variant {
            id: "v".to_string(),
            repo: repo.to_string(),
            revision: "abc123".to_string(),
            files: files.iter().map(ToString::to_string).collect(),
            format,
            quant: None,
            download_bytes: 1,
            extra_repos: Vec::new(),
            measured: HashMap::new(),
        }
    }

    fn entry(id: &str, variants: Vec<Variant>) -> ModelEntry {
        ModelEntry {
            id: id.to_string(),
            display_name: id.to_string(),
            family: "qwen3_5".to_string(),
            model_type: "qwen3_5".to_string(),
            params: 1,
            native_context: 32768,
            capabilities: vec![Capability::Text],
            license: "mit".to_string(),
            gated: false,
            supports: Supports {
                isq: false,
                kv_quant: false,
                kv_swap: false,
                vision: false,
            },
            reasoning_effort: Some(ReasoningEffort::Medium),
            variants,
        }
    }

    #[test]
    fn entry_for_path_matches_gguf_by_name_and_safetensors_by_repo_revision() {
        let catalog = Catalog {
            schema_version: 1,
            updated: String::new(),
            models: vec![
                entry(
                    "split",
                    vec![variant(
                        "org/Split-GGUF",
                        &["Q/m-00001-of-00002.gguf", "Q/m-00002-of-00002.gguf"],
                        Format::Gguf,
                    )],
                ),
                entry(
                    "dense",
                    vec![variant(
                        "org/Dense",
                        &["config.json", "model.safetensors"],
                        Format::Safetensors,
                    )],
                ),
            ],
        };
        let id = |p: &str| catalog.entry_for_path(Path::new(p)).map(|m| m.id.as_str());

        assert_eq!(
            id("/models/org/Split-GGUF/abc123/m-00001-of-00002.gguf"),
            Some("split")
        );
        assert_eq!(id("/elsewhere/m-00001-of-00002.gguf"), Some("split"));
        assert_eq!(id("/models/org/Dense/abc123"), Some("dense"));
        assert_eq!(id("/models/org/Dense/other-sha"), None);
        assert_eq!(id("/models/unrelated.gguf"), None);
    }
}
