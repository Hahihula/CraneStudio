//! Catalog schema, per PLAN.md §8.1. RON on disk (§5) — not JSON, not TOML.

use std::collections::HashMap;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
    pub format: Format,
    pub quant: Option<String>,
    pub download_bytes: u64,
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
}
