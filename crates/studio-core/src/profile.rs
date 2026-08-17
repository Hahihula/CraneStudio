//! Named, saved launch configurations, per PLAN.md §5. A profile is a
//! reusable "this model, launched this way" snapshot the user names and
//! keeps — separate from `measurement`'s append-only history, which is
//! keyed by a config fingerprint rather than a name and exists whether or
//! not the user ever saved a profile.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::catalog::schema::{Format, KvQuant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub catalog_id: Option<String>,
    pub path: String,
    /// HF commit sha, never a floating tag — see §5's warning that a
    /// profile would otherwise silently change meaning when the upstream
    /// repo moves.
    pub revision: Option<String>,
    pub model_type: String,
    pub format: Format,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub quant: Option<String>,
    pub dtype: Option<String>,
    pub kv_quant: Option<KvQuant>,
    pub max_seq_len: usize,
    pub max_concurrent: usize,
    pub gpu_memory_limit: Option<String>,
    pub device: usize,
    pub text_only: bool,
    pub prefill_chunk: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredSummary {
    pub peak_vram_bytes: u64,
    pub decode_tokens_sec: f32,
    pub max_verified_depth: usize,
    pub kv_swaps_observed: u64,
    pub verified_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    pub created: String,
    pub last_used: String,
    pub model: ModelRef,
    pub runtime: RuntimeConfig,
    pub measured: Option<MeasuredSummary>,
}

fn file_name(name: &str) -> String {
    format!("{name}.ron")
}

/// # Errors
/// Any I/O failure creating `dir`, serializing, or writing the file.
pub fn save(profile: &Profile, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let text = ron::ser::to_string_pretty(profile, ron::ser::PrettyConfig::default()).map_err(std::io::Error::other)?;
    std::fs::write(dir.join(file_name(&profile.name)), text)
}

/// A missing or unparsable profile is just `None`, not an error — matches
/// the "never fail to start" convention used throughout `studio-core`.
#[must_use]
pub fn load(name: &str, dir: &Path) -> Option<Profile> {
    let text = std::fs::read_to_string(dir.join(file_name(name))).ok()?;
    ron::from_str(&text).ok()
}

/// # Errors
/// If the profile file doesn't exist or can't be removed.
pub fn delete(name: &str, dir: &Path) -> std::io::Result<()> {
    std::fs::remove_file(dir.join(file_name(name)))
}

/// An unreadable or absent `dir` just yields no profiles.
#[must_use]
pub fn list(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "ron"))
        .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn sample(name: &str) -> Profile {
        Profile {
            name: name.to_string(),
            created: "2026-08-16T13:00:00Z".to_string(),
            last_used: "2026-08-16T13:00:00Z".to_string(),
            model: ModelRef {
                catalog_id: Some("qwen3.5-9b-q4km".to_string()),
                path: "/home/u/models/Qwen3.5-9B-Q4_K_M.gguf".to_string(),
                revision: Some("a1b2c3d".to_string()),
                model_type: "qwen3_5".to_string(),
                format: Format::Gguf,
            },
            runtime: RuntimeConfig {
                quant: None,
                dtype: Some("bf16".to_string()),
                kv_quant: Some(KvQuant::Int4),
                max_seq_len: 262_144,
                max_concurrent: 1,
                gpu_memory_limit: Some("0.92".to_string()),
                device: 0,
                text_only: false,
                prefill_chunk: None,
            },
            measured: None,
        }
    }

    #[test]
    fn saves_loads_and_deletes_a_profile() {
        let dir = TempDir::new().unwrap();
        let profile = sample("qwen35-9b-coding");
        save(&profile, dir.path()).unwrap();

        let loaded = load("qwen35-9b-coding", dir.path()).unwrap();
        assert_eq!(loaded.model.model_type, "qwen3_5");
        assert_eq!(loaded.runtime.kv_quant, Some(KvQuant::Int4));

        delete("qwen35-9b-coding", dir.path()).unwrap();
        assert!(load("qwen35-9b-coding", dir.path()).is_none());
    }

    #[test]
    fn lists_saved_profiles_sorted_by_name() {
        let dir = TempDir::new().unwrap();
        save(&sample("zebra"), dir.path()).unwrap();
        save(&sample("alpha"), dir.path()).unwrap();

        assert_eq!(list(dir.path()), vec!["alpha".to_string(), "zebra".to_string()]);
    }

    #[test]
    fn missing_dir_yields_no_profiles_rather_than_panicking() {
        assert_eq!(list(Path::new("/does/not/exist")).len(), 0);
    }
}
