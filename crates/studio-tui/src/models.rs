//! The launchpad's model list is a *thing on disk*, not a catalog entry — so
//! it needs the details a catalog row already carries and a filesystem scan
//! doesn't: how big the weights are, which quantization they are, and a
//! human-readable name that isn't a 90-character path.
//!
//! Collected on a background task (`app::spawn_local_scan`) because it stats
//! every candidate file.

use std::path::Path;

use studio_core::catalog::Classification;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::schema::Format;
use studio_core::estimator::safetensors_dir_bytes;

#[derive(Debug, Clone)]
pub struct LocalModel {
    pub candidate: LocalCandidate,
    /// Filename (GGUF) or directory name (safetensors), without extension.
    pub name: String,
    /// `org/repo`, recovered from the models-directory layout when the file
    /// lives under one — that's where the downloader puts things.
    pub repo: Option<String>,
    /// `Q4_K_M`, `BF16`, … as advertised in the filename. Not derived from the
    /// GGUF header: reading every candidate's header just to label a list row
    /// would make the scan much slower than the name lookup it replaces.
    pub quant: Option<String>,
    pub size: u64,
    pub supported: bool,
    /// Crane's `--model-type` for this file, when it's supported.
    pub model_type: Option<String>,
    /// Why it can't be launched, when it can't.
    pub reason: Option<String>,
}

impl LocalModel {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.candidate.path
    }

    #[must_use]
    pub fn format_label(&self) -> &'static str {
        match self.candidate.format {
            Format::Gguf => "GGUF",
            Format::Safetensors => "safetensors",
        }
    }
}

/// Scans `root` and enriches every candidate. Supported models sort first —
/// the launchpad's whole point is that the top of the list is launchable.
#[must_use]
pub fn collect(root: &Path) -> Vec<LocalModel> {
    let mut models: Vec<LocalModel> = studio_core::catalog::local::scan(root)
        .into_iter()
        .map(|candidate| describe(root, candidate))
        .collect();
    models.sort_by(|a, b| {
        b.supported
            .cmp(&a.supported)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    models
}

fn describe(root: &Path, candidate: LocalCandidate) -> LocalModel {
    let name = display_name(&candidate.path, candidate.format);
    let size = match candidate.format {
        Format::Gguf => std::fs::metadata(&candidate.path).map_or(0, |m| m.len()),
        Format::Safetensors => safetensors_dir_bytes(&candidate.path).unwrap_or(0),
    };
    let (supported, model_type, reason) = match &candidate.classification {
        Classification::Supported { model_type, .. } => {
            (true, Some((*model_type).to_string()), None)
        }
        Classification::Unsupported { reason, .. } | Classification::Unknown { reason } => {
            (false, None, Some(reason.clone()))
        }
    };

    LocalModel {
        repo: repo_of(root, &candidate.path),
        quant: quant_of(&name),
        name,
        size,
        supported,
        model_type,
        reason,
        candidate,
    }
}

fn display_name(path: &Path, format: Format) -> String {
    let raw = match format {
        Format::Gguf => path.file_stem(),
        Format::Safetensors => path.file_name(),
    };
    raw.map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().to_string(),
    )
}

/// `<models_dir>/<org>/<repo>/<revision>/file.gguf` → `org/repo`. Anything
/// outside the models directory (an "add local path" model) has no repo, and
/// gets its parent directory shown instead by the caller.
fn repo_of(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let parts: Vec<String> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if parts.len() >= 3 {
        Some(format!("{}/{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// Pulls a quantization label out of a filename: `…-Q4_K_M.gguf` → `Q4_K_M`.
fn quant_of(name: &str) -> Option<String> {
    const FLOATS: [&str; 5] = ["BF16", "FP16", "F16", "F32", "FP32"];
    let upper = name.to_uppercase();
    for token in upper.split(['-', '.', '_', ' ']) {
        if FLOATS.contains(&token) {
            return Some(token.to_string());
        }
    }
    // Q-quants carry underscores of their own (Q4_K_M), so they're matched
    // against the whole name rather than a split token.
    let bytes: Vec<char> = upper.chars().collect();
    for (i, ch) in bytes.iter().enumerate() {
        if *ch != 'Q' || (i > 0 && bytes[i - 1].is_alphanumeric()) {
            continue;
        }
        let mut end = i + 1;
        if !bytes.get(end).is_some_and(char::is_ascii_digit) {
            continue;
        }
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == '_') {
            end += 1;
        }
        let token: String = bytes[i..end].iter().collect();
        let token = token.trim_end_matches('_').to_string();
        if token.len() >= 2 {
            return Some(token);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_q_quants_and_float_types_in_filenames() {
        assert_eq!(
            quant_of("qwen3.5-9b-instruct-Q4_K_M"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(quant_of("gemma-4-4b-it-q6_k"), Some("Q6_K".to_string()));
        assert_eq!(quant_of("model-bf16"), Some("BF16".to_string()));
        assert_eq!(quant_of("mystery-model"), None);
    }

    #[test]
    fn recovers_the_repo_from_the_models_directory_layout() {
        let root = Path::new("/models");
        assert_eq!(
            repo_of(root, Path::new("/models/unsloth/Qwen3.5-9B-GGUF/abc123/m.gguf")),
            Some("unsloth/Qwen3.5-9B-GGUF".to_string())
        );
        assert_eq!(repo_of(root, Path::new("/elsewhere/m.gguf")), None);
    }
}
