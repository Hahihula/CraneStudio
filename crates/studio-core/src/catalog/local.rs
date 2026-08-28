//! Local filesystem model scan, per PLAN.md §8.3: walk the models directory
//! (plus, in the TUI, an explicit "add local path" action not modeled
//! here), and run the same classification logic §8.2 uses on HF search
//! results, so a local checkpoint gets exactly the same verdict a remote
//! one would.

use std::fs;
use std::path::{Path, PathBuf};

use super::classify::{self, Classification, ConfigJson};
use super::gguf;
use super::schema::Format;

#[derive(Debug, Clone)]
pub struct LocalCandidate {
    pub path: PathBuf,
    pub format: Format,
    pub classification: Classification,
}

const MAX_DEPTH: usize = 6;

/// `root` need not exist — an absent or unreadable directory just yields no
/// candidates, matching §6's "never fail to start" spirit.
///
/// `root` itself counts as a candidate when it *is* a checkpoint (a directory
/// holding `config.json`, or a single `.gguf` file). Without that, scanning a
/// model directory directly — `cranestudio catalog scan ~/models/some-model`,
/// or the post-download classification of a safetensors repo, whose destination
/// directory is exactly the checkpoint — reported "no local models found" for a
/// model that was plainly right there.
#[must_use]
pub fn scan(root: &Path) -> Vec<LocalCandidate> {
    let mut out = Vec::new();
    if root.is_file() {
        if root
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(classify_gguf_file(root));
        }
        return out;
    }
    if root.join("config.json").is_file() {
        out.push(classify_safetensors_dir(root));
        return out;
    }
    walk(root, 0, &mut out);
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<LocalCandidate>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            // A directory with its own config.json is one candidate, not a
            // subtree to keep descending into.
            if path.join("config.json").is_file() {
                out.push(classify_safetensors_dir(&path));
            } else {
                subdirs.push(path);
            }
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(classify_gguf_file(&path));
        }
    }

    for subdir in subdirs {
        walk(&subdir, depth + 1, out);
    }
}

fn classify_safetensors_dir(path: &Path) -> LocalCandidate {
    let classification =
        read_config_json(&path.join("config.json")).unwrap_or_else(|| Classification::Unknown {
            reason: "config.json is missing or unreadable".to_string(),
        });
    let classification = classify::apply_path_hint(classification, &path.to_string_lossy());
    LocalCandidate {
        path: path.to_path_buf(),
        format: Format::Safetensors,
        classification,
    }
}

/// Mirrors `detect_model_type`'s own precedence: even a standalone `.gguf`
/// *file* is checked against a sibling `config.json` in its parent
/// directory first (some GGUF repos ship both), before falling back to the
/// GGUF header itself.
fn classify_gguf_file(path: &Path) -> LocalCandidate {
    let sibling_config = path.parent().map(|parent| parent.join("config.json"));
    let classification = sibling_config
        .filter(|p| p.is_file())
        .and_then(|p| read_config_json(&p))
        .or_else(|| classify_from_gguf_header(path))
        .unwrap_or_else(|| Classification::Unknown {
            reason: "could not read a GGUF architecture header".to_string(),
        });
    let classification = classify::apply_path_hint(classification, &path.to_string_lossy());
    LocalCandidate {
        path: path.to_path_buf(),
        format: Format::Gguf,
        classification,
    }
}

fn read_config_json(path: &Path) -> Option<Classification> {
    let bytes = fs::read(path).ok()?;
    let config: ConfigJson = serde_json::from_slice(&bytes).ok()?;
    Some(config.classify())
}

fn classify_from_gguf_header(path: &Path) -> Option<Classification> {
    let mut file = fs::File::open(path).ok()?;
    let arch = gguf::read_architecture(&mut file)?;
    Some(super::classify::classify_gguf_architecture(&arch))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn classifies_a_supported_safetensors_dir() {
        let root = TempDir::new().unwrap();
        let model_dir = root.path().join("Qwen3.5-4B");
        fs::create_dir(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type": "qwen3_5"}"#,
        )
        .unwrap();

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].format, Format::Safetensors);
        assert!(matches!(
            candidates[0].classification,
            Classification::Supported {
                model_type: "qwen3_5",
                ..
            }
        ));
    }

    #[test]
    fn classifies_an_unsupported_safetensors_dir_with_a_reason() {
        let root = TempDir::new().unwrap();
        let model_dir = root.path().join("Llama-3");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        match &candidates[0].classification {
            Classification::Unsupported { reason, .. } => assert!(reason.contains("llama")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn a_config_json_directory_is_not_recursed_into() {
        let root = TempDir::new().unwrap();
        let model_dir = root.path().join("outer");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
        // A nested checkpoint-shaped directory that should never be visited,
        // since `outer` itself is already one candidate.
        let nested = model_dir.join("checkpoint-500");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("config.json"), r#"{"model_type": "qwen25"}"#).unwrap();

        assert_eq!(scan(root.path()).len(), 1);
    }

    /// Scanning a checkpoint directly is what `cranestudio catalog scan <dir>`
    /// and the post-download classification of a safetensors repo both do, and
    /// both used to come back empty-handed.
    #[test]
    fn the_scan_root_can_itself_be_the_checkpoint() {
        let root = TempDir::new().unwrap();
        fs::write(
            root.path().join("config.json"),
            r#"{"model_type": "qwen3"}"#,
        )
        .unwrap();
        fs::write(root.path().join("model.safetensors"), b"weights").unwrap();

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].path, root.path());
        assert_eq!(candidates[0].format, Format::Safetensors);
    }

    #[test]
    fn a_scan_root_that_is_a_single_gguf_file_is_still_one_candidate() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("model.gguf");
        fs::write(&path, b"not really a gguf").unwrap();

        let candidates = scan(&path);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].path, path);
        assert_eq!(candidates[0].format, Format::Gguf);
    }

    #[test]
    fn a_scan_root_that_is_an_unrelated_file_yields_nothing() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("notes.txt");
        fs::write(&path, b"hello").unwrap();

        assert!(scan(&path).is_empty());
    }

    #[test]
    fn a_bare_gguf_file_falls_back_to_the_gguf_header() {
        let root = TempDir::new().unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let value = "qwen35";
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
        fs::write(root.path().join("model.gguf"), buf).unwrap();

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].format, Format::Gguf);
        assert!(matches!(
            candidates[0].classification,
            Classification::Supported {
                model_type: "qwen3_5",
                ..
            }
        ));
    }

    fn write_gguf_with_architecture(path: &Path, arch: &str) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch.as_bytes());
        fs::write(path, buf).unwrap();
    }

    /// `MiniCPM5`'s real ambiguity (§ the `architecture::from_path_name` docs):
    /// its GGUF header says plain "llama", so a directory named after the
    /// real HF repo must still classify as supported via the path-name
    /// fallback — the exact gap that made a genuinely-downloaded `MiniCPM5`
    /// GGUF show up as unsupported in the Local tab.
    #[test]
    fn a_llama_gguf_under_a_minicpm_named_dir_is_reclassified() {
        let root = TempDir::new().unwrap();
        let dir = root.path().join("openbmb").join("MiniCPM5-1B-GGUF");
        fs::create_dir_all(&dir).unwrap();
        write_gguf_with_architecture(&dir.join("MiniCPM5-1B-Q8_0.gguf"), "llama");

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        assert!(
            matches!(
                candidates[0].classification,
                Classification::Supported {
                    model_type: "minicpm5",
                    ..
                }
            ),
            "{:?}",
            candidates[0].classification
        );
    }

    /// A real, unrelated Llama checkpoint must not be swept up by the
    /// `MiniCPM5` path-name fallback just because "llama" was detected.
    #[test]
    fn a_real_llama_gguf_stays_unsupported() {
        let root = TempDir::new().unwrap();
        write_gguf_with_architecture(&root.path().join("Llama-3.2-1B.gguf"), "llama");

        let candidates = scan(root.path());
        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].classification,
            Classification::Unsupported { .. }
        ));
    }

    #[test]
    fn missing_root_yields_no_candidates_rather_than_panicking() {
        assert_eq!(scan(Path::new("/does/not/exist")).len(), 0);
    }
}
