//! Finding the `DSpark` drafter that goes with a MiniCPM-V 4.7 model, so the
//! launch can offer speculative decoding (crane-serve's `--dspark`).
//!
//! A catalog download knows its drafter exactly (`Variant::drafter`). A model
//! found anywhere else is matched by the layouts `OpenBMB` publishes under:
//! a `dspark-*.gguf` in the model's own directory (the GGUF repos ship an
//! F16 one), or a sibling directory named after the model plus `-DSpark`
//! (`MiniCPM-V-4.7-1B` → `MiniCPM-V-4.7-1B-DSpark`, `MiniCPM-V-4.7-1B-gguf`
//! → `MiniCPM-V-4.7-1B-DSpark-gguf`).

use std::fs;
use std::path::{Path, PathBuf};

use super::classify::{ConfigJson, DRAFTER_GGUF_ARCHITECTURE};
use super::gguf;
use super::schema::{Catalog, Format};

/// The drafter to offer for `model`, best match first; `None` when there is
/// none on disk. Only MiniCPM-V has drafters — callers check the model type.
#[must_use]
pub fn find(
    model: &Path,
    format: Format,
    catalog: Option<&Catalog>,
    models_dir: &Path,
) -> Option<PathBuf> {
    if let Some(drafter) = catalog
        .and_then(|c| c.variant_for_path(model))
        .and_then(|(_, variant)| variant.drafter.as_ref())
    {
        let path = drafter.local_path(models_dir);
        if is_drafter(&path) {
            return Some(path);
        }
    }
    find_nearby(model, format)
}

fn find_nearby(model: &Path, format: Format) -> Option<PathBuf> {
    // The directory that names the model: the checkpoint directory itself,
    // or the one holding the GGUF.
    let model_dir = match format {
        Format::Safetensors => model,
        Format::Gguf => model.parent()?,
    };
    let mut found = Vec::new();
    if format == Format::Gguf {
        found.extend(drafters_in(model_dir));
    }
    let stem = model_dir
        .file_name()?
        .to_string_lossy()
        .to_lowercase()
        .trim_end_matches("-gguf")
        .to_string();
    if let Some(entries) = model_dir.parent().and_then(|p| fs::read_dir(p).ok()) {
        for dir in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if name
                .strip_prefix(&stem)
                .is_some_and(|rest| rest.starts_with("-dspark"))
            {
                if is_drafter(&dir) {
                    found.push(dir);
                } else {
                    found.extend(drafters_in(&dir));
                }
            }
        }
    }

    let quant = match format {
        Format::Gguf => gguf_quant(model),
        Format::Safetensors => None,
    };
    found.sort_by_key(|p| {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let drafter_format = if is_gguf(p) {
            Format::Gguf
        } else {
            Format::Safetensors
        };
        (
            drafter_format != format,
            !quant.as_ref().is_some_and(|q| name.contains(q.as_str())),
            !name.contains("q8_0"),
            name,
        )
    });
    found.into_iter().next()
}

/// Working memory a loaded drafter adds beyond its weights — its own KV
/// cache and draft buffers. Measured on an RTX 3090 with the 1B drafter:
/// +480 MiB in total for 407 MiB of BF16 weights, +576 MiB for the `Q8_0`
/// GGUF (427 MiB once dequantized).
const DRAFTER_WORKING_BYTES: u64 = 160 * 1024 * 1024;

/// Predicted VRAM a drafter takes once loaded. crane-core dequantizes a GGUF
/// drafter to the compute dtype at load, so a quantized file is scaled back
/// up to 16 bits a weight rather than counted at its size on disk.
#[must_use]
pub fn loaded_bytes(drafter: &Path) -> u64 {
    let weights = if drafter.is_dir() {
        crate::estimator::safetensors_dir_bytes(drafter).unwrap_or(0)
    } else {
        let on_disk = fs::metadata(drafter).map_or(0, |m| m.len());
        let bits = gguf_quant(drafter).map_or(16.0, |q| bits_per_weight(&q));
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let dequantized = (on_disk as f64 * 16.0 / bits) as u64;
        dequantized
    };
    weights + DRAFTER_WORKING_BYTES
}

/// Effective bits a weight for llama.cpp's quant types — block scales
/// included, so a file's size divided by it gives its weight count.
fn bits_per_weight(quant: &str) -> f64 {
    match quant {
        "q8_0" => 8.5,
        "q6_k" => 6.56,
        "q5_1" => 6.0,
        "q5_0" | "q5_k_s" | "q5_k_m" => 5.5,
        "q4_1" => 5.0,
        "q4_k_m" => 4.85,
        "q4_0" | "q4_k_s" => 4.5,
        "f32" => 32.0,
        _ => 16.0,
    }
}

/// A drafter checkpoint directory or `DSpark` GGUF file.
fn is_drafter(path: &Path) -> bool {
    if path.is_dir() {
        fs::read(path.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ConfigJson>(&bytes).ok())
            .is_some_and(|config| config.is_drafter())
    } else {
        is_gguf(path)
            && fs::File::open(path)
                .ok()
                .and_then(|mut f| gguf::read_architecture(&mut f))
                .is_some_and(|arch| arch.eq_ignore_ascii_case(DRAFTER_GGUF_ARCHITECTURE))
    }
}

fn drafters_in(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_drafter(p))
                .collect()
        })
        .unwrap_or_default()
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// `MiniCPM-V-4_7-1B-Q8_0.gguf` → `q8_0`: the last `-` segment of the stem.
fn gguf_quant(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().to_lowercase();
    stem.rsplit_once('-').map(|(_, quant)| quant.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::TempDir;

    use super::super::schema::{Drafter, Variant};
    use super::*;

    fn write_gguf(path: &Path, arch: &str) {
        let key = "general.architecture";
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch.as_bytes());
        fs::write(path, buf).unwrap();
    }

    fn write_drafter_dir(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type": "qwen3", "architectures": ["Qwen3DSparkModel"]}"#,
        )
        .unwrap();
    }

    /// The layout `crane-model-download` (and a plain `hf download` per
    /// repo) leaves behind: every repo in its own sibling directory.
    fn openbmb_layout(root: &Path) {
        let target = root.join("MiniCPM-V-4.7-1B");
        fs::create_dir_all(&target).unwrap();
        fs::write(
            target.join("config.json"),
            r#"{"model_type": "minicpmv4_7"}"#,
        )
        .unwrap();
        write_drafter_dir(&root.join("MiniCPM-V-4.7-1B-DSpark"));

        let gguf = root.join("MiniCPM-V-4.7-1B-gguf");
        fs::create_dir_all(&gguf).unwrap();
        write_gguf(&gguf.join("MiniCPM-V-4_7-1B-Q4_K_M.gguf"), "qwen35");
        write_gguf(&gguf.join("MiniCPM-V-4_7-1B-Q8_0.gguf"), "qwen35");
        let drafters = root.join("MiniCPM-V-4.7-1B-DSpark-gguf");
        fs::create_dir_all(&drafters).unwrap();
        write_gguf(
            &drafters.join("dspark-MiniCPM-V-4_7-1B-Q4_K_M.gguf"),
            "dflash",
        );
        write_gguf(
            &drafters.join("dspark-MiniCPM-V-4_7-1B-Q8_0.gguf"),
            "dflash",
        );

        // A different model's drafter must never be picked up.
        write_drafter_dir(&root.join("MiniCPM-V-4.7-35B-A3B-DSpark"));
    }

    #[test]
    fn safetensors_target_gets_the_sibling_drafter_dir() {
        let root = TempDir::new().unwrap();
        openbmb_layout(root.path());
        let found = find(
            &root.path().join("MiniCPM-V-4.7-1B"),
            Format::Safetensors,
            None,
            root.path(),
        );
        assert_eq!(found, Some(root.path().join("MiniCPM-V-4.7-1B-DSpark")));
    }

    #[test]
    fn gguf_target_gets_the_drafter_of_the_same_quant() {
        let root = TempDir::new().unwrap();
        openbmb_layout(root.path());
        let gguf = root.path().join("MiniCPM-V-4.7-1B-gguf");
        let drafters = root.path().join("MiniCPM-V-4.7-1B-DSpark-gguf");
        assert_eq!(
            find(
                &gguf.join("MiniCPM-V-4_7-1B-Q4_K_M.gguf"),
                Format::Gguf,
                None,
                root.path()
            ),
            Some(drafters.join("dspark-MiniCPM-V-4_7-1B-Q4_K_M.gguf"))
        );
        assert_eq!(
            find(
                &gguf.join("MiniCPM-V-4_7-1B-Q8_0.gguf"),
                Format::Gguf,
                None,
                root.path()
            ),
            Some(drafters.join("dspark-MiniCPM-V-4_7-1B-Q8_0.gguf"))
        );
    }

    #[test]
    fn a_drafter_in_the_gguf_repo_itself_is_found() {
        let root = TempDir::new().unwrap();
        let gguf = root.path().join("MiniCPM-V-4.7-1B-gguf");
        fs::create_dir_all(&gguf).unwrap();
        write_gguf(&gguf.join("MiniCPM-V-4_7-1B-Q8_0.gguf"), "qwen35");
        write_gguf(&gguf.join("dspark-MiniCPM-V-4_7-1B-F16.gguf"), "dflash");
        assert_eq!(
            find(
                &gguf.join("MiniCPM-V-4_7-1B-Q8_0.gguf"),
                Format::Gguf,
                None,
                root.path()
            ),
            Some(gguf.join("dspark-MiniCPM-V-4_7-1B-F16.gguf"))
        );
    }

    #[test]
    fn a_quantized_gguf_drafter_is_predicted_at_its_dequantized_size() {
        let root = TempDir::new().unwrap();
        let q8 = root.path().join("dspark-MiniCPM-V-4_7-1B-Q8_0.gguf");
        let f16 = root.path().join("dspark-MiniCPM-V-4_7-1B-F16.gguf");
        // The real 1B drafter files' sizes.
        fs::File::create(&q8).unwrap().set_len(237_879_552).unwrap();
        fs::File::create(&f16)
            .unwrap()
            .set_len(438_065_472)
            .unwrap();
        let (q8, f16) = (loaded_bytes(&q8), loaded_bytes(&f16));
        let diff = q8.abs_diff(f16);
        assert!(diff < f16 / 20, "Q8_0 {q8} vs F16 {f16}");
    }

    #[test]
    fn no_drafter_means_none() {
        let root = TempDir::new().unwrap();
        let target = root.path().join("MiniCPM-V-4.6");
        fs::create_dir_all(&target).unwrap();
        assert_eq!(find(&target, Format::Safetensors, None, root.path()), None);
    }

    #[test]
    fn a_catalog_variant_uses_its_own_downloaded_drafter() {
        let models = TempDir::new().unwrap();
        let target = models.path().join("openbmb/MiniCPM-V-4.7-1B-gguf/abc");
        fs::create_dir_all(&target).unwrap();
        write_gguf(&target.join("MiniCPM-V-4_7-1B-Q8_0.gguf"), "qwen35");
        let drafter_dir = models
            .path()
            .join("openbmb/MiniCPM-V-4.7-1B-DSpark-gguf/def");
        fs::create_dir_all(&drafter_dir).unwrap();
        write_gguf(
            &drafter_dir.join("dspark-MiniCPM-V-4_7-1B-Q8_0.gguf"),
            "dflash",
        );

        let catalog: Catalog = Catalog {
            schema_version: 1,
            updated: String::new(),
            models: vec![super::super::schema::ModelEntry {
                id: "m".to_string(),
                display_name: "m".to_string(),
                family: "minicpmv4_6".to_string(),
                model_type: "minicpmv4_6".to_string(),
                params: 1,
                native_context: 262_144,
                capabilities: Vec::new(),
                license: "apache-2.0".to_string(),
                gated: false,
                supports: super::super::schema::Supports {
                    isq: false,
                    kv_quant: false,
                    kv_swap: false,
                    vision: true,
                },
                reasoning_effort: None,
                variants: vec![Variant {
                    id: "v".to_string(),
                    repo: "openbmb/MiniCPM-V-4.7-1B-gguf".to_string(),
                    revision: "abc".to_string(),
                    files: vec!["MiniCPM-V-4_7-1B-Q8_0.gguf".to_string()],
                    format: Format::Gguf,
                    quant: None,
                    download_bytes: 1,
                    extra_repos: Vec::new(),
                    drafter: Some(Drafter {
                        repo: "openbmb/MiniCPM-V-4.7-1B-DSpark-gguf".to_string(),
                        revision: "def".to_string(),
                        files: vec!["dspark-MiniCPM-V-4_7-1B-Q8_0.gguf".to_string()],
                        format: Format::Gguf,
                        download_bytes: 1,
                    }),
                    measured: HashMap::new(),
                }],
            }],
        };
        assert_eq!(
            find(
                &target.join("MiniCPM-V-4_7-1B-Q8_0.gguf"),
                Format::Gguf,
                Some(&catalog),
                models.path()
            ),
            Some(drafter_dir.join("dspark-MiniCPM-V-4_7-1B-Q8_0.gguf"))
        );
    }
}
