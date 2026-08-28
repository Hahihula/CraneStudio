//! Catalog loading, per PLAN.md §8.1: try the live GitHub raw URL first,
//! fall back to the last-fetched copy cached on disk, fall back again to the
//! copy baked into the binary at compile time. Never fail to start because
//! the fetch failed.

use std::path::Path;
use std::time::Duration;

use super::schema::Catalog;

/// The in-repo copy, versioned alongside `CraneStudio` and embedded at
/// compile time — this is what a fully offline, first-ever run sees.
const BUNDLED: &str = include_str!("../../../../catalog/models.ron");

pub const DEFAULT_REMOTE_URL: &str =
    "https://raw.githubusercontent.com/hahihula/CraneStudio/main/catalog/models.ron";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Remote,
    Cached,
    Bundled,
}

/// # Panics
/// Only if `catalog/models.ron` itself fails to parse — a build-time asset
/// bug, not a runtime condition, so this is deliberately not a `Result`.
#[must_use]
pub fn bundled() -> Catalog {
    parse(BUNDLED).expect("bundled catalog/models.ron must parse — this is a build-time asset bug")
}

fn parse(text: &str) -> Result<Catalog, ron::error::SpannedError> {
    ron::from_str(text)
}

/// `cache_path` is read/written best-effort — a failure to persist the
/// cache never fails the load, since the bundled copy is always a valid
/// fallback.
pub async fn load(remote_url: &str, cache_path: &Path) -> (Catalog, Source) {
    if let Some(catalog) = fetch_remote(remote_url).await {
        let _ = std::fs::create_dir_all(cache_path.parent().unwrap_or(Path::new(".")));
        let _ = std::fs::write(
            cache_path,
            ron::ser::to_string_pretty(&catalog, ron::ser::PrettyConfig::default())
                .unwrap_or_default(),
        );
        return (catalog, Source::Remote);
    }

    if let Ok(text) = std::fs::read_to_string(cache_path)
        && let Ok(catalog) = parse(&text)
    {
        return (catalog, Source::Cached);
    }

    (bundled(), Source::Bundled)
}

async fn fetch_remote(url: &str) -> Option<Catalog> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let text = client
        .get(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .await
        .ok()?;
    parse(&text).ok()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::catalog::FAMILIES;
    use crate::catalog::schema::{Capability, Format};
    use crate::estimator::CONTEXT_FLOOR;

    fn has_extension(file: &str, extension: &str) -> bool {
        std::path::Path::new(file)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(extension))
    }

    #[test]
    fn bundled_catalog_parses_and_is_non_empty() {
        let catalog = bundled();
        assert!(!catalog.models.is_empty());
    }

    /// §8.1's promise is that "nothing it offers can fail to load", and the
    /// first way to break that is naming a `--model-type` Crane has never
    /// heard of. Everything below this test guards one such promise, so that
    /// adding a model stays a data-only change that can't quietly ship a
    /// broken entry.
    #[test]
    fn every_entry_names_a_family_crane_supports() {
        for model in bundled().models {
            assert!(
                FAMILIES.iter().any(|f| f.model_type == model.model_type),
                "{}: model_type {:?} is not a family Crane supports",
                model.id,
                model.model_type
            );
        }
    }

    #[test]
    fn ids_are_unique_across_models_and_variants() {
        let catalog = bundled();
        let mut model_ids = HashSet::new();
        let mut variant_ids = HashSet::new();
        for model in &catalog.models {
            assert!(
                model_ids.insert(&model.id),
                "duplicate model id {}",
                model.id
            );
            for variant in &model.variants {
                assert!(
                    variant_ids.insert(&variant.id),
                    "duplicate variant id {}",
                    variant.id
                );
            }
        }
    }

    /// Crane's `create_backend` rejects `--quant` outright for every model type
    /// except `qwen3_5`, and KV-cache compression is that same family's
    /// feature (§2.8) — an entry claiming either elsewhere would offer the
    /// wizard a knob that makes the launch fail.
    #[test]
    fn only_the_hybrid_qwen_family_claims_isq_or_kv_quant() {
        for model in bundled().models {
            if model.model_type == "qwen3_5" {
                continue;
            }
            assert!(
                !model.supports.isq,
                "{}: only qwen3_5 accepts --quant",
                model.id
            );
            assert!(
                !model.supports.kv_quant,
                "{}: KV-cache compression is qwen3_5-only (§2.8)",
                model.id
            );
        }
    }

    /// §2.11b: no family v1 targets supports KV swap, which is what pins
    /// concurrency to 1 and makes 256k reachable at all.
    #[test]
    fn no_entry_claims_kv_swap() {
        for model in bundled().models {
            assert!(
                !model.supports.kv_swap,
                "{}: kv_swap must be false",
                model.id
            );
        }
    }

    #[test]
    fn vision_is_declared_consistently_with_the_family() {
        for model in bundled().models {
            let family = FAMILIES
                .iter()
                .find(|f| f.model_type == model.model_type)
                .expect("checked by every_entry_names_a_family_crane_supports");
            assert_eq!(
                model.supports.vision, family.vision,
                "{}: supports.vision disagrees with the {} family",
                model.id, family.model_type
            );
            assert_eq!(
                model.capabilities.contains(&Capability::Vision),
                model.supports.vision,
                "{}: the Vision capability and supports.vision must agree",
                model.id
            );
        }
    }

    /// A variant below the 32k floor could never serve a usable session
    /// (§7.0, §10.3), so it doesn't belong in a curated list.
    #[test]
    fn every_entry_reaches_the_context_floor() {
        for model in bundled().models {
            assert!(
                model.native_context >= CONTEXT_FLOOR,
                "{}: native context {} is below the {CONTEXT_FLOOR} floor",
                model.id,
                model.native_context
            );
        }
    }

    /// §5: pin to a commit sha, never a floating branch — a saved profile (or
    /// a catalog entry) must not silently change meaning later.
    #[test]
    fn every_variant_pins_a_commit_sha() {
        for model in bundled().models {
            for variant in &model.variants {
                assert!(
                    variant.revision.len() == 40
                        && variant.revision.chars().all(|c| c.is_ascii_hexdigit()),
                    "{}: revision {:?} is not a commit sha",
                    variant.id,
                    variant.revision
                );
                assert!(
                    variant.download_bytes > 0,
                    "{}: download_bytes must be the real size",
                    variant.id
                );
            }
        }
    }

    /// The launcher builds a model path from a variant's own file list — one
    /// `.gguf` file for GGUF, or a directory Crane's `from_pretrained` can
    /// read for safetensors (config + tokenizer + weights). Getting this
    /// wrong surfaces as a failed launch *after* a multi-gigabyte download.
    #[test]
    fn every_variant_lists_the_files_its_format_needs() {
        for model in bundled().models {
            for variant in &model.variants {
                match variant.format {
                    Format::Gguf => {
                        let ggufs: Vec<_> = variant
                            .files
                            .iter()
                            .filter(|f| has_extension(f, "gguf"))
                            .collect();
                        assert_eq!(
                            ggufs.len(),
                            1,
                            "{}: a GGUF variant names exactly one .gguf file, got {:?}",
                            variant.id,
                            variant.files
                        );
                        assert!(
                            variant.quant.is_some(),
                            "{}: a GGUF variant states its quantization",
                            variant.id
                        );
                    }
                    Format::Safetensors => {
                        for required in ["config.json", "tokenizer.json"] {
                            assert!(
                                variant.files.iter().any(|f| f == required),
                                "{}: safetensors variants must include {required}",
                                variant.id
                            );
                        }
                        assert!(
                            variant
                                .files
                                .iter()
                                .any(|f| has_extension(f, "safetensors")),
                            "{}: safetensors variants must include weights",
                            variant.id
                        );
                        assert!(
                            variant.quant.is_none(),
                            "{}: safetensors weights are unquantized",
                            variant.id
                        );
                    }
                }
            }
        }
    }

    /// Crane resolves the chat template from `tokenizer_config.json`, falling
    /// back to a standalone `chat_template.jinja` — a safetensors entry that
    /// downloads neither gets Crane's Hunyuan fallback template applied to a
    /// model that isn't Hunyuan, which produces confidently wrong output
    /// rather than an error.
    #[test]
    fn every_safetensors_variant_downloads_a_chat_template() {
        for model in bundled().models {
            for variant in &model.variants {
                if variant.format != Format::Safetensors {
                    continue;
                }
                assert!(
                    variant
                        .files
                        .iter()
                        .any(|f| f == "tokenizer_config.json" || f == "chat_template.jinja"),
                    "{}: no chat template among {:?}",
                    variant.id,
                    variant.files
                );
            }
        }
    }

    #[tokio::test]
    async fn unreachable_remote_and_missing_cache_falls_back_to_bundled() {
        let cache = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(cache.path()).unwrap(); // ensure no cache exists
        let (catalog, source) = load("http://127.0.0.1:1/nope.ron", cache.path()).await;
        assert_eq!(source, Source::Bundled);
        assert!(!catalog.models.is_empty());
    }

    // Hits the real GitHub raw URL — not run by default (§12). The
    // CraneStudio repo doesn't publish this path yet, so today this also
    // exercises the fallback-to-bundled path, just over a real network call
    // instead of a deliberately unroutable address.
    #[tokio::test]
    #[ignore = "hits a real network API — not run by default (§12)"]
    async fn default_remote_url_falls_back_gracefully_when_unpublished() {
        let cache = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(cache.path()).unwrap();
        let (catalog, _source) = load(DEFAULT_REMOTE_URL, cache.path()).await;
        assert!(!catalog.models.is_empty());
    }

    /// The one way a "data-only" catalog entry rots: a repo owner renames a
    /// file, re-quantizes it in place, or takes the repo down. Everything the
    /// catalog claims about a download is checkable without downloading it —
    /// the file exists at the pinned sha, and its size is exactly what the
    /// entry says — so check all of it, and let CI do so on a schedule (§12).
    ///
    /// Sizes are compared as the sum over a variant's whole file list, which is
    /// what `download_bytes` means and what the progress bar divides by.
    #[tokio::test]
    #[ignore = "hits the real HuggingFace API — not run by default (§12)"]
    async fn every_variant_is_still_downloadable_at_its_pinned_sha() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let mut problems = Vec::new();

        for model in bundled().models {
            for variant in &model.variants {
                let mut total = 0u64;
                for file in &variant.files {
                    let url = format!(
                        "https://huggingface.co/{}/resolve/{}/{file}",
                        variant.repo, variant.revision
                    );
                    match client.head(&url).send().await {
                        Ok(response) if response.status().is_success() => {
                            // LFS files answer with the real size in
                            // `x-linked-size` before the redirect, and with a
                            // plain `content-length` from the CDN after it.
                            let size = response
                                .headers()
                                .get("x-linked-size")
                                .and_then(|v| v.to_str().ok())
                                .and_then(|v| v.parse::<u64>().ok())
                                .or_else(|| response.content_length());
                            match size {
                                Some(size) => total += size,
                                None => problems
                                    .push(format!("{}: {file}: no size reported", variant.id)),
                            }
                        }
                        Ok(response) => problems.push(format!(
                            "{}: {file}: HTTP {}",
                            variant.id,
                            response.status()
                        )),
                        Err(e) => problems.push(format!("{}: {file}: {e}", variant.id)),
                    }
                }
                if total != variant.download_bytes {
                    problems.push(format!(
                        "{}: download_bytes says {} but the files add up to {total}",
                        variant.id, variant.download_bytes
                    ));
                }
            }
        }

        assert!(
            problems.is_empty(),
            "the catalog no longer matches what HuggingFace serves:\n  {}",
            problems.join("\n  ")
        );
    }
}
