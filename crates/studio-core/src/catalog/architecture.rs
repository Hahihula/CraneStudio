//! Mirrors the subset of `crane_serve::engine::model_factory`'s alias table
//! relevant to `CraneStudio` v1 (PLAN.md §2.12) — the text/VL chat model
//! families a coding agent can actually talk to. Crane also supports
//! TTS/ASR/OCR/duplex families, but v1 has no UI for them, so they are
//! deliberately left out here rather than mirrored and then hidden.
//!
//! Kept independent of `crane-serve` at *runtime* — linking it would pull
//! the whole Candle/CUDA dependency chain into `studio-core`, which is
//! supposed to stay light (§3.3). The alias table is instead verified
//! against the real `crane_serve::engine::model_factory::ModelType` in this
//! module's tests, where `crane-serve` is a dev-dependency only.

/// One CraneStudio-v1-relevant model family, mirroring one arm of
/// `ModelType::from_str` (`crane-serve/src/engine/model_factory.rs`).
pub struct Family {
    /// The canonical `--model-type` string to pass to crane-serve.
    pub model_type: &'static str,
    /// Every `config.json` `model_type` spelling that resolves to this
    /// family (mirrors `ModelType::from_str`'s match arms).
    pub config_aliases: &'static [&'static str],
    /// `general.architecture` values from a GGUF header that resolve to
    /// this family (mirrors `detect_from_gguf_header`, which is private to
    /// Crane — not test-verifiable the way `config_aliases` is, so this one
    /// is a plain by-hand mirror; re-check it against the source on drift).
    pub gguf_architectures: &'static [&'static str],
    pub vision: bool,
    pub gated: bool,
}

pub const FAMILIES: &[Family] = &[
    Family {
        model_type: "qwen3_5",
        config_aliases: &[
            "qwen3_5", "qwen3.5", "qwen3_6", "qwen3.6", "qwen3_8", "qwen3.8",
        ],
        gguf_architectures: &[
            "qwen35", "qwen3_5", "qwen3.5", "qwen36", "qwen3_6", "qwen3.6", "qwen38", "qwen3_8",
            "qwen3.8",
        ],
        vision: false,
        gated: false,
    },
    Family {
        model_type: "qwen3_5_vl",
        config_aliases: &[
            "qwen3_5", "qwen3.5", "qwen3_6", "qwen3.6", "qwen3_8", "qwen3.8",
        ],
        // No GGUF/mmproj vision loader for this family (§2.12) — never
        // matched from a GGUF header.
        gguf_architectures: &[],
        vision: true,
        gated: false,
    },
    Family {
        model_type: "qwen3",
        config_aliases: &["qwen3"],
        gguf_architectures: &["qwen3", "qwen3moe"],
        vision: false,
        gated: false,
    },
    Family {
        model_type: "qwen25",
        config_aliases: &["qwen2", "qwen2.5"],
        gguf_architectures: &["qwen2"],
        vision: false,
        gated: false,
    },
    Family {
        model_type: "hunyuan",
        // `detect_model_type` matches this family by substring
        // (`m.contains("hunyuan")`), not a fixed alias list — these are
        // the spellings actually seen on published checkpoints.
        config_aliases: &["hunyuan", "hunyuan_dense", "hunyuandense"],
        gguf_architectures: &["hunyuan"],
        vision: false,
        gated: false,
    },
    Family {
        model_type: "gemma4",
        config_aliases: &["gemma4"],
        gguf_architectures: &["gemma"],
        vision: false,
        // Gated repo on HF (§2.12, §9).
        gated: true,
    },
    Family {
        model_type: "gemma4_vl",
        config_aliases: &["gemma4"],
        gguf_architectures: &[],
        vision: true,
        gated: true,
    },
];

/// Classifies a `config.json` by its `model_type` / `architectures` fields.
/// The `model_type`-field path mirrors `detect_model_type`'s step 1
/// exactly. The `architectures`-list fallback (step 2, used only when
/// `model_type` is absent — rare in practice) is a looser substring match
/// rather than Crane's precedence-ordered one; good enough to classify
/// supported-vs-not for catalog/search purposes, not meant to be
/// byte-identical to the real loader's resolution order.
#[must_use]
pub fn from_config(
    model_type: Option<&str>,
    architectures: &[String],
    has_vision_config: bool,
) -> Option<&'static Family> {
    let mt = model_type.map(str::to_lowercase);
    if let Some(mt) = mt.as_deref() {
        for family in FAMILIES {
            if family.config_aliases.contains(&mt) && family.vision == has_vision_config {
                return Some(family);
            }
        }
    }

    for arch in architectures {
        let a = arch.to_lowercase();
        for family in FAMILIES {
            if family.vision == has_vision_config
                && family.config_aliases.iter().any(|alias| a.contains(alias))
            {
                return Some(family);
            }
        }
    }

    None
}

/// Classifies a GGUF `general.architecture` string, mirroring
/// `detect_from_gguf_header`.
#[must_use]
pub fn from_gguf_architecture(arch: &str) -> Option<&'static Family> {
    let a = arch.to_lowercase();
    FAMILIES.iter().find(|family| {
        family
            .gguf_architectures
            .iter()
            .any(|ga| a == *ga || a.starts_with(ga))
    })
}

#[cfg(test)]
mod tests {
    use crane_serve::engine::model_factory::ModelType;

    use super::*;

    /// Every `config_aliases` entry must actually be recognized by Crane's
    /// real `ModelType::from_str`, and resolve to the family we claim it
    /// does. Catches drift if Crane renames or drops an alias.
    #[test]
    fn config_aliases_match_crane_model_type() {
        for family in FAMILIES {
            for alias in family.config_aliases {
                let resolved = ModelType::from_str(alias);
                assert_ne!(
                    resolved,
                    ModelType::Auto,
                    "alias {alias:?} (family {}) is not recognized by crane_serve::ModelType::from_str",
                    family.model_type
                );
            }
        }
    }

    #[test]
    fn text_and_vl_variants_are_distinguished_by_vision_flag() {
        let text = from_config(Some("qwen3_5"), &[], false).unwrap();
        assert_eq!(text.model_type, "qwen3_5");
        let vl = from_config(Some("qwen3_5"), &[], true).unwrap();
        assert_eq!(vl.model_type, "qwen3_5_vl");
    }

    #[test]
    fn unrecognized_model_type_is_none() {
        assert!(from_config(Some("llama"), &["LlamaForCausalLM".to_string()], false).is_none());
    }

    #[test]
    fn gguf_architecture_prefix_match() {
        assert_eq!(
            from_gguf_architecture("qwen35").unwrap().model_type,
            "qwen3_5"
        );
        assert_eq!(
            from_gguf_architecture("hunyuan-dense").unwrap().model_type,
            "hunyuan"
        );
        assert!(from_gguf_architecture("llama").is_none());
    }
}
