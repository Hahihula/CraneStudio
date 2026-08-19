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
    Family {
        model_type: "minicpmv4_6",
        config_aliases: &["minicpmv4_6", "minicpmv4.6"],
        // Same hybrid GDN + attention text tower as Qwen 3.5 under
        // `text_config` (verified live against a real checkpoint) — but
        // never ships as GGUF (§2.12-style: no mmproj/vision loader for
        // any GGUF path), so never matched from a header.
        gguf_architectures: &[],
        vision: true,
        gated: false,
    },
    Family {
        model_type: "minicpm5",
        // Deliberately empty, not omitted: real checkpoints' `config.json`
        // declares `"model_type": "llama"` / `architectures:
        // ["LlamaForCausalLM"]` — indistinguishable from an actual Llama
        // checkpoint by content alone (verified against a real
        // `openbmb/MiniCPM5-1B` checkpoint). Adding "llama" as an alias
        // here would misclassify every genuine (unsupported) Llama
        // checkpoint as supported, so `from_config`/`from_gguf_architecture`
        // must never match this entry directly. Two other paths reach it
        // instead, both verified against the real
        // `crane-serve/src/engine/model_factory.rs::detect_model_type`:
        // `from_path_name` mirrors its own path-name-heuristic last
        // resort (used by `catalog::local`/`catalog::hf` when content
        // classification finds "llama"), and the catalog (§8.1) sets
        // `model_type: "minicpm5"` explicitly on its entry, bypassing
        // detection entirely (`studio_tui::screens::download::known_candidate`).
        config_aliases: &[],
        gguf_architectures: &[],
        vision: false,
        gated: false,
    },
];

/// Path-name fallback for the one family whose real content (`config.json`
/// or GGUF header) is genuinely indistinguishable from an unsupported
/// architecture: `MiniCPM5` declares itself plain `"llama"`. Mirrors
/// `detect_model_type`'s own last-resort path-name heuristic, including its
/// precedence — "MiniCPM-V"/"MiniCPM-O" are checked first since both names
/// also contain the bare "minicpm" substring, and would otherwise be
/// mis-claimed by it (verified against the real source; MiniCPM-V's own
/// `config.json` is unambiguous in practice, but a corrupted/incomplete one
/// falling through to this path should still not be misread as `MiniCPM5`).
///
/// Only meant to run over a *local path or repo id*, and only when content
/// classification specifically found `"llama"` — see `classify::apply_path_hint`,
/// the caller responsible for that gating.
#[must_use]
pub fn from_path_name(path: &str) -> Option<&'static Family> {
    let p = path.to_lowercase();
    if p.contains("minicpm-v")
        || p.contains("minicpmv")
        || p.contains("minicpm-o")
        || p.contains("minicpmo")
    {
        return None;
    }
    if p.contains("minicpm") {
        return FAMILIES.iter().find(|f| f.model_type == "minicpm5");
    }
    None
}

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
    fn from_path_name_finds_minicpm5_but_not_v_or_o_variants() {
        assert_eq!(
            from_path_name("/home/u/models/openbmb/MiniCPM5-1B-GGUF/model.gguf")
                .unwrap()
                .model_type,
            "minicpm5"
        );
        assert_eq!(
            from_path_name("/home/u/models/MiniCPM-V-4.6").map(|f| f.model_type),
            None
        );
        assert_eq!(
            from_path_name("/home/u/models/MiniCPM-o-4_5").map(|f| f.model_type),
            None
        );
        assert_eq!(
            from_path_name("/home/u/models/Llama-3.2-1B").map(|f| f.model_type),
            None
        );
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
    fn minicpm_v_config_type_resolves_to_the_vl_family() {
        let vl = from_config(Some("minicpmv4_6"), &[], true).unwrap();
        assert_eq!(vl.model_type, "minicpmv4_6");
        assert!(from_config(Some("minicpmv4_6"), &[], false).is_none());
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
