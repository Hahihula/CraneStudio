//! Shared "is this architecture supported" verdict, used by both the local
//! filesystem scan (§8.3) and `HuggingFace` search (§8.2) so the two agree.

use serde::Deserialize;

use super::architecture;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Classification {
    Supported {
        model_type: &'static str,
        vision: bool,
        gated: bool,
        audio: bool,
    },
    /// A real architecture was found, but it's not one of Crane's.
    Unsupported {
        detected: Option<String>,
        reason: String,
    },
    /// Couldn't even determine an architecture (no config.json, no GGUF
    /// header we could read).
    Unknown { reason: String },
}

/// Minimal subset of a `HuggingFace` `config.json`, mirroring the
/// `HfConfig` crane-serve's own `model_factory.rs` reads for detection.
#[derive(Debug, Deserialize, Default)]
pub struct ConfigJson {
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    /// The singular `architecture` field; some configs (VoxCPM2) use only this.
    pub architecture: Option<String>,
    pub vision_config: Option<serde_json::Value>,
}

impl ConfigJson {
    #[must_use]
    pub fn classify(&self) -> Classification {
        classify_config(
            self.model_type.as_deref().or(self.architecture.as_deref()),
            &self.architectures,
            self.vision_config.is_some(),
        )
    }
}

#[must_use]
pub fn classify_config(
    model_type: Option<&str>,
    architectures: &[String],
    has_vision_config: bool,
) -> Classification {
    let Some(family) = architecture::from_config(model_type, architectures, has_vision_config)
    else {
        let detected = model_type
            .map(str::to_string)
            .or_else(|| architectures.first().cloned());
        let reason = match &detected {
            Some(mt) => format!("Crane does not support this architecture ({mt})"),
            None => "Crane could not determine this repo's architecture".to_string(),
        };
        return Classification::Unsupported { detected, reason };
    };
    Classification::Supported {
        model_type: family.model_type,
        vision: family.vision,
        gated: family.gated,
        audio: family.audio,
    }
}

#[must_use]
pub fn classify_gguf_architecture(arch: &str) -> Classification {
    match architecture::from_gguf_architecture(arch) {
        Some(family) => Classification::Supported {
            model_type: family.model_type,
            vision: family.vision,
            gated: family.gated,
            audio: family.audio,
        },
        None => Classification::Unsupported {
            detected: Some(arch.to_string()),
            reason: format!("Crane does not support this architecture ({arch})"),
        },
    }
}

/// Last-resort override for the one real ambiguity content-based
/// classification can't resolve on its own: `MiniCPM5` checkpoints declare
/// themselves plain `"llama"` (see `architecture::from_path_name`'s docs).
/// Gated tightly — only fires when classification specifically detected
/// `"llama"`, never for any other unsupported architecture — so a
/// coincidental "minicpm" substring elsewhere in a path can't misclassify
/// an actual Llama checkpoint as supported.
#[must_use]
pub fn apply_path_hint(classification: Classification, path_hint: &str) -> Classification {
    let Classification::Unsupported {
        detected: Some(detected),
        ..
    } = &classification
    else {
        return classification;
    };
    if !detected.eq_ignore_ascii_case("llama") {
        return classification;
    }
    architecture::from_path_name(path_hint).map_or(classification, |family| {
        Classification::Supported {
            model_type: family.model_type,
            vision: family.vision,
            gated: family.gated,
            audio: family.audio,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detected_llama() -> Classification {
        Classification::Unsupported {
            detected: Some("llama".to_string()),
            reason: "Crane does not support this architecture (llama)".to_string(),
        }
    }

    #[test]
    fn voxcpm2_singular_architecture_field_classifies_as_supported_audio() {
        let config: ConfigJson = serde_json::from_str(r#"{"architecture": "voxcpm2"}"#).unwrap();
        assert!(matches!(
            config.classify(),
            Classification::Supported {
                model_type: "voxcpm2",
                audio: true,
                vision: false,
                ..
            }
        ));
    }

    #[test]
    fn llama_under_a_minicpm_path_is_reclassified_as_supported() {
        let result = apply_path_hint(detected_llama(), "openbmb/MiniCPM5-1B-GGUF");
        assert!(matches!(
            result,
            Classification::Supported {
                model_type: "minicpm5",
                ..
            }
        ));
    }

    #[test]
    fn llama_under_an_unrelated_path_stays_unsupported() {
        let result = apply_path_hint(detected_llama(), "meta-llama/Llama-3.2-1B");
        assert!(matches!(result, Classification::Unsupported { .. }));
    }

    #[test]
    fn a_non_llama_unsupported_verdict_is_never_touched() {
        let mistral = Classification::Unsupported {
            detected: Some("mistral".to_string()),
            reason: "Crane does not support this architecture (mistral)".to_string(),
        };
        let result = apply_path_hint(mistral, "some/minicpm-flavored-repo-name");
        assert!(matches!(result, Classification::Unsupported { .. }));
    }

    #[test]
    fn supported_and_unknown_verdicts_pass_through_unchanged() {
        let supported = Classification::Supported {
            model_type: "qwen3_5",
            vision: false,
            gated: false,
            audio: false,
        };
        assert_eq!(
            apply_path_hint(supported.clone(), "anything/minicpm"),
            supported
        );

        let unknown = Classification::Unknown {
            reason: "no config.json".to_string(),
        };
        assert_eq!(
            apply_path_hint(unknown.clone(), "anything/minicpm"),
            unknown
        );
    }
}
