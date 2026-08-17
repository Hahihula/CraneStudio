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
    pub vision_config: Option<serde_json::Value>,
}

impl ConfigJson {
    #[must_use]
    pub fn classify(&self) -> Classification {
        classify_config(
            self.model_type.as_deref(),
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
    }
}

#[must_use]
pub fn classify_gguf_architecture(arch: &str) -> Classification {
    match architecture::from_gguf_architecture(arch) {
        Some(family) => Classification::Supported {
            model_type: family.model_type,
            vision: family.vision,
            gated: family.gated,
        },
        None => Classification::Unsupported {
            detected: Some(arch.to_string()),
            reason: format!("Crane does not support this architecture ({arch})"),
        },
    }
}
