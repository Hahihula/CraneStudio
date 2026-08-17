//! Architecture dimensions read from a real `config.json`, per PLAN.md §7.1.
//! Two real shapes exist in the wild and both must parse: Qwen 3.5's
//! checkpoints nest everything under `text_config` (verified live against
//! `Qwen/Qwen3.5-9B`); plain Qwen 3/2.5 put the same fields at the top
//! level (verified against a local `Qwen3-0.6B-Instruct`).

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub has_vision_config: bool,
    /// `Some` only for the Qwen 3.5/3.6/3.8 hybrid family — presence of
    /// this is what marks a config as hybrid throughout the estimator.
    pub hybrid: Option<HybridConfig>,
}

/// GDN (linear-attention) layer dimensions, per PLAN.md §7.1's hybrid
/// architecture note — verified against `crane-core/src/ops/gdn` by a
/// research pass reading the real source (see M4 commit notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridConfig {
    /// Every `full_attention_interval`-th layer (1-indexed) is full
    /// attention; the rest are GDN. Mirrors
    /// `TextConfig::layer_types()` in `crane-core/src/models/qwen3_5/config.rs`
    /// — NOT read from `config.json`'s own `layer_types` array, which Crane
    /// ignores and recomputes from this interval instead.
    pub full_attention_interval: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl ModelConfig {
    /// # Errors
    /// If the JSON doesn't parse, or the fields needed to estimate memory
    /// are missing from both the top level and `text_config`.
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: RawConfig = serde_json::from_str(json).map_err(|e| e.to_string())?;
        raw.into_model_config()
    }
}

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default)]
    vision_config: Option<serde_json::Value>,
    text_config: Option<RawFields>,
    #[serde(flatten)]
    top: RawFields,
}

#[derive(Deserialize, Default)]
struct RawFields {
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    num_hidden_layers: Option<usize>,
    max_position_embeddings: Option<usize>,
    vocab_size: Option<usize>,
    full_attention_interval: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
}

impl RawConfig {
    fn into_model_config(self) -> Result<ModelConfig, String> {
        let has_vision_config = self.vision_config.is_some();
        // text_config's fields win when present; fall back to top-level.
        let fields = self.text_config.unwrap_or(self.top);

        macro_rules! required {
            ($field:ident) => {
                fields
                    .$field
                    .ok_or_else(|| format!("config.json is missing `{}`", stringify!($field)))?
            };
        }

        let hybrid = fields
            .full_attention_interval
            .map(|full_attention_interval| {
                Ok::<_, String>(HybridConfig {
                    full_attention_interval,
                    linear_num_key_heads: fields
                        .linear_num_key_heads
                        .ok_or("missing `linear_num_key_heads` on a hybrid config")?,
                    linear_num_value_heads: fields
                        .linear_num_value_heads
                        .ok_or("missing `linear_num_value_heads` on a hybrid config")?,
                    linear_key_head_dim: fields
                        .linear_key_head_dim
                        .ok_or("missing `linear_key_head_dim` on a hybrid config")?,
                    linear_value_head_dim: fields
                        .linear_value_head_dim
                        .ok_or("missing `linear_value_head_dim` on a hybrid config")?,
                    linear_conv_kernel_dim: fields
                        .linear_conv_kernel_dim
                        .ok_or("missing `linear_conv_kernel_dim` on a hybrid config")?,
                })
            });
        let hybrid = hybrid.transpose()?;

        Ok(ModelConfig {
            hidden_size: required!(hidden_size),
            num_attention_heads: required!(num_attention_heads),
            num_key_value_heads: required!(num_key_value_heads),
            head_dim: required!(head_dim),
            num_hidden_layers: required!(num_hidden_layers),
            max_position_embeddings: required!(max_position_embeddings),
            vocab_size: required!(vocab_size),
            has_vision_config,
            hybrid,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real config.json, fetched live from Qwen/Qwen3.5-0.8B (§7.1's own
    // worked-example family, smallest member) — nested `text_config` shape.
    const QWEN3_5_0_8B: &str = include_str!("../../testdata/qwen3.5-0.8b-config.json");
    // Real config.json, fetched live from Qwen/Qwen3.8-27B — the exact
    // model PLAN.md §7.0 uses for its 64 KB/token worked example.
    const QWEN3_8_27B: &str = include_str!("../../testdata/qwen3.8-27b-config.json");
    // Real local file, /home/hahihula/mywork/ai/additional_models — flat
    // top-level shape, no hybrid fields.
    const QWEN3_0_6B: &str = include_str!("../../testdata/qwen3-0.6b-config.json");

    #[test]
    fn parses_nested_hybrid_config() {
        let cfg = ModelConfig::parse(QWEN3_5_0_8B).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert!(cfg.has_vision_config);
        let hybrid = cfg.hybrid.unwrap();
        assert_eq!(hybrid.full_attention_interval, 4);
    }

    #[test]
    fn parses_the_27b_worked_example() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.hybrid.unwrap().full_attention_interval, 4);
    }

    #[test]
    fn parses_flat_non_hybrid_config() {
        let cfg = ModelConfig::parse(QWEN3_0_6B).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert!(cfg.hybrid.is_none());
        assert!(!cfg.has_vision_config);
    }

    #[test]
    fn missing_required_field_is_a_clear_error() {
        let err = ModelConfig::parse(r#"{"hidden_size": 1024}"#).unwrap_err();
        assert!(err.contains("num_attention_heads"), "{err}");
    }
}
