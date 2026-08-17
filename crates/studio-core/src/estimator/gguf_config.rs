//! Builds a [`ModelConfig`] straight from a local GGUF file's own embedded
//! metadata — no companion `config.json`, no network access. Verified
//! against three real local files (Qwen3.5-0.8B, Qwen3.5-4B,
//! Qwen3.8-27B-Heretic) by cross-checking every derived field against their
//! real HF `config.json` values.
//!
//! llama.cpp's GGUF writer names dimensions `<arch>.<key>`; the mapping to
//! `config.json`'s field names is direct except for the hybrid/GDN fields,
//! which live under `<arch>.ssm.*` and don't share config.json's names:
//! `ssm.group_count` is `linear_num_key_heads`, `ssm.time_step_rank` is
//! `linear_num_value_heads`, and `ssm.state_size` is used for both
//! `linear_key_head_dim` and `linear_value_head_dim` (identical in every
//! real checkpoint observed).

use std::collections::HashMap;
use std::io::Read;

use crate::catalog::gguf::{MetaValue, read_all_scalars};

use super::config::{HybridConfig, ModelConfig};

/// # Errors
/// If the file isn't a valid GGUF file, or is missing a dimension the
/// estimator needs.
pub fn read_model_config<R: Read>(reader: &mut R) -> Result<ModelConfig, String> {
    let scalars = read_all_scalars(reader).ok_or("not a valid GGUF file")?;
    model_config_from_scalars(&scalars)
}

fn model_config_from_scalars(scalars: &HashMap<String, MetaValue>) -> Result<ModelConfig, String> {
    let arch = scalars.get("general.architecture").and_then(MetaValue::as_str).ok_or("missing general.architecture")?;

    let get = |suffix: &str| -> Option<usize> { scalars.get(&format!("{arch}.{suffix}")).and_then(MetaValue::as_usize) };
    let required = |suffix: &str| get(suffix).ok_or_else(|| format!("GGUF metadata is missing `{arch}.{suffix}`"));

    let hybrid = get("full_attention_interval")
        .map(|full_attention_interval| {
            Ok::<_, String>(HybridConfig {
                full_attention_interval,
                linear_num_key_heads: required("ssm.group_count")?,
                linear_num_value_heads: required("ssm.time_step_rank")?,
                linear_key_head_dim: required("ssm.state_size")?,
                linear_value_head_dim: required("ssm.state_size")?,
                linear_conv_kernel_dim: required("ssm.conv_kernel")?,
            })
        })
        .transpose()?;

    Ok(ModelConfig {
        hidden_size: required("embedding_length")?,
        num_attention_heads: required("attention.head_count")?,
        num_key_value_heads: required("attention.head_count_kv")?,
        head_dim: required("attention.key_length")?,
        num_hidden_layers: required("block_count")?,
        max_position_embeddings: required("context_length")?,
        // Not exposed as a single scalar GGUF key (only recoverable by
        // counting the tokenizer's vocab array) and unused by any current
        // VRAM computation — see ModelConfig::vocab_size's callers.
        vocab_size: 0,
        has_vision_config: false,
        hybrid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::Path;

    fn load(path: &str) -> Option<ModelConfig> {
        let p = Path::new(path);
        if !p.exists() {
            return None;
        }
        let mut f = File::open(p).unwrap();
        Some(read_model_config(&mut f).unwrap())
    }

    #[test]
    fn reads_real_local_qwen3_5_0_8b_gguf() {
        let Some(cfg) = load("/home/hahihula/mywork/ai/additional_models/Qwen3.5-0.8B-Q8_0.gguf") else {
            return;
        };
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        let hybrid = cfg.hybrid.unwrap();
        assert_eq!(hybrid.full_attention_interval, 4);
        assert_eq!(hybrid.linear_num_key_heads, 16);
        assert_eq!(hybrid.linear_num_value_heads, 16);
        assert_eq!(hybrid.linear_key_head_dim, 128);
        assert_eq!(hybrid.linear_value_head_dim, 128);
        assert_eq!(hybrid.linear_conv_kernel_dim, 4);
    }

    #[test]
    fn reads_real_local_qwen3_5_4b_gguf_matches_hf_config() {
        // Cross-checked against the real Qwen/Qwen3.5-4B config.json fetched
        // live in M4: linear_num_value_heads=32, linear_num_key_heads=16.
        let Some(cfg) = load("/home/hahihula/mywork/ai/additional_models/Qwen3.5-4B-Q6_K.gguf") else {
            return;
        };
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_key_value_heads, 4);
        let hybrid = cfg.hybrid.unwrap();
        assert_eq!(hybrid.linear_num_key_heads, 16);
        assert_eq!(hybrid.linear_num_value_heads, 32);
    }
}
