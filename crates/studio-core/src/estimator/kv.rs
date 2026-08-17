//! Per-token KV-cache bytes and per-sequence GDN recurrent-state bytes, per
//! PLAN.md §7.1's hybrid-architecture note. Verified against the real
//! source, not guessed (PLAN.md §15.2):
//!
//! - Full- vs linear-attention layer split is *derived* from
//!   `full_attention_interval`, exactly mirroring
//!   `TextConfig::layer_types()` in `crane-core/src/models/qwen3_5/config.rs`
//!   — Crane does **not** read `config.json`'s own `layer_types` array.
//! - Per-token KV is `[B, num_kv_heads, seq_len, head_dim]` for K and V,
//!   full-attention layers only (`crane-core/src/models/qwen3_5/modeling.rs`).
//! - GDN state is two tensors per linear-attention layer, per sequence,
//!   independent of context length (`crane-core/src/ops/gdn/cache.rs`):
//!   a conv state at the compute dtype, and a recurrent state that is
//!   **always f32 regardless of compute dtype** — easy to under-count
//!   otherwise.
//!
//! Every dimension here (layer/head counts, `head_dim`) is small enough that
//! `usize as f64` never loses precision in practice — this module allows
//! that lint file-wide rather than annotating every arithmetic line.
#![allow(clippy::cast_precision_loss)]

use super::config::ModelConfig;
use crate::catalog::schema::KvQuant;

#[must_use]
pub fn kv_dtype_bytes(kv_quant: Option<KvQuant>) -> f64 {
    match kv_quant {
        None => 2.0,
        Some(KvQuant::Int8) => 1.0,
        Some(KvQuant::Int4) => 0.5,
    }
}

/// Layers that are full attention (contribute per-token KV) — every layer
/// for a non-hybrid model, or every `full_attention_interval`-th layer
/// (1-indexed) for a hybrid one.
#[must_use]
pub fn full_attention_layers(cfg: &ModelConfig) -> usize {
    match cfg.hybrid {
        Some(h) if h.full_attention_interval > 0 => {
            cfg.num_hidden_layers / h.full_attention_interval
        }
        _ => cfg.num_hidden_layers,
    }
}

#[must_use]
pub fn linear_attention_layers(cfg: &ModelConfig) -> usize {
    cfg.num_hidden_layers - full_attention_layers(cfg)
}

/// Bytes of KV cache consumed per token of context, for one sequence.
#[must_use]
pub fn kv_bytes_per_token(cfg: &ModelConfig, kv_quant: Option<KvQuant>) -> f64 {
    2.0 * full_attention_layers(cfg) as f64
        * cfg.num_key_value_heads as f64
        * cfg.head_dim as f64
        * kv_dtype_bytes(kv_quant)
}

/// Fixed per-sequence GDN state (conv + recurrent), independent of context
/// length — `0.0` for a non-hybrid model. `compute_dtype_bytes` is the
/// model's compute dtype (2.0 for bf16/f16, 4.0 for f32); the recurrent
/// tensor ignores it and is always counted at f32.
#[must_use]
pub fn gdn_state_bytes_per_sequence(cfg: &ModelConfig, compute_dtype_bytes: f64) -> f64 {
    let Some(h) = cfg.hybrid else { return 0.0 };
    let gdn_layers = linear_attention_layers(cfg) as f64;

    let key_dim = h.linear_num_key_heads * h.linear_key_head_dim;
    let value_dim = h.linear_num_value_heads * h.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    let conv_state = conv_dim as f64 * h.linear_conv_kernel_dim as f64 * compute_dtype_bytes;
    let recurrent_state = h.linear_num_value_heads as f64
        * h.linear_key_head_dim as f64
        * h.linear_value_head_dim as f64
        * 4.0;

    gdn_layers * (conv_state + recurrent_state)
}

#[cfg(test)]
// These byte counts are exact integer arithmetic cast to f64 (no
// accumulated rounding involved) — equality is the right check, not a
// fuzzy epsilon comparison.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const QWEN3_8_27B: &str = include_str!("../../testdata/qwen3.8-27b-config.json");
    const QWEN3_5_9B: &str = include_str!("../../testdata/qwen3.5-9b-config.json");
    const QWEN3_0_6B: &str = include_str!("../../testdata/qwen3-0.6b-config.json");

    /// PLAN.md §7.0's own worked example, from real config.json: "KV cache
    /// is ~64 KB/token on the 27B (16 full-attn layers × 4 KV heads × 256 ×
    /// 2 × 2 B)". This is the M4 accept-criteria test.
    #[test]
    fn reproduces_the_27b_64kb_per_token_figure() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        assert_eq!(full_attention_layers(&cfg), 16);
        assert_eq!(linear_attention_layers(&cfg), 48);
        assert_eq!(kv_bytes_per_token(&cfg, None), 65_536.0);
    }

    #[test]
    fn kv_quant_scales_the_27b_figure() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        assert_eq!(kv_bytes_per_token(&cfg, Some(KvQuant::Int8)), 32_768.0);
        assert_eq!(kv_bytes_per_token(&cfg, Some(KvQuant::Int4)), 16_384.0);
    }

    /// At 262144 (256k) native context, f16/int8/int4 KV give 16/8/4 GiB —
    /// PLAN.md §7.0's table, reproduced from the real config.
    #[test]
    fn kv_at_256k_matches_the_plan_table() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        let gib = 1024.0 * 1024.0 * 1024.0;
        let at_256k = |q| kv_bytes_per_token(&cfg, q) * 262_144.0 / gib;
        assert!((at_256k(None) - 16.0).abs() < 0.01);
        assert!((at_256k(Some(KvQuant::Int8)) - 8.0).abs() < 0.01);
        assert!((at_256k(Some(KvQuant::Int4)) - 4.0).abs() < 0.01);
    }

    /// Cross-check against the 9B member of the same family, real config:
    /// 32 layers / interval 4 = 8 full-attention layers, 4 KV heads, `head_dim`
    /// 256 → 2×8×4×256×2 = 32 KiB/token.
    #[test]
    fn qwen3_5_9b_kv_bytes_per_token() {
        let cfg = ModelConfig::parse(QWEN3_5_9B).unwrap();
        assert_eq!(full_attention_layers(&cfg), 8);
        assert_eq!(kv_bytes_per_token(&cfg, None), 32_768.0);
    }

    /// Real dims (48 linear layers, 16/32 key/value heads, 128/128 head
    /// dims, kernel 4): conv ≈3.75 MiB total + recurrent ≈144 MiB total ≈
    /// 148 MiB per sequence — independent of context length.
    #[test]
    fn gdn_state_matches_hand_computed_27b_figure() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        let mib = 1024.0 * 1024.0;
        let total = gdn_state_bytes_per_sequence(&cfg, 2.0) / mib;
        assert!((total - 147.75).abs() < 1.0, "got {total} MiB");
    }

    #[test]
    fn non_hybrid_model_has_no_gdn_state_and_every_layer_is_full_attention() {
        let cfg = ModelConfig::parse(QWEN3_0_6B).unwrap();
        assert_eq!(full_attention_layers(&cfg), cfg.num_hidden_layers);
        assert_eq!(gdn_state_bytes_per_sequence(&cfg, 2.0), 0.0);
    }
}
