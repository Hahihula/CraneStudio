//! Assembles a predicted peak-VRAM breakdown, per PLAN.md §7.1:
//! `weights + kv_cache + recurrent_state + vision_tower + activation +
//! runtime_overhead`.
//!
//! Byte counts here stay well under 2^52 for any real model, so the
//! usize/u64⇄f64 conversions this module's arithmetic needs are allowed
//! file-wide rather than annotated line by line.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use super::config::ModelConfig;
use super::kv::{gdn_state_bytes_per_sequence, kv_bytes_per_token};
use crate::catalog::schema::KvQuant;
use crate::hardware::Backend;

/// Qwen 3.5-VL's `ViT`, per PLAN.md §7.1: "~600M params... at bf16 that is
/// ~1.2 GiB, and it is never quantized" (§2.9). A flat constant, not
/// computed from `vision_config` dims — this is the one figure the plan
/// hands over as ground truth rather than asking it to be derived.
const VISION_TOWER_BYTES: u64 = 1_200 * 1024 * 1024;

/// llama.cpp/Crane's default prefill chunk (`CRANE_PREFILL_CHUNK` unset),
/// per `crane-core/src/models/qwen3_5/prefill.rs`'s own `DEFAULT_CHUNK`.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

/// CUDA/ROCm driver context overhead, per PLAN.md §7.1 ("roughly 300-600
/// MiB per process") — midpoint until a real measurement replaces it (§7.3).
/// Metal and CPU have no equivalent driver context to account for.
#[must_use]
pub fn runtime_overhead_bytes(backend: Backend) -> u64 {
    match backend {
        Backend::Cuda | Backend::Rocm => 450 * 1024 * 1024,
        Backend::Metal | Backend::Cpu => 0,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PredictionInputs {
    pub weight_bytes: u64,
    pub context: usize,
    pub concurrency: usize,
    pub kv_quant: Option<KvQuant>,
    /// Compute dtype size in bytes (2.0 for bf16/f16, 4.0 for f32) — feeds
    /// the GDN conv-state term and, later, the activation term.
    pub compute_dtype_bytes: f64,
    pub prefill_chunk: usize,
    pub backend: Backend,
    /// Whether *this launch* actually loads the vision tower — **not**
    /// the same as `cfg.has_vision_config`. Verified live: a checkpoint's
    /// `config.json` can declare `vision_config` while a specific GGUF
    /// variant of it (no accompanying `mmproj`) or a `--text-only` launch
    /// never materialises those weights at all. The caller decides this
    /// from the launch spec (model type resolved to a `*_vl` variant, or a
    /// GGUF with a paired `mmproj` file), not from the base config alone.
    pub vision: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub weights: u64,
    pub kv_cache: u64,
    pub recurrent_state: u64,
    pub vision_tower: u64,
    pub activation: u64,
    pub runtime_overhead: u64,
}

impl Prediction {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.weights
            + self.kv_cache
            + self.recurrent_state
            + self.vision_tower
            + self.activation
            + self.runtime_overhead
    }
}

/// Peak activation working set during prefill. Per a research pass reading
/// `crane-core/src/models/qwen3_5/prefill.rs`'s own module docs: a
/// full-attention layer materialises a `[B, heads, chunk, chunk]` score
/// matrix, so this is **quadratic in `prefill_chunk`**, not linear — one
/// layer's matrix assumed live at a time (conservative; PLAN.md §7.1 says
/// to start conservative and let measurement correct it, §7.3).
#[must_use]
fn activation_bytes(cfg: &ModelConfig, prefill_chunk: usize) -> u64 {
    let bytes = cfg.num_attention_heads as f64 * (prefill_chunk as f64).powi(2) * 4.0; // f32 scores
    bytes as u64
}

#[must_use]
pub fn predict(cfg: &ModelConfig, inputs: &PredictionInputs) -> Prediction {
    let kv_cache = (kv_bytes_per_token(cfg, inputs.kv_quant)
        * inputs.context as f64
        * inputs.concurrency as f64) as u64;
    let recurrent_state = (gdn_state_bytes_per_sequence(cfg, inputs.compute_dtype_bytes)
        * inputs.concurrency as f64) as u64;
    let vision_tower = if inputs.vision { VISION_TOWER_BYTES } else { 0 };
    let activation = activation_bytes(cfg, inputs.prefill_chunk);

    Prediction {
        weights: inputs.weight_bytes,
        kv_cache,
        recurrent_state,
        vision_tower,
        activation,
        runtime_overhead: runtime_overhead_bytes(inputs.backend),
    }
}

/// M4's accept criterion ("predictions for three known models land within
/// 20% of measured reality") verified for real on this project's dev
/// machine (RTX 3090): loaded each model via `cranestudio __serve`, sent a
/// prompt reaching close to the configured `max_seq_len`, and diffed
/// `nvidia-smi`'s total-used delta against this module's `predict()`.
/// Recorded here as a permanent, fast regression check against those real
/// numbers rather than re-launching a model on every test run.
#[cfg(test)]
mod calibration {
    use super::*;
    use crate::catalog::schema::KvQuant;

    struct Case {
        config_json: &'static str,
        weight_bytes: u64,
        context: usize,
        measured_bytes: u64,
    }

    // Real launches, 2026-08-17, all --model-type qwen3_5, GGUF (no
    // vision/mmproj), CRANE_KV_QUANT unset (f16 KV), concurrency forced to
    // 1. `measured_bytes` = nvidia-smi total-used delta (after a real chat
    // request reaching `context` tokens) minus the pre-launch baseline.
    const CASES: &[Case] = &[
        Case {
            // Qwen3.5-0.8B-Q8_0.gguf
            config_json: include_str!("../../testdata/qwen3.5-0.8b-config.json"),
            weight_bytes: 811_843_840,
            context: 7020,
            measured_bytes: 1384 * 1024 * 1024,
        },
        Case {
            // Qwen3.5-4B-Q6_K.gguf
            config_json: include_str!("../../testdata/qwen3.5-4b-config.json"),
            weight_bytes: 3_525_956_768,
            context: 7020,
            measured_bytes: 4392 * 1024 * 1024,
        },
        Case {
            // Qwen3.8-27B-Heretic-Q4_K_M.gguf, --text-only
            config_json: include_str!("../../testdata/qwen3.8-27b-config.json"),
            weight_bytes: 16_547_400_032,
            context: 3860,
            measured_bytes: 16_868 * 1024 * 1024,
        },
    ];

    #[test]
    fn predictions_are_within_20_percent_of_real_measurements() {
        for case in CASES {
            let cfg = ModelConfig::parse(case.config_json).unwrap();
            let inputs = PredictionInputs {
                weight_bytes: case.weight_bytes,
                context: case.context,
                concurrency: 1,
                kv_quant: None::<KvQuant>,
                compute_dtype_bytes: 2.0,
                prefill_chunk: DEFAULT_PREFILL_CHUNK,
                backend: Backend::Cuda,
                vision: false,
            };
            let predicted = predict(&cfg, &inputs).total();
            let error =
                (predicted as f64 - case.measured_bytes as f64).abs() / case.measured_bytes as f64;
            assert!(
                error < 0.20,
                "{} vs {} — {:.1}% error (weights={})",
                predicted,
                case.measured_bytes,
                error * 100.0,
                case.weight_bytes
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_8_27B: &str = include_str!("../../testdata/qwen3.8-27b-config.json");

    #[test]
    fn total_sums_every_term() {
        let cfg = ModelConfig::parse(QWEN3_8_27B).unwrap();
        let inputs = PredictionInputs {
            weight_bytes: 16_547_400_032, // real local Qwen3.8-27B-Heretic-Q4_K_M.gguf size
            context: 262_144,
            concurrency: 1,
            kv_quant: Some(KvQuant::Int4),
            compute_dtype_bytes: 2.0,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            backend: Backend::Cuda,
            vision: true,
        };
        let p = predict(&cfg, &inputs);
        assert_eq!(
            p.total(),
            p.weights
                + p.kv_cache
                + p.recurrent_state
                + p.vision_tower
                + p.activation
                + p.runtime_overhead
        );
        // int4 KV at 256k should land at ~4 GiB per §7.0's table.
        let gib = 1024.0 * 1024.0 * 1024.0;
        assert!((p.kv_cache as f64 / gib - 4.0).abs() < 0.01);
        assert_eq!(p.vision_tower, VISION_TOWER_BYTES);
    }
}
