//! The context-length solver, per PLAN.md §7.2: given hardware and a model,
//! find a configuration reaching the target context (default 256k, §7.0).
//! The wizard leads with this answer, not a blank form of knobs (§4.4).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use super::config::ModelConfig;
use super::kv::kv_bytes_per_token;
use super::predict::{Prediction, PredictionInputs, predict};
use crate::catalog::schema::KvQuant;
use crate::hardware::Backend;

/// The wizard's rounded context choices (§4.4) — a solved context is always
/// one of these, never an arbitrary number.
const CONTEXT_BUCKETS: [usize; 6] = [8192, 16_384, 32_768, 65_536, 131_072, 262_144];

/// PLAN.md §10.3: below this, coding agents don't function at all — the
/// wizard refuses rather than warns.
pub const CONTEXT_FLOOR: usize = 32768;

#[derive(Debug, Clone)]
pub struct Variant {
    pub label: String,
    pub weight_bytes: u64,
    /// Whether this is a published GGUF (`false`) or an in-situ-quantized
    /// estimate (`true`) — used only for the `Unusable` "not yet measured"
    /// framing; ranking is driven entirely by variant order (§2.9: caller
    /// passes real GGUFs before any ISQ fallback).
    pub is_isq: bool,
}

pub struct SolveRequest<'a> {
    pub cfg: &'a ModelConfig,
    /// Best quality first — GGUF variants (largest/least-quantized first),
    /// with any ISQ fallback last (§2.9).
    pub variants: &'a [Variant],
    pub supports_kv_quant: bool,
    pub supports_kv_swap: bool,
    pub native_context: usize,
    pub usable_vram: u64,
    pub backend: Backend,
    pub compute_dtype_bytes: f64,
    pub vision: bool,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub variant_label: String,
    pub kv_quant: Option<KvQuant>,
    pub context: usize,
    pub concurrency: usize,
    pub predicted: Prediction,
}

#[derive(Debug, Clone)]
pub struct Blocker {
    pub description: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub enum Suggestion {
    SmallerVariant {
        label: String,
        achievable_context: usize,
    },
    NeedMoreVram {
        additional_bytes_for_floor: u64,
    },
}

#[derive(Debug, Clone)]
pub enum SolveResult {
    Reaches(Vec<Config>),
    Short {
        best: Config,
        achieved_context: usize,
        blockers: Vec<Blocker>,
    },
    Unusable {
        achieved_context: usize,
        suggestions: Vec<Suggestion>,
    },
}

fn round_down_to_bucket(ctx: usize, native_context: usize) -> usize {
    let capped = ctx.min(native_context);
    CONTEXT_BUCKETS
        .iter()
        .rev()
        .find(|&&bucket| bucket <= capped)
        .copied()
        .unwrap_or(0)
}

/// `fixed_bytes` is everything in a `Prediction` except `kv_cache` — what's
/// left of `usable_vram` after that is entirely down to context depth.
fn max_context_for(
    cfg: &ModelConfig,
    fixed_bytes: u64,
    kv_quant: Option<KvQuant>,
    concurrency: usize,
    usable_vram: u64,
    native_context: usize,
) -> usize {
    if fixed_bytes >= usable_vram {
        return 0;
    }
    let budget = (usable_vram - fixed_bytes) as f64;
    let per_token = kv_bytes_per_token(cfg, kv_quant) * concurrency as f64;
    if per_token <= 0.0 {
        return native_context;
    }
    let raw_ctx = (budget / per_token) as usize;
    round_down_to_bucket(raw_ctx, native_context)
}

fn evaluate(
    request: &SolveRequest,
    variant: &Variant,
    kv_quant: Option<KvQuant>,
    concurrency: usize,
    context: usize,
) -> Config {
    let inputs = PredictionInputs {
        weight_bytes: variant.weight_bytes,
        context,
        concurrency,
        kv_quant,
        compute_dtype_bytes: request.compute_dtype_bytes,
        prefill_chunk: super::predict::DEFAULT_PREFILL_CHUNK,
        backend: request.backend,
        vision: request.vision,
    };
    let predicted = predict(request.cfg, &inputs);
    Config {
        variant_label: variant.label.clone(),
        kv_quant,
        context,
        concurrency,
        predicted,
    }
}

fn kv_quant_options(request: &SolveRequest) -> Vec<Option<KvQuant>> {
    if request.supports_kv_quant {
        vec![None, Some(KvQuant::Int8), Some(KvQuant::Int4)]
    } else {
        vec![None]
    }
}

fn concurrency_options(request: &SolveRequest) -> Vec<usize> {
    if request.supports_kv_swap && request.max_concurrent > 1 {
        vec![1, request.max_concurrent]
    } else {
        vec![1]
    }
}

#[must_use]
pub fn solve(request: &SolveRequest, target_context: usize) -> SolveResult {
    let mut best: Option<Config> = None;
    let mut reaching = Vec::new();

    for variant in request.variants {
        for &concurrency in &concurrency_options(request) {
            for kv_quant in kv_quant_options(request) {
                // Fixed terms (everything but kv_cache) don't depend on
                // context, so compute them once at context 0 to find the
                // achievable depth, then re-predict at that depth for the
                // real breakdown.
                let probe = evaluate(request, variant, kv_quant, concurrency, 0);
                let fixed_bytes = probe.predicted.total() - probe.predicted.kv_cache;
                let achieved = max_context_for(
                    request.cfg,
                    fixed_bytes,
                    kv_quant,
                    concurrency,
                    request.usable_vram,
                    request.native_context,
                );

                let config = evaluate(request, variant, kv_quant, concurrency, achieved);

                if achieved >= target_context.min(request.native_context) {
                    reaching.push(config);
                    continue;
                }
                if best.as_ref().is_none_or(|b| achieved > b.context) {
                    best = Some(config);
                }
            }
        }
    }

    if !reaching.is_empty() {
        return SolveResult::Reaches(reaching);
    }

    let Some(best) = best else {
        return SolveResult::Unusable {
            achieved_context: 0,
            suggestions: vec![Suggestion::NeedMoreVram {
                additional_bytes_for_floor: request.usable_vram,
            }],
        };
    };

    if best.context < CONTEXT_FLOOR {
        return SolveResult::Unusable {
            achieved_context: best.context,
            suggestions: unusable_suggestions(request, &best),
        };
    }

    SolveResult::Short {
        achieved_context: best.context,
        blockers: blockers_for(&best),
        best,
    }
}

fn blockers_for(config: &Config) -> Vec<Blocker> {
    let p = &config.predicted;
    let mut blockers = vec![Blocker {
        description: "weights".to_string(),
        bytes: p.weights,
    }];
    if p.runtime_overhead > 0 {
        blockers.push(Blocker {
            description: "runtime overhead".to_string(),
            bytes: p.runtime_overhead,
        });
    }
    if p.vision_tower > 0 {
        blockers.push(Blocker {
            description: "vision tower".to_string(),
            bytes: p.vision_tower,
        });
    }
    blockers.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    blockers
}

fn unusable_suggestions(request: &SolveRequest, best: &Config) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();
    // A smaller variant might already be in the candidate list — if the
    // worst-fitting variant achieved less than the floor, a
    // smaller/further-quantized one is the concrete next step.
    if let Some(smallest) = request.variants.iter().min_by_key(|v| v.weight_bytes)
        && smallest.label != best.variant_label
    {
        suggestions.push(Suggestion::SmallerVariant {
            label: smallest.label.clone(),
            achievable_context: best.context,
        });
    }
    let floor_bytes = (best.predicted.total() - best.predicted.kv_cache) as f64
        + kv_bytes_per_token(request.cfg, Some(KvQuant::Int4)) * CONTEXT_FLOOR as f64;
    let additional = (floor_bytes - request.usable_vram as f64).max(0.0) as u64;
    if additional > 0 {
        suggestions.push(Suggestion::NeedMoreVram {
            additional_bytes_for_floor: additional,
        });
    }
    suggestions
}

#[derive(Debug, Clone)]
pub struct OomAdviceOption {
    pub description: String,
    pub saves_bytes: u64,
}

/// Quantitative advice for a launch that just OOM'd, per PLAN.md §7.4:
/// "Advice must be quantitative, not vague." Each option is a single knob
/// change and exactly how much it frees up, computed from the same
/// `predict()` math the solver uses — not a canned message.
#[must_use]
pub fn oom_advice(
    cfg: &ModelConfig,
    failed: &PredictionInputs,
    supports_kv_quant: bool,
    supports_kv_swap: bool,
) -> Vec<OomAdviceOption> {
    let mut options = Vec::new();
    let per_token = kv_bytes_per_token(cfg, failed.kv_quant) * failed.concurrency as f64;

    // Reduce context to the next bucket down.
    if failed.context > 0 && per_token > 0.0 {
        let lower = CONTEXT_BUCKETS
            .iter()
            .rev()
            .find(|&&b| b < failed.context)
            .copied();
        if let Some(new_ctx) = lower {
            let saved = per_token * (failed.context - new_ctx) as f64;
            options.push(OomAdviceOption {
                description: format!("context {} → {new_ctx}", failed.context),
                saves_bytes: saved as u64,
            });
        }
    }

    // Deepen KV quantization one step.
    if supports_kv_quant {
        let next = match failed.kv_quant {
            None => Some((KvQuant::Int8, "f16", "int8")),
            Some(KvQuant::Int8) => Some((KvQuant::Int4, "int8", "int4")),
            Some(KvQuant::Int4) => None,
        };
        if let Some((next_quant, from_label, to_label)) = next {
            let current = kv_bytes_per_token(cfg, failed.kv_quant)
                * failed.context as f64
                * failed.concurrency as f64;
            let next_bytes = kv_bytes_per_token(cfg, Some(next_quant))
                * failed.context as f64
                * failed.concurrency as f64;
            options.push(OomAdviceOption {
                description: format!("KV cache {from_label} → {to_label}"),
                saves_bytes: (current - next_bytes) as u64,
            });
        }
    }

    // Halve concurrency, if it's above the floor of 1.
    if supports_kv_swap && failed.concurrency > 1 {
        let half = (failed.concurrency / 2).max(1);
        let saved = kv_bytes_per_token(cfg, failed.kv_quant)
            * failed.context as f64
            * (failed.concurrency - half) as f64;
        options.push(OomAdviceOption {
            description: format!("concurrency {} → {half}", failed.concurrency),
            saves_bytes: saved as u64,
        });
    }

    options
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_5_9B: &str = include_str!("../../testdata/qwen3.5-9b-config.json");

    fn gib(n: u64) -> u64 {
        n * 1024 * 1024 * 1024
    }

    /// M4's own accept criterion: Qwen 3.5 9B on a 24 GiB card reaches
    /// 256k. Real weight sizes fetched live from `unsloth/Qwen3.5-9B-GGUF`.
    #[test]
    fn qwen3_5_9b_reaches_256k_on_a_24gib_card() {
        let cfg = ModelConfig::parse(QWEN3_5_9B).unwrap();
        let variants = vec![
            Variant {
                label: "Q5_K_M".to_string(),
                weight_bytes: 6_577_841_376,
                is_isq: false,
            },
            Variant {
                label: "Q4_K_M".to_string(),
                weight_bytes: 5_680_522_464,
                is_isq: false,
            },
        ];
        // usable_vram per §6: (24 GiB - 512 MiB) * 0.95.
        let usable_vram = ((gib(24) - 512 * 1024 * 1024) as f64 * 0.95) as u64;
        let request = SolveRequest {
            cfg: &cfg,
            variants: &variants,
            supports_kv_quant: true,
            supports_kv_swap: false,
            native_context: 262_144,
            usable_vram,
            backend: Backend::Cuda,
            compute_dtype_bytes: 2.0,
            vision: false,
            max_concurrent: 1,
        };

        match solve(&request, 262_144) {
            SolveResult::Reaches(configs) => {
                assert!(!configs.is_empty());
                let top = &configs[0];
                assert_eq!(top.context, 262_144);
                assert!(
                    top.predicted.total() <= usable_vram,
                    "{} > {usable_vram}",
                    top.predicted.total()
                );
                // §7.2's ranking rule: prefer the best weight quality that
                // still reaches target, spending KV quant before weight
                // quant — Q5_K_M (best) should win here since there's
                // plenty of headroom even before touching KV quant.
                assert_eq!(top.variant_label, "Q5_K_M");
            }
            other => panic!("expected Reaches, got {other:?}"),
        }
    }

    #[test]
    fn tiny_vram_budget_is_short_with_blockers() {
        let cfg = ModelConfig::parse(QWEN3_5_9B).unwrap();
        let variants = vec![Variant {
            label: "Q4_K_M".to_string(),
            weight_bytes: 5_680_522_464,
            is_isq: false,
        }];
        // Just over the weight size — reaches the 32k floor but not 256k.
        let usable_vram = 6_500_000_000;
        let request = SolveRequest {
            cfg: &cfg,
            variants: &variants,
            supports_kv_quant: true,
            supports_kv_swap: false,
            native_context: 262_144,
            usable_vram,
            backend: Backend::Cuda,
            compute_dtype_bytes: 2.0,
            vision: false,
            max_concurrent: 1,
        };

        match solve(&request, 262_144) {
            SolveResult::Short {
                achieved_context,
                blockers,
                best,
            } => {
                assert!(achieved_context >= CONTEXT_FLOOR, "{achieved_context}");
                assert!(achieved_context < 262_144);
                assert_eq!(blockers[0].description, "weights");
                assert_eq!(best.kv_quant, Some(KvQuant::Int4)); // deepest quant, still short
            }
            other => panic!("expected Short, got {other:?}"),
        }
    }

    #[test]
    fn vram_too_small_for_the_floor_is_unusable_with_suggestions() {
        let cfg = ModelConfig::parse(QWEN3_5_9B).unwrap();
        let variants = vec![Variant {
            label: "Q4_K_M".to_string(),
            weight_bytes: 5_680_522_464,
            is_isq: false,
        }];
        // Not even enough for weights + a 32k KV buffer.
        let usable_vram = gib(4);
        let request = SolveRequest {
            cfg: &cfg,
            variants: &variants,
            supports_kv_quant: true,
            supports_kv_swap: false,
            native_context: 262_144,
            usable_vram,
            backend: Backend::Cuda,
            compute_dtype_bytes: 2.0,
            vision: false,
            max_concurrent: 1,
        };

        match solve(&request, 262_144) {
            SolveResult::Unusable {
                achieved_context,
                suggestions,
            } => {
                assert!(achieved_context < CONTEXT_FLOOR);
                assert!(!suggestions.is_empty());
                assert!(
                    suggestions
                        .iter()
                        .any(|s| matches!(s, Suggestion::NeedMoreVram { .. }))
                );
            }
            other => panic!("expected Unusable, got {other:?}"),
        }
    }

    /// M5's accept criterion: "a deliberately over-provisioned launch
    /// produces a correct OOM classification with quantitative advice."
    #[test]
    fn oom_advice_is_quantitative_and_actionable() {
        let cfg = ModelConfig::parse(QWEN3_5_9B).unwrap();
        let failed = PredictionInputs {
            weight_bytes: 5_680_522_464,
            context: 262_144,
            concurrency: 1,
            kv_quant: None,
            compute_dtype_bytes: 2.0,
            prefill_chunk: super::super::predict::DEFAULT_PREFILL_CHUNK,
            backend: Backend::Cuda,
            vision: false,
        };
        let options = oom_advice(&cfg, &failed, true, false);
        assert!(!options.is_empty());
        // Every option must name a real, non-zero saving — "quantitative,
        // not vague" per §7.4.
        for option in &options {
            assert!(option.saves_bytes > 0, "{option:?}");
            assert!(!option.description.is_empty());
        }
        assert!(options.iter().any(|o| o.description.contains("KV cache")));
    }
}
