//! VRAM prediction and the context-length solver, per PLAN.md §7. Trust
//! measurement over prediction wherever they disagree — see PLAN.md §15.4.

mod config;
mod gguf_config;
mod kv;
mod predict;
mod solve;
mod weights;

pub use config::{HybridConfig, ModelConfig};
pub use gguf_config::read_model_config as read_model_config_from_gguf;
pub use kv::{
    full_attention_layers, gdn_state_bytes_per_sequence, kv_bytes_per_token, kv_dtype_bytes,
    linear_attention_layers,
};
pub use predict::{
    DEFAULT_PREFILL_CHUNK, Prediction, PredictionInputs, predict, runtime_overhead_bytes,
};
pub use solve::{
    Blocker, CONTEXT_FLOOR, Config, OomAdviceOption, SolveRequest, SolveResult, Suggestion,
    Variant, oom_advice, solve,
};
pub use weights::{ParamCounts, isq_weight_bytes, param_counts, safetensors_dir_bytes};
