//! `LaunchSpec` → argv/envp for a `cranestudio __serve` child, per PLAN.md
//! §3.3. Pure — no process spawning here (that's `studio-supervisor`'s job);
//! this only decides *what* to run.

use crate::catalog::schema::KvQuant;

/// Every `--model-type qwen3_5`/`CRANE_ISQ` quantization level Crane
/// accepts, mirroring `crane-core/src/ops/linear.rs::parse_ggml_dtype`
/// exactly (case-insensitive, `_k` normalized to `k`) — verified against
/// the real source rather than guessed, since a bad value there panics the
/// child instead of erroring cleanly (§2.7), which is exactly what this
/// validation exists to prevent.
const VALID_ISQ_LEVELS: &[&str] = &[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k", "q4k", "q5k", "q6k",
];

#[must_use]
pub fn normalize_isq_level(level: &str) -> String {
    level.trim().to_lowercase().replace("_k", "k")
}

/// # Errors
/// If `level` isn't one of Crane's accepted quantization levels.
pub fn validate_isq_level(level: &str) -> Result<(), String> {
    let normalized = normalize_isq_level(level);
    if VALID_ISQ_LEVELS.contains(&normalized.as_str()) {
        Ok(())
    } else {
        Err(format!(
            "unknown quantization level '{level}' (expected one of {})",
            VALID_ISQ_LEVELS.join(", ")
        ))
    }
}

/// Everything needed to spawn one `cranestudio __serve` child, per the CLI
/// surface in PLAN.md §2.13. Children always bind loopback-only (§10.1) —
/// that's not a field here, it's hardcoded in `argv()`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LaunchSpec {
    pub model_path: String,
    pub model_type: String,
    pub model_name: Option<String>,
    pub port: u16,
    pub cpu: bool,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    pub format: Option<String>,
    /// ISQ level, passed as `--quant` (never via the `CRANE_ISQ` env var —
    /// §2.7 says that path panics on a bad value; this one, `--quant`,
    /// returns a normal error, verified against the real source).
    pub quant: Option<String>,
    pub dtype: Option<String>,
    pub max_seq_len: usize,
    pub gpu_memory_limit: Option<String>,
    pub text_only: bool,
    /// Qwen 3.5-family-only (§2.8) — the caller is responsible for only
    /// setting this when the model actually supports it.
    pub kv_quant: Option<KvQuant>,
    pub prefill_chunk: Option<usize>,
    /// Which GPU this child should see, via `CUDA_VISIBLE_DEVICES` (§2.6) —
    /// not a real multi-GPU capability, just per-child GPU pinning.
    pub device: usize,
}

impl LaunchSpec {
    /// # Errors
    /// If `quant` is set to a level Crane doesn't accept.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(quant) = &self.quant {
            validate_isq_level(quant)?;
        }
        Ok(())
    }

    /// Args for `cranestudio __serve <these>`, i.e. everything after the
    /// hidden subcommand name.
    #[must_use]
    pub fn argv(&self) -> Vec<String> {
        let mut argv = vec![
            "-m".to_string(),
            self.model_path.clone(),
            "--model-type".to_string(),
            self.model_type.clone(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "-p".to_string(),
            self.port.to_string(),
        ];

        if let Some(name) = &self.model_name {
            argv.push("--model-name".to_string());
            argv.push(name.clone());
        }
        if self.cpu {
            argv.push("--cpu".to_string());
        }
        argv.push("-c".to_string());
        argv.push(self.max_concurrent.to_string());
        argv.push("--decode-tokens-per-seq".to_string());
        argv.push(self.decode_tokens_per_seq.to_string());
        if let Some(format) = &self.format {
            argv.push("--format".to_string());
            argv.push(format.clone());
        }
        if let Some(quant) = &self.quant {
            argv.push("--quant".to_string());
            argv.push(quant.clone());
        }
        if let Some(dtype) = &self.dtype {
            argv.push("--dtype".to_string());
            argv.push(dtype.clone());
        }
        argv.push("--max-seq-len".to_string());
        argv.push(self.max_seq_len.to_string());
        if let Some(limit) = &self.gpu_memory_limit {
            argv.push("--gpu-memory-limit".to_string());
            argv.push(limit.clone());
        }
        if self.text_only {
            argv.push("--text-only".to_string());
        }
        argv
    }

    /// Env vars that have no CLI flag equivalent (§2.7) — must be set on
    /// the child process environment, not passed as arguments.
    #[must_use]
    pub fn envp(&self) -> Vec<(String, String)> {
        let mut env = Vec::new();
        if let Some(kv_quant) = self.kv_quant {
            let value = match kv_quant {
                KvQuant::Int8 => "int8",
                KvQuant::Int4 => "int4",
            };
            env.push(("CRANE_KV_QUANT".to_string(), value.to_string()));
        }
        if let Some(chunk) = self.prefill_chunk {
            env.push(("CRANE_PREFILL_CHUNK".to_string(), chunk.to_string()));
        }
        env.push(("CUDA_VISIBLE_DEVICES".to_string(), self.device.to_string()));
        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_spec() -> LaunchSpec {
        LaunchSpec {
            model_path: "/models/qwen3.5-0.8b.gguf".to_string(),
            model_type: "qwen3_5".to_string(),
            model_name: None,
            port: 41001,
            cpu: false,
            max_concurrent: 1,
            decode_tokens_per_seq: 16,
            format: None,
            quant: None,
            dtype: None,
            max_seq_len: 262_144,
            gpu_memory_limit: None,
            text_only: false,
            kv_quant: None,
            prefill_chunk: None,
            device: 0,
        }
    }

    #[test]
    fn children_always_bind_loopback_only() {
        let argv = minimal_spec().argv();
        let host_index = argv.iter().position(|a| a == "--host").unwrap();
        assert_eq!(argv[host_index + 1], "127.0.0.1");
    }

    #[test]
    fn kv_quant_and_device_become_env_vars_not_args() {
        let mut spec = minimal_spec();
        spec.kv_quant = Some(KvQuant::Int4);
        spec.device = 1;
        let env = spec.envp();
        assert!(env.contains(&("CRANE_KV_QUANT".to_string(), "int4".to_string())));
        assert!(env.contains(&("CUDA_VISIBLE_DEVICES".to_string(), "1".to_string())));
        // Never as CLI args.
        assert!(!spec.argv().iter().any(|a| a.contains("CRANE_KV_QUANT")));
    }

    #[test]
    fn valid_isq_levels_pass_every_real_spelling() {
        for level in [
            "q4_0", "Q4_1", "q5_0", "q5_1", "Q8_0", "q2k", "q3_k", "Q4_K", "q5k", "q6K",
        ] {
            assert!(validate_isq_level(level).is_ok(), "{level} should be valid");
        }
    }

    #[test]
    fn invalid_isq_level_is_rejected_before_spawn() {
        let mut spec = minimal_spec();
        spec.quant = Some("q7_bogus".to_string());
        assert!(spec.validate().is_err());
    }
}
