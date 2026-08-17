//! Weight-size estimation, per PLAN.md §7.1.
//!
//! GGUF and safetensors are just file sizes on disk — no math needed. ISQ
//! needs real per-tensor parameter counts (`embed_tokens` stays dense —
//! §2.9 — so it must be split out from everything else), read from
//! safetensors headers rather than guessed from architecture dims: the
//! header is a small JSON blob at the front of the file, cheap to read
//! without touching tensor data, mirroring the GGUF-header-only approach
//! `catalog::gguf` already uses for the same reason.
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

/// Sum of every `.safetensors` file's size in `dir` — the weight size for
/// an unquantized (or already-quantized-by-the-publisher) safetensors
/// checkpoint.
///
/// # Errors
/// If `dir` can't be read.
pub fn safetensors_dir_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry
            .path()
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

/// A tensor's element count and whether it's an embedding/LM-head tensor
/// (the ones ISQ leaves dense — §2.9).
#[derive(Debug, Clone, Copy)]
struct TensorInfo {
    elements: u64,
    dtype_bytes: f64,
    is_embedding: bool,
}

#[derive(Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
}

/// Reads just the header (a length-prefixed JSON blob) of one safetensors
/// file — never the tensor data itself.
fn read_header(path: &Path) -> std::io::Result<Vec<TensorInfo>> {
    let mut file = fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header_buf = vec![0u8; usize::try_from(header_len).map_err(std::io::Error::other)?];
    file.read_exact(&mut header_buf)?;

    let raw: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_buf).map_err(std::io::Error::other)?;

    Ok(raw
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .filter_map(|(name, value)| {
            let entry: RawTensorEntry = serde_json::from_value(value).ok()?;
            let elements = entry.shape.iter().product();
            let dtype_bytes = dtype_size_bytes(&entry.dtype)?;
            let is_embedding = name.contains("embed_tokens") || name.contains("lm_head");
            Some(TensorInfo {
                elements,
                dtype_bytes,
                is_embedding,
            })
        })
        .collect())
}

fn dtype_size_bytes(dtype: &str) -> Option<f64> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8.0),
        "F32" | "I32" | "U32" => Some(4.0),
        "F16" | "BF16" | "I16" | "U16" => Some(2.0),
        "I8" | "U8" | "BOOL" => Some(1.0),
        _ => None,
    }
}

#[derive(Debug)]
pub struct ParamCounts {
    pub embedding_params: u64,
    pub embedding_dtype_bytes: f64,
    pub non_embedding_params: u64,
}

/// Reads every `.safetensors` file's header under `dir` (a sharded
/// checkpoint has one header per shard) and splits parameters into
/// embedding/LM-head vs everything else.
///
/// # Errors
/// If `dir` can't be read or contains no readable safetensors headers.
pub fn param_counts(dir: &Path) -> std::io::Result<ParamCounts> {
    let mut embedding_params = 0u64;
    let mut embedding_dtype_bytes = 2.0;
    let mut non_embedding_params = 0u64;
    let mut found_any = false;

    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            continue;
        }
        for tensor in read_header(&path)? {
            found_any = true;
            if tensor.is_embedding {
                embedding_params += tensor.elements;
                embedding_dtype_bytes = tensor.dtype_bytes;
            } else {
                non_embedding_params += tensor.elements;
            }
        }
    }

    if !found_any {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no readable .safetensors headers in this directory",
        ));
    }
    Ok(ParamCounts {
        embedding_params,
        embedding_dtype_bytes,
        non_embedding_params,
    })
}

/// Estimated weight bytes after in-situ quantization, per PLAN.md §2.9:
/// `embed_tokens`/`lm_head` stay at their original dtype; everything else
/// compresses toward `isq_bits_per_weight`.
#[must_use]
pub fn isq_weight_bytes(counts: &ParamCounts, isq_bits_per_weight: f64) -> f64 {
    (counts.non_embedding_params as f64 * isq_bits_per_weight / 8.0)
        + (counts.embedding_params as f64 * counts.embedding_dtype_bytes)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    /// Hand-builds a minimal valid safetensors file (header + one byte of
    /// dummy data per tensor — `param_counts` never reads the data section).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn write_fake_safetensors(dir: &Path, name: &str, tensors: &[(&str, &str, &[u64])]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (tensor_name, dtype, shape) in tensors {
            let elements: u64 = shape.iter().product();
            let size = elements * dtype_size_bytes(dtype).unwrap() as u64;
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}),
            );
            offset += size;
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(dir.join(name)).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&vec![0u8; offset as usize]).unwrap();
    }

    #[test]
    fn sums_safetensors_file_sizes() {
        let dir = TempDir::new().unwrap();
        write_fake_safetensors(dir.path(), "a.safetensors", &[("w", "F32", &[100])]);
        write_fake_safetensors(dir.path(), "b.safetensors", &[("w", "F32", &[100])]);
        fs::write(dir.path().join("readme.txt"), "not a tensor file").unwrap();

        let total = safetensors_dir_bytes(dir.path()).unwrap();
        // Each file: 8-byte length prefix + header JSON + 400 bytes of data.
        assert!(total > 800, "{total}");
    }

    // `embedding_dtype_bytes` is a fixed lookup-table value (2.0 for
    // BF16), not an accumulated computation — exact equality is correct.
    #[allow(clippy::float_cmp)]
    #[test]
    fn splits_embedding_from_non_embedding_params() {
        let dir = TempDir::new().unwrap();
        write_fake_safetensors(
            dir.path(),
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[1000, 128]),
                ("model.layers.0.mlp.gate_proj.weight", "BF16", &[512, 128]),
            ],
        );

        let counts = param_counts(dir.path()).unwrap();
        assert_eq!(counts.embedding_params, 1000 * 128);
        assert_eq!(counts.non_embedding_params, 512 * 128);
        assert_eq!(counts.embedding_dtype_bytes, 2.0);
    }

    // Reads a real local checkpoint's safetensors headers — machine-specific
    // path, not run by default. `cargo test -p studio-core -- --ignored`.
    #[test]
    #[ignore = "reads a real local checkpoint at a machine-specific path — not run by default"]
    fn real_qwen3_5_4b_param_counts_are_plausible() {
        let dir = Path::new("/home/hahihula/mywork/ai/additional_models/Qwen3.5-4B");
        let counts = param_counts(dir).unwrap();
        let total = counts.embedding_params + counts.non_embedding_params;
        // "4B" is a rounded model name; real dense param counts for
        // published "4B"-class checkpoints are typically ~3.5-4.5B.
        assert!((3_000_000_000..5_000_000_000).contains(&total), "{total}");
        // vocab_size 248320 × hidden_size 1024 (real Qwen3.5-4B config) —
        // embed_tokens should be a small fraction of the total, not most of it.
        assert!(counts.embedding_params < total / 4, "{counts:?}");
    }

    #[test]
    fn isq_estimate_keeps_embedding_dense() {
        let counts = ParamCounts {
            embedding_params: 1_000_000,
            embedding_dtype_bytes: 2.0,
            non_embedding_params: 10_000_000,
        };
        // q4k ≈ 4.5 bits/weight in practice; use a round 4.0 for the test.
        let bytes = isq_weight_bytes(&counts, 4.0);
        let expected = 10_000_000.0 * 4.0 / 8.0 + 1_000_000.0 * 2.0;
        assert!((bytes - expected).abs() < 1.0);
    }
}
