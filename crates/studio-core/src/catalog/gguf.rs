//! Minimal GGUF metadata reader — extracts `general.architecture` without
//! depending on candle-core or downloading a whole checkpoint. Deliberately
//! stops at the end of the metadata key/value section; the tensor info
//! table that follows (which candle's own parser also reads, since it needs
//! tensor offsets to actually load weights) is never touched, and never
//! needed for architecture detection. That's what makes a bounded HTTP
//! range request enough for §8.2 — llama.cpp's GGUF writer emits `general.*`
//! keys first, so `general.architecture` is normally within the first few
//! KB regardless of how large the rest of the file's metadata (e.g. a large
//! tokenizer vocab array) is.

use std::collections::HashMap;
use std::io::Read;

/// A scalar GGUF metadata value, widened to a common representation. Only
/// scalars are kept — arrays (tokenizer vocab lists, rope dimension
/// sections, etc.) are read past and discarded, since none of the
/// architecture-dimension keys the estimator needs are ever array-typed.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Int(u64),
    Str(String),
}

impl MetaValue {
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            MetaValue::Int(n) => usize::try_from(*n).ok(),
            MetaValue::Str(_) => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            MetaValue::Int(_) => None,
        }
    }
}

/// Reads every scalar key/value pair out of a GGUF file's metadata section.
/// Unlike [`read_architecture`], this reads to the end of the kv section
/// (no early exit) — meant for local files opened directly, where the whole
/// section is cheaply available, not a bounded HTTP range fetch. This is
/// what lets the wizard build a full `ModelConfig` from a local `.gguf`
/// file's own embedded dimensions, with no companion `config.json` and no
/// network access — see `estimator::gguf_config`.
pub fn read_all_scalars<R: Read>(reader: &mut R) -> Option<HashMap<String, MetaValue>> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(reader)?;
    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    let mut map = HashMap::with_capacity(usize::try_from(kv_count).unwrap_or(64));
    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)?;
        match value_type {
            0 | 1 | 7 => {
                let mut b = [0u8; 1];
                reader.read_exact(&mut b).ok()?;
                map.insert(key, MetaValue::Int(u64::from(b[0])));
            }
            2 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b).ok()?;
                map.insert(key, MetaValue::Int(u64::from(u16::from_le_bytes(b))));
            }
            3 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b).ok()?;
                map.insert(
                    key,
                    MetaValue::Int(u64::from(i16::from_le_bytes(b).cast_unsigned())),
                );
            }
            4 | 5 => {
                let v = read_u32(reader)?;
                map.insert(key, MetaValue::Int(u64::from(v)));
            }
            6 => {
                let mut b = [0u8; 4];
                reader.read_exact(&mut b).ok()?;
                map.insert(
                    key,
                    MetaValue::Int(u64::from(f32::from_le_bytes(b).to_bits())),
                );
            }
            8 => {
                let s = read_string(reader)?;
                map.insert(key, MetaValue::Str(s));
            }
            9 => {
                skip_value(reader, 9)?;
            }
            10 | 11 => {
                let v = read_u64(reader)?;
                map.insert(key, MetaValue::Int(v));
            }
            12 => {
                let mut b = [0u8; 8];
                reader.read_exact(&mut b).ok()?;
                map.insert(key, MetaValue::Int(f64::from_le_bytes(b).to_bits()));
            }
            _ => return None,
        }
    }
    Some(map)
}

pub fn read_architecture<R: Read>(reader: &mut R) -> Option<String> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(reader)?;
    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)?;
        if key == "general.architecture" {
            return if value_type == 8 {
                read_string(reader)
            } else {
                None
            };
        }
        skip_value(reader, value_type)?;
    }
    None
}

fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Option<String> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    r.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn skip_bytes<R: Read>(r: &mut R, n: usize) -> Option<()> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).ok()
}

/// GGUF metadata value type ids: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32
/// 7=bool 8=string 9=array 10=u64 11=i64 12=f64.
fn skip_value<R: Read>(r: &mut R, value_type: u32) -> Option<()> {
    match value_type {
        0 | 1 | 7 => skip_bytes(r, 1),
        2 | 3 => skip_bytes(r, 2),
        4..=6 => skip_bytes(r, 4),
        10..=12 => skip_bytes(r, 8),
        8 => read_string(r).map(|_| ()),
        9 => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            for _ in 0..len {
                skip_value(r, elem_type)?;
            }
            Some(())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        write_string(buf, value);
    }

    fn write_kv_u32_array(buf: &mut Vec<u8>, key: &str, values: &[u32]) {
        write_string(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes()); // type: array
        buf.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn header(kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf
    }

    #[test]
    fn finds_architecture_as_the_first_key() {
        let mut buf = header(1);
        write_kv_string(&mut buf, "general.architecture", "qwen35");
        assert_eq!(read_architecture(&mut &buf[..]).as_deref(), Some("qwen35"));
    }

    #[test]
    fn skips_unrelated_keys_including_arrays() {
        let mut buf = header(3);
        write_kv_u32_array(&mut buf, "tokenizer.ggml.token_type", &[1, 2, 3, 4, 5]);
        write_kv_string(&mut buf, "general.name", "Some Model");
        write_kv_string(&mut buf, "general.architecture", "hunyuan-dense");
        assert_eq!(
            read_architecture(&mut &buf[..]).as_deref(),
            Some("hunyuan-dense")
        );
    }

    #[test]
    fn missing_key_returns_none() {
        let mut buf = header(1);
        write_kv_string(&mut buf, "general.name", "Some Model");
        assert_eq!(read_architecture(&mut &buf[..]), None);
    }

    #[test]
    fn rejects_bad_magic() {
        let buf = b"NOPE".to_vec();
        assert_eq!(read_architecture(&mut &buf[..]), None);
    }

    #[test]
    fn truncated_buffer_returns_none_instead_of_panicking() {
        let mut buf = header(1);
        write_string(&mut buf, "general.architecture");
        // cut off mid-value — simulates a too-small HTTP range fetch
        assert_eq!(read_architecture(&mut &buf[..]), None);
    }

    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type: u32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn reads_all_scalars_and_skips_arrays() {
        let mut buf = header(3);
        write_kv_string(&mut buf, "general.architecture", "qwen35");
        write_kv_u32(&mut buf, "qwen35.block_count", 24);
        write_kv_u32_array(&mut buf, "tokenizer.ggml.token_type", &[1, 2, 3]);
        let map = read_all_scalars(&mut &buf[..]).unwrap();
        assert_eq!(
            map.get("general.architecture").and_then(MetaValue::as_str),
            Some("qwen35")
        );
        assert_eq!(
            map.get("qwen35.block_count").and_then(MetaValue::as_usize),
            Some(24)
        );
        assert!(!map.contains_key("tokenizer.ggml.token_type"));
    }
}
