//! The local measurement DB, per PLAN.md §7.3: "every launch is a data
//! point." Records are keyed by a config fingerprint (model + variant + KV
//! quant + context + concurrency + backend class), not by profile name, so
//! "has this exact configuration ever been measured" can be answered
//! without a saved profile existing at all. Local measurements always take
//! precedence over the catalog's ship-time reference figures
//! (`catalog::schema::Variant::measured`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::catalog::schema::KvQuant;
use crate::hardware::{Backend, GpuInfo};

pub const SCHEMA_VERSION: u32 = 1;

/// Current time as RFC3339 (`"2026-08-16T14:25:00Z"`), for `MeasurementRecord::at`.
#[must_use]
pub fn now_iso() -> String {
    time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Ok,
    Oom,
    Thrashed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementRecord {
    pub schema_version: u32,
    pub key: String,
    pub predicted_bytes: u64,
    pub measured_peak_bytes: u64,
    /// The deepest context a real request actually exercised — §7.3: "a
    /// run that never exceeded 8k tokens has not verified that the
    /// configuration works at 256k." Never the configured `max_seq_len`.
    pub max_depth_reached: usize,
    pub kv_swaps: u64,
    pub decode_tokens_sec: f32,
    pub outcome: Outcome,
    pub at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasurementDb {
    pub records: Vec<MeasurementRecord>,
}

impl MeasurementDb {
    /// A missing or unparsable file loads as an empty DB, matching the
    /// "never fail to start" convention used throughout `studio-core`.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|text| ron::from_str(&text).ok()).unwrap_or_default()
    }

    /// # Errors
    /// Any I/O failure creating the parent directory, serializing, or
    /// writing the file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).map_err(std::io::Error::other)?;
        std::fs::write(path, text)
    }

    /// The most recent record for `key` — records are append-only, so the
    /// last match is the latest by construction.
    #[must_use]
    pub fn latest_for(&self, key: &str) -> Option<&MeasurementRecord> {
        self.records.iter().rev().find(|r| r.key == key)
    }

    /// The most recent record for `key` with a meaningful peak-bytes
    /// figure to show — `Oom`/`Failed` runs have none, so an OOM after an
    /// earlier success must not shadow that success's real number here
    /// (the OOM itself is what `prior_oom_for` surfaces instead).
    #[must_use]
    pub fn latest_successful_for(&self, key: &str) -> Option<&MeasurementRecord> {
        self.records.iter().rev().find(|r| r.key == key && matches!(r.outcome, Outcome::Ok | Outcome::Thrashed))
    }

    /// The most recent `Oom` record for `key`, if this exact configuration
    /// has ever OOM'd — what pre-warns the wizard before it's tried again.
    #[must_use]
    pub fn prior_oom_for(&self, key: &str) -> Option<&MeasurementRecord> {
        self.records.iter().rev().find(|r| r.key == key && r.outcome == Outcome::Oom)
    }

    /// Mean of measured÷predicted across every non-failed record with a
    /// real prediction to compare against — `Oom`/`Failed` runs have no
    /// meaningful peak-bytes figure to ratio. `None` until there's at
    /// least one data point, so callers know to fall back to raw
    /// predictions rather than multiply by a fabricated `1.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn correction_factor(&self) -> Option<f64> {
        let ratios: Vec<f64> = self
            .records
            .iter()
            .filter(|r| matches!(r.outcome, Outcome::Ok | Outcome::Thrashed) && r.predicted_bytes > 0)
            .map(|r| r.measured_peak_bytes as f64 / r.predicted_bytes as f64)
            .collect();
        if ratios.is_empty() {
            return None;
        }
        Some(ratios.iter().sum::<f64>() / ratios.len() as f64)
    }
}

/// Appends `entry` and persists the DB.
///
/// # Errors
/// Any I/O failure persisting the updated DB.
pub fn record(entry: MeasurementRecord, path: &Path) -> std::io::Result<()> {
    let mut db = MeasurementDb::load(path);
    db.records.push(entry);
    db.save(path)
}

/// Backend class string used in measurement keys, e.g. `"cuda_sm86"`,
/// `"metal"`, `"cpu"` — mirrors the format the catalog's own ship-time
/// figures already use (`catalog::schema::Variant::measured`'s keys).
#[must_use]
pub fn backend_class(backend: Backend, gpu: Option<&GpuInfo>) -> String {
    match backend {
        Backend::Cuda => gpu.and_then(|g| g.compute_capability).map_or_else(|| "cuda".to_string(), |(major, minor)| format!("cuda_sm{major}{minor}")),
        Backend::Metal => "metal".to_string(),
        Backend::Rocm => "rocm".to_string(),
        Backend::Cpu => "cpu".to_string(),
    }
}

#[must_use]
pub fn build_key(model_id: &str, variant_label: &str, kv_quant: Option<KvQuant>, context: usize, concurrency: usize, backend_class: &str) -> String {
    let kv = kv_quant.map_or_else(|| "none".to_string(), |q| match q { KvQuant::Int8 => "int8".to_string(), KvQuant::Int4 => "int4".to_string() });
    format!("{model_id}|{variant_label}|kv:{kv}|ctx:{context}|conc:{concurrency}|{backend_class}")
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;

    fn ok_record(key: &str, predicted: u64, measured: u64) -> MeasurementRecord {
        MeasurementRecord {
            schema_version: SCHEMA_VERSION,
            key: key.to_string(),
            predicted_bytes: predicted,
            measured_peak_bytes: measured,
            max_depth_reached: 131_072,
            kv_swaps: 0,
            decode_tokens_sec: 62.4,
            outcome: Outcome::Ok,
            at: "2026-08-16T14:25:00Z".to_string(),
        }
    }

    #[test]
    fn round_trips_through_ron() {
        let file = NamedTempFile::new().unwrap();
        std::fs::remove_file(file.path()).unwrap();
        record(ok_record("k1", 100, 110), file.path()).unwrap();

        let db = MeasurementDb::load(file.path());
        assert_eq!(db.records.len(), 1);
        assert_eq!(db.latest_for("k1").unwrap().measured_peak_bytes, 110);
    }

    #[test]
    fn latest_for_returns_the_most_recently_appended_match() {
        let mut db = MeasurementDb::default();
        db.records.push(ok_record("k1", 100, 110));
        db.records.push(ok_record("k1", 100, 120));
        db.records.push(ok_record("k2", 50, 55));

        assert_eq!(db.latest_for("k1").unwrap().measured_peak_bytes, 120);
        assert_eq!(db.latest_for("k2").unwrap().measured_peak_bytes, 55);
        assert!(db.latest_for("nope").is_none());
    }

    #[test]
    fn an_oom_after_a_success_does_not_shadow_the_successful_measurement() {
        let mut db = MeasurementDb::default();
        db.records.push(ok_record("k1", 100, 110));
        let mut oom = ok_record("k1", 100, 0);
        oom.outcome = Outcome::Oom;
        db.records.push(oom);

        // latest_for sees the OOM (it's the most recent record overall)...
        assert_eq!(db.latest_for("k1").unwrap().outcome, Outcome::Oom);
        // ...but the display-facing lookup still finds the real measurement.
        assert_eq!(db.latest_successful_for("k1").unwrap().measured_peak_bytes, 110);
    }

    #[test]
    fn prior_oom_is_found_even_after_a_later_success() {
        let mut db = MeasurementDb::default();
        let mut oom = ok_record("k1", 100, 0);
        oom.outcome = Outcome::Oom;
        db.records.push(oom);
        db.records.push(ok_record("k1", 80, 75));

        assert!(db.prior_oom_for("k1").is_some());
        assert!(db.prior_oom_for("k2").is_none());
    }

    #[test]
    fn correction_factor_is_the_mean_measured_over_predicted_ratio() {
        let mut db = MeasurementDb::default();
        db.records.push(ok_record("k1", 100, 110)); // 1.1
        db.records.push(ok_record("k2", 200, 180)); // 0.9
        let mut failed = ok_record("k3", 999, 0);
        failed.outcome = Outcome::Failed;
        db.records.push(failed);

        let factor = db.correction_factor().unwrap();
        assert!((factor - 1.0).abs() < 1e-9, "{factor}");
    }

    #[test]
    fn no_correction_factor_without_any_data() {
        assert!(MeasurementDb::default().correction_factor().is_none());
    }

    #[test]
    fn backend_class_includes_compute_capability_for_cuda() {
        let gpu = GpuInfo {
            index: 0,
            name: "RTX 3090".to_string(),
            vram_total: 0,
            vram_free: 0,
            compute_capability: Some((8, 6)),
            driver_version: None,
            unified_memory: false,
        };
        assert_eq!(backend_class(Backend::Cuda, Some(&gpu)), "cuda_sm86");
        assert_eq!(backend_class(Backend::Cpu, None), "cpu");
    }
}
