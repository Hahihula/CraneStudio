//! Peak-VRAM + `/v1/stats` sampling for one launched child, per PLAN.md
//! §7.3: "every launch is a data point." Runs for the child's whole
//! lifetime at ~1Hz and writes one `MeasurementRecord` once it's gone —
//! only when the caller supplied a `measurement_key` (the wizard does;
//! the raw `cranestudio launch`/`register` CLI paths don't, and simply
//! skip measurement).
//!
//! Exit detection doesn't rely on catching `Supervisor`'s exact
//! `ChildState::Exited` transition: an explicitly-stopped child is
//! `forget()`-ed by `Supervisor::stop` almost immediately after exiting,
//! which usually wins the race against this sampler's ~1s tick. Instead,
//! `state(id)` returning `None` (the child is simply no longer tracked) is
//! treated as "it's gone" on its own, and the outcome is inferred from
//! whatever was observed while it was still alive — see `outcome_for`.

use std::time::Duration;

use serde::Deserialize;
use studio_core::launch::LaunchSpec;
use studio_core::measurement::{MeasurementRecord, Outcome, SCHEMA_VERSION};
use studio_supervisor::{ChildId, ChildState, ExitClassification, Supervisor};

const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const STATS_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Deserialize, Default)]
struct StatsSnapshot {
    #[serde(default)]
    total_kv_swaps: u64,
    #[serde(default)]
    avg_decode_tokens_per_sec: f64,
    #[serde(default)]
    total_prompt_tokens: u64,
    #[serde(default)]
    total_completion_tokens: u64,
}

/// Starts the sampler as a background task and returns its handle — not
/// awaited by the launch handler itself (a slow sampler must never delay a
/// launch response), but the caller registers it with `Daemon` so a
/// whole-process shutdown can wait for it to finish writing its final
/// record before the tokio runtime (and every detached task still on it)
/// gets torn down.
pub fn spawn(
    supervisor: Supervisor,
    id: ChildId,
    spec: LaunchSpec,
    measurement_key: String,
    predicted_bytes: u64,
    baseline_vram_used: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let observed = sample_until_gone(&supervisor, id, &spec, baseline_vram_used).await;
        let record = MeasurementRecord {
            schema_version: SCHEMA_VERSION,
            key: measurement_key,
            predicted_bytes,
            measured_peak_bytes: observed.peak_vram_delta,
            max_depth_reached: observed.max_depth_reached,
            kv_swaps: observed.kv_swaps,
            decode_tokens_sec: observed.decode_tokens_sec,
            outcome: observed.outcome,
            at: studio_core::measurement::now_iso(),
        };
        let _ = studio_core::measurement::record(record, &studio_core::paths::measurements_file());
    })
}

/// What the sampler learns over a child's lifetime — everything a
/// `MeasurementRecord` needs except the caller-supplied key/prediction.
struct Observed {
    peak_vram_delta: u64,
    max_depth_reached: usize,
    kv_swaps: u64,
    decode_tokens_sec: f32,
    outcome: Outcome,
}

async fn sample_until_gone(
    supervisor: &Supervisor,
    id: ChildId,
    spec: &LaunchSpec,
    baseline_vram_used: u64,
) -> Observed {
    let client = reqwest::Client::new();
    let stats_url = format!("http://127.0.0.1:{}/v1/stats", spec.port);

    let mut peak_delta: u64 = 0;
    let mut last_kv_swaps: u64 = 0;
    let mut last_decode_tps: f32 = 0.0;
    let mut max_depth_reached: usize = 0;
    let mut prev_total_tokens: u64 = 0;
    let mut ever_healthy = false;
    let mut final_classification: Option<ExitClassification> = None;

    loop {
        tokio::time::sleep(SAMPLE_INTERVAL).await;

        match supervisor.state(id) {
            Some(ChildState::Healthy) => ever_healthy = true,
            Some(ChildState::Exited { classification, .. }) => {
                final_classification = Some(classification);
            }
            Some(ChildState::Starting) | None => {}
        }

        if let Some(gpu) = studio_core::hardware::probe_gpus()
            .into_iter()
            .find(|g| g.index == spec.device)
        {
            let used = gpu.vram_total.saturating_sub(gpu.vram_free);
            peak_delta = peak_delta.max(used.saturating_sub(baseline_vram_used));
        }

        if let Ok(resp) = client.get(&stats_url).timeout(STATS_TIMEOUT).send().await
            && let Ok(stats) = resp.json::<StatsSnapshot>().await
        {
            last_kv_swaps = stats.total_kv_swaps;
            #[allow(clippy::cast_possible_truncation)]
            {
                last_decode_tps = stats.avg_decode_tokens_per_sec as f32;
            }
            let total = stats.total_prompt_tokens + stats.total_completion_tokens;
            max_depth_reached = max_depth_reached.max(
                usize::try_from(total.saturating_sub(prev_total_tokens)).unwrap_or(usize::MAX),
            );
            prev_total_tokens = total;
        }

        let gone = final_classification.is_some() || supervisor.state(id).is_none();
        if gone {
            break;
        }
    }

    Observed {
        peak_vram_delta: peak_delta,
        max_depth_reached,
        kv_swaps: last_kv_swaps,
        decode_tokens_sec: last_decode_tps,
        outcome: outcome_for(final_classification, ever_healthy, last_kv_swaps),
    }
}

/// No `ChildState::Exited` was ever observed for most explicitly-stopped
/// children (see module docs) — `ever_healthy` is what distinguishes "ran
/// fine, then was stopped" (`Ok`/`Thrashed`) from "vanished without ever
/// working" (`Failed`) in that common case.
fn outcome_for(
    classification: Option<ExitClassification>,
    ever_healthy: bool,
    kv_swaps: u64,
) -> Outcome {
    match classification {
        Some(
            ExitClassification::OomAtLoad
            | ExitClassification::OomAtPrefill
            | ExitClassification::HostOomKilled,
        ) => Outcome::Oom,
        Some(ExitClassification::Stopped) | None if ever_healthy => {
            if kv_swaps > 0 {
                Outcome::Thrashed
            } else {
                Outcome::Ok
            }
        }
        _ => Outcome::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_classification_wins_regardless_of_health() {
        assert_eq!(
            outcome_for(Some(ExitClassification::OomAtPrefill), true, 0),
            Outcome::Oom
        );
        assert_eq!(
            outcome_for(Some(ExitClassification::HostOomKilled), false, 0),
            Outcome::Oom
        );
    }

    #[test]
    fn clean_stop_after_healthy_is_ok_unless_it_thrashed() {
        assert_eq!(
            outcome_for(Some(ExitClassification::Stopped), true, 0),
            Outcome::Ok
        );
        assert_eq!(
            outcome_for(Some(ExitClassification::Stopped), true, 3),
            Outcome::Thrashed
        );
    }

    #[test]
    fn vanishing_without_ever_becoming_healthy_is_failed() {
        assert_eq!(outcome_for(None, false, 0), Outcome::Failed);
        assert_eq!(
            outcome_for(Some(ExitClassification::CleanEarlyExit), false, 0),
            Outcome::Failed
        );
    }

    #[test]
    fn forgotten_child_with_no_classification_falls_back_to_health_history() {
        // The common race: `stop()` forgot the entry before we polled it.
        assert_eq!(outcome_for(None, true, 0), Outcome::Ok);
    }
}
