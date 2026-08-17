//! Discovering where a running daemon actually bound its control/gateway
//! ports, per PLAN.md §7.4's "port in use → retry on a different port
//! automatically" — applied to the daemon's own listening ports, not just
//! a model child's. A fresh machine's default gateway port (`:1234`)
//! collides with other local tools (LM Studio uses the same default), so
//! the daemon falls forward to the next free port instead of hard-failing,
//! and writes what it actually bound here so every other CraneStudio
//! process (the TUI, `cranestudio status`/`stop`/`attach`, …) can find it
//! without needing an env var set.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Endpoints {
    pub control_port: u16,
    pub gateway_port: u16,
}

fn file_path() -> PathBuf {
    crate::paths::data_dir().join("daemon-endpoints.ron")
}

/// Best-effort — an unwritable data dir just means other processes fall
/// back to env vars/defaults instead of finding this daemon automatically.
pub fn save(endpoints: Endpoints) {
    let path = file_path();
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
    if let Ok(text) = ron::ser::to_string_pretty(&endpoints, ron::ser::PrettyConfig::default()) {
        let _ = std::fs::write(&path, text);
    }
}

#[must_use]
pub fn load() -> Option<Endpoints> {
    let text = std::fs::read_to_string(file_path()).ok()?;
    ron::from_str(&text).ok()
}

/// Called on clean daemon shutdown so a later process doesn't try to reuse
/// a port this daemon no longer holds. A crash/`kill -9` leaves the file
/// behind — harmless, since callers only trust it after confirming the
/// daemon actually answers on the port it names (see `DaemonClient`).
pub fn clear() {
    let _ = std::fs::remove_file(file_path());
}

/// Resolution order: an explicit env var is a deliberate override and
/// always wins; otherwise prefer whatever the last daemon reported binding
/// (right, even after a port-conflict fallback); otherwise the hardcoded
/// default.
#[must_use]
pub fn resolve_control_port(default: u16) -> u16 {
    std::env::var("CRANESTUDIO_CONTROL_PORT").ok().and_then(|v| v.parse().ok()).or_else(|| load().map(|e| e.control_port)).unwrap_or(default)
}

#[must_use]
pub fn resolve_gateway_port(default: u16) -> u16 {
    std::env::var("CRANESTUDIO_GATEWAY_PORT").ok().and_then(|v| v.parse().ok()).or_else(|| load().map(|e| e.gateway_port)).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let text = ron::ser::to_string_pretty(&Endpoints { control_port: 41999, gateway_port: 1234 }, ron::ser::PrettyConfig::default()).unwrap();
        let back: Endpoints = ron::from_str(&text).unwrap();
        assert_eq!(back.control_port, 41999);
        assert_eq!(back.gateway_port, 1234);
    }
}
