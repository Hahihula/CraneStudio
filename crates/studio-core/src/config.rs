//! Global settings, per PLAN.md §5 (`config.ron`). Only the piece §9
//! actually needs right now — the HF token, stored `0600` and never logged
//! — is implemented; the rest of §5's schema (gateway host/port, headroom
//! fraction, telemetry, …) grows here as the milestones that consume them
//! land.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// `HuggingFace` access token, for gated repos (§9). Never logged.
    #[serde(default)]
    pub hf_token: Option<String>,
    /// §7.3: there's no telemetry endpoint yet, so this only distinguishes
    /// "never asked" from "asked and declined" for whenever one exists —
    /// no prompt is shown and nothing is ever uploaded in v1.
    #[serde(default)]
    pub telemetry: Telemetry,
}

/// §7.3's telemetry consent state — `Unasked` today and for the whole of
/// v1, since asking would be asking for a server that doesn't exist yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Telemetry {
    #[default]
    Unasked,
    Enabled,
    Declined,
}

impl Config {
    /// A missing or unparsable file is treated as an empty config, not an
    /// error — matches the "never fail to start" spirit applied to the
    /// catalog loader (§8.1).
    #[must_use]
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|text| ron::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// # Errors
    /// Any I/O failure creating the parent directory, serializing, writing,
    /// or setting owner-only permissions on the file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| io::Error::other(e.to_string()))?;
        fs::write(path, text)?;
        restrict_permissions(path)
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn round_trips_and_restricts_permissions() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.ron");

        let config = Config {
            hf_token: Some("hf_secret".to_string()),
            ..Config::default()
        };
        config.save(&path).unwrap();

        let loaded = Config::load(&path);
        assert_eq!(loaded.hf_token.as_deref(), Some("hf_secret"));
        assert_eq!(loaded.telemetry, Telemetry::Unasked);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn missing_file_loads_as_empty_default() {
        let config = Config::load(Path::new("/does/not/exist/config.ron"));
        assert_eq!(config.hf_token, None);
    }
}
