//! Global settings, per PLAN.md §5 (`config.ron`). Only the piece §9
//! actually needs right now — the HF token, stored `0600` and never logged
//! — is implemented; the rest of §5's schema (gateway host/port, headroom
//! fraction, telemetry, …) grows here as the milestones that consume them
//! land.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Default system prompt for the chat playground (§4.6) — small models left
/// with no language guidance at all have been observed drifting into a
/// single language regardless of what the user writes in.
pub const DEFAULT_SYSTEM_PROMPT: &str = "Respond in the same language the user writes in.";

/// crane-serve itself defaults `max_tokens` to 512 when a request omits it
/// (`openai_api.rs`'s `default_max_tokens`) — too low for anything beyond a
/// short reply (verified live: a multi-file code answer got hard-cut mid
/// sentence). The server clamps whatever it's sent to the model's actual
/// remaining context, so a generous default here costs nothing on requests
/// that don't need it.
pub const DEFAULT_MAX_TOKENS: usize = 4096;

/// Mirrors crane-serve's own fallback (`openai.rs`'s handler) when a request
/// omits `temperature`, so sending it explicitly changes nothing by default.
pub const DEFAULT_TEMPERATURE: f64 = 0.8;

fn default_system_prompt() -> String {
    DEFAULT_SYSTEM_PROMPT.to_string()
}

fn default_max_tokens() -> usize {
    DEFAULT_MAX_TOKENS
}

fn default_temperature() -> f64 {
    DEFAULT_TEMPERATURE
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// `HuggingFace` access token, for gated repos (§9). Never logged.
    #[serde(default)]
    pub hf_token: Option<String>,
    /// §7.3: there's no telemetry endpoint yet, so this only distinguishes
    /// "never asked" from "asked and declined" for whenever one exists —
    /// no prompt is shown and nothing is ever uploaded in v1.
    #[serde(default)]
    pub telemetry: Telemetry,
    /// System prompt sent with every chat playground request (§4.6),
    /// editable from the chat screen and remembered as "last used" here.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// `max_tokens` sent with every chat playground request, editable from
    /// the chat screen (`Ctrl-L`) and remembered as "last used" here.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// `temperature` sent with every chat playground request, editable from
    /// the chat screen (`Ctrl-T`) and remembered as "last used" here.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// TUI color palette, cycled with `F2` from any screen and remembered
    /// here.
    #[serde(default)]
    pub theme: ThemeName,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hf_token: None,
            telemetry: Telemetry::default(),
            system_prompt: default_system_prompt(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            theme: ThemeName::default(),
        }
    }
}

/// Named color palettes for the TUI (`studio-tui`'s `theme` module maps each
/// to concrete colors) — kept here, not in `studio-tui`, since it's
/// persisted config data like everything else in this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThemeName {
    /// `CraneStudio`'s own palette — the default.
    #[default]
    Crane,
    Monokai,
    Dracula,
    Plain,
}

impl ThemeName {
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            ThemeName::Crane => ThemeName::Monokai,
            ThemeName::Monokai => ThemeName::Dracula,
            ThemeName::Dracula => ThemeName::Plain,
            ThemeName::Plain => ThemeName::Crane,
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ThemeName::Crane => "Crane",
            ThemeName::Monokai => "Monokai",
            ThemeName::Dracula => "Dracula",
            ThemeName::Plain => "Plain",
        }
    }
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
        assert_eq!(loaded.system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(loaded.max_tokens, DEFAULT_MAX_TOKENS);
        assert!((loaded.temperature - DEFAULT_TEMPERATURE).abs() < f64::EPSILON);
        assert_eq!(loaded.theme, ThemeName::Crane);

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
        assert_eq!(config.system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(config.max_tokens, DEFAULT_MAX_TOKENS);
        assert!((config.temperature - DEFAULT_TEMPERATURE).abs() < f64::EPSILON);
        assert_eq!(config.theme, ThemeName::Crane);
    }

    #[test]
    fn theme_cycles_through_every_palette_and_back() {
        assert_eq!(ThemeName::Crane.next(), ThemeName::Monokai);
        assert_eq!(ThemeName::Monokai.next(), ThemeName::Dracula);
        assert_eq!(ThemeName::Dracula.next(), ThemeName::Plain);
        assert_eq!(ThemeName::Plain.next(), ThemeName::Crane);
    }

    #[test]
    fn a_config_file_written_before_system_prompt_existed_still_loads() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.ron");
        fs::write(&path, "(hf_token: None, telemetry: Unasked)").unwrap();

        let config = Config::load(&path);
        assert_eq!(config.system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(config.max_tokens, DEFAULT_MAX_TOKENS);
        assert!((config.temperature - DEFAULT_TEMPERATURE).abs() < f64::EPSILON);
    }
}
