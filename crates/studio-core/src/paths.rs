//! Standard config/data directories, per PLAN.md §5. Linux:
//! `~/.config/cranestudio/`, `~/.local/share/cranestudio/`. macOS:
//! `~/Library/Application Support/cranestudio/` for both.

use std::path::PathBuf;

use directories::ProjectDirs;

fn project_dirs() -> ProjectDirs {
    ProjectDirs::from("", "", "cranestudio").expect("no home directory found for the current user")
}

#[must_use]
pub fn config_dir() -> PathBuf {
    project_dirs().config_dir().to_path_buf()
}

#[must_use]
pub fn data_dir() -> PathBuf {
    project_dirs().data_dir().to_path_buf()
}

#[must_use]
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

#[must_use]
pub fn profiles_dir() -> PathBuf {
    data_dir().join("profiles")
}

#[must_use]
pub fn measurements_file() -> PathBuf {
    data_dir().join("measurements.ron")
}
