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

#[must_use]
pub fn expand_tilde(input: &str) -> PathBuf {
    let Some(dirs) = directories::BaseDirs::new() else {
        return PathBuf::from(input);
    };
    if input == "~" {
        return dirs.home_dir().to_path_buf();
    }
    if let Some(rest) = input.strip_prefix("~/") {
        return dirs.home_dir().join(rest);
    }
    PathBuf::from(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_bare_and_prefixed_tilde() {
        let home = directories::BaseDirs::new()
            .unwrap()
            .home_dir()
            .to_path_buf();
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/models/foo"), home.join("models/foo"));
    }

    #[test]
    fn leaves_non_tilde_paths_unchanged() {
        assert_eq!(expand_tilde("/mnt/models"), PathBuf::from("/mnt/models"));
        assert_eq!(
            expand_tilde("relative/path"),
            PathBuf::from("relative/path")
        );
        // Not `~user` shell semantics — left untouched rather than guessing.
        assert_eq!(expand_tilde("~bob/models"), PathBuf::from("~bob/models"));
    }
}
