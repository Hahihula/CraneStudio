//! Catalog loading, per PLAN.md §8.1: try the live GitHub raw URL first,
//! fall back to the last-fetched copy cached on disk, fall back again to the
//! copy baked into the binary at compile time. Never fail to start because
//! the fetch failed.

use std::path::Path;
use std::time::Duration;

use super::schema::Catalog;

/// The in-repo copy, versioned alongside `CraneStudio` and embedded at
/// compile time — this is what a fully offline, first-ever run sees.
const BUNDLED: &str = include_str!("../../../../catalog/models.ron");

pub const DEFAULT_REMOTE_URL: &str =
    "https://raw.githubusercontent.com/hahihula/CraneStudio/main/catalog/models.ron";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Remote,
    Cached,
    Bundled,
}

/// # Panics
/// Only if `catalog/models.ron` itself fails to parse — a build-time asset
/// bug, not a runtime condition, so this is deliberately not a `Result`.
#[must_use]
pub fn bundled() -> Catalog {
    parse(BUNDLED).expect("bundled catalog/models.ron must parse — this is a build-time asset bug")
}

fn parse(text: &str) -> Result<Catalog, ron::error::SpannedError> {
    ron::from_str(text)
}

/// `cache_path` is read/written best-effort — a failure to persist the
/// cache never fails the load, since the bundled copy is always a valid
/// fallback.
pub async fn load(remote_url: &str, cache_path: &Path) -> (Catalog, Source) {
    if let Some(catalog) = fetch_remote(remote_url).await {
        let _ = std::fs::create_dir_all(cache_path.parent().unwrap_or(Path::new(".")));
        let _ = std::fs::write(
            cache_path,
            ron::ser::to_string_pretty(&catalog, ron::ser::PrettyConfig::default())
                .unwrap_or_default(),
        );
        return (catalog, Source::Remote);
    }

    if let Ok(text) = std::fs::read_to_string(cache_path)
        && let Ok(catalog) = parse(&text)
    {
        return (catalog, Source::Cached);
    }

    (bundled(), Source::Bundled)
}

async fn fetch_remote(url: &str) -> Option<Catalog> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let text = client
        .get(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .await
        .ok()?;
    parse(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_parses_and_is_non_empty() {
        let catalog = bundled();
        assert!(!catalog.models.is_empty());
    }

    #[tokio::test]
    async fn unreachable_remote_and_missing_cache_falls_back_to_bundled() {
        let cache = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(cache.path()).unwrap(); // ensure no cache exists
        let (catalog, source) = load("http://127.0.0.1:1/nope.ron", cache.path()).await;
        assert_eq!(source, Source::Bundled);
        assert!(!catalog.models.is_empty());
    }

    // Hits the real GitHub raw URL — not run by default (§12). The
    // CraneStudio repo doesn't publish this path yet, so today this also
    // exercises the fallback-to-bundled path, just over a real network call
    // instead of a deliberately unroutable address.
    #[tokio::test]
    #[ignore = "hits a real network API — not run by default (§12)"]
    async fn default_remote_url_falls_back_gracefully_when_unpublished() {
        let cache = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(cache.path()).unwrap();
        let (catalog, _source) = load(DEFAULT_REMOTE_URL, cache.path()).await;
        assert!(!catalog.models.is_empty());
    }
}
