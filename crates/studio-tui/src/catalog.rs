//! Plain-text rendering for the catalog browser and search (§8, §4.3) —
//! same "print and exit" treatment M1 gave `doctor`, ahead of the real
//! ratatui screens in M7.

use std::fmt::Write as _;

use studio_core::catalog::hf::HfCandidate;
use studio_core::catalog::local::LocalCandidate;
use studio_core::catalog::{Catalog, Classification, Source};

#[must_use]
pub fn render_catalog(catalog: &Catalog, source: Source) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Catalog ({}, updated {}):",
        source_label(source),
        catalog.updated
    );
    if catalog.models.is_empty() {
        let _ = writeln!(out, "  (empty)");
    }
    for model in &catalog.models {
        let _ = writeln!(out, "  {} — {}", model.id, model.display_name);
        for variant in &model.variants {
            let quant = variant.quant.as_deref().unwrap_or("unquantized");
            let _ = writeln!(
                out,
                "    {} ({quant}, {}) — {}",
                variant.id,
                crate::fmt::bytes(variant.download_bytes),
                variant.repo
            );
        }
    }
    out
}

fn source_label(source: Source) -> &'static str {
    match source {
        Source::Remote => "fetched live",
        Source::Cached => "cached",
        Source::Bundled => "bundled",
    }
}

#[must_use]
pub fn render_local_candidates(candidates: &[LocalCandidate]) -> String {
    let mut out = String::new();
    if candidates.is_empty() {
        let _ = writeln!(out, "No local models found.");
        return out;
    }
    for candidate in candidates {
        let _ = writeln!(out, "{}", candidate.path.display());
        let _ = writeln!(out, "  format: {:?}", candidate.format);
        render_classification(&candidate.classification, &mut out);
    }
    out
}

#[must_use]
pub fn render_hf_candidates(candidates: &[HfCandidate]) -> String {
    let mut out = String::new();
    if candidates.is_empty() {
        let _ = writeln!(out, "No results.");
        return out;
    }
    for candidate in candidates {
        let gated = if candidate.gated { " [gated]" } else { "" };
        let _ = writeln!(out, "{}{gated}", candidate.repo_id);
        render_classification(&candidate.classification, &mut out);
    }
    out
}

fn render_classification(classification: &Classification, out: &mut String) {
    match classification {
        Classification::Supported {
            model_type,
            vision,
            gated,
            audio,
        } => {
            let vision = if *vision { ", vision" } else { "" };
            let gated = if *gated { ", gated" } else { "" };
            let audio = if *audio { ", audio" } else { "" };
            let _ = writeln!(
                out,
                "  \u{25cf} supported — model_type: {model_type}{vision}{gated}{audio}"
            );
        }
        Classification::Unsupported { reason, .. } => {
            let _ = writeln!(out, "  \u{2715} {reason}");
        }
        Classification::Unknown { reason } => {
            let _ = writeln!(out, "  ? {reason}");
        }
    }
}
