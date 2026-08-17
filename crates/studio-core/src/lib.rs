//! Shared domain types and pure logic: hardware probing, the model catalog,
//! the VRAM estimator, the download manager, profiles, and launch-spec
//! construction. No terminal-rendering code lives here — see PLAN.md §3.3.

pub mod catalog;
pub mod config;
pub mod download;
pub mod hardware;
pub mod paths;
