//! The `CraneStudio` TUI: a centered, `btop`-flavored studio for running local
//! models — splash, launchpad, downloads, and the apps you point at a running
//! model. See PLAN.md §4.

pub mod app;
pub mod catalog;
pub mod daemon_client;
pub mod doctor;
pub mod download;
mod fmt;
pub mod models;
#[cfg(test)]
mod preview;
pub mod screens;
pub mod theme;
pub mod ui;
