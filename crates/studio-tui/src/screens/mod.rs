//! One module per screen, per PLAN.md §4: dashboard, hardware report,
//! model browser, launch wizard, connect screen, chat playground, plus the
//! quit-lease popup (§3.1a) that can render over any of them.

pub mod browser;
pub mod chat;
pub mod connect;
pub mod doctor;
pub mod home;
pub mod quit_prompt;
pub mod wizard;
