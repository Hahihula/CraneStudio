//! One module per screen: splash, launchpad (§4.1), hardware report (§4.2),
//! model browser (§4.3), download (§9), launch options (§4.4), the ready screen
//! and its app launcher (§4.5), and the chat app (§4.6) — plus the quit-lease
//! modal (§3.1a) that can render over any of them.

pub mod browser;
pub mod chat;
pub mod download;
pub mod hardware;
pub mod launchpad;
pub mod quit_prompt;
pub mod ready;
pub mod splash;
pub mod wizard;
