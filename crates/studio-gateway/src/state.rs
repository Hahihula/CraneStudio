//! State shared between the control API (§3.1a) and the `/v1/*` gateway
//! (§3.2) — they're two listeners on two ports, but both live inside one
//! daemon process and need the same view of "what's configured" and
//! "what's running".

use std::sync::{Arc, RwLock};

use studio_core::launch::LaunchSpec;

use crate::control::Daemon;
use crate::registry::ModelRegistry;

pub type SpawnBuilder =
    fn(&LaunchSpec) -> std::io::Result<(studio_supervisor::SpawnRequest, String)>;

/// What the gateway is actually bound to right now — updated by
/// `serve_with_rebind` after every (re)bind, so a control-API status query
/// always reflects reality rather than a stale start-up value.
#[derive(Debug, Clone)]
pub struct BoundInfo {
    pub host: String,
    pub port: u16,
}

pub struct GatewayState {
    pub daemon: Arc<Daemon>,
    pub registry: ModelRegistry,
    pub client: reqwest::Client,
    /// How to turn a `LaunchSpec` into the actual `SpawnRequest` for an
    /// on-demand start. Always `control::spawn_request_for` (a real
    /// `cranestudio __serve` re-exec) via `new` — `with_spawn_builder` is
    /// the seam integration tests use to point on-demand starts at a real
    /// but lightweight stand-in process instead of a real crane-serve
    /// model.
    pub(crate) spawn_builder: SpawnBuilder,
    /// PLAN.md §10.1: set only when the gateway is bound off loopback —
    /// every `/v1/*` request must then carry `Authorization: Bearer
    /// <token>`. `None` (the loopback default) requires no auth. A
    /// `RwLock` (not a plain field) because the TUI can flip LAN exposure
    /// on/off at runtime via the control API, with running models
    /// undisturbed — see `crate::control`'s `/control/gateway/public`.
    auth_token: RwLock<Option<String>>,
    bound: RwLock<BoundInfo>,
}

impl GatewayState {
    #[must_use]
    pub fn new(daemon: Arc<Daemon>) -> Arc<Self> {
        Self::with_spawn_builder(daemon, crate::control::spawn_request_for)
    }

    #[must_use]
    pub fn with_spawn_builder(daemon: Arc<Daemon>, spawn_builder: SpawnBuilder) -> Arc<Self> {
        Arc::new(GatewayState {
            daemon,
            registry: ModelRegistry::new(),
            client: reqwest::Client::new(),
            spawn_builder,
            auth_token: RwLock::new(None),
            bound: RwLock::new(BoundInfo {
                host: "127.0.0.1".to_string(),
                port: 0,
            }),
        })
    }

    #[must_use]
    pub fn auth_token(&self) -> Option<String> {
        self.auth_token
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn set_auth_token(&self, token: Option<String>) {
        *self
            .auth_token
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = token;
    }

    #[must_use]
    pub fn bound(&self) -> BoundInfo {
        self.bound
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn set_bound(&self, host: String, port: u16) {
        *self
            .bound
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = BoundInfo { host, port };
    }
}
