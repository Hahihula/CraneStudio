//! State shared between the control API (§3.1a) and the `/v1/*` gateway
//! (§3.2) — they're two listeners on two ports, but both live inside one
//! daemon process and need the same view of "what's configured" and
//! "what's running".

use std::sync::Arc;

use studio_core::launch::LaunchSpec;

use crate::control::Daemon;
use crate::registry::ModelRegistry;

pub type SpawnBuilder =
    fn(&LaunchSpec) -> std::io::Result<(studio_supervisor::SpawnRequest, String)>;

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
        })
    }
}
