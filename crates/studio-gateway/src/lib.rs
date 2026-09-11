//! axum control API and `/v1/*` multiplexing across child model servers.
//! See PLAN.md §3.2.

mod control;
mod measure;
mod proxy;
mod registry;
mod runtime;
mod state;

pub use control::{
    DEFAULT_CONTROL_PORT, DETACH_GRACE_PERIOD, Daemon, GatewayStatus, reap_stale_children, router,
};
pub use proxy::{DEFAULT_GATEWAY_PORT, router as gateway_router};
pub use registry::ModelRegistry;
pub use runtime::{MAX_PORT_ATTEMPTS, bind_with_fallback, generate_auth_token, serve_with_rebind};
pub use state::{BoundInfo, GatewayState, SpawnBuilder};
