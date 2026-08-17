//! axum control API and `/v1/*` multiplexing across child model servers.
//! See PLAN.md §3.2.

mod control;
mod measure;
mod proxy;
mod registry;
mod state;

pub use control::{DEFAULT_CONTROL_PORT, DETACH_GRACE_PERIOD, Daemon, reap_stale_children, router};
pub use proxy::{DEFAULT_GATEWAY_PORT, router as gateway_router};
pub use registry::ModelRegistry;
pub use state::{GatewayState, SpawnBuilder};
