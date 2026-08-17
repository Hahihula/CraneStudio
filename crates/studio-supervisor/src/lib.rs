//! Child process lifecycle: spawn/kill/health, log capture, exit
//! classification, and the detach lease. See PLAN.md §3.1, §3.1a, §7.4.

mod classify;
mod health;
mod log_ring;
mod registry;
mod spawn;

pub use classify::{ExitClassification, ExitContext, classify};
pub use log_ring::LogRing;
pub use registry::{ChildId, ChildInfo, ChildState, Supervisor, reap_stale_children};
pub use spawn::SpawnRequest;
