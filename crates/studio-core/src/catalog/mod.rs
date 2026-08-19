//! The model catalog, per PLAN.md §8: the curated, versioned catalog
//! (§8.1), filtered `HuggingFace` search (§8.2), and local filesystem
//! scanning (§8.3). All three share one architecture-classification core
//! (`architecture`/`classify`/`gguf`) so a model gets the same verdict
//! whether it's a catalog entry, a search result, or a file on disk.

pub mod architecture;
mod classify;
pub(crate) mod gguf;
pub mod hf;
pub mod load;
pub mod local;
pub mod schema;

pub use architecture::{FAMILIES, Family};
pub use classify::Classification;
pub use load::{Source, load};
pub use schema::Catalog;
