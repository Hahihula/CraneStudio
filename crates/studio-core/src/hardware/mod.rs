//! Hardware probing, per PLAN.md §6: GPU/CPU/RAM/disk. Reports *free* VRAM,
//! not just total — fit verdicts must be based on what's actually available,
//! since a desktop's compositor and browser routinely hold several GiB.

mod cpu;
pub(crate) mod disk;
mod gpu;

use std::path::Path;

pub use cpu::CpuInfo;
pub use disk::DiskInfo;
pub use gpu::GpuInfo;
use sysinfo::System;

/// What *this binary* was built with — see PLAN.md §13. Backend-conditional
/// code should match on this exhaustively, so adding a variant (`ROCm` today)
/// produces compile errors at exactly the sites needing work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    Rocm,
    Cpu,
}

impl Backend {
    /// Exactly one of `cuda`/`metal` is enabled per build (mutually
    /// exclusive build features), mirroring `crane_serve::run`'s device
    /// priority (§2.6): cuda → rocm → metal → cpu.
    #[must_use]
    pub fn current() -> Self {
        #[cfg(feature = "cuda")]
        {
            Backend::Cuda
        }
        #[cfg(all(not(feature = "cuda"), target_os = "macos", feature = "metal"))]
        {
            Backend::Metal
        }
        #[cfg(not(any(feature = "cuda", all(target_os = "macos", feature = "metal"))))]
        {
            Backend::Cpu
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareReport {
    pub gpus: Vec<GpuInfo>,
    pub cpu: CpuInfo,
    pub ram_total: u64,
    pub ram_available: u64,
    pub disk: Vec<DiskInfo>,
    pub backend: Backend,
}

/// `models_dir` need not exist yet — disk reporting matches it against
/// mount points by path prefix, not by `stat`ing it.
#[must_use]
pub fn probe(models_dir: &Path) -> HardwareReport {
    let sys = System::new_all();

    HardwareReport {
        gpus: gpu::probe(),
        cpu: cpu::probe(&sys),
        ram_total: sys.total_memory(),
        ram_available: sys.available_memory(),
        disk: disk::probe(models_dir),
        backend: Backend::current(),
    }
}
