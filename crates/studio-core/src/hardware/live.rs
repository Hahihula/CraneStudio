//! Cheap repeat sampling of the values that actually move while the TUI is
//! open — CPU load, RAM pressure, VRAM pressure — for the live `btop`-style
//! meters on the launchpad (§4.1). Deliberately *not* `probe()`: that runs a
//! full `System::new_all()` scan (processes, disks, the works) and is meant
//! to be called once, whereas this refreshes only CPU usage and RAM at ~2Hz
//! for the whole life of the session.

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

use super::gpu::{self, GpuInfo};

#[derive(Debug, Clone)]
pub struct Sample {
    /// 0–100, averaged over all logical cores.
    pub cpu_total: f32,
    /// 0–100 per logical core, in the OS's own core order.
    pub per_core: Vec<f32>,
    pub ram_total: u64,
    pub ram_available: u64,
    /// Re-probed every sample: another process (or a model we just started)
    /// taking VRAM is exactly the thing a live meter exists to show.
    pub gpus: Vec<GpuInfo>,
}

pub struct Sampler {
    sys: System,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler {
    #[must_use]
    pub fn new() -> Self {
        Sampler {
            sys: System::new_with_specifics(
                RefreshKind::nothing()
                    .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
                    .with_memory(MemoryRefreshKind::nothing().with_ram()),
            ),
        }
    }

    /// The first sample after construction reports 0% CPU — sysinfo needs two
    /// refreshes at least `MINIMUM_CPU_UPDATE_INTERVAL` apart to have a delta
    /// to divide. Callers poll on a timer, so the second sample is already
    /// meaningful; nothing here sleeps.
    pub fn sample(&mut self) -> Sample {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        Sample {
            cpu_total: self.sys.global_cpu_usage(),
            per_core: self
                .sys
                .cpus()
                .iter()
                .map(sysinfo::Cpu::cpu_usage)
                .collect(),
            ram_total: self.sys.total_memory(),
            ram_available: self.sys.available_memory(),
            gpus: gpu::probe(),
        }
    }
}
