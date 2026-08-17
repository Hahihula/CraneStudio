#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: Option<String>,
    pub unified_memory: bool,
}

pub(super) fn probe() -> Vec<GpuInfo> {
    #[cfg(feature = "cuda")]
    {
        cuda::probe()
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos", feature = "metal"))]
    {
        metal::probe()
    }
    #[cfg(not(any(feature = "cuda", all(target_os = "macos", feature = "metal"))))]
    {
        Vec::new()
    }
}
