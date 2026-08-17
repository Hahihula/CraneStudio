//! Metal GPU probe, per PLAN.md §6/§2.10: crane-serve's own VRAM query
//! returns `(0, 0)` on Metal, so this is independent of Crane. Apple Silicon
//! is unified memory — there is no separate VRAM pool to query — so
//! `vram_total` is the OS-recommended working set, and `vram_free` is that
//! headroom further capped by whatever system RAM is actually free right
//! now, since the GPU and every other process share the same pool.

use super::GpuInfo;

pub(super) fn probe() -> Vec<GpuInfo> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram_available = sys.available_memory();

    metal::Device::all()
        .into_iter()
        .enumerate()
        .map(|(index, device)| {
            let working_set = device.recommended_max_working_set_size();
            let allocated = device.current_allocated_size();
            let headroom = working_set.saturating_sub(allocated);
            GpuInfo {
                index,
                name: device.name().to_string(),
                vram_total: working_set,
                vram_free: headroom.min(ram_available),
                compute_capability: None,
                driver_version: None,
                unified_memory: device.has_unified_memory(),
            }
        })
        .collect()
}
