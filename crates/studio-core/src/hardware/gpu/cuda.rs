//! Three-tier CUDA GPU probe, per PLAN.md §6: NVML first, then `nvidia-smi`
//! (exactly the query `crane/src/utils.rs` uses), then `cudarc` as a last
//! resort. NVML and `nvidia-smi` only need the NVIDIA driver; `cudarc`
//! additionally initialises a real CUDA context per device, so it is
//! reserved for machines where neither of the lighter paths worked.

use std::process::Command;

use super::GpuInfo;

pub(super) fn probe() -> Vec<GpuInfo> {
    probe_nvml()
        .or_else(probe_nvidia_smi)
        .or_else(probe_cudarc)
        .unwrap_or_default()
}

fn probe_nvml() -> Option<Vec<GpuInfo>> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let driver_version = nvml.sys_driver_version().ok();
    let count = nvml.device_count().ok()?;

    let gpus: Vec<GpuInfo> = (0..count)
        .filter_map(|index| {
            let device = nvml.device_by_index(index).ok()?;
            let mem = device.memory_info().ok()?;
            let name = device.name().unwrap_or_else(|_| format!("GPU {index}"));
            let compute_capability = device
                .cuda_compute_capability()
                .ok()
                .map(|cc| (cc.major.cast_unsigned(), cc.minor.cast_unsigned()));
            Some(GpuInfo {
                index: index as usize,
                name,
                vram_total: mem.total,
                vram_free: mem.free,
                compute_capability,
                driver_version: driver_version.clone(),
                unified_memory: false,
            })
        })
        .collect();

    if gpus.is_empty() { None } else { Some(gpus) }
}

fn probe_nvidia_smi() -> Option<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let gpus: Vec<GpuInfo> = stdout.lines().filter_map(parse_nvidia_smi_line).collect();

    if gpus.is_empty() { None } else { Some(gpus) }
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    let [index, name, total_mb, free_mb, driver_version, compute_cap] = parts[..] else {
        return None;
    };
    Some(GpuInfo {
        index: index.parse().ok()?,
        name: name.to_string(),
        vram_total: total_mb.parse::<u64>().ok()? * 1024 * 1024,
        vram_free: free_mb.parse::<u64>().ok()? * 1024 * 1024,
        compute_capability: parse_compute_cap(compute_cap),
        driver_version: Some(driver_version.to_string()),
        unified_memory: false,
    })
}

fn parse_compute_cap(s: &str) -> Option<(u32, u32)> {
    let (major, minor) = s.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn probe_cudarc() -> Option<Vec<GpuInfo>> {
    let count = cudarc::driver::CudaContext::device_count().ok()?;
    let count = usize::try_from(count).unwrap_or(0);

    let gpus: Vec<GpuInfo> = (0..count)
        .filter_map(|index| {
            let ctx = cudarc::driver::CudaContext::new(index).ok()?;
            let (free, total) = ctx.mem_get_info().ok()?;
            let name = ctx.name().unwrap_or_else(|_| format!("GPU {index}"));
            Some(GpuInfo {
                index,
                name,
                vram_total: total as u64,
                vram_free: free as u64,
                compute_capability: None,
                driver_version: None,
                unified_memory: false,
            })
        })
        .collect();

    if gpus.is_empty() { None } else { Some(gpus) }
}
