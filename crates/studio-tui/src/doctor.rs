//! Plain-text rendering of a `HardwareReport`, per PLAN.md §4.2 — this is
//! what `cranestudio doctor` prints, and the first thing to ask a user for
//! in a bug report. No ratatui widgets needed for a print-and-exit screen.

use std::fmt::Write as _;

use studio_core::hardware::{Backend, HardwareReport};

#[must_use]
pub fn render(report: &HardwareReport) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "Backend: {}", backend_label(report.backend));
    let _ = writeln!(
        out,
        "CPU: {} ({} physical / {} logical cores)",
        report.cpu.model, report.cpu.physical_cores, report.cpu.logical_cores
    );
    let _ = writeln!(
        out,
        "RAM: {} available / {} total",
        fmt_bytes(report.ram_available),
        fmt_bytes(report.ram_total)
    );

    render_gpus(report, &mut out);
    render_disks(&report.disk, &mut out);

    out
}

fn render_gpus(report: &HardwareReport, out: &mut String) {
    if report.gpus.is_empty() {
        let _ = writeln!(out, "GPU: none detected");
        if matches!(
            report.backend,
            Backend::Cuda | Backend::Metal | Backend::Rocm
        ) {
            let _ = writeln!(
                out,
                "  \u{26a0} this build expects a {} GPU but none was found — inference will fall back to CPU and be very slow",
                backend_label(report.backend)
            );
        }
        return;
    }

    for gpu in &report.gpus {
        let _ = writeln!(out, "GPU {}: {}", gpu.index, gpu.name);
        let _ = writeln!(
            out,
            "  VRAM: {} free / {} total",
            fmt_bytes(gpu.vram_free),
            fmt_bytes(gpu.vram_total)
        );
        if let Some((major, minor)) = gpu.compute_capability {
            let _ = writeln!(out, "  Compute capability: {major}.{minor}");
        }
        if let Some(driver) = &gpu.driver_version {
            let _ = writeln!(out, "  Driver: {driver}");
        }
        if gpu.unified_memory {
            let _ = writeln!(
                out,
                "  Unified memory: shared with system RAM and other apps"
            );
        }
    }
}

fn render_disks(disks: &[studio_core::hardware::DiskInfo], out: &mut String) {
    if disks.is_empty() {
        let _ = writeln!(
            out,
            "Disk: could not determine free space for the models directory"
        );
        return;
    }
    for disk in disks {
        let _ = writeln!(
            out,
            "Disk ({}): {} available / {} total",
            disk.mount_point,
            fmt_bytes(disk.available),
            fmt_bytes(disk.total)
        );
    }
}

fn backend_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "CUDA",
        Backend::Metal => "Metal",
        Backend::Rocm => "ROCm",
        Backend::Cpu => "CPU",
    }
}

// Precision loss above 2^52 bytes (4 EiB) is irrelevant for VRAM/RAM/disk
// sizes; this is display formatting only.
#[allow(clippy::cast_precision_loss)]
fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use studio_core::hardware::CpuInfo;

    use super::*;

    fn empty_report(backend: Backend) -> HardwareReport {
        HardwareReport {
            gpus: Vec::new(),
            cpu: CpuInfo {
                model: "test CPU".to_string(),
                physical_cores: 4,
                logical_cores: 8,
            },
            ram_total: 16 * 1024 * 1024 * 1024,
            ram_available: 8 * 1024 * 1024 * 1024,
            disk: Vec::new(),
            backend,
        }
    }

    #[test]
    fn warns_when_gpu_backend_finds_no_gpu() {
        let out = render(&empty_report(Backend::Cuda));
        assert!(out.contains("none was found"), "{out}");
    }

    #[test]
    fn no_gpu_warning_on_cpu_backend() {
        let out = render(&empty_report(Backend::Cpu));
        assert!(!out.contains("none was found"), "{out}");
    }

    #[test]
    fn fmt_bytes_picks_the_right_unit() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(2048), "2.0 KiB");
        assert_eq!(fmt_bytes(24 * 1024 * 1024 * 1024), "24.0 GiB");
    }
}
