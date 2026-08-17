use sysinfo::System;

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
}

pub(super) fn probe(sys: &System) -> CpuInfo {
    let model = sys
        .cpus()
        .first()
        .map_or_else(|| "unknown CPU".to_string(), |cpu| cpu.brand().to_string());
    let logical_cores = sys.cpus().len();
    let physical_cores = System::physical_core_count().unwrap_or(logical_cores);

    CpuInfo {
        model,
        physical_cores,
        logical_cores,
    }
}
