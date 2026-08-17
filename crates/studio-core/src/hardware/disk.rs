use std::path::Path;

use sysinfo::Disks;

#[derive(Debug, Clone)]
pub struct DiskInfo {
    pub mount_point: String,
    pub total: u64,
    pub available: u64,
}

/// Reports the filesystem that contains `target` — the models directory —
/// rather than every disk on the machine, per PLAN.md §6. Picks the disk
/// whose mount point is the longest matching prefix of `target`, the same
/// resolution `df` uses.
pub(super) fn probe(target: &Path) -> Vec<DiskInfo> {
    find(target).into_iter().collect()
}

/// Same lookup, reused by the download manager's disk-space precheck (§9)
/// so both agree on which filesystem a given path lives on.
pub(crate) fn find(target: &Path) -> Option<DiskInfo> {
    let disks = Disks::new_with_refreshed_list();
    let target = target.to_string_lossy();

    disks
        .list()
        .iter()
        .filter(|disk| target.starts_with(disk.mount_point().to_string_lossy().as_ref()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| DiskInfo {
            mount_point: disk.mount_point().to_string_lossy().into_owned(),
            total: disk.total_space(),
            available: disk.available_space(),
        })
}
