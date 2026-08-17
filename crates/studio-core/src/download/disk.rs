//! Disk-space precheck, per PLAN.md §9: "refuse to start if free space <
//! size × 1.1, with a clear message. Do not discover this at 94%."

use std::path::Path;

const HEADROOM_FACTOR: f64 = 1.1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsufficientSpace {
    pub needed_bytes: u64,
    pub free_bytes: u64,
    pub mount_point: String,
}

impl std::fmt::Display for InsufficientSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "not enough free space on {} — need {} (with headroom), only {} free",
            self.mount_point,
            crate::download::fmt_bytes(self.needed_bytes),
            crate::download::fmt_bytes(self.free_bytes)
        )
    }
}

impl std::error::Error for InsufficientSpace {}

/// `dest_dir` need not exist yet — checked the same way hardware probing
/// checks the models directory (§6): by filesystem mount-point prefix, not
/// by `stat`.
///
/// # Errors
/// `InsufficientSpace` if `dest_dir`'s filesystem doesn't have at least
/// `total_bytes_needed * 1.1` free, or if no matching filesystem could be
/// found at all (treated as "can't confirm there's room", not "assume
/// there is").
pub fn check(dest_dir: &Path, total_bytes_needed: u64) -> Result<(), InsufficientSpace> {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let needed_with_headroom = (total_bytes_needed as f64 * HEADROOM_FACTOR) as u64;

    let disk = crate::hardware::disk::find(dest_dir);
    let (free_bytes, mount_point) = match &disk {
        Some(disk) => (disk.available, disk.mount_point.clone()),
        None => (0, dest_dir.display().to_string()),
    };

    if free_bytes < needed_with_headroom {
        return Err(InsufficientSpace {
            needed_bytes: needed_with_headroom,
            free_bytes,
            mount_point,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_when_free_space_is_short() {
        // Real filesystem, absurd size requirement — guaranteed to be short
        // on any machine.
        let err = check(std::path::Path::new("/"), u64::MAX / 2).unwrap_err();
        assert!(err.free_bytes < err.needed_bytes);
    }

    #[test]
    fn allows_a_trivially_small_request() {
        assert!(check(std::path::Path::new("/"), 1).is_ok());
    }
}
