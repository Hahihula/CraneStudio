//! Shared plain-text formatting helpers for the print-and-exit screens
//! (`doctor`, `catalog`) ahead of the real ratatui UI in M7.

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn bytes(bytes: u64) -> String {
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
    use super::*;

    #[test]
    fn picks_the_right_unit() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2.0 KiB");
        assert_eq!(bytes(24 * 1024 * 1024 * 1024), "24.0 GiB");
    }
}
