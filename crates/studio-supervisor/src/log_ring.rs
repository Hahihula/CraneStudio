//! Bounded capture of a child's stderr, per PLAN.md §3.3/§7.4 — the
//! supervisor "captures the child's stderr into a ring buffer" for exit
//! classification and for `cranestudio`'s log tail pane (§4.5).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Cheap to clone and share between the reader task and whoever wants to
/// inspect the tail (classification, a future log-tail UI).
#[derive(Clone)]
pub struct LogRing {
    inner: Arc<Mutex<VecDeque<String>>>,
    capacity: usize,
}

impl LogRing {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        LogRing {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    pub fn push_line(&self, line: String) {
        let Ok(mut buf) = self.inner.lock() else {
            return;
        };
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(line);
    }

    #[must_use]
    pub fn tail(&self) -> String {
        let Ok(buf) = self.inner.lock() else {
            return String::new();
        };
        buf.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        let Ok(buf) = self.inner.lock() else {
            return Vec::new();
        };
        buf.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_lines_past_capacity() {
        let ring = LogRing::new(3);
        for i in 0..5 {
            ring.push_line(format!("line {i}"));
        }
        assert_eq!(ring.lines(), vec!["line 2", "line 3", "line 4"]);
    }

    #[test]
    fn tail_joins_with_newlines() {
        let ring = LogRing::new(10);
        ring.push_line("a".to_string());
        ring.push_line("b".to_string());
        assert_eq!(ring.tail(), "a\nb");
    }
}
