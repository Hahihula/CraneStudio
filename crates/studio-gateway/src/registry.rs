//! The gateway's model registry, per PLAN.md §3.2: which models it knows
//! about (whether or not a child is currently running for them), and which
//! ones are running right now. Registration is in-memory for M6 — M8's
//! profile persistence is the natural future source for populating this on
//! daemon startup, not something this module needs to know about.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use studio_core::launch::LaunchSpec;
use studio_supervisor::ChildId;

#[derive(Clone)]
struct RunningEntry {
    child_id: ChildId,
    port: u16,
    last_used: Instant,
}

#[derive(Clone)]
pub struct ModelRegistry {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    configured: HashMap<String, LaunchSpec>,
    running: HashMap<String, RunningEntry>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        ModelRegistry {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    pub fn register(&self, name: String, spec: LaunchSpec) {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .configured
            .insert(name, spec);
    }

    #[must_use]
    pub fn configured_names(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .configured
            .keys()
            .cloned()
            .collect()
    }

    #[must_use]
    pub fn spec_for(&self, name: &str) -> Option<LaunchSpec> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .configured
            .get(name)
            .cloned()
    }

    /// If `name` is already running, touches its LRU timestamp and returns
    /// its port. `None` means the caller needs to start it.
    #[must_use]
    pub fn touch_if_running(&self, name: &str) -> Option<u16> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = inner.running.get_mut(name)?;
        entry.last_used = Instant::now();
        Some(entry.port)
    }

    /// The id of the child currently serving `name`, if one is running.
    #[must_use]
    pub fn running_child_id(&self, name: &str) -> Option<ChildId> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .running
            .get(name)
            .map(|entry| entry.child_id)
    }

    pub fn mark_running(&self, name: String, child_id: ChildId, port: u16) {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .running
            .insert(
                name,
                RunningEntry {
                    child_id,
                    port,
                    last_used: Instant::now(),
                },
            );
    }

    pub fn forget_running(&self, name: &str) {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .running
            .remove(name);
    }

    #[must_use]
    pub fn running_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .running
            .len()
    }

    /// The running model that's gone longest without a request, excluding
    /// `except` (the one we're about to serve) — the LRU eviction
    /// candidate (§3.2).
    #[must_use]
    pub fn least_recently_used(&self, except: &str) -> Option<(String, ChildId)> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner
            .running
            .iter()
            .filter(|(name, _)| name.as_str() != except)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(name, entry)| (name.clone(), entry.child_id))
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use super::*;

    fn spec() -> LaunchSpec {
        LaunchSpec {
            model_path: "/models/x.gguf".to_string(),
            model_type: "qwen3_5".to_string(),
            model_name: None,
            port: 0,
            cpu: false,
            max_concurrent: 1,
            decode_tokens_per_seq: 16,
            format: None,
            quant: None,
            dtype: None,
            max_seq_len: 8192,
            gpu_memory_limit: None,
            text_only: false,
            kv_quant: None,
            prefill_chunk: None,
            device: 0,
        }
    }

    #[test]
    fn configured_models_are_listed_whether_or_not_running() {
        let registry = ModelRegistry::new();
        registry.register("a".to_string(), spec());
        registry.register("b".to_string(), spec());
        let mut names = registry.configured_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(registry.running_count(), 0);
    }

    #[test]
    fn touch_if_running_updates_lru_order() {
        let registry = ModelRegistry::new();
        registry.mark_running("a".to_string(), ChildId(1), 100);
        sleep(Duration::from_millis(5));
        registry.mark_running("b".to_string(), ChildId(2), 200);

        // "a" is older right now.
        assert_eq!(registry.least_recently_used("").unwrap().0, "a");

        // Touching "a" makes "b" the LRU one instead.
        sleep(Duration::from_millis(5));
        let _ = registry.touch_if_running("a");
        assert_eq!(registry.least_recently_used("").unwrap().0, "b");
    }

    #[test]
    fn least_recently_used_excludes_the_model_about_to_be_served() {
        let registry = ModelRegistry::new();
        registry.mark_running("a".to_string(), ChildId(1), 100);
        assert!(registry.least_recently_used("a").is_none());
    }
}
