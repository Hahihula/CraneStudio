//! `Supervisor`: the registry of running children, per PLAN.md §3.3. Ties
//! together spawning, health polling, log capture, and exit classification;
//! also the reference point for §3.1a's stale-pidfile reaping.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::AsyncBufReadExt;

use crate::classify::{ExitClassification, ExitContext, classify};
use crate::log_ring::LogRing;
use crate::spawn::{SpawnRequest, spawn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ChildId(pub u64);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChildInfo {
    pub id: ChildId,
    pub pid: u32,
    pub label: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ChildState {
    Starting,
    Healthy,
    Exited {
        classification: ExitClassification,
        exit_code: Option<i32>,
    },
}

struct ChildEntry {
    info: ChildInfo,
    state: Arc<Mutex<ChildState>>,
    log: LogRing,
    stop_requested: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct Supervisor {
    children: Arc<Mutex<HashMap<ChildId, ChildEntry>>>,
    next_id: Arc<AtomicU64>,
    pidfile: Option<PathBuf>,
}

const LOG_CAPACITY: usize = 500;
const STOP_GRACE: Duration = Duration::from_secs(3);

impl Supervisor {
    #[must_use]
    pub fn new() -> Self {
        Supervisor {
            children: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
            pidfile: None,
        }
    }

    /// Every child's pid is written here (one per line) after each change,
    /// so a future daemon startup can reap anything left behind by a crash
    /// (§3.1a). Best-effort — a write failure never fails the launch.
    #[must_use]
    pub fn with_pidfile(mut self, path: PathBuf) -> Self {
        self.pidfile = Some(path);
        self
    }

    /// Spawns one child and starts tracking it. `health_url` is `None` for
    /// health-check-less spawns (used in tests that don't run a real
    /// crane-serve server).
    ///
    /// # Errors
    /// If the process fails to spawn.
    pub fn launch(
        &self,
        request: &SpawnRequest,
        health_url: Option<String>,
        label: String,
    ) -> std::io::Result<ChildId> {
        let mut child = spawn(request)?;
        let pid = child.id().unwrap_or(0);
        let id = ChildId(self.next_id.fetch_add(1, Ordering::SeqCst));

        let state = Arc::new(Mutex::new(ChildState::Starting));
        let log = LogRing::new(LOG_CAPACITY);
        let stop_requested = Arc::new(AtomicBool::new(false));

        if let Some(stderr) = child.stderr.take() {
            let log = log.clone();
            tokio::spawn(async move {
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    log.push_line(line);
                }
            });
        }

        let health_ok = Arc::new(AtomicBool::new(false));
        if let Some(url) = health_url {
            let health_ok = health_ok.clone();
            let state = state.clone();
            tokio::spawn(async move {
                if crate::health::wait_for_health(&url, Duration::from_secs(300)).await {
                    health_ok.store(true, Ordering::SeqCst);
                    let mut guard = state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if matches!(*guard, ChildState::Starting) {
                        *guard = ChildState::Healthy;
                    }
                }
            });
        }

        let state_for_wait = state.clone();
        let log_for_wait = log.clone();
        let stop_for_wait = stop_requested.clone();
        tokio::spawn(async move {
            let result = child.wait().await;
            let (exit_code, signal) = exit_status_parts(&result);
            let ctx = ExitContext {
                exit_code,
                signal,
                health_ok_observed: health_ok.load(Ordering::SeqCst),
                stderr_tail: &log_for_wait.tail(),
                requested_stop: stop_for_wait.load(Ordering::SeqCst),
            };
            let classification = classify(&ctx);
            *state_for_wait
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = ChildState::Exited {
                classification,
                exit_code,
            };
        });

        self.children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                id,
                ChildEntry {
                    info: ChildInfo { id, pid, label },
                    state,
                    log,
                    stop_requested,
                },
            );
        self.write_pidfile();
        Ok(id)
    }

    #[must_use]
    pub fn state(&self, id: ChildId) -> Option<ChildState> {
        let children = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        children.get(&id).map(|entry| {
            entry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        })
    }

    #[must_use]
    pub fn log_tail(&self, id: ChildId) -> Option<String> {
        let children = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        children.get(&id).map(|entry| entry.log.tail())
    }

    #[must_use]
    pub fn list(&self) -> Vec<(ChildInfo, ChildState)> {
        let children = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        children
            .values()
            .map(|entry| {
                (
                    entry.info.clone(),
                    entry
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                )
            })
            .collect()
    }

    /// Sends `SIGTERM`, waits `STOP_GRACE` for a clean exit, escalates to
    /// `SIGKILL` if it's still running — "never leave a process behind"
    /// (§15.13) trumps a clean shutdown.
    pub async fn stop(&self, id: ChildId) -> bool {
        let Some((pid, stop_requested)) = self.pid_and_stop_flag(id) else {
            return false;
        };
        stop_requested.store(true, Ordering::SeqCst);
        send_signal(pid, Signal::Term);

        let deadline = tokio::time::Instant::now() + STOP_GRACE;
        while tokio::time::Instant::now() < deadline {
            if matches!(self.state(id), Some(ChildState::Exited { .. })) {
                self.forget(id);
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        send_signal(pid, Signal::Kill);
        self.forget(id);
        true
    }

    /// Stops every tracked child concurrently — used for the detach
    /// lease's "last client disconnected" shutdown (§3.1a) and for daemon
    /// shutdown generally.
    pub async fn stop_all(&self) {
        let ids: Vec<ChildId> = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .copied()
            .collect();
        let stops = ids.into_iter().map(|id| self.stop(id));
        futures_util::future::join_all(stops).await;
    }

    fn pid_and_stop_flag(&self, id: ChildId) -> Option<(u32, Arc<AtomicBool>)> {
        let children = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        children
            .get(&id)
            .map(|entry| (entry.info.pid, entry.stop_requested.clone()))
    }

    fn forget(&self, id: ChildId) {
        self.children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
        self.write_pidfile();
    }

    fn write_pidfile(&self) {
        let Some(path) = &self.pidfile else { return };
        let children = self
            .children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let contents = children
            .values()
            .map(|entry| entry.info.pid.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let _ = std::fs::write(path, contents);
    }
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
enum Signal {
    Term,
    Kill,
}

#[cfg(unix)]
fn send_signal(pid: u32, signal: Signal) {
    let sig = match signal {
        Signal::Term => libc::SIGTERM,
        Signal::Kill => libc::SIGKILL,
    };
    // SAFETY: kill(2) with a plain pid and signal number has no memory
    // safety implications; a stale/exited pid just yields ESRCH, ignored.
    unsafe {
        libc::kill(pid.cast_signed(), sig);
    }
}

/// Windows has nothing to send that means "please exit" to another console
/// process: `GenerateConsoleCtrlEvent` only reaches processes sharing our own
/// console (and would then hit the daemon itself), and `WM_CLOSE` only reaches
/// windowed apps. So both requests terminate the child outright — which is safe
/// here precisely because a crane-serve child holds nothing unsaved: it is a
/// stateless inference server whose only durable state is the model file it
/// read. `/T` takes any grandchildren with it, and without `/F` a busy
/// inference process would ignore the request entirely.
///
/// The alternative — keeping the `tokio::process::Child` handle around to call
/// `kill()` on — doesn't fit: the handle is owned by the task awaiting the
/// child's exit, and this registry deliberately tracks pids so that a *previous*
/// daemon's children can be reaped too (`reap_stale_children`).
#[cfg(windows)]
fn send_signal(pid: u32, signal: Signal) {
    let _ = signal;
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(any(unix, windows)))]
fn send_signal(_pid: u32, _signal: Signal) {}

fn exit_status_parts(
    result: &std::io::Result<std::process::ExitStatus>,
) -> (Option<i32>, Option<i32>) {
    match result {
        Ok(status) => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                (status.code(), status.signal())
            }
            #[cfg(not(unix))]
            {
                (status.code(), None)
            }
        }
        Err(_) => (None, None),
    }
}

/// Scans `pidfile` for pids left over from a previous, crashed daemon
/// (§3.1a) and kills anything still alive. Returns the pids it reaped.
/// Linux-only for now: a pid is only ever killed after confirming
/// `/proc/<pid>/cmdline` actually names our own `__serve` re-exec, so a
/// reused pid belonging to an unrelated process is never touched.
#[cfg(target_os = "linux")]
#[must_use]
pub fn reap_stale_children(pidfile: &std::path::Path) -> Vec<u32> {
    let Ok(contents) = std::fs::read_to_string(pidfile) else {
        return Vec::new();
    };
    let mut reaped = Vec::new();
    for line in contents.lines() {
        let Ok(pid) = line.trim().parse::<u32>() else {
            continue;
        };
        let cmdline_path = format!("/proc/{pid}/cmdline");
        let Ok(cmdline) = std::fs::read_to_string(&cmdline_path) else {
            continue;
        };
        if cmdline.contains("__serve") {
            send_signal(pid, Signal::Kill);
            reaped.push(pid);
        }
    }
    let _ = std::fs::remove_file(pidfile);
    reaped
}

#[cfg(not(any(target_os = "linux", windows)))]
#[must_use]
pub fn reap_stale_children(_pidfile: &std::path::Path) -> Vec<u32> {
    Vec::new()
}

/// The Windows counterpart, with the same guard the Linux version applies for
/// the same reason: Windows recycles pids eagerly, so a pid from a previous run
/// is only killed after confirming it still belongs to *our* executable running
/// the `__serve` re-exec. Windows exposes a process's command line through WMI
/// rather than a file, which is what the PowerShell call reads; if that lookup
/// fails for any reason the pid is left alone, because killing the wrong process
/// is far worse than leaving a stale one for the user to notice.
#[cfg(windows)]
#[must_use]
pub fn reap_stale_children(pidfile: &std::path::Path) -> Vec<u32> {
    let Ok(contents) = std::fs::read_to_string(pidfile) else {
        return Vec::new();
    };
    let image = std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_lowercase())
        })
        .unwrap_or_else(|| "cranestudio.exe".to_string());

    let mut reaped = Vec::new();
    for line in contents.lines() {
        let Ok(pid) = line.trim().parse::<u32>() else {
            continue;
        };
        let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &format!("(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"),
            ])
            .output()
        else {
            continue;
        };
        let command_line = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if command_line.contains(&image) && command_line.contains("__serve") {
            send_signal(pid, Signal::Kill);
            reaped.push(pid);
        }
    }
    let _ = std::fs::remove_file(pidfile);
    reaped
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    fn shell_request(script: &str) -> SpawnRequest {
        SpawnRequest {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), script.to_string()],
            envs: vec![],
        }
    }

    #[tokio::test]
    async fn tracks_a_child_through_to_a_clean_exit() {
        let supervisor = Supervisor::new();
        let id = supervisor
            .launch(&shell_request("exit 0"), None, "test".to_string())
            .unwrap();

        for _ in 0..50 {
            if matches!(supervisor.state(id), Some(ChildState::Exited { .. })) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        match supervisor.state(id) {
            Some(ChildState::Exited {
                classification,
                exit_code,
            }) => {
                assert_eq!(exit_code, Some(0));
                assert_eq!(classification, ExitClassification::CleanEarlyExit);
            }
            other => panic!("expected Exited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stop_terminates_the_child_and_marks_it_stopped() {
        let supervisor = Supervisor::new();
        let id = supervisor
            .launch(&shell_request("sleep 300"), None, "test".to_string())
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(supervisor.stop(id).await);
        assert!(
            supervisor.state(id).is_none(),
            "stopped child should be forgotten"
        );
    }

    /// M5's accept criterion: killing a child does not kill the daemon —
    /// i.e. the supervisor keeps running and correctly observes the exit.
    #[tokio::test]
    async fn killing_a_child_does_not_affect_the_supervisor() {
        let supervisor = Supervisor::new();
        let id = supervisor
            .launch(&shell_request("sleep 300"), None, "victim".to_string())
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let pid = supervisor
            .list()
            .into_iter()
            .find(|(info, _)| info.id == id)
            .unwrap()
            .0
            .pid;
        send_signal(pid, Signal::Kill);

        for _ in 0..50 {
            if matches!(supervisor.state(id), Some(ChildState::Exited { .. })) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(matches!(
            supervisor.state(id),
            Some(ChildState::Exited { .. })
        ));
        // The supervisor itself is still fully usable — launch another.
        let id2 = supervisor
            .launch(&shell_request("exit 0"), None, "second".to_string())
            .unwrap();
        assert!(id2 != id);
    }

    #[test]
    fn reap_stale_children_ignores_unrelated_processes() {
        let dir = TempDir::new().unwrap();
        let pidfile = dir.path().join("children.pids");
        // Our own test process's pid — very much alive, but its cmdline
        // will never contain "__serve", so it must not be touched.
        std::fs::write(&pidfile, std::process::id().to_string()).unwrap();

        let reaped = reap_stale_children(&pidfile);
        assert!(reaped.is_empty());
        assert!(!pidfile.exists(), "pidfile should still be cleared");
    }
}
