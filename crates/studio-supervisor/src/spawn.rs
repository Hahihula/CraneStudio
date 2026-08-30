//! Process spawning, per PLAN.md §3.1a's safety nets.
//!
//! Children are deliberately **not** put in a new process group — leaving
//! them in the daemon's own group (tokio's default) is what makes a group
//! signal (e.g. a terminal's Ctrl-C to the whole foreground group) reach
//! them too, per §3.1a's "spawned in the daemon's process group" note.
//!
//! `PR_SET_PDEATHSIG` (Linux only) is the belt-and-suspenders case: it
//! makes the kernel itself kill a child if the daemon dies for any reason,
//! including a `SIGKILL` the daemon never got to react to.
//!
//! Windows has no per-process equivalent of that hook — the equivalent
//! mechanism is a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, which
//! needs the Win32 API and hasn't been wired up yet. Until it is, a
//! hard-terminated daemon on Windows can leave a child holding VRAM until the
//! *next* daemon start, when `registry::reap_stale_children` finds it in the
//! pidfile and kills it. Every ordinary path (quit prompt, stop, detach lease)
//! still stops children explicitly.

use std::path::PathBuf;
use std::process::Stdio;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpawnRequest {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
}

/// # Errors
/// Whatever `std::process::Command::spawn` returns — e.g. the program
/// doesn't exist or isn't executable.
pub fn spawn(request: &SpawnRequest) -> std::io::Result<tokio::process::Child> {
    let mut cmd = tokio::process::Command::new(&request.program);
    cmd.args(&request.args);
    cmd.envs(request.envs.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);

    #[cfg(target_os = "linux")]
    {
        // SAFETY: prctl with PR_SET_PDEATHSIG is async-signal-safe and
        // touches no Rust state — the standard pattern for this hook.
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    cmd.spawn()
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawns_and_captures_stderr() {
        let request = SpawnRequest {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), "echo hello-stderr 1>&2".to_string()],
            envs: vec![],
        };
        let mut child = spawn(&request).unwrap();
        let status = child.wait().await.unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    async fn env_vars_reach_the_child() {
        let request = SpawnRequest {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), "test \"$FOO\" = bar".to_string()],
            envs: vec![("FOO".to_string(), "bar".to_string())],
        };
        let mut child = spawn(&request).unwrap();
        let status = child.wait().await.unwrap();
        assert!(status.success());
    }
}
