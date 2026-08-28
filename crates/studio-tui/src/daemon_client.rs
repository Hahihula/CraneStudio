//! Talks to the daemon's control (§3.1a) and gateway (§3.2) APIs, and
//! spawns the daemon itself if one isn't already running — this is what
//! lets the TUI "just work" with no separate `cranestudio daemon` step
//! (§1's user story: "no install, no Python, no toolchain").

use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use studio_core::launch::LaunchSpec;
use tokio_tungstenite::WebSocketStream;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Deserialize)]
pub struct ChildInfo {
    pub id: u64,
    pub pid: u32,
    pub label: String,
}

/// Mirrors `studio_supervisor::ChildState`'s serde shape: unit variants as
/// bare strings, the struct variant as `{"Exited": {...}}}`.
#[derive(Debug, Clone)]
pub enum ChildState {
    Starting,
    Healthy,
    Exited {
        classification: String,
        exit_code: Option<i64>,
    },
    Unknown,
}

impl ChildState {
    #[must_use]
    pub fn is_running(&self) -> bool {
        matches!(self, ChildState::Starting | ChildState::Healthy)
    }
}

impl<'de> Deserialize<'de> for ChildState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(match &value {
            serde_json::Value::String(s) if s == "Starting" => ChildState::Starting,
            serde_json::Value::String(s) if s == "Healthy" => ChildState::Healthy,
            serde_json::Value::Object(map) => {
                map.get("Exited")
                    .map_or(ChildState::Unknown, |exited| ChildState::Exited {
                        classification: exited
                            .get("classification")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown")
                            .to_string(),
                        exit_code: exited.get("exit_code").and_then(serde_json::Value::as_i64),
                    })
            }
            _ => ChildState::Unknown,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChildSummary {
    pub info: ChildInfo,
    pub state: ChildState,
}

#[derive(Debug, Deserialize)]
pub struct DaemonStatus {
    pub detached: bool,
    pub attached_clients: usize,
    pub running_children: usize,
}

pub struct DaemonClient {
    http: reqwest::Client,
    control_base: String,
    gateway_base: String,
    control_port: u16,
    gateway_port: u16,
    attach_socket:
        Option<WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>,
}

impl DaemonClient {
    /// Connects to an already-running daemon, or spawns one and waits for
    /// it to come up — the user never has to know the daemon exists.
    ///
    /// `preferred_control_port`/`preferred_gateway_port` are only a
    /// starting preference: the daemon falls forward to the next free port
    /// on a bind conflict (e.g. LM Studio also defaults to gateway `:1234`)
    /// rather than hard-failing, and reports back what it actually bound —
    /// see `studio_core::endpoints`.
    ///
    /// # Errors
    /// If the daemon can't be reached even after trying to spawn one.
    pub async fn connect_or_spawn(
        preferred_control_port: u16,
        preferred_gateway_port: u16,
    ) -> anyhow::Result<Self> {
        let http = reqwest::Client::new();

        let known_control_port =
            studio_core::endpoints::resolve_control_port(preferred_control_port);
        let known_gateway_port =
            studio_core::endpoints::resolve_gateway_port(preferred_gateway_port);
        let known_control_base = format!("http://127.0.0.1:{known_control_port}");

        let (control_port, gateway_port) = if http
            .get(format!("{known_control_base}/control/status"))
            .timeout(Duration::from_secs(1))
            .send()
            .await
            .is_ok()
        {
            (known_control_port, known_gateway_port)
        } else {
            spawn_daemon(preferred_control_port, preferred_gateway_port)?;
            wait_for_daemon(&http).await?
        };

        Ok(DaemonClient {
            http,
            control_base: format!("http://127.0.0.1:{control_port}"),
            gateway_base: format!("http://127.0.0.1:{gateway_port}"),
            control_port,
            gateway_port,
            attach_socket: None,
        })
    }

    /// Opens the interactive-control-client websocket (§3.1a) — holding
    /// this open is what the detach lease uses to decide whether anyone's
    /// still watching.
    ///
    /// # Errors
    /// If the websocket handshake fails.
    pub async fn attach(&mut self) -> anyhow::Result<()> {
        let url = self.control_base.replacen("http://", "ws://", 1) + "/control/attach";
        let (stream, _) = tokio_tungstenite::connect_async(url).await?;
        self.attach_socket = Some(stream);
        Ok(())
    }

    /// Closes the attach connection, if open — a clean disconnect looks
    /// the same to the daemon as an abrupt one (§3.1a), so this alone is
    /// enough to trigger (or not, if detached first) the grace-period
    /// shutdown.
    pub async fn detach_connection(&mut self) {
        if let Some(mut socket) = self.attach_socket.take() {
            let _ = socket.close(None).await;
        }
    }

    /// # Errors
    /// If the request fails.
    pub async fn status(&self) -> anyhow::Result<DaemonStatus> {
        Ok(self
            .http
            .get(format!("{}/control/status", self.control_base))
            .send()
            .await?
            .json()
            .await?)
    }

    /// # Errors
    /// If the request fails.
    pub async fn list(&self) -> anyhow::Result<Vec<ChildSummary>> {
        Ok(self
            .http
            .get(format!("{}/control/list", self.control_base))
            .send()
            .await?
            .json()
            .await?)
    }

    /// # Errors
    /// If the request fails.
    pub async fn any_running(&self) -> anyhow::Result<bool> {
        Ok(self.list().await?.iter().any(|c| c.state.is_running()))
    }

    /// §3.1a: "Choosing 'keep serving' sends an explicit `detach` command."
    ///
    /// # Errors
    /// If the request fails.
    pub async fn keep_serving(&self) -> anyhow::Result<()> {
        self.http
            .post(format!("{}/control/detach", self.control_base))
            .send()
            .await?;
        Ok(())
    }

    /// Stops every model and shuts the daemon down right now — used for
    /// the quit prompt's "stop everything" choice, which should be
    /// immediate rather than waiting out the disconnect grace period.
    ///
    /// # Errors
    /// If the request fails.
    pub async fn stop_everything(&self) -> anyhow::Result<()> {
        self.http
            .post(format!("{}/control/shutdown", self.control_base))
            .send()
            .await?;
        Ok(())
    }

    /// Registers with the gateway (§3.2, `/v1/models` aggregation) and
    /// starts it immediately (via `/control/launch`) so its status is
    /// visible right away rather than waiting for a client's first
    /// request.
    ///
    /// # Errors
    /// If either request fails, or the spec fails validation.
    pub async fn register_and_launch(&self, name: &str, spec: &LaunchSpec) -> anyhow::Result<u64> {
        let register = self
            .http
            .post(format!("{}/register", self.gateway_base))
            .json(&json!({"name": name, "spec": spec}))
            .send()
            .await?;
        if !register.status().is_success() {
            anyhow::bail!(
                "register failed: {}",
                register.text().await.unwrap_or_default()
            );
        }

        let launch = self
            .http
            .post(format!("{}/control/launch", self.control_base))
            .json(&json!({"spec": spec, "label": name}))
            .send()
            .await?;
        if !launch.status().is_success() {
            anyhow::bail!("launch failed: {}", launch.text().await.unwrap_or_default());
        }
        let body: serde_json::Value = launch.json().await?;
        Ok(body["id"].as_u64().unwrap_or(0))
    }

    #[must_use]
    pub fn control_port(&self) -> u16 {
        self.control_port
    }

    #[must_use]
    pub fn gateway_port(&self) -> u16 {
        self.gateway_port
    }

    /// A client that talks to nothing, for rendering tests — every screen takes
    /// the whole `App`, so previewing one needs an `App`, and an `App` needs a
    /// `DaemonClient` even when no daemon is involved.
    #[cfg(test)]
    pub(crate) fn offline(control_port: u16, gateway_port: u16) -> Self {
        DaemonClient {
            http: reqwest::Client::new(),
            control_base: format!("http://127.0.0.1:{control_port}"),
            gateway_base: format!("http://127.0.0.1:{gateway_port}"),
            control_port,
            gateway_port,
            attach_socket: None,
        }
    }
}

fn spawn_daemon(control_port: u16, gateway_port: u16) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let log_dir = studio_core::paths::data_dir().join("logs");
    std::fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join("daemon.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let log_file_err = log_file.try_clone()?;

    std::process::Command::new(exe)
        .arg("daemon")
        .env("CRANESTUDIO_CONTROL_PORT", control_port.to_string())
        .env("CRANESTUDIO_GATEWAY_PORT", gateway_port.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file_err))
        .spawn()?;
    Ok(())
}

/// Polls the endpoints file rather than a fixed port — the daemon we just
/// spawned may have fallen forward to a different port than requested if
/// its preferred one was taken, so the port to actually check for isn't
/// known until the daemon reports it.
async fn wait_for_daemon(http: &reqwest::Client) -> anyhow::Result<(u16, u16)> {
    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        if let Some(endpoints) = studio_core::endpoints::load() {
            let control_base = format!("http://127.0.0.1:{}", endpoints.control_port);
            if http
                .get(format!("{control_base}/control/status"))
                .timeout(Duration::from_secs(1))
                .send()
                .await
                .is_ok()
            {
                return Ok((endpoints.control_port, endpoints.gateway_port));
            }
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    anyhow::bail!(
        "daemon did not come up in time — check {}",
        studio_core::paths::data_dir()
            .join("logs/daemon.log")
            .display()
    )
}
