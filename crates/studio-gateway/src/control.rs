//! The daemon's control API, per PLAN.md §3.1a/§3.2: launch/stop/list
//! children, plus the detach lease. `/v1/*` model multiplexing is M6 — this
//! is control-plane only.
//!
//! The detach lease (§3.1a): `GET /control/attach` is a websocket an
//! interactive client holds open for its whole session. Its connection
//! closing — by a graceful client shutdown *or* the client process getting
//! `SIGKILL`'d — looks identical from here: the socket read just returns.
//! When the last attached client goes away and nobody called `/detach`
//! first, a grace-period timer stops every child and shuts the daemon
//! down. A new attach before the timer fires cancels it (via the shutdown
//! epoch counter) and resets `detached` to `false`, so the next quit asks
//! again.

use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use studio_core::launch::LaunchSpec;
use studio_supervisor::{ChildId, ChildInfo, ChildState, Supervisor};
use tokio::sync::watch;

/// §3.1a: "Give a short grace period (~5s) so a TUI restart or a transient
/// reconnect does not kill a session."
pub const DETACH_GRACE_PERIOD: Duration = Duration::from_secs(5);

/// Loopback-only control port (§10.1's loopback-binding rule applies here
/// too, not just the future `/v1` gateway) — distinct from the gateway's
/// own default `:1234` (§3.2), since the control plane and the model
/// multiplexer are separate concerns that happen to share a process.
pub const DEFAULT_CONTROL_PORT: u16 = 41999;

pub struct Daemon {
    supervisor: Supervisor,
    detached: AtomicBool,
    attached_clients: AtomicUsize,
    shutdown_epoch: AtomicU64,
    shutdown_tx: watch::Sender<bool>,
    grace_period: Duration,
    /// §7.3 measurement samplers (`measure::spawn`) are detached
    /// background tasks on the same tokio runtime this process tears down
    /// on exit — without waiting for them here first, a launch stopped as
    /// part of "stop everything" would have its final measurement silently
    /// lost to the runtime shutdown racing the sampler's next ~1s tick.
    sampler_tasks: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl Daemon {
    #[must_use]
    pub fn new(supervisor: Supervisor) -> (Arc<Self>, watch::Receiver<bool>) {
        Self::with_grace_period(supervisor, DETACH_GRACE_PERIOD)
    }

    /// Same as `new`, with a configurable grace period — tests use a short
    /// one so the orphan-test suite doesn't need to wait out the real 5s.
    #[must_use]
    pub fn with_grace_period(
        supervisor: Supervisor,
        grace_period: Duration,
    ) -> (Arc<Self>, watch::Receiver<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let daemon = Arc::new(Daemon {
            supervisor,
            detached: AtomicBool::new(false),
            attached_clients: AtomicUsize::new(0),
            shutdown_epoch: AtomicU64::new(0),
            shutdown_tx,
            grace_period,
            sampler_tasks: std::sync::Mutex::new(Vec::new()),
        });
        (daemon, shutdown_rx)
    }

    #[must_use]
    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    #[must_use]
    pub fn is_detached(&self) -> bool {
        self.detached.load(Ordering::SeqCst)
    }

    #[must_use]
    pub fn attached_client_count(&self) -> usize {
        self.attached_clients.load(Ordering::SeqCst)
    }

    fn register_sampler(&self, handle: tokio::task::JoinHandle<()>) {
        self.sampler_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle);
    }

    /// Waits (up to a bounded timeout each) for every measurement sampler
    /// started since the last call to finish writing its record. Called
    /// before this process's tokio runtime — and every task still on it —
    /// gets torn down on a whole-daemon shutdown.
    async fn wait_for_samplers(&self) {
        let handles: Vec<_> = std::mem::take(
            &mut *self
                .sampler_tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for handle in handles {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
    }
}

pub fn router(daemon: Arc<Daemon>) -> Router {
    Router::new()
        .route("/control/launch", post(launch))
        .route("/control/stop/{id}", post(stop))
        .route("/control/list", get(list))
        .route("/control/status", get(status))
        .route("/control/detach", post(detach))
        .route("/control/shutdown", post(shutdown))
        .route("/control/attach", get(attach))
        .with_state(daemon)
}

#[derive(Deserialize)]
struct LaunchRequest {
    spec: LaunchSpec,
    label: String,
    /// §7.3: when the caller (the wizard) supplies both of these, the
    /// launch is measured — a background sampler (`measure::spawn`) tracks
    /// this child's peak VRAM and `/v1/stats` for its whole lifetime and
    /// records one `MeasurementRecord`. Absent for the raw `cranestudio
    /// launch`/`register` CLI paths, which just don't get measured.
    #[serde(default)]
    measurement_key: Option<String>,
    #[serde(default)]
    predicted_bytes: Option<u64>,
}

#[derive(Serialize)]
struct LaunchResponse {
    id: u64,
}

/// Builds the `cranestudio __serve` re-exec `SpawnRequest` and health-check
/// URL for a `LaunchSpec` — shared by the `/control/launch` handler and the
/// gateway's on-demand start (§3.2), so both spawn children identically.
///
/// # Errors
/// If `current_exe()` can't be resolved.
pub(crate) fn spawn_request_for(
    spec: &LaunchSpec,
) -> std::io::Result<(studio_supervisor::SpawnRequest, String)> {
    let program = std::env::current_exe()?;
    let mut args = vec!["__serve".to_string()];
    args.extend(spec.argv());
    let request = studio_supervisor::SpawnRequest {
        program,
        args,
        envs: spec.envp(),
    };
    let health_url = format!("http://127.0.0.1:{}/health", spec.port);
    Ok((request, health_url))
}

async fn launch(
    State(daemon): State<Arc<Daemon>>,
    Json(req): Json<LaunchRequest>,
) -> Result<Json<LaunchResponse>, (StatusCode, String)> {
    req.spec
        .validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let (request, health_url) = spawn_request_for(&req.spec)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Baselined before spawning — §7.3: "sample total-used delta against
    // the pre-spawn baseline," so this has to be read now, not from
    // inside the sampler task after the child may already be loading.
    let baseline_vram_used = studio_core::hardware::probe_gpus()
        .into_iter()
        .find(|g| g.index == req.spec.device)
        .map_or(0, |g| g.vram_total.saturating_sub(g.vram_free));

    let id = daemon
        .supervisor
        .launch(&request, Some(health_url), req.label)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if let (Some(key), Some(predicted_bytes)) = (req.measurement_key, req.predicted_bytes) {
        let handle = crate::measure::spawn(
            daemon.supervisor.clone(),
            id,
            req.spec,
            key,
            predicted_bytes,
            baseline_vram_used,
        );
        daemon.register_sampler(handle);
    }

    Ok(Json(LaunchResponse { id: id.0 }))
}

async fn stop(State(daemon): State<Arc<Daemon>>, Path(id): Path<u64>) -> StatusCode {
    if daemon.supervisor.stop(ChildId(id)).await {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

#[derive(Serialize)]
struct ChildSummary {
    info: ChildInfo,
    state: ChildState,
}

async fn list(State(daemon): State<Arc<Daemon>>) -> Json<Vec<ChildSummary>> {
    Json(
        daemon
            .supervisor
            .list()
            .into_iter()
            .map(|(info, state)| ChildSummary { info, state })
            .collect(),
    )
}

#[derive(Serialize)]
struct StatusResponse {
    detached: bool,
    attached_clients: usize,
    running_children: usize,
}

async fn status(State(daemon): State<Arc<Daemon>>) -> Json<StatusResponse> {
    Json(StatusResponse {
        detached: daemon.is_detached(),
        attached_clients: daemon.attached_client_count(),
        running_children: daemon.supervisor.list().len(),
    })
}

async fn detach(State(daemon): State<Arc<Daemon>>) -> StatusCode {
    daemon.detached.store(true, Ordering::SeqCst);
    StatusCode::OK
}

async fn shutdown(State(daemon): State<Arc<Daemon>>) -> StatusCode {
    daemon.supervisor.stop_all().await;
    daemon.wait_for_samplers().await;
    let _ = daemon.shutdown_tx.send(true);
    StatusCode::OK
}

async fn attach(State(daemon): State<Arc<Daemon>>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_attach(socket, daemon))
}

async fn handle_attach(mut socket: WebSocket, daemon: Arc<Daemon>) {
    daemon.attached_clients.fetch_add(1, Ordering::SeqCst);
    // §3.1a: "`detached` resets to `false` when a new interactive client
    // attaches, so the next quit asks again."
    daemon.detached.store(false, Ordering::SeqCst);
    daemon.shutdown_epoch.fetch_add(1, Ordering::SeqCst);

    // Hold the connection open. A graceful close and an abrupt kill of the
    // client both surface identically here: `recv` stops yielding `Some`.
    while let Some(Ok(message)) = socket.recv().await {
        if matches!(message, Message::Close(_)) {
            break;
        }
    }

    let remaining = daemon.attached_clients.fetch_sub(1, Ordering::SeqCst) - 1;
    if remaining == 0 && !daemon.is_detached() {
        let epoch_at_disconnect = daemon.shutdown_epoch.load(Ordering::SeqCst);
        let grace_period = daemon.grace_period;
        tokio::spawn(async move {
            tokio::time::sleep(grace_period).await;
            let still_current = daemon.shutdown_epoch.load(Ordering::SeqCst) == epoch_at_disconnect;
            let still_empty = daemon.attached_client_count() == 0;
            if still_current && still_empty && !daemon.is_detached() {
                daemon.supervisor.stop_all().await;
                daemon.wait_for_samplers().await;
                let _ = daemon.shutdown_tx.send(true);
            }
        });
    }
}

/// Reaps `__serve` processes left behind by a previous, crashed daemon run
/// (§3.1a), before the new `Supervisor` starts tracking anything. Returns
/// the pids it killed, for the startup log line.
#[must_use]
pub fn reap_stale_children(pidfile: &FsPath) -> Vec<u32> {
    studio_supervisor::reap_stale_children(pidfile)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use studio_supervisor::{ChildState, SpawnRequest};
    use tokio::net::TcpListener;

    use super::*;

    fn sleep_forever_request() -> SpawnRequest {
        SpawnRequest {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), "sleep 300".to_string()],
            envs: vec![],
        }
    }

    /// Starts the router on a real TCP listener and returns its base URL —
    /// a real network round trip, not an in-process `tower::Service` call,
    /// so a dropped websocket connection behaves exactly like a real
    /// client dying.
    async fn spawn_server(daemon: Arc<Daemon>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = router(daemon);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("127.0.0.1:{}", addr.port())
    }

    /// M5's own accept criterion, verified literally: attach a control
    /// client over a real websocket, launch a real child through the
    /// supervisor, drop the connection without a clean close (exactly what
    /// `kill -9` on the client process looks like from the server's side —
    /// the socket just stops), and assert the child is gone within the
    /// grace period.
    #[tokio::test]
    async fn kill_9_on_the_control_client_reaps_every_child_within_the_grace_period() {
        let (daemon, _shutdown_rx) =
            Daemon::with_grace_period(Supervisor::new(), Duration::from_millis(200));
        let addr = spawn_server(daemon.clone()).await;

        let id = daemon
            .supervisor()
            .launch(&sleep_forever_request(), None, "victim".to_string())
            .unwrap();
        assert!(
            daemon.supervisor().state(id).is_some(),
            "child should be tracked before attach"
        );

        let (ws_stream, _) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/control/attach"))
                .await
                .unwrap();
        // Give the server a moment to register the attach.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(daemon.attached_client_count(), 1);

        // Simulate `kill -9`: drop the stream without sending a close
        // frame. The underlying TCP connection just disappears.
        drop(ws_stream);

        // Wait past the grace period and assert the child was reaped.
        tokio::time::sleep(
            Duration::from_millis(200) + DETACH_GRACE_PERIOD.min(Duration::from_millis(500)),
        )
        .await;
        for _ in 0..20 {
            if daemon.supervisor().state(id).is_none() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            daemon.supervisor().state(id).is_none(),
            "child should have been stopped and forgotten"
        );
    }

    #[tokio::test]
    async fn detach_prevents_the_shutdown_on_disconnect() {
        let (daemon, _shutdown_rx) =
            Daemon::with_grace_period(Supervisor::new(), Duration::from_millis(150));
        let addr = spawn_server(daemon.clone()).await;

        let id = daemon
            .supervisor()
            .launch(&sleep_forever_request(), None, "kept-alive".to_string())
            .unwrap();

        let (ws_stream, _) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/control/attach"))
                .await
                .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Explicit detach, then the client goes away.
        let client = reqwest::Client::new();
        client
            .post(format!("http://{addr}/control/detach"))
            .send()
            .await
            .unwrap();
        drop(ws_stream);

        tokio::time::sleep(Duration::from_millis(150) + Duration::from_millis(300)).await;
        assert!(
            matches!(
                daemon.supervisor().state(id),
                Some(ChildState::Starting | ChildState::Healthy)
            ),
            "detached daemon must not stop its children on disconnect"
        );

        daemon.supervisor().stop(id).await;
    }

    #[tokio::test]
    async fn a_second_attach_before_the_grace_period_cancels_the_shutdown() {
        let (daemon, _shutdown_rx) =
            Daemon::with_grace_period(Supervisor::new(), Duration::from_millis(300));
        let addr = spawn_server(daemon.clone()).await;
        let id = daemon
            .supervisor()
            .launch(&sleep_forever_request(), None, "reconnecting".to_string())
            .unwrap();

        let (first, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/control/attach"))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(first);

        // Reconnect well within the 300ms grace period.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (_second, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/control/attach"))
            .await
            .unwrap();

        // Wait past when the *first* disconnect's timer would have fired.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            daemon.supervisor().state(id).is_some(),
            "reconnecting before the grace period elapses must cancel the shutdown"
        );

        daemon.supervisor().stop(id).await;
    }

    #[tokio::test]
    async fn status_and_list_endpoints_report_a_launched_child() {
        let (daemon, _shutdown_rx) =
            Daemon::with_grace_period(Supervisor::new(), Duration::from_secs(5));
        let addr = spawn_server(daemon.clone()).await;
        daemon
            .supervisor()
            .launch(&sleep_forever_request(), None, "visible".to_string())
            .unwrap();

        let client = reqwest::Client::new();
        let status: serde_json::Value = client
            .get(format!("http://{addr}/control/status"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["running_children"], 1);

        let list: serde_json::Value = client
            .get(format!("http://{addr}/control/list"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.as_array().unwrap().len(), 1);
        assert_eq!(list[0]["info"]["label"], "visible");
    }

    #[tokio::test]
    async fn shutdown_endpoint_stops_children_and_signals_the_main_loop() {
        let (daemon, mut shutdown_rx) =
            Daemon::with_grace_period(Supervisor::new(), Duration::from_secs(5));
        let addr = spawn_server(daemon.clone()).await;
        let id = daemon
            .supervisor()
            .launch(&sleep_forever_request(), None, "to-shutdown".to_string())
            .unwrap();

        let client = reqwest::Client::new();
        client
            .post(format!("http://{addr}/control/shutdown"))
            .send()
            .await
            .unwrap();

        shutdown_rx.changed().await.unwrap();
        assert!(*shutdown_rx.borrow());
        assert!(daemon.supervisor().state(id).is_none());
    }
}
