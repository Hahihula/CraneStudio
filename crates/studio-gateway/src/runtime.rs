//! Binding the gateway listener, and rebinding it live when the TUI flips
//! LAN exposure on/off (§10.1) — the control API and every running model
//! child are completely unaffected by a gateway rebind; only this one
//! listener is torn down and reopened.

use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::state::GatewayState;

pub const MAX_PORT_ATTEMPTS: u16 = 20;

/// Binds `preferred`, or the next free port after it (up to `max_tries`
/// attempts) if that one's already taken — PLAN.md §7.4: "port in use →
/// retry on a different port automatically," applied to the daemon's own
/// listening ports rather than hard-failing over a collision with something
/// unrelated (LM Studio's default is also `:1234`).
///
/// # Errors
/// If every port from `preferred` up to `preferred + max_tries` is taken.
pub async fn bind_with_fallback(
    host: &str,
    preferred: u16,
    max_tries: u16,
) -> anyhow::Result<(TcpListener, u16)> {
    for offset in 0..max_tries {
        let port = preferred.saturating_add(offset);
        if let Ok(listener) = TcpListener::bind((host, port)).await {
            // Not just `port`: a `preferred` of `0` (as tests use, to get
            // an OS-assigned ephemeral port) means the OS chose a port
            // other than the literal `0` requested.
            let bound_port = listener.local_addr().map_or(port, |addr| addr.port());
            return Ok((listener, bound_port));
        }
    }
    anyhow::bail!("could not find a free port starting from {preferred} after {max_tries} attempts")
}

/// A fresh bearer token for LAN exposure (§10.1) — 32 alphanumeric
/// characters, plenty for a value that's typed into a header once per
/// client setup and otherwise just compared byte-for-byte.
#[must_use]
pub fn generate_auth_token() -> String {
    use rand::Rng;
    rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

/// Serves the gateway, rebinding whenever `bind_rx` reports a new desired
/// host — e.g. the TUI toggling "public access" via
/// `/control/gateway/public`, which calls `Daemon::set_gateway_public` and
/// that in turn sends the new host through this channel. Each (re)bind
/// records what actually got bound in `state.set_bound` and re-saves
/// `studio_core::endpoints`, so status queries and other `cranestudio`
/// processes never see a stale host/port.
///
/// # Errors
/// If every port from `preferred_port` up to `preferred_port +
/// MAX_PORT_ATTEMPTS` is taken, or the listener otherwise fails.
pub async fn serve_with_rebind(
    app: Router,
    state: Arc<GatewayState>,
    control_port: u16,
    preferred_port: u16,
    mut bind_rx: watch::Receiver<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    loop {
        let host = bind_rx.borrow_and_update().clone();
        let (listener, port) =
            bind_with_fallback(&host, preferred_port, MAX_PORT_ATTEMPTS).await?;
        println!("gateway (/v1/*) listening on {host}:{port}");
        state.set_bound(host, port);
        studio_core::endpoints::save(studio_core::endpoints::Endpoints {
            control_port,
            gateway_port: port,
        });

        let mut rebind_rx = bind_rx.clone();
        let mut shutdown_rx_inner = shutdown_rx.clone();
        axum::serve(listener, app.clone())
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = shutdown_rx_inner.changed() => {}
                    _ = rebind_rx.changed() => {}
                }
            })
            .await?;

        if *shutdown_rx.borrow_and_update() {
            return Ok(());
        }
        // Otherwise `bind_rx` changed — loop back and rebind to the new host.
    }
}
