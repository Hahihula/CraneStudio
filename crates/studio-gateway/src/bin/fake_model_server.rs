//! Test-only stand-in for a `crane-serve` child, used by `proxy.rs`'s
//! integration tests: a real, separate process (so it exercises the real
//! `Supervisor::launch` path, not an in-process mock), identifying itself
//! in every response so a test can confirm which model actually served a
//! request. Never shipped — only ever invoked from `#[cfg(test)]` code via
//! `env!("CARGO_BIN_EXE_fake_model_server")`.

use axum::extract::{Path, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::{Router, http};
use futures_util::StreamExt as _;

#[derive(Clone)]
struct Identity(String);

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let mut port = 0u16;
    let mut id = "unknown".to_string();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--id" => id = args.next().unwrap_or_default(),
            _ => {}
        }
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(echo))
        .route("/v1/stream/{n}", post(stream))
        .fallback(echo)
        .with_state(Identity(id));

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn echo(
    State(identity): State<Identity>,
    uri: http::Uri,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "served_by": identity.0,
        "path": uri.path(),
        "echoed_body": String::from_utf8_lossy(&body),
    }))
}

/// Streams `n` small SSE chunks with a short delay between them, so a test
/// can confirm the gateway forwards a stream incrementally rather than
/// buffering the whole thing before responding.
async fn stream(
    State(identity): State<Identity>,
    Path(n): Path<u32>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = identity.0;
    let events = (0..n).map(move |i| Ok(Event::default().data(format!("{id}-chunk-{i}"))));
    let stream = futures_util::stream::iter(events).then(|event| async move {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        event
    });
    Sse::new(stream)
}
