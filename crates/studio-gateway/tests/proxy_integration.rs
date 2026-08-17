//! Integration tests for the `/v1/*` gateway (§3.2) that need a real,
//! separate child process — `fake_model_server` (see `src/bin/`), a
//! lightweight stand-in for `crane-serve` that still exercises the real
//! `Supervisor::launch` path. Lives here rather than in `src/proxy.rs`
//! because `CARGO_BIN_EXE_fake_model_server` is only set for integration
//! test targets, not a lib's own unit tests.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{StatusCode, header};
use studio_core::launch::LaunchSpec;
use studio_gateway::{Daemon, GatewayState};
use studio_supervisor::{SpawnRequest, Supervisor};
use tokio::net::TcpListener;

/// `spec.model_path` doubles as the fake server's self-reported identity —
/// a test-fixture convenience; production never reads it that way.
// Must return `io::Result` to match `studio_gateway::SpawnBuilder`'s
// signature, even though this particular implementation can't fail.
#[allow(clippy::unnecessary_wraps)]
fn fake_spawn_builder(spec: &LaunchSpec) -> std::io::Result<(SpawnRequest, String)> {
    let program = PathBuf::from(env!("CARGO_BIN_EXE_fake_model_server"));
    let args = vec![
        "--port".to_string(),
        spec.port.to_string(),
        "--id".to_string(),
        spec.model_path.clone(),
    ];
    let health_url = format!("http://127.0.0.1:{}/health", spec.port);
    Ok((
        SpawnRequest {
            program,
            args,
            envs: vec![],
        },
        health_url,
    ))
}

fn spec(identity: &str, port: u16) -> LaunchSpec {
    LaunchSpec {
        model_path: identity.to_string(),
        model_type: "qwen3_5".to_string(),
        model_name: None,
        port,
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

async fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn spawn_gateway(state: Arc<GatewayState>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = studio_gateway::gateway_router().with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("127.0.0.1:{}", addr.port())
}

fn test_state() -> Arc<GatewayState> {
    let (daemon, _rx) = Daemon::with_grace_period(Supervisor::new(), Duration::from_secs(5));
    GatewayState::with_spawn_builder(daemon, fake_spawn_builder)
}

async fn wait_for_health(url: &str) {
    let client = reqwest::Client::new();
    for _ in 0..50 {
        if let Ok(r) = client.get(url).send().await
            && r.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("fake server never became healthy at {url}");
}

/// M6's own accept criterion: with two models configured, a single
/// unchanged base URL serves both, selecting by the `model` field —
/// including starting the second one on demand, mid-session.
#[tokio::test]
async fn two_models_one_base_url_second_one_starts_on_demand() {
    let state = test_state();
    let port_a = free_port().await;
    let port_b = free_port().await;
    state
        .registry
        .register("model-a".to_string(), spec("model-a", port_a));
    state
        .registry
        .register("model-b".to_string(), spec("model-b", port_b));
    let addr = spawn_gateway(state.clone()).await;

    // Pre-start only "model-a" directly through the real spawn path, to
    // prove the *other* request path (already running) too.
    let (request, health_url) = fake_spawn_builder(&spec("model-a", port_a)).unwrap();
    let child_id = state
        .daemon
        .supervisor()
        .launch(&request, Some(health_url), "model-a".to_string())
        .unwrap();
    wait_for_health(&format!("http://127.0.0.1:{port_a}/health")).await;
    state
        .registry
        .mark_running("model-a".to_string(), child_id, port_a);

    let client = reqwest::Client::new();

    // Same base URL (`addr`) serves "model-a", already running.
    let resp_a: serde_json::Value = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "model-a"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp_a["served_by"], "model-a");

    // Same base URL serves "model-b" — nothing was pre-started for it, so
    // this is the on-demand path.
    assert_eq!(state.registry.running_count(), 1);
    let resp_b: serde_json::Value = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "model-b"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp_b["served_by"], "model-b");
    assert_eq!(state.registry.running_count(), 2);

    state.daemon.supervisor().stop_all().await;
}

/// "Streams SSE straight through" (§3.2) — verified by timing: if the
/// gateway buffered the whole response before replying, all chunks would
/// arrive at once; forwarded live, they arrive tens of ms apart.
#[tokio::test]
async fn sse_responses_are_streamed_not_buffered() {
    let state = test_state();
    let port = free_port().await;
    state
        .registry
        .register("streamer".to_string(), spec("streamer", port));
    let addr = spawn_gateway(state.clone()).await;

    let client = reqwest::Client::new();
    let mut resp = client
        .post(format!("http://{addr}/v1/stream/4"))
        .json(&serde_json::json!({"model": "streamer"}))
        .send()
        .await
        .unwrap();
    assert!(
        resp.headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("event-stream")
    );

    let mut arrival_times = Vec::new();
    let start = tokio::time::Instant::now();
    while let Some(chunk) = resp.chunk().await.unwrap() {
        if !chunk.is_empty() {
            arrival_times.push(start.elapsed());
        }
    }
    assert!(arrival_times.len() >= 2, "{arrival_times:?}");
    let gap = arrival_times[arrival_times.len() - 1]
        .checked_sub(arrival_times[0])
        .unwrap();
    assert!(
        gap >= Duration::from_millis(50),
        "chunks arrived too close together to be streamed: {arrival_times:?}"
    );

    state.daemon.supervisor().stop_all().await;
}

#[tokio::test]
async fn on_demand_start_of_an_unregistered_model_fails_cleanly() {
    let state = test_state();
    let addr = spawn_gateway(state).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "never-registered"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}
