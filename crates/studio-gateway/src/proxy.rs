//! The `/v1/*` gateway, per PLAN.md §3.2: "a single stable port... that
//! aggregates `/v1/models`... routes each request to the correct child by
//! the request's `model` field... starts a child on demand... evicts by
//! LRU when VRAM is insufficient... streams SSE straight through."
//!
//! This is the connect-instructions win the whole product is built around:
//! the base URL a coding agent points at never changes when the user
//! switches models.

use std::time::Duration;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use studio_core::launch::LaunchSpec;

use crate::state::GatewayState;

/// Default gateway port, per §3.2 — distinct from the control port (§3.1a).
pub const DEFAULT_GATEWAY_PORT: u16 = 1234;

const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);

/// Builds the fully-stated gateway router, with the auth-token check (§10.1)
/// layered in — takes `state` directly (rather than the more usual
/// `Router::with_state` at the call site) because `from_fn_with_state` needs
/// the actual `Arc<GatewayState>` value to compare tokens against, not just
/// its type.
pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/register", post(register))
        .route("/restart", post(restart))
        .fallback(proxy)
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .with_state(state)
}

/// PLAN.md §10.1: LAN exposure "must require a bearer token." A `None`
/// `auth_token` (the loopback default) means every request passes through
/// unchecked, matching today's no-auth behavior.
async fn require_auth(
    State(state): State<Arc<GatewayState>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(token) = state.auth_token() else {
        return next.run(request).await;
    };
    let provided = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(provided) if constant_time_eq(provided.as_bytes(), token.as_bytes()) => {
            next.run(request).await
        }
        _ => (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response(),
    }
}

/// Avoids leaking the token's length/prefix through response-time timing —
/// cheap insurance for a value that's otherwise typed into a header once
/// per client setup.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[derive(Deserialize)]
struct RegisterRequest {
    name: String,
    spec: LaunchSpec,
}

#[derive(Deserialize)]
struct RestartRequest {
    name: String,
}

/// Stops the child currently serving `name`, if any, so the next request for it
/// spawns a fresh process. The model stays registered. Idempotent.
async fn restart(
    State(state): State<std::sync::Arc<GatewayState>>,
    Json(req): Json<RestartRequest>,
) -> StatusCode {
    if let Some(child_id) = state.registry.running_child_id(&req.name) {
        state.daemon.supervisor().stop(child_id).await;
        state.registry.forget_running(&req.name);
    }
    StatusCode::OK
}

/// Adds a model to the gateway's aggregate `/v1/models` list and makes it
/// eligible for on-demand start — the wizard (M7) or a saved profile (M8)
/// is the intended real caller.
async fn register(
    State(state): State<std::sync::Arc<GatewayState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    req.spec
        .validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    state.registry.register(req.name, req.spec);
    Ok(StatusCode::OK)
}

async fn list_models(State(state): State<std::sync::Arc<GatewayState>>) -> Json<serde_json::Value> {
    let data: Vec<_> = state.registry.configured_names().into_iter().map(|id| serde_json::json!({"id": id, "object": "model", "created": 0, "owned_by": "cranestudio"})).collect();
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// Everything that isn't `/v1/models` or `/register`: read the `model`
/// field, make sure that child is running (starting it on demand, evicting
/// an LRU one if needed), and reverse-proxy the request straight through —
/// streamed, not buffered, so SSE passes through untouched.
async fn proxy(
    State(state): State<std::sync::Arc<GatewayState>>,
    request: Request,
) -> Result<Response, (StatusCode, String)> {
    let (parts, body) = request.into_parts();
    let bytes = axum::body::to_bytes(body, 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let model_name = extract_model_field(&bytes).ok_or((
        StatusCode::BAD_REQUEST,
        "request body has no 'model' field".to_string(),
    ))?;

    let port = ensure_running(&state, &model_name)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e))?;

    // The model's default thinking level, for clients that don't choose one
    // (crane-serve only reads it per request).
    let bytes = match state
        .registry
        .spec_for(&model_name)
        .and_then(|spec| spec.reasoning_effort)
    {
        Some(effort) if parts.uri.path().ends_with("/chat/completions") => {
            with_default_reasoning_effort(&bytes, effort.as_str()).map_or(bytes, Into::into)
        }
        _ => bytes,
    };

    let path_and_query = parts
        .uri
        .path_and_query()
        .map_or("/", axum::http::uri::PathAndQuery::as_str);
    let target_url = format!("http://127.0.0.1:{port}{path_and_query}");

    let mut outbound = state.client.request(parts.method.clone(), &target_url);
    outbound = outbound.headers(forwarded_headers(&parts.headers));
    outbound = outbound.body(bytes);

    let upstream = outbound.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("upstream request failed: {e}"),
        )
    })?;
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response_headers = upstream.headers().clone();
    let body = Body::from_stream(upstream.bytes_stream());

    let mut response = Response::builder().status(status);
    for (name, value) in &response_headers {
        if *name == header::TRANSFER_ENCODING || *name == header::CONNECTION {
            continue;
        }
        response = response.header(name, value);
    }
    response
        .body(body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        .map(IntoResponse::into_response)
}

fn forwarded_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if *name == header::HOST || *name == header::CONTENT_LENGTH {
            continue;
        }
        out.insert(name.clone(), value.clone());
    }
    out
}

fn extract_model_field(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(str::to_string)
}

/// `body` with a top-level `reasoning_effort` added, or `None` when the client
/// already chose one — as the top-level request field or through
/// `chat_template_kwargs`, the two places crane-serve reads it from.
fn with_default_reasoning_effort(body: &[u8], effort: &str) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let object = value.as_object_mut()?;
    let chosen = |v: Option<&serde_json::Value>| v.is_some_and(|v| !v.is_null());
    if chosen(object.get("reasoning_effort"))
        || chosen(
            object
                .get("chat_template_kwargs")
                .and_then(|k| k.get("reasoning_effort")),
        )
    {
        return None;
    }
    object.insert("reasoning_effort".to_string(), effort.into());
    serde_json::to_vec(&value).ok()
}

/// Starts `model_name` if it isn't already running, evicting the
/// least-recently-used other model first if there isn't likely to be room
/// (§3.2's LRU eviction) — grounded in a real number (the new model's
/// weight size on disk), not a guess.
async fn ensure_running(state: &GatewayState, model_name: &str) -> Result<u16, String> {
    if let Some(port) = state.registry.touch_if_running(model_name) {
        return Ok(port);
    }

    let spec = state.registry.spec_for(model_name).ok_or_else(|| {
        format!("unknown model '{model_name}' — not registered with this gateway")
    })?;

    make_room_for(state, &spec, model_name).await;

    let (request, health_url) = (state.spawn_builder)(&spec).map_err(|e| e.to_string())?;
    let child_id = state
        .daemon
        .supervisor()
        .launch(&request, Some(health_url.clone()), model_name.to_string())
        .map_err(|e| e.to_string())?;

    if !wait_for_health(&health_url).await {
        return Err(format!(
            "'{model_name}' did not become healthy in time — check its logs"
        ));
    }

    state
        .registry
        .mark_running(model_name.to_string(), child_id, spec.port);
    Ok(spec.port)
}

async fn wait_for_health(url: &str) -> bool {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + HEALTH_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        if let Ok(response) = client.get(url).timeout(Duration::from_secs(2)).send().await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    false
}

/// Evicts running models, least-recently-used first, while the free VRAM
/// looks too small for `incoming`'s weights — repeats until either it
/// looks like it'll fit or there's nothing left to evict. Not a full §7
/// prediction (that needs the model's architecture config, which isn't
/// available from a `LaunchSpec` alone) — just the one number always
/// available for free: the weight file size on disk, plus headroom.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn make_room_for(state: &GatewayState, incoming: &LaunchSpec, incoming_name: &str) {
    let Some(needed) = weight_bytes(&incoming.model_path) else {
        return;
    };
    let needed_with_headroom = (needed as f64 * 1.2) as u64;

    loop {
        let free = studio_core::hardware::probe(&studio_core::paths::models_dir())
            .gpus
            .first()
            .map_or(u64::MAX, |gpu| gpu.vram_free);
        if free >= needed_with_headroom {
            return;
        }
        let Some((victim_name, victim_id)) = state.registry.least_recently_used(incoming_name)
        else {
            return; // nothing left to evict; let it try and possibly OOM (classified per §7.4)
        };
        state.daemon.supervisor().stop(victim_id).await;
        state.registry.forget_running(&victim_name);
    }
}

fn weight_bytes(model_path: &str) -> Option<u64> {
    let path = std::path::Path::new(model_path);
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        std::fs::metadata(path).ok().map(|m| m.len())
    } else {
        studio_core::estimator::safetensors_dir_bytes(path).ok()
    }
}

// Tests that need to actually spawn `fake_model_server` live in
// tests/proxy_integration.rs — `CARGO_BIN_EXE_*` is only set for
// integration test targets, not for a lib's own `#[cfg(test)]` modules.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use studio_supervisor::{SpawnRequest, Supervisor};
    use tokio::net::TcpListener;

    use super::*;
    use crate::control::Daemon;

    // Never actually invoked in this module's tests (none of them reach
    // the on-demand-start path) — a real builder is only needed by the
    // integration tests in tests/proxy_integration.rs, where
    // `CARGO_BIN_EXE_fake_model_server` is available.
    fn unreachable_spawn_builder(_spec: &LaunchSpec) -> std::io::Result<(SpawnRequest, String)> {
        unreachable!("this test never starts a model on demand")
    }

    #[test]
    fn default_reasoning_effort_is_added_only_when_the_client_sets_none() {
        let added = with_default_reasoning_effort(br#"{"model":"m","messages":[]}"#, "medium")
            .map(|b| serde_json::from_slice::<serde_json::Value>(&b).unwrap());
        assert_eq!(added.unwrap()["reasoning_effort"], "medium");

        assert!(
            with_default_reasoning_effort(br#"{"model":"m","reasoning_effort":"low"}"#, "xhigh")
                .is_none()
        );
        assert!(
            with_default_reasoning_effort(
                br#"{"model":"m","chat_template_kwargs":{"reasoning_effort":"low"}}"#,
                "xhigh"
            )
            .is_none()
        );
        // An explicit null counts as unset.
        let added =
            with_default_reasoning_effort(br#"{"model":"m","reasoning_effort":null}"#, "xhigh")
                .map(|b| serde_json::from_slice::<serde_json::Value>(&b).unwrap());
        assert_eq!(added.unwrap()["reasoning_effort"], "xhigh");
        assert!(with_default_reasoning_effort(b"not json", "xhigh").is_none());
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
            reasoning_effort: None,
            dspark: None,
        }
    }

    async fn spawn_gateway(state: Arc<GatewayState>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app: Router = router(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("127.0.0.1:{}", addr.port())
    }

    fn test_state() -> Arc<GatewayState> {
        let (daemon, _rx) = Daemon::with_grace_period(Supervisor::new(), Duration::from_secs(5));
        GatewayState::with_spawn_builder(daemon, unreachable_spawn_builder)
    }

    #[tokio::test]
    async fn v1_models_lists_configured_models_whether_or_not_running() {
        let state = test_state();
        state
            .registry
            .register("configured-only".to_string(), spec("configured-only", 0));
        let addr = spawn_gateway(state).await;

        let client = reqwest::Client::new();
        let body: serde_json::Value = client
            .get(format!("http://{addr}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let ids: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["configured-only"]);
    }

    #[tokio::test]
    async fn no_auth_token_configured_lets_requests_through_unchecked() {
        let state = test_state();
        state
            .registry
            .register("configured-only".to_string(), spec("configured-only", 0));
        let addr = spawn_gateway(state).await;

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_configured_auth_token_rejects_requests_without_it() {
        let state = test_state();
        state.set_auth_token(Some("secret-token".to_string()));
        let addr = spawn_gateway(state).await;

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_configured_auth_token_rejects_the_wrong_bearer_value() {
        let state = test_state();
        state.set_auth_token(Some("secret-token".to_string()));
        let addr = spawn_gateway(state).await;

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/v1/models"))
            .bearer_auth("wrong-token")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_configured_auth_token_accepts_the_matching_bearer_value() {
        let state = test_state();
        state.set_auth_token(Some("secret-token".to_string()));
        let addr = spawn_gateway(state).await;

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/v1/models"))
            .bearer_auth("secret-token")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_model_is_a_clear_error_not_a_hang() {
        let state = test_state();
        let addr = spawn_gateway(state).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "nope"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert!(resp.text().await.unwrap().contains("unknown model"));
    }

    #[tokio::test]
    async fn restart_is_idempotent_when_nothing_is_running() {
        let state = test_state();
        state.registry.register("m".to_string(), spec("m", 0));
        let addr = spawn_gateway(state).await;
        let resp = reqwest::Client::new()
            .post(format!("http://{addr}/restart"))
            .json(&serde_json::json!({"name": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn register_rejects_an_invalid_isq_level_before_it_can_ever_be_launched() {
        let state = test_state();
        let addr = spawn_gateway(state).await;
        let client = reqwest::Client::new();
        let mut bad_spec = spec("bad", 0);
        bad_spec.quant = Some("not-a-real-level".to_string());
        let resp = client
            .post(format!("http://{addr}/register"))
            .json(&serde_json::json!({"name": "bad", "spec": bad_spec}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
