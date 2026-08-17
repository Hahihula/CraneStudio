//! Polls a child's `/health` endpoint, per PLAN.md §2.14: "`GET /health`
//! returns `{"status":"ok"}` — use it as the child readiness probe."

use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(300);
const PER_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Polls until `url` returns a successful status, or `timeout` elapses
/// (whichever first) — returns whether it became healthy.
pub async fn wait_for_health(url: &str, timeout: Duration) -> bool {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + timeout;

    while tokio::time::Instant::now() < deadline {
        if let Ok(response) = client.get(url).timeout(PER_REQUEST_TIMEOUT).send().await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    false
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn returns_true_once_the_endpoint_responds_ok() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/health");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}").await;
            let _ = socket.shutdown().await;
        });

        assert!(wait_for_health(&url, Duration::from_secs(5)).await);
    }

    #[tokio::test]
    async fn gives_up_after_the_timeout_when_nothing_is_listening() {
        // Port 1 is a reserved low port nothing will ever bind in a test
        // sandbox — connection is refused immediately, every poll.
        let ok = wait_for_health("http://127.0.0.1:1/health", Duration::from_millis(700)).await;
        assert!(!ok);
    }

    #[tokio::test]
    async fn keeps_polling_through_a_slow_start() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/health");
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = attempts.clone();

        tokio::spawn(async move {
            // First connection: refuse to answer (drop it), simulating a
            // not-yet-ready server. Second: answer for real.
            let (socket, _) = listener.accept().await.unwrap();
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            drop(socket);

            let (mut socket, _) = listener.accept().await.unwrap();
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            let mut buf = vec![0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}").await;
            let _ = socket.shutdown().await;
        });

        assert!(wait_for_health(&url, Duration::from_secs(5)).await);
        assert!(attempts.load(Ordering::SeqCst) >= 2);
    }
}
