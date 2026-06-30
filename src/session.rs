//! MCP session manager for the FastMCP backend.
//!
//! Maintains a persistent MCP HTTP session (JSON-RPC over StreamableHTTP).
//! Handles initialization, reconnection on failure, and health checks.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::error::ProxyError;

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    message: String,
}

#[derive(Debug)]
struct SessionState {
    session_id: Option<String>,
    last_health_check: Option<Instant>,
    failed: bool,
}

impl SessionState {
    fn new() -> Self {
        Self {
            session_id: None,
            last_health_check: None,
            failed: false,
        }
    }
}

/// Manages a persistent MCP HTTP session to the backend.
pub struct McpSession {
    backend_url: String,
    client: reqwest::Client,
    state: Arc<Mutex<SessionState>>,
    health_check_interval: Duration,
}

impl McpSession {
    pub fn new(backend_url: String, timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            backend_url,
            client,
            state: Arc::new(Mutex::new(SessionState::new())),
            health_check_interval: Duration::from_secs(60),
        }
    }

    /// Get or initialize a session ID. Reconnects if the session was lost.
    pub async fn ensure_session(&self) -> Result<String, ProxyError> {
        let mut state = self.state.lock().await;

        // Return existing session if connected and healthy
        if let Some(sid) = state.session_id.clone() {
            if !state.failed {
                // Refresh health check timestamp periodically without reconnecting
                if let Some(last) = state.last_health_check {
                    if last.elapsed() > self.health_check_interval {
                        state.last_health_check = Some(Instant::now());
                    }
                }
                return Ok(sid);
            }
        }

        // Need to initialize a new session
        debug!("Initializing MCP session to {}", self.backend_url);

        let init_request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: next_id(),
            method: "initialize".into(),
            params: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "chord-proxy",
                    "version": "0.1.0"
                }
            })),
        };

        let response = self
            .client
            .post(format!("{}/mcp", self.backend_url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&init_request)
            .send()
            .await
            .map_err(|e| ProxyError::Session(format!("Failed to connect to MCP backend: {e}")))?;

        // Extract session ID from response header
        let session_id = response
            .headers()
            .get("mcp-session-id")
            .or_else(|| response.headers().get("Mcp-Session-Id"))
            .and_then(|v| v.to_str().ok())
            .map(String::from)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        debug!("MCP session initialized: {}", session_id);

        // Consume the body (may be empty or contain the initialize result)
        let _ = response.bytes().await;

        // Send initialized notification
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });
        let _ = self
            .client
            .post(format!("{}/mcp", self.backend_url))
            .header("Content-Type", "application/json")
            .header("Mcp-Session-Id", &session_id)
            .json(&notif)
            .send()
            .await;

        state.session_id = Some(session_id.clone());
        state.last_health_check = Some(Instant::now());
        state.failed = false;

        Ok(session_id)
    }

    /// Send a JSON-RPC request to the MCP backend and return the result.
    pub async fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, ProxyError> {
        let session_id = self.ensure_session().await?;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: next_id(),
            method: method.into(),
            params,
        };

        let response = self
            .client
            .post(format!("{}/mcp", self.backend_url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Session-Id", &session_id)
            .json(&request)
            .send()
            .await
            .map_err(|e| ProxyError::McpBackend(format!("Request failed: {e}")))?;

        let status = response.status();
        let body = response.text().await.map_err(ProxyError::Http)?;

        if !status.is_success() {
            return Err(ProxyError::McpBackend(format!(
                "MCP backend returned HTTP {status}: {body}"
            )));
        }

        // FastMCP SSE responses may start with "event: message\ndata: {...}\n\n"
        // or just "data: {...}\n\n", or plain JSON. Extract the data: line.
        let json_str = if body.contains("data:") {
            body.lines()
                .find(|l| l.starts_with("data:"))
                .map(|l| l.trim_start_matches("data:").trim())
                .unwrap_or(&body)
                .to_string()
        } else {
            body
        };

        if json_str.is_empty() {
            // Some MCP notifications return empty body — treat as ok/null result
            return Ok(Value::Null);
        }

        let rpc_response: JsonRpcResponse = serde_json::from_str(&json_str)
            .map_err(|e| ProxyError::McpBackend(format!("Invalid JSON response: {e} — body: {json_str}")))?;

        if let Some(err) = rpc_response.error {
            return Err(ProxyError::McpBackend(err.message));
        }

        Ok(rpc_response.result.unwrap_or(Value::Null))
    }

    /// Force session reset (e.g., after a fatal error).
    pub async fn reset(&self) {
        let mut state = self.state.lock().await;
        warn!("Resetting MCP session");
        state.session_id = None;
        state.failed = true;
    }

    /// Returns true if the session has been successfully initialized.
    pub async fn is_connected(&self) -> bool {
        let state = self.state.lock().await;
        state.session_id.is_some() && !state.failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_initial() {
        let state = SessionState::new();
        assert!(state.session_id.is_none());
        assert!(state.last_health_check.is_none());
        assert!(!state.failed);
    }

    #[test]
    fn test_next_id_increments() {
        let a = next_id();
        let b = next_id();
        assert!(b > a);
    }

    #[tokio::test]
    async fn test_session_is_not_connected_initially() {
        let session = McpSession::new("http://does-not-exist-for-test:9999".into(), 5);
        assert!(!session.is_connected().await);
    }

    #[tokio::test]
    async fn test_reset_clears_session() {
        let session = McpSession::new("http://does-not-exist-for-test:9999".into(), 5);
        // Manually set state as if connected
        {
            let mut state = session.state.lock().await;
            state.session_id = Some("test-session".into());
            state.failed = false;
        }
        assert!(session.is_connected().await);

        session.reset().await;
        assert!(!session.is_connected().await);
    }

    #[tokio::test]
    async fn test_session_connects_to_mock_backend() {
        let mock_server = httpmock::MockServer::start_async().await;

        // Mock the initialize call (exact method match)
        mock_server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/mcp")
                .body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "test-session-abc")
                .header("Content-Type", "application/json")
                .json_body(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05", "capabilities": {}}
                }));
        });

        // Mock the initialized notification and any other calls
        mock_server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/mcp");
            then.status(200).body("");
        });

        let session = McpSession::new(mock_server.base_url(), 10);
        let session_id = session.ensure_session().await.unwrap();
        assert_eq!(session_id, "test-session-abc");
        assert!(session.is_connected().await);
    }

    #[tokio::test]
    async fn test_session_returns_existing_id_without_reinit() {
        let mock_server = httpmock::MockServer::start_async().await;

        // Track call count — use exact method match to avoid matching notifications/initialized
        let init_mock = mock_server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/mcp")
                .body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "session-xyz")
                .header("Content-Type", "application/json")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        // Catch-all for notifications/initialized and other calls
        mock_server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/mcp");
            then.status(200).body("");
        });

        let session = McpSession::new(mock_server.base_url(), 10);
        let id1 = session.ensure_session().await.unwrap();
        let id2 = session.ensure_session().await.unwrap();
        assert_eq!(id1, id2);
        // initialize called only once
        init_mock.assert_hits(1);
    }

    #[tokio::test]
    async fn test_send_request_uses_session_header() {
        let mock_server = httpmock::MockServer::start_async().await;

        // Initialize (exact method match to avoid ambiguity with notifications/initialized)
        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "abc123")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        // Catch-all for notifications and other non-tools/list calls
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        // tools/list — check it sends the session header
        let tools_mock = mock_server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/mcp")
                .header("Mcp-Session-Id", "abc123")
                .body_contains("tools/list");
            then.status(200)
                .header("Content-Type", "application/json")
                .json_body(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"tools": []}
                }));
        });

        let session = McpSession::new(mock_server.base_url(), 10);
        let result = session.send_request("tools/list", None).await.unwrap();
        assert!(result["tools"].is_array());
        tools_mock.assert();
    }

    #[tokio::test]
    async fn test_send_request_parses_sse_response() {
        let mock_server = httpmock::MockServer::start_async().await;

        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "sse-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        // SSE-framed response
        mock_server.mock(|when, then| {
            when.body_contains("tools/call");
            then.status(200)
                .header("Content-Type", "text/event-stream")
                .body("data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}\n\n");
        });

        let session = McpSession::new(mock_server.base_url(), 10);
        let result = session
            .send_request(
                "tools/call",
                Some(serde_json::json!({"name": "test_tool", "arguments": {}})),
            )
            .await
            .unwrap();
        assert_eq!(result["content"][0]["text"], "ok");
    }

    #[tokio::test]
    async fn test_backend_error_propagated() {
        let mock_server = httpmock::MockServer::start_async().await;

        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "err-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/call");
            then.status(200)
                .json_body(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32601, "message": "Method not found"}
                }));
        });

        let session = McpSession::new(mock_server.base_url(), 10);
        let err = session
            .send_request("tools/call", Some(serde_json::json!({})))
            .await
            .unwrap_err();
        assert!(matches!(err, ProxyError::McpBackend(_)));
        assert!(err.to_string().contains("Method not found"));
    }
}
