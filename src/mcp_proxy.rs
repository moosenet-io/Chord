//! McpProxy: the core of CHORD-01.
//!
//! Routes tool requests between lumina-core and the MCP backend.
//! Falls back to in-process Rust tools (terminus-rs, added by CHORD-05) when
//! the backend is unavailable.

use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::catalog::{extract_tool_result, parse_mcp_tools, ToolCatalog, ToolEntry};
use crate::config::Config;
use crate::error::ProxyError;
use crate::session::McpSession;

/// A Rust fallback tool. Implemented by terminus-rs tool modules (CHORD-05 onward).
/// This trait lives here so chord-proxy can call fallback tools without depending
/// on terminus-rs directly — the fallback registry is populated at startup.
#[async_trait::async_trait]
pub trait FallbackTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<String, ProxyError>;
}

/// Registry of Rust fallback tools.
#[derive(Default)]
pub struct FallbackRegistry {
    tools: Vec<Box<dyn FallbackTool>>,
}

impl FallbackRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn FallbackTool>) {
        self.tools.push(tool);
    }

    /// Returns true if a tool with this name is registered in the Rust fallback.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name() == name)
    }

    pub fn as_catalog_entries(&self) -> Vec<ToolEntry> {
        self.tools
            .iter()
            .map(|t| ToolEntry::from_rust(
                t.name().into(),
                t.description().into(),
                t.parameters(),
            ))
            .collect()
    }

    pub async fn call(&self, name: &str, args: Value) -> Option<Result<String, ProxyError>> {
        let tool = self.tools.iter().find(|t| t.name() == name)?;
        Some(tool.execute(args).await)
    }
}

/// The unified MCP proxy.
pub struct McpProxy {
    session: McpSession,
    catalog: Mutex<ToolCatalog>,
    fallback: Arc<FallbackRegistry>,
    timeout: Duration,
}

impl McpProxy {
    pub fn new(config: &Config, fallback: Arc<FallbackRegistry>) -> Self {
        Self {
            session: McpSession::new(config.mcp_backend_url.clone(), config.tool_timeout_secs),
            catalog: Mutex::new(ToolCatalog::new(config.catalog_cache_secs)),
            fallback,
            timeout: Duration::from_secs(config.tool_timeout_secs),
        }
    }

    /// Return the merged tool catalog, refreshing from MCP backend if stale.
    pub async fn tool_list(&self) -> Result<Vec<ToolEntry>, ProxyError> {
        let mut cat = self.catalog.lock().await;
        if !cat.is_stale() {
            return Ok(cat.all().to_vec());
        }

        debug!("Refreshing tool catalog from MCP backend");

        let rust_tools = self.fallback.as_catalog_entries();

        // Attempt to fetch from MCP backend
        let mcp_tools = match self.fetch_mcp_tools().await {
            Ok(tools) => {
                debug!("Fetched {} MCP tools", tools.len());
                tools
            }
            Err(e) => {
                warn!("Failed to fetch MCP tools: {e}. Using Rust-only catalog.");
                vec![]
            }
        };

        cat.update(mcp_tools, rust_tools);
        Ok(cat.all().to_vec())
    }

    /// Execute a tool call. Routes based on catalog source, then falls back if needed.
    ///
    /// Routing:
    ///   1. If the catalog shows source="chord" (Rust-only), call Rust directly.
    ///   2. Otherwise try MCP first; if MCP fails or returns an error, try Rust.
    ///
    /// Returns `(result_text, source)` where source is "mcp" or "chord" (Rust fallback).
    pub async fn tool_call(&self, name: &str, args: Value) -> Result<(String, &'static str), ProxyError> {
        // If the tool is in the Rust fallback registry and NOT in the warmed MCP
        // catalog (or the catalog isn't warmed yet), skip MCP entirely.
        // This avoids the case where MCP returns HTTP 200 "Unknown tool: X" which
        // looks like a success and blocks the fallback path.
        let in_rust = self.fallback.contains(name);
        let in_mcp = {
            let cat = self.catalog.lock().await;
            cat.find(name).map(|e| e.source.as_str() == "mcp").unwrap_or(false)
        };
        if in_rust && !in_mcp {
            if let Some(result) = self.fallback.call(name, args.clone()).await {
                return result.map(|r| (r, "chord"));
            }
        }

        // Try MCP backend
        match self.call_mcp(name, args.clone()).await {
            Ok(result) => return Ok((result, "mcp")),
            Err(e) => {
                debug!("MCP call failed for {name}: {e}. Trying Rust fallback.");
            }
        }

        // Rust fallback (for tools where MCP failed unexpectedly)
        if let Some(result) = self.fallback.call(name, args).await {
            return result.map(|r| (r, "chord"));
        }

        Err(ProxyError::ToolNotFound(format!(
            "Tool '{name}' not available (MCP failed, no Rust fallback)"
        )))
    }

    /// Discover tools matching a query, up to max_results.
    pub async fn tool_discover(&self, query: &str, max_results: usize) -> Result<Vec<ToolEntry>, ProxyError> {
        let _ = self.tool_list().await?; // ensure catalog is warm
        let cat = self.catalog.lock().await;
        Ok(cat.discover(query, max_results))
    }

    async fn fetch_mcp_tools(&self) -> Result<Vec<ToolEntry>, ProxyError> {
        let result = tokio::time::timeout(
            self.timeout,
            self.session.send_request("tools/list", None),
        )
        .await
        .map_err(|_| ProxyError::Timeout("tools/list".into()))??;
        // On failure, ensure_session() will reconnect on the next call automatically.

        Ok(parse_mcp_tools(&result))
    }

    async fn call_mcp(&self, name: &str, args: Value) -> Result<String, ProxyError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": args
        });

        let result = tokio::time::timeout(
            self.timeout,
            self.session.send_request("tools/call", Some(params)),
        )
        .await
        .map_err(|_| ProxyError::Timeout(name.into()))??;

        Ok(extract_tool_result(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RateLimitConfig;

    struct EchoTool;

    #[async_trait::async_trait]
    impl FallbackTool for EchoTool {
        fn name(&self) -> &str { "echo_test" }
        fn description(&self) -> &str { "Echo the input" }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn execute(&self, args: Value) -> Result<String, ProxyError> {
            Ok(args.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
        }
    }

    struct AlwaysErrorTool;

    #[async_trait::async_trait]
    impl FallbackTool for AlwaysErrorTool {
        fn name(&self) -> &str { "error_tool" }
        fn description(&self) -> &str { "Always fails" }
        fn parameters(&self) -> Value { serde_json::json!({}) }
        async fn execute(&self, _args: Value) -> Result<String, ProxyError> {
            Err(ProxyError::ToolExecution("always fails".into()))
        }
    }

    fn make_registry_with_echo() -> Arc<FallbackRegistry> {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(EchoTool));
        Arc::new(reg)
    }

    #[test]
    fn test_fallback_registry_as_catalog_entries() {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(EchoTool));
        let entries = reg.as_catalog_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "echo_test");
        assert_eq!(entries[0].source, "chord");
    }

    #[tokio::test]
    async fn test_fallback_registry_call_found() {
        let mut reg = FallbackRegistry::new();
        reg.register(Box::new(EchoTool));
        let result = reg.call("echo_test", serde_json::json!({"text": "hello"})).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().unwrap(), "hello");
    }

    #[tokio::test]
    async fn test_fallback_registry_call_not_found() {
        let reg = FallbackRegistry::new();
        let result = reg.call("nonexistent", serde_json::json!({})).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_tool_call_uses_rust_fallback_when_mcp_fails() {
        let mock_server = httpmock::MockServer::start_async().await;

        // MCP backend: initialize succeeds, then tools/call fails
        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "test-xyz")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/call");
            then.status(500).body("internal error");
        });

        let config = Config {
            mcp_backend_url: mock_server.base_url(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
        };

        let proxy = McpProxy::new(&config, make_registry_with_echo());
        let (result, source) = proxy
            .tool_call("echo_test", serde_json::json!({"text": "fallback works"}))
            .await
            .unwrap();
        assert_eq!(result, "fallback works");
        assert_eq!(source, "chord"); // served by Rust fallback
    }

    #[tokio::test]
    async fn test_tool_call_not_found_when_both_fail() {
        let mock_server = httpmock::MockServer::start_async().await;

        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "nf-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/call");
            then.status(404);
        });

        let config = Config {
            mcp_backend_url: mock_server.base_url(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
        };

        let reg = Arc::new(FallbackRegistry::new()); // no tools registered
        let proxy = McpProxy::new(&config, reg);
        let err = proxy.tool_call("nonexistent_tool", serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ProxyError::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn test_tool_list_merges_mcp_and_rust() {
        let mock_server = httpmock::MockServer::start_async().await;

        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "list-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/list");
            then.status(200)
                .json_body(serde_json::json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {
                        "tools": [
                            {"name": "mcp_tool_a", "description": "From MCP", "inputSchema": {}}
                        ]
                    }
                }));
        });

        let config = Config {
            mcp_backend_url: mock_server.base_url(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
        };

        let proxy = McpProxy::new(&config, make_registry_with_echo());
        let tools = proxy.tool_list().await.unwrap();

        assert!(tools.len() >= 2); // at least mcp_tool_a + echo_test
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"mcp_tool_a"));
        assert!(names.contains(&"echo_test"));
    }

    #[tokio::test]
    async fn test_tool_list_rust_only_when_mcp_down() {
        let config = Config {
            mcp_backend_url: "http://does-not-exist-for-test:9999".into(),
            jwt_secret: String::new(),
            tool_timeout_secs: 1, // short timeout
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
        };

        let proxy = McpProxy::new(&config, make_registry_with_echo());
        let tools = proxy.tool_list().await.unwrap();

        // Should still return Rust tools even when MCP is down
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "echo_test"));
    }

    #[tokio::test]
    async fn test_tool_discover_returns_relevant_tools() {
        let mock_server = httpmock::MockServer::start_async().await;

        mock_server.mock(|when, then| {
            when.body_contains(r#""method":"initialize""#);
            then.status(200)
                .header("Mcp-Session-Id", "disc-test")
                .json_body(serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}}));
        });
        mock_server.mock(|when, then| {
            when.body_contains("notifications/initialized");
            then.status(200).body("");
        });
        mock_server.mock(|when, then| {
            when.body_contains("tools/list");
            then.status(200)
                .json_body(serde_json::json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {
                        "tools": [
                            {"name": "calendar_today", "description": "Get calendar events today"},
                            {"name": "email_inbox", "description": "Read email inbox"}
                        ]
                    }
                }));
        });

        let config = Config {
            mcp_backend_url: mock_server.base_url(),
            jwt_secret: String::new(),
            tool_timeout_secs: 5,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: std::collections::HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: vec![],
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
        };

        let reg = Arc::new(FallbackRegistry::new());
        let proxy = McpProxy::new(&config, reg);
        let results = proxy.tool_discover("calendar events", 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].name, "calendar_today");
    }
}
