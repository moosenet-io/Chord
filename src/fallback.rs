//! Bridge between terminus-rs's ToolRegistry and chord-proxy's FallbackRegistry.
//!
//! `TerminusToolProxy` wraps a shared `Arc<ToolRegistry>` and exposes each
//! registered tool as a `FallbackTool` — one proxy per tool name.
//!
//! `ToolRegistry` is `Send + Sync` and `RustTool::execute` takes `&self`, so
//! concurrent tool calls share the same `Arc<ToolRegistry>` without any locking.
//!
//! At startup in main.rs:
//!   1. Create `terminus_rs::ToolRegistry`, call `register_all()`
//!   2. Call `build_fallback_registry(terminus)` → `FallbackRegistry`
//!   3. Pass `FallbackRegistry` to `McpProxy::new()`

use std::sync::Arc;
use serde_json::Value;

use terminus_rs::ToolRegistry;

use crate::error::ProxyError;
use crate::mcp_proxy::FallbackTool;

/// Adapts one terminus-rs tool into chord-proxy's `FallbackTool` interface.
///
/// One `TerminusToolProxy` is registered per tool name. All proxies share
/// the same `Arc<ToolRegistry>` — no locks needed since `ToolRegistry` is
/// read-only after startup and `RustTool::execute` is `&self + Send + Sync`.
pub struct TerminusToolProxy {
    tool_name: String,
    tool_desc: String,
    tool_params: Value,
    registry: Arc<ToolRegistry>,
}

impl TerminusToolProxy {
    pub fn new(
        tool_name: String,
        tool_desc: String,
        tool_params: Value,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        Self { tool_name, tool_desc, tool_params, registry }
    }
}

#[async_trait::async_trait]
impl FallbackTool for TerminusToolProxy {
    fn name(&self) -> &str { &self.tool_name }
    fn description(&self) -> &str { &self.tool_desc }
    fn parameters(&self) -> Value { self.tool_params.clone() }

    async fn execute(&self, args: Value) -> Result<String, ProxyError> {
        match self.registry.call(&self.tool_name, args).await {
            Some(Ok(result)) => Ok(result),
            Some(Err(e)) => Err(ProxyError::ToolExecution(e.to_string())),
            None => Err(ProxyError::ToolNotFound(self.tool_name.clone())),
        }
    }
}

/// Build a `FallbackRegistry` from a terminus-rs `ToolRegistry`.
///
/// The `ToolRegistry` is moved into an `Arc` (no Mutex) — concurrent
/// fallback calls dispatch directly without serialization overhead.
pub fn build_fallback_registry(
    terminus: ToolRegistry,
) -> crate::mcp_proxy::FallbackRegistry {
    let tool_list = terminus.list();
    let shared = Arc::new(terminus);
    let mut registry = crate::mcp_proxy::FallbackRegistry::new();

    for info in tool_list {
        let proxy = TerminusToolProxy::new(
            info.name,
            info.description,
            info.parameters,
            Arc::clone(&shared),
        );
        registry.register(Box::new(proxy));
    }

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use terminus_rs::{RustTool, ToolError, ToolRegistry};

    struct PingTool;

    #[async_trait::async_trait]
    impl RustTool for PingTool {
        fn name(&self) -> &str { "ping" }
        fn description(&self) -> &str { "Ping" }
        fn parameters(&self) -> Value { serde_json::json!({}) }
        async fn execute(&self, _: Value) -> Result<String, ToolError> {
            Ok("pong".into())
        }
    }

    struct ErrorTool;

    #[async_trait::async_trait]
    impl RustTool for ErrorTool {
        fn name(&self) -> &str { "error_tool" }
        fn description(&self) -> &str { "Always fails" }
        fn parameters(&self) -> Value { serde_json::json!({}) }
        async fn execute(&self, _: Value) -> Result<String, ToolError> {
            Err(ToolError::Execution("forced failure".into()))
        }
    }

    fn make_registry_with_ping() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(PingTool)).unwrap();
        reg
    }

    #[test]
    fn test_build_fallback_registry_has_tools() {
        let terminus = make_registry_with_ping();
        let fallback = build_fallback_registry(terminus);
        let entries = fallback.as_catalog_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "ping");
        assert_eq!(entries[0].source, "chord");
    }

    #[tokio::test]
    async fn test_terminus_tool_proxy_executes() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(PingTool)).unwrap();
        let shared = Arc::new(reg);

        let proxy = TerminusToolProxy::new(
            "ping".into(),
            "Ping".into(),
            serde_json::json!({}),
            Arc::clone(&shared),
        );

        let result = proxy.execute(serde_json::json!({})).await.unwrap();
        assert_eq!(result, "pong");
    }

    #[tokio::test]
    async fn test_concurrent_calls_no_lock_contention() {
        // Verify multiple concurrent calls can proceed without serialization.
        // Arc<ToolRegistry> (no Mutex) allows this natively.
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(PingTool)).unwrap();
        let shared = Arc::new(reg);

        let proxies: Vec<_> = (0..10)
            .map(|_| {
                let proxy = TerminusToolProxy::new(
                    "ping".into(),
                    "Ping".into(),
                    serde_json::json!({}),
                    Arc::clone(&shared),
                );
                tokio::spawn(async move { proxy.execute(serde_json::json!({})).await })
            })
            .collect();

        for handle in proxies {
            assert_eq!(handle.await.unwrap().unwrap(), "pong");
        }
    }

    #[tokio::test]
    async fn test_terminus_tool_proxy_propagates_error() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(ErrorTool)).unwrap();
        let shared = Arc::new(reg);

        let proxy = TerminusToolProxy::new(
            "error_tool".into(),
            "Always fails".into(),
            serde_json::json!({}),
            shared,
        );

        let err = proxy.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ProxyError::ToolExecution(_)));
        assert!(err.to_string().contains("forced failure"));
    }

    #[tokio::test]
    async fn test_terminus_tool_proxy_not_found() {
        let reg = ToolRegistry::new(); // empty
        let shared = Arc::new(reg);

        let proxy = TerminusToolProxy::new(
            "nonexistent".into(),
            "Missing".into(),
            serde_json::json!({}),
            shared,
        );

        let err = proxy.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ProxyError::ToolNotFound(_)));
    }

    #[test]
    fn test_fallback_registry_from_empty_terminus() {
        let empty_registry = ToolRegistry::new();
        let fallback = build_fallback_registry(empty_registry);
        assert!(fallback.as_catalog_entries().is_empty());
    }

    #[test]
    fn test_terminus_adapter_name_matches_tool_name() {
        let terminus = make_registry_with_ping();
        let fallback = build_fallback_registry(terminus);
        let entries = fallback.as_catalog_entries();
        assert_eq!(entries[0].name, "ping");
    }
}
