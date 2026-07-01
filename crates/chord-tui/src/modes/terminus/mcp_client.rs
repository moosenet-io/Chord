//! Terminus MCP client (S91 CTUI-04 connect+status, extended by CTUI-05 config
//! writes).
//!
//! Connects to a Terminus MCP server over stdio OR HTTP, lists tools, and reads
//! health/status. Reuses CTUI-01's async model: all I/O is behind an injectable
//! [`McpConn`] transport so a slow/dead instance is bounded by timeout and one
//! instance's failure is isolated to that instance.
//!
//! Robustness requirements handled here:
//!   - **unexpected MCP version** → surfaced as [`ConnState::Incompatible`], no
//!     crash; tools are not trusted from an incompatible server.
//!   - **stdio process death**    → [`ConnState::Disconnected`] with
//!     `retriable = true`.
//!   - **remote auth failure**    → [`ConnState::AuthFailed`] scoped to that
//!     instance only.
//!
//! Auth tokens are resolved from the vault by the caller and passed per-call;
//! values are never logged and never stored in a status snapshot.

use async_trait::async_trait;

use crate::secret::SecretValue;

/// The MCP protocol version this client speaks. A server advertising a version
/// outside the supported set is treated as incompatible (read-only, no trust of
/// its tool list for mutations).
pub const SUPPORTED_MCP_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26"];

/// Connection/health state of one instance, as shown per-instance. Never carries
/// secret material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnState {
    /// Not yet connected.
    Idle,
    /// Connected; MCP handshake OK and version supported.
    Connected { version: String },
    /// Reachable but advertising an unsupported MCP version — shown as
    /// incompatible; we do NOT crash and do NOT trust it for mutations.
    Incompatible { version: String },
    /// The stdio child process died (or HTTP conn dropped). Retriable.
    Disconnected { retriable: bool, detail: String },
    /// Remote authentication failed (e.g. 401/403). Scoped to THIS instance.
    AuthFailed { detail: String },
    /// Any other transport error, isolated to this instance.
    Error { detail: String },
}

impl ConnState {
    pub fn label(&self) -> &'static str {
        match self {
            ConnState::Idle => "idle",
            ConnState::Connected { .. } => "connected",
            ConnState::Incompatible { .. } => "incompatible",
            ConnState::Disconnected { .. } => "disconnected",
            ConnState::AuthFailed { .. } => "auth-failed",
            ConnState::Error { .. } => "error",
        }
    }

    /// True only when the handshake succeeded with a supported version — the
    /// precondition for trusting the tool inventory and offering mutations.
    pub fn is_usable(&self) -> bool {
        matches!(self, ConnState::Connected { .. })
    }
}

/// One tool as reported by an MCP server. Read-only inventory row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolInfo {
    pub name: String,
    /// Whether the server reports the tool currently enabled. `None` if the
    /// server does not expose enable/disable state (capability absent).
    pub enabled: Option<bool>,
    /// Domain / module the tool belongs to, if reported.
    pub domain: Option<String>,
    /// Human description, if reported.
    pub description: Option<String>,
}

/// A per-instance status snapshot (state + inventory). No secrets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceReport {
    pub state: ConnState,
    pub tool_count: usize,
    pub tools: Vec<ToolInfo>,
    /// Capabilities the server advertises — decides which CTUI-05 controls are
    /// offered vs shown read-only ("not supported by this instance").
    pub caps: ServerCaps,
}

/// Server-advertised control capabilities. When a capability is absent, the UI
/// shows the control read-only and NEVER fakes a mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ServerCaps {
    /// Server exposes per-tool enable/disable control.
    pub tool_toggle: bool,
    /// Server exposes scope view/edit.
    pub scope_edit: bool,
}

/// Errors from an MCP transport. Never carries secret material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpError {
    /// stdio child process died / HTTP connection dropped.
    ProcessDied(String),
    /// Auth rejected by a remote (401/403).
    Auth(String),
    /// Server MCP version not in the supported set.
    Version(String),
    /// Any other transport error.
    Transport(String),
}

/// Abstraction over the underlying MCP transport (stdio child OR HTTP). Injected
/// so unit tests can simulate process death, auth failure, and version drift
/// without real sockets or processes. One transport instance ⇒ one server; a
/// failure here is isolated to that instance.
#[async_trait]
pub trait McpConn: Send + Sync {
    /// Perform the MCP `initialize` handshake and return the server's advertised
    /// protocol version + capabilities.
    async fn handshake(&self, token: Option<&SecretValue>) -> Result<(String, ServerCaps), McpError>;

    /// List tools (only meaningful after a compatible handshake).
    async fn list_tools(&self, token: Option<&SecretValue>) -> Result<Vec<ToolInfo>, McpError>;
}

/// Connect + read status for ONE instance. This is the CTUI-04 status entry
/// point. It classifies handshake results into a [`ConnState`] and, on success,
/// fetches the tool inventory. Any failure is returned as this instance's own
/// report — it can never affect another instance.
pub async fn connect_and_report(conn: &dyn McpConn, token: Option<&SecretValue>) -> InstanceReport {
    match conn.handshake(token).await {
        Ok((version, caps)) => {
            if !SUPPORTED_MCP_VERSIONS.contains(&version.as_str()) {
                // Unexpected MCP version → incompatible, no crash, no tool trust.
                return InstanceReport {
                    state: ConnState::Incompatible { version },
                    tool_count: 0,
                    tools: Vec::new(),
                    caps: ServerCaps::default(),
                };
            }
            // Compatible: fetch inventory; inventory failure degrades to an
            // empty list but keeps the connected state (drift tolerant).
            let tools = conn.list_tools(token).await.unwrap_or_default();
            InstanceReport {
                state: ConnState::Connected { version },
                tool_count: tools.len(),
                tools,
                caps,
            }
        }
        Err(McpError::Auth(d)) => InstanceReport {
            state: ConnState::AuthFailed { detail: d },
            tool_count: 0,
            tools: Vec::new(),
            caps: ServerCaps::default(),
        },
        Err(McpError::ProcessDied(d)) => InstanceReport {
            state: ConnState::Disconnected { retriable: true, detail: d },
            tool_count: 0,
            tools: Vec::new(),
            caps: ServerCaps::default(),
        },
        Err(McpError::Version(v)) => InstanceReport {
            state: ConnState::Incompatible { version: v },
            tool_count: 0,
            tools: Vec::new(),
            caps: ServerCaps::default(),
        },
        Err(McpError::Transport(d)) => InstanceReport {
            state: ConnState::Error { detail: d },
            tool_count: 0,
            tools: Vec::new(),
            caps: ServerCaps::default(),
        },
    }
}

#[cfg(test)]
pub(crate) mod mock {
    //! Test doubles simulating each failure mode without real I/O.
    use super::*;

    /// A conn that succeeds with a chosen version + caps + tools.
    pub struct OkConn {
        pub version: String,
        pub caps: ServerCaps,
        pub tools: Vec<ToolInfo>,
    }

    #[async_trait]
    impl McpConn for OkConn {
        async fn handshake(&self, _t: Option<&SecretValue>) -> Result<(String, ServerCaps), McpError> {
            Ok((self.version.clone(), self.caps))
        }
        async fn list_tools(&self, _t: Option<&SecretValue>) -> Result<Vec<ToolInfo>, McpError> {
            Ok(self.tools.clone())
        }
    }

    /// A conn whose handshake fails with a fixed error.
    pub struct FailConn(pub McpError);

    #[async_trait]
    impl McpConn for FailConn {
        async fn handshake(&self, _t: Option<&SecretValue>) -> Result<(String, ServerCaps), McpError> {
            Err(self.0.clone())
        }
        async fn list_tools(&self, _t: Option<&SecretValue>) -> Result<Vec<ToolInfo>, McpError> {
            Err(self.0.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::*;
    use super::*;

    fn tool(n: &str) -> ToolInfo {
        ToolInfo { name: n.into(), enabled: Some(true), domain: Some("plane".into()), description: None }
    }

    #[tokio::test]
    async fn compatible_server_connects_and_lists_tools() {
        let conn = OkConn {
            version: "2024-11-05".into(),
            caps: ServerCaps { tool_toggle: true, scope_edit: false },
            tools: vec![tool("plane_create_issue"), tool("gitea_list_repos")],
        };
        let rep = connect_and_report(&conn, None).await;
        assert!(rep.state.is_usable());
        assert_eq!(rep.tool_count, 2);
        assert!(rep.caps.tool_toggle);
    }

    /// NEGATIVE: an unexpected MCP version is shown incompatible, does NOT crash,
    /// and its tools are NOT trusted (empty inventory, not usable).
    #[tokio::test]
    async fn unexpected_version_is_incompatible_not_crash() {
        let conn = OkConn {
            version: "1999-01-01".into(),
            caps: ServerCaps { tool_toggle: true, scope_edit: true },
            tools: vec![tool("x")],
        };
        let rep = connect_and_report(&conn, None).await;
        assert!(matches!(rep.state, ConnState::Incompatible { .. }));
        assert!(!rep.state.is_usable());
        assert_eq!(rep.tool_count, 0, "tools from incompatible server are not trusted");
        assert!(!rep.caps.tool_toggle, "no caps trusted from incompatible server");
    }

    /// NEGATIVE: stdio process death → disconnected + retriable, isolated.
    #[tokio::test]
    async fn stdio_process_death_is_retriable_disconnected() {
        let conn = FailConn(McpError::ProcessDied("stdio child exited".into()));
        let rep = connect_and_report(&conn, None).await;
        match rep.state {
            ConnState::Disconnected { retriable, .. } => assert!(retriable),
            other => panic!("expected disconnected, got {other:?}"),
        }
    }

    /// NEGATIVE: remote auth failure is a per-instance error only.
    #[tokio::test]
    async fn remote_auth_fail_is_per_instance() {
        let conn = FailConn(McpError::Auth("401".into()));
        let rep = connect_and_report(&conn, None).await;
        assert!(matches!(rep.state, ConnState::AuthFailed { .. }));
        assert_eq!(rep.state.label(), "auth-failed");
    }

    /// NEGATIVE / isolation: one unreachable instance does not affect another.
    /// We report two independent conns and assert the good one is unaffected by
    /// the bad one's failure.
    #[tokio::test]
    async fn one_unreachable_instance_does_not_break_others() {
        let dead = FailConn(McpError::ProcessDied("gone".into()));
        let live = OkConn {
            version: "2025-03-26".into(),
            caps: ServerCaps::default(),
            tools: vec![tool("ok")],
        };
        let dead_rep = connect_and_report(&dead, None).await;
        let live_rep = connect_and_report(&live, None).await;
        assert!(matches!(dead_rep.state, ConnState::Disconnected { .. }));
        assert!(live_rep.state.is_usable(), "live instance unaffected by dead one");
        assert_eq!(live_rep.tool_count, 1);
    }
}
