//! Terminus fleet model (S91 CTUI-04).
//!
//! The fleet is the operator's list of Terminus MCP instances. Each instance
//! declares:
//!   - a [`Transport`] (stdio or HTTP),
//!   - an endpoint (a URL for HTTP, or a command for stdio) — supplied by the
//!     operator via config, NEVER a hardcoded literal,
//!   - a [`FleetKind`] (local | remote | chord-embedded).
//!
//! A chord-embedded Terminus is just another fleet instance (kind = embedded).
//!
//! CRITICAL isolation invariant: one unreachable instance must NOT break the
//! others. The fleet stores config only; all I/O is per-instance in
//! [`super::mcp_client`], and every per-instance status is independent.
//!
//! Per-instance auth tokens are resolved from the vault at connect time via a
//! [`crate::secret::SecretRef`]; they are NEVER stored in the fleet or config as
//! literals.

use serde::{Deserialize, Serialize};

use crate::secret::SecretRef;

/// How the TUI reaches a Terminus MCP server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "transport")]
pub enum Transport {
    /// Spawn a local process and speak MCP over its stdin/stdout. `command` is
    /// the operator-supplied launcher (e.g. a wrapper script path); NOT a
    /// hardcoded infra value.
    Stdio {
        /// Launcher command (path/argv0) from config.
        command: String,
        /// Optional argv, from config.
        #[serde(default)]
        args: Vec<String>,
    },
    /// Connect over HTTP to an MCP endpoint. `endpoint` is operator-supplied.
    Http {
        /// Base URL of the MCP endpoint from config (never a literal here).
        endpoint: String,
    },
}

impl Transport {
    /// Short label for the transport column.
    pub fn label(&self) -> &'static str {
        match self {
            Transport::Stdio { .. } => "stdio",
            Transport::Http { .. } => "http",
        }
    }

    /// A display-safe endpoint string (command or URL). Carries no secret.
    pub fn endpoint_display(&self) -> String {
        match self {
            Transport::Stdio { command, args } => {
                if args.is_empty() {
                    command.clone()
                } else {
                    format!("{command} {}", args.join(" "))
                }
            }
            Transport::Http { endpoint } => endpoint.clone(),
        }
    }
}

/// Where a fleet instance lives, for grouping/labeling in the UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FleetKind {
    /// A Terminus running on the operator's own host.
    Local,
    /// A Terminus on another host, reached over HTTP.
    Remote,
    /// A Terminus embedded inside a Chord process — just another fleet instance.
    ChordEmbedded,
}

impl FleetKind {
    pub fn label(self) -> &'static str {
        match self {
            FleetKind::Local => "local",
            FleetKind::Remote => "remote",
            FleetKind::ChordEmbedded => "chord-embedded",
        }
    }
}

/// One configured Terminus MCP instance. Contains NO secret value — only a
/// [`SecretRef`] naming a vault key resolved at connect time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetInstance {
    /// Operator-facing label. May collide with another instance's name — the UI
    /// disambiguates by name + endpoint (see [`FleetInstance::disambiguated`]).
    pub name: String,
    pub kind: FleetKind,
    pub transport: Transport,
    /// Vault key name for this instance's auth token. `None` → unauthenticated
    /// (e.g. a local stdio dev instance). NEVER a literal token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_secret_ref: Option<SecretRef>,
}

impl FleetInstance {
    /// A stable identity that disambiguates same-name instances by pairing the
    /// name with its endpoint. Two instances named "terminus" pointing at
    /// different endpoints get distinct identities.
    pub fn disambiguated(&self) -> String {
        format!("{} @ {}", self.name, self.transport.endpoint_display())
    }
}

/// The fleet: an ordered list of instances plus a selection cursor. Pure state;
/// no I/O. Isolation is structural — the fleet never performs a network call, so
/// a dead instance cannot affect operations on another.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fleet {
    instances: Vec<FleetInstance>,
    selected: usize,
}

impl Fleet {
    pub fn new(instances: Vec<FleetInstance>) -> Self {
        Fleet { instances, selected: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn instances(&self) -> &[FleetInstance] {
        &self.instances
    }

    /// Add an instance. Same-name instances are permitted (disambiguated by
    /// endpoint); an exact duplicate (same name AND endpoint) is rejected so the
    /// list stays unambiguous.
    pub fn add(&mut self, inst: FleetInstance) -> Result<(), String> {
        let id = inst.disambiguated();
        if self.instances.iter().any(|i| i.disambiguated() == id) {
            return Err(format!("instance '{id}' already exists"));
        }
        self.instances.push(inst);
        Ok(())
    }

    /// Remove the instance at `idx`. Fixes up the selection cursor. Returns the
    /// removed instance, or `None` if out of range.
    pub fn remove(&mut self, idx: usize) -> Option<FleetInstance> {
        if idx >= self.instances.len() {
            return None;
        }
        let removed = self.instances.remove(idx);
        if self.selected >= self.instances.len() {
            self.selected = self.instances.len().saturating_sub(1);
        }
        Some(removed)
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn selected(&self) -> Option<&FleetInstance> {
        self.instances.get(self.selected)
    }

    pub fn select_next(&mut self) {
        if !self.instances.is_empty() {
            self.selected = (self.selected + 1) % self.instances.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.instances.is_empty() {
            self.selected = (self.selected + self.instances.len() - 1) % self.instances.len();
        }
    }
}

// ── Fleet-mode panels ─────────────────────────────────────────────────────────

/// Which fleet-mode panel is focused. CTUI-04 adds per-instance detail; CTUI-05
/// adds the tools/secrets/transport config panels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FleetPanel {
    /// The fleet list (health from the shared connection manager).
    Instances,
    /// Per-instance status + tool inventory (CTUI-04).
    Detail,
    /// Per-instance tool enable/disable + scopes (CTUI-05).
    Tools,
    /// Per-instance vault-backed secret management — names/status only (CTUI-05).
    Secrets,
    /// Per-instance transport config (CTUI-05).
    Transport,
}

impl FleetPanel {
    pub const ALL: [FleetPanel; 5] = [
        FleetPanel::Instances,
        FleetPanel::Detail,
        FleetPanel::Tools,
        FleetPanel::Secrets,
        FleetPanel::Transport,
    ];

    pub fn title(self) -> &'static str {
        match self {
            FleetPanel::Instances => "Instances",
            FleetPanel::Detail => "Detail",
            FleetPanel::Tools => "Tools",
            FleetPanel::Secrets => "Secrets",
            FleetPanel::Transport => "Transport",
        }
    }

    pub fn next(self) -> FleetPanel {
        let all = Self::ALL;
        let i = all.iter().position(|p| *p == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }

    pub fn prev(self) -> FleetPanel {
        let all = Self::ALL;
        let i = all.iter().position(|p| *p == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

/// Short scaffold description shown in the fleet pane when empty.
pub const FLEET_SCAFFOLD_NOTE: &str =
    "Terminus-fleet mode — shares plumbing with Chord mode; add instances with 'a'.";

#[cfg(test)]
mod tests {
    use super::*;

    fn http(name: &str, endpoint: &str) -> FleetInstance {
        FleetInstance {
            name: name.into(),
            kind: FleetKind::Remote,
            transport: Transport::Http { endpoint: endpoint.into() },
            auth_secret_ref: None,
        }
    }

    #[test]
    fn fleet_panel_has_title_and_cycles() {
        assert_eq!(FleetPanel::Instances.title(), "Instances");
        assert_eq!(FleetPanel::Instances.next(), FleetPanel::Detail);
        assert_eq!(FleetPanel::Instances.prev(), FleetPanel::Transport);
    }

    #[test]
    fn add_select_remove_roundtrip() {
        let mut f = Fleet::default();
        f.add(http("a", "http://host-a.invalid/mcp")).unwrap();
        f.add(http("b", "http://host-b.invalid/mcp")).unwrap();
        assert_eq!(f.len(), 2);
        f.select_next();
        assert_eq!(f.selected().unwrap().name, "b");
        f.select_next(); // wraps
        assert_eq!(f.selected().unwrap().name, "a");
        let removed = f.remove(0).unwrap();
        assert_eq!(removed.name, "a");
        assert_eq!(f.len(), 1);
        assert_eq!(f.selected().unwrap().name, "b");
    }

    /// Same-name instances are allowed and disambiguated by endpoint; an exact
    /// duplicate (name + endpoint) is rejected.
    #[test]
    fn same_name_instances_disambiguate_by_endpoint() {
        let mut f = Fleet::default();
        f.add(http("terminus", "http://host-1.invalid/mcp")).unwrap();
        // Same name, different endpoint → allowed and distinguishable.
        f.add(http("terminus", "http://host-2.invalid/mcp")).unwrap();
        assert_eq!(f.len(), 2);
        assert_ne!(
            f.instances()[0].disambiguated(),
            f.instances()[1].disambiguated()
        );
        // Exact duplicate rejected.
        let err = f.add(http("terminus", "http://host-1.invalid/mcp")).unwrap_err();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn chord_embedded_is_just_another_instance() {
        let mut f = Fleet::default();
        f.add(FleetInstance {
            name: "embedded".into(),
            kind: FleetKind::ChordEmbedded,
            transport: Transport::Http { endpoint: "http://chord.invalid/mcp".into() },
            auth_secret_ref: None,
        })
        .unwrap();
        assert_eq!(f.instances()[0].kind, FleetKind::ChordEmbedded);
        assert_eq!(f.instances()[0].kind.label(), "chord-embedded");
    }

    /// NEGATIVE / invariant: the fleet holds config only and never a secret
    /// value — only a SecretRef name is serializable.
    #[test]
    fn instance_serializes_ref_not_value() {
        let inst = FleetInstance {
            name: "auth".into(),
            kind: FleetKind::Remote,
            transport: Transport::Http { endpoint: "http://remote.invalid/mcp".into() },
            auth_secret_ref: Some(SecretRef::new("TERMINUS_REMOTE_TOKEN")),
        };
        let toml = toml::to_string(&inst).unwrap();
        assert!(toml.contains("TERMINUS_REMOTE_TOKEN"), "ref name persisted");
        assert!(!toml.to_lowercase().contains("bearer"), "no token literal");
    }

    #[test]
    fn stdio_and_http_endpoint_display() {
        let s = Transport::Stdio { command: "/opt/terminus/stdio.sh".into(), args: vec!["--fleet".into()] };
        assert_eq!(s.label(), "stdio");
        assert_eq!(s.endpoint_display(), "/opt/terminus/stdio.sh --fleet");
        let h = Transport::Http { endpoint: "http://x.invalid/mcp".into() };
        assert_eq!(h.label(), "http");
        assert_eq!(h.endpoint_display(), "http://x.invalid/mcp");
    }
}
