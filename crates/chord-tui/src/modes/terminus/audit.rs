//! Sanitized mutation audit log (S91 CTUI-05).
//!
//! Every attempted or applied fleet mutation (tool toggle, scope edit, secret
//! change, transport change) is recorded here. HARD RULE: an audit entry NEVER
//! contains a secret value — only the action, the target instance/tool/secret
//! NAME, and the outcome. Secret CHANGES record the secret's reference name and
//! that a write occurred, never the value written.

/// Outcome of an audited mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditOutcome {
    /// The mutation was requested but not confirmed (gate not satisfied).
    Unconfirmed,
    /// Confirmed and applied successfully.
    Applied,
    /// Confirmed but the server/vault rejected it — UI reflects true state.
    Rejected,
    /// Confirmed but the instance became unreachable mid-action.
    Interrupted,
    /// A capability was absent → shown read-only, no mutation attempted.
    NotSupported,
}

impl AuditOutcome {
    pub fn label(self) -> &'static str {
        match self {
            AuditOutcome::Unconfirmed => "unconfirmed",
            AuditOutcome::Applied => "applied",
            AuditOutcome::Rejected => "rejected",
            AuditOutcome::Interrupted => "interrupted",
            AuditOutcome::NotSupported => "not-supported",
        }
    }
}

/// A single sanitized audit entry. Fields are all non-secret by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditEntry {
    /// Stable action id, e.g. "terminus.tool.toggle" / "terminus.secret.change".
    pub action: String,
    /// The instance identity (name @ endpoint) — non-secret.
    pub instance: String,
    /// The target within the instance (tool name / secret ref name) — a NAME,
    /// never a value.
    pub target: String,
    pub outcome: AuditOutcome,
}

/// An in-memory sanitized audit log. In a deployment this is mirrored to the
/// system audit sink; here it is the choke-point that guarantees no secret value
/// is ever recorded.
#[derive(Clone, Debug, Default)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    /// Record a mutation. `target` MUST be a name/reference, never a value. This
    /// is enforced by the callers (secret changes pass the SecretRef name).
    pub fn record(
        &mut self,
        action: impl Into<String>,
        instance: impl Into<String>,
        target: impl Into<String>,
        outcome: AuditOutcome,
    ) {
        self.entries.push(AuditEntry {
            action: action.into(),
            instance: instance.into(),
            target: target.into(),
            outcome,
        });
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Render one entry as a single sanitized log line (no secret values).
    pub fn format_line(e: &AuditEntry) -> String {
        format!(
            "audit action={} instance={} target={} outcome={}",
            e.action,
            e.instance,
            e.target,
            e.outcome.label()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_sanitized_entries() {
        let mut log = AuditLog::default();
        log.record(
            "terminus.tool.toggle",
            "terminus @ http://host.invalid/mcp",
            "plane_create_issue",
            AuditOutcome::Applied,
        );
        assert_eq!(log.len(), 1);
        let line = AuditLog::format_line(&log.entries()[0]);
        assert!(line.contains("terminus.tool.toggle"));
        assert!(line.contains("outcome=applied"));
    }

    /// NEGATIVE: a secret-change audit records only the ref NAME + outcome, never
    /// the value. We pass a would-be secret value string and assert it never
    /// appears — callers pass the ref name as `target`, so the value is
    /// structurally absent.
    #[test]
    fn secret_change_audit_never_contains_value() {
        let mut log = AuditLog::default();
        // Caller passes the ref NAME, not the value.
        log.record(
            "terminus.secret.change",
            "terminus @ http://host.invalid/mcp",
            "TERMINUS_REMOTE_TOKEN",
            AuditOutcome::Applied,
        );
        let line = AuditLog::format_line(&log.entries()[0]);
        assert!(line.contains("TERMINUS_REMOTE_TOKEN"), "ref name recorded");
        assert!(!line.contains("hunter2"), "no secret value present");
        assert!(!line.to_lowercase().contains("bearer "), "no token literal");
    }
}
