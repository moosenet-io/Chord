//! Mutation confirmation gating (S91 control-plane safety).
//!
//! The TUI is read-mostly. Any state-changing action must pass through this
//! gate:
//!   - **Simple mutations** (e.g. pull/archive a model) require an explicit
//!     confirm keystroke (`y`).
//!   - **DESTRUCTIVE mutations** (unload a live model, clean-swap, change a
//!     secret) require a TYPED confirmation: the operator must type the action's
//!     exact challenge phrase, not a single key.
//!
//! Separately, stubbed / not-yet-wired mutations (S85 seam) are gated by a
//! runtime flag ([`crate::config::Settings::enable_stubbed_mutations`]) AND a
//! compile-time guard, so they stay INERT until S85 is verified.

/// Severity of a mutation, deciding how confirmation must be given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Explicit single-key confirm (`y`) is enough.
    Simple,
    /// Requires typing the exact challenge phrase.
    Destructive { challenge: String },
}

/// A pending mutation awaiting operator confirmation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingMutation {
    /// Stable id of the action (e.g. "chord.model.pull").
    pub action: String,
    /// Human description shown in the confirm prompt.
    pub description: String,
    pub severity: Severity,
    /// True if this action is a not-yet-wired S85 stub (inert unless the flag
    /// is on).
    pub is_stub: bool,
}

impl PendingMutation {
    pub fn simple(action: impl Into<String>, description: impl Into<String>) -> Self {
        PendingMutation {
            action: action.into(),
            description: description.into(),
            severity: Severity::Simple,
            is_stub: false,
        }
    }

    pub fn destructive(
        action: impl Into<String>,
        description: impl Into<String>,
        challenge: impl Into<String>,
    ) -> Self {
        PendingMutation {
            action: action.into(),
            description: description.into(),
            severity: Severity::Destructive { challenge: challenge.into() },
            is_stub: false,
        }
    }

    /// Mark this mutation as an S85 stub (inert unless the flag enables it).
    pub fn as_stub(mut self) -> Self {
        self.is_stub = true;
        self
    }

    /// Does a single explicit confirm keystroke satisfy this mutation?
    /// True only for [`Severity::Simple`].
    pub fn satisfied_by_keystroke(&self, key: char) -> bool {
        matches!(self.severity, Severity::Simple) && (key == 'y' || key == 'Y')
    }

    /// Does the typed input satisfy a destructive confirmation? Requires an
    /// exact match of the challenge phrase. Simple mutations are NOT satisfied
    /// by typing (they use the keystroke path).
    pub fn satisfied_by_typed(&self, input: &str) -> bool {
        match &self.severity {
            Severity::Destructive { challenge } => input == challenge,
            Severity::Simple => false,
        }
    }
}

/// The result of attempting to execute a confirmed mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecOutcome {
    /// A real op would run (only reached for non-stub, confirmed actions).
    Executed,
    /// A stubbed action whose flag is OFF: NO real operation performed.
    InertStub,
    /// Confirmation was not satisfied; nothing ran.
    Rejected,
}

/// Decide the outcome for a confirmed mutation given the stub-enable flag. This
/// is the single choke-point proving stubbed actions are inert when the flag is
/// off. It performs no I/O — it only classifies.
pub fn resolve_execution(m: &PendingMutation, stub_flag_enabled: bool) -> ExecOutcome {
    if m.is_stub && !stub_flag_enabled {
        // INERT: S85 not wired / flag off → never a real op.
        ExecOutcome::InertStub
    } else {
        ExecOutcome::Executed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_mutation_requires_confirm_keystroke() {
        let m = PendingMutation::simple("chord.model.pull", "Pull cold model");
        // A non-confirm key does nothing.
        assert!(!m.satisfied_by_keystroke('n'));
        assert!(!m.satisfied_by_keystroke('\n'));
        // The explicit confirm keystroke satisfies it.
        assert!(m.satisfied_by_keystroke('y'));
        assert!(m.satisfied_by_keystroke('Y'));
        // Typing does NOT satisfy a simple mutation.
        assert!(!m.satisfied_by_typed("y"));
    }

    #[test]
    fn destructive_mutation_requires_typed_confirmation() {
        let m = PendingMutation::destructive(
            "chord.cleanswap",
            "Clean-swap live model",
            "CLEAN-SWAP",
        );
        // A single keystroke is NOT enough for destructive.
        assert!(!m.satisfied_by_keystroke('y'));
        // Wrong phrase rejected.
        assert!(!m.satisfied_by_typed("clean-swap"));
        assert!(!m.satisfied_by_typed("CLEANSWAP"));
        // Exact challenge phrase accepted.
        assert!(m.satisfied_by_typed("CLEAN-SWAP"));
    }

    /// NEGATIVE TEST: a stubbed clean-swap performs NO real op while the flag is
    /// off, even after a valid typed confirmation.
    #[test]
    fn stubbed_cleanswap_is_inert_when_flag_off() {
        let m = PendingMutation::destructive(
            "chord.cleanswap",
            "Clean-swap (pending S85)",
            "CLEAN-SWAP",
        )
        .as_stub();
        assert!(m.satisfied_by_typed("CLEAN-SWAP"), "confirmation UX still works");
        // Flag OFF → inert, no real operation.
        assert_eq!(resolve_execution(&m, false), ExecOutcome::InertStub);
        // Flag ON → would execute (real wiring lands in S85).
        assert_eq!(resolve_execution(&m, true), ExecOutcome::Executed);
    }

    #[test]
    fn non_stub_confirmed_mutation_executes() {
        let m = PendingMutation::simple("chord.model.pull", "Pull cold model");
        assert_eq!(resolve_execution(&m, false), ExecOutcome::Executed);
    }
}
