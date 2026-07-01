//! S85 clean-swap panel (S91 CTUI-03) — STUBBED, pending S85.
//!
//! Clean-swap is the most DESTRUCTIVE control action (tears down + reloads a
//! live model). This module builds the confirm-gated action for it. Until S85 is
//! wired, executing it is INERT (no real op) via the mock in `serving.rs` and the
//! `enable_stubbed_mutations` flag.

use crate::confirm::PendingMutation;

/// The exact phrase an operator must TYPE to confirm a clean-swap. Chosen to be
/// hard to hit accidentally.
pub const CLEAN_SWAP_CHALLENGE: &str = "CLEAN-SWAP";

/// Build the destructive, stubbed clean-swap mutation for `model`. It is:
///   - destructive → requires typed confirmation of [`CLEAN_SWAP_CHALLENGE`],
///   - a stub      → inert unless `enable_stubbed_mutations` is on.
pub fn clean_swap_mutation(model: &str) -> PendingMutation {
    PendingMutation::destructive(
        "chord.cleanswap",
        format!("Clean-swap live model '{model}' (pending S85 — inert)"),
        CLEAN_SWAP_CHALLENGE,
    )
    .as_stub()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::confirm::{resolve_execution, ExecOutcome, Severity};

    #[test]
    fn clean_swap_is_destructive_and_stubbed() {
        let m = clean_swap_mutation("qwen3-coder:30b");
        assert!(m.is_stub);
        assert!(matches!(m.severity, Severity::Destructive { .. }));
        // Single keystroke is NOT enough.
        assert!(!m.satisfied_by_keystroke('y'));
        // Typed challenge required.
        assert!(m.satisfied_by_typed(CLEAN_SWAP_CHALLENGE));
    }

    /// NEGATIVE TEST: even correctly typed, a stubbed clean-swap is inert with
    /// the flag off — no real operation.
    #[test]
    fn typed_clean_swap_still_inert_when_flag_off() {
        let m = clean_swap_mutation("qwen3-coder:30b");
        assert!(m.satisfied_by_typed(CLEAN_SWAP_CHALLENGE));
        assert_eq!(resolve_execution(&m, false), ExecOutcome::InertStub);
    }
}
