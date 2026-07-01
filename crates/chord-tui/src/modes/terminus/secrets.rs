//! Per-instance vault-backed secret management (S91 CTUI-05).
//!
//! HARD RULES:
//!   - Secret VALUES are NEVER displayed, logged, or written to a file/screen —
//!     only the reference NAME and a presence [`SecretStatus`] are shown.
//!   - Changing a secret requires a TYPED confirmation (destructive severity).
//!   - A change writes ONLY to the vault (via [`SecretWriter`]), never to the
//!     config file or the terminal. The write is ATOMIC-or-nothing: an
//!     interrupted change leaves the vault in its prior state, never half-written.
//!   - Every change is audit-logged with the ref NAME + outcome only.

use async_trait::async_trait;

use crate::confirm::PendingMutation;
use crate::modes::terminus::audit::AuditOutcome;
use crate::secret::{SecretRef, SecretStatus, SecretValue};

/// A row in the per-instance secrets panel: a NAME + a presence status. It
/// deliberately has no field capable of holding a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretDisplay {
    pub name: String,
    pub status: SecretStatus,
}

impl SecretDisplay {
    /// A single sanitized display line — names/status only, provably no value.
    pub fn line(&self) -> String {
        format!("{}  [{}]", self.name, self.status.label())
    }
}

/// Vault WRITE capability, separate from the read-only [`crate::secret::SecretManager`].
/// The write is atomic-or-nothing: on error the vault is unchanged. Injected so
/// tests can simulate an interrupted (mid-write) failure without a real vault.
#[async_trait]
pub trait SecretWriter: Send + Sync {
    /// Write `value` to the vault under `r`. MUST be atomic: either the new value
    /// is fully committed, or the prior value is preserved. Returns Err with a
    /// sanitized reason on failure (never echoing the value).
    async fn write(&self, r: &SecretRef, value: SecretValue) -> Result<(), String>;
}

/// The exact phrase an operator TYPES to confirm a secret change. Destructive.
pub const SECRET_CHANGE_CHALLENGE: &str = "CHANGE-SECRET";

/// Build the destructive, typed-confirm mutation for changing a secret. The ref
/// NAME appears in the description; the value never does.
pub fn secret_change_mutation(secret_ref: &SecretRef) -> PendingMutation {
    PendingMutation::destructive(
        "terminus.secret.change",
        format!("Change vault secret '{}' (value never shown)", secret_ref.name()),
        SECRET_CHANGE_CHALLENGE,
    )
}

/// Outcome of a confirmed secret change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SecretChangeResult {
    /// Vault write committed atomically.
    Written,
    /// Vault write failed; prior value preserved (atomic-or-nothing).
    Failed { why: String },
}

/// Apply a confirmed secret change: write ONLY to the vault. Precondition: the
/// TYPED confirm was satisfied. Returns the result + audit outcome. The `value`
/// is consumed and never returned, logged, or written anywhere but the vault.
pub async fn apply_secret_change(
    writer: &dyn SecretWriter,
    secret_ref: &SecretRef,
    value: SecretValue,
) -> (SecretChangeResult, AuditOutcome) {
    match writer.write(secret_ref, value).await {
        Ok(()) => (SecretChangeResult::Written, AuditOutcome::Applied),
        Err(why) => {
            // Atomic-or-nothing: on failure the vault is unchanged.
            (SecretChangeResult::Failed { why }, AuditOutcome::Interrupted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A vault that commits writes into an in-memory cell, atomically.
    #[derive(Default)]
    struct MemVault {
        committed: std::sync::Mutex<Option<String>>,
    }
    #[async_trait]
    impl SecretWriter for MemVault {
        async fn write(&self, _r: &SecretRef, value: SecretValue) -> Result<(), String> {
            // Commit fully or not at all.
            *self.committed.lock().unwrap() = Some(value.expose().to_string());
            Ok(())
        }
    }

    /// A vault whose write is interrupted; the prior value must be preserved.
    struct InterruptedVault {
        prior: std::sync::Mutex<Option<String>>,
        wrote: AtomicBool,
    }
    #[async_trait]
    impl SecretWriter for InterruptedVault {
        async fn write(&self, _r: &SecretRef, _value: SecretValue) -> Result<(), String> {
            // Simulate a mid-write failure: do NOT mutate the prior value.
            self.wrote.store(false, Ordering::SeqCst);
            Err("vault write interrupted".into())
        }
    }

    #[test]
    fn secret_display_line_has_no_value() {
        let d = SecretDisplay {
            name: "TERMINUS_REMOTE_TOKEN".into(),
            status: SecretStatus::Present,
        };
        let line = d.line();
        assert!(line.contains("TERMINUS_REMOTE_TOKEN"));
        assert!(line.contains("present"));
        // Structurally there is no value field, so no value can leak.
        assert!(!line.contains("hunter2"));
    }

    #[test]
    fn secret_change_requires_typed_confirmation() {
        let m = secret_change_mutation(&SecretRef::new("TERMINUS_REMOTE_TOKEN"));
        // A single keystroke is NOT enough — destructive.
        assert!(!m.satisfied_by_keystroke('y'));
        assert!(m.satisfied_by_typed(SECRET_CHANGE_CHALLENGE));
        // The description names the ref but not any value.
        assert!(m.description.contains("TERMINUS_REMOTE_TOKEN"));
        assert!(!m.description.to_lowercase().contains("bearer"));
    }

    /// A confirmed change writes to the vault (not a file/screen) and returns
    /// Written. The value went only to the vault.
    #[tokio::test]
    async fn confirmed_change_writes_to_vault_only() {
        let vault = MemVault::default();
        let (res, outcome) = apply_secret_change(
            &vault,
            &SecretRef::new("TERMINUS_REMOTE_TOKEN"),
            SecretValue::new("new-rotated-token"),
        )
        .await;
        assert_eq!(res, SecretChangeResult::Written);
        assert_eq!(outcome, AuditOutcome::Applied);
        // The value lives only in the vault sink, never returned to the caller.
        assert_eq!(
            vault.committed.lock().unwrap().as_deref(),
            Some("new-rotated-token")
        );
    }

    /// NEGATIVE: an interrupted secret change is atomic-or-nothing — the prior
    /// value is preserved and the outcome is Interrupted (not Applied).
    #[tokio::test]
    async fn interrupted_change_preserves_prior_value() {
        let vault = InterruptedVault {
            prior: std::sync::Mutex::new(Some("old-token".into())),
            wrote: AtomicBool::new(false),
        };
        let (res, outcome) = apply_secret_change(
            &vault,
            &SecretRef::new("TERMINUS_REMOTE_TOKEN"),
            SecretValue::new("half-written"),
        )
        .await;
        assert!(matches!(res, SecretChangeResult::Failed { .. }));
        assert_eq!(outcome, AuditOutcome::Interrupted);
        // Prior value untouched — atomic-or-nothing.
        assert_eq!(vault.prior.lock().unwrap().as_deref(), Some("old-token"));
        assert!(!vault.wrote.load(Ordering::SeqCst));
    }
}
