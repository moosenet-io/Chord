//! Secret handling (S91 CTUI-01).
//!
//! HARD RULE: secret VALUES are never displayed, logged, or written to the
//! config file. The TUI only ever knows a secret by its *name/reference*; the
//! actual value is resolved at connection time from a [`SecretManager`] backed
//! by the vault (<secret-manager> / environment injected by the vault agent), never
//! from a literal stored in config.
//!
//! [`SecretRef`] is the only secret-shaped thing that is ever serialized. It
//! holds a reference (a vault key name), NOT a value. [`SecretValue`] wraps a
//! resolved value, redacts itself in every Debug/Display, and is deliberately
//! not `Serialize`, so it is a compile error to persist one.

use std::fmt;

use async_trait::async_trait;

/// A *reference* to a secret — a vault key name. This is the only secret-shaped
/// type that is ever serialized into config. It contains no secret material.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SecretRef(pub String);

impl SecretRef {
    pub fn new(name: impl Into<String>) -> Self {
        SecretRef(name.into())
    }
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// A resolved secret value. Redacts itself everywhere and is intentionally
/// **not** `Serialize` — persisting one is a compile error, which is how the
/// "secrets never written to config" invariant is enforced structurally.
#[derive(Clone)]
pub struct SecretValue(String);

impl SecretValue {
    pub fn new(v: impl Into<String>) -> Self {
        SecretValue(v.into())
    }
    /// Expose the raw value for the single legitimate use: putting it into an
    /// outbound `Authorization` header. Callers must never log the result.
    pub fn expose(&self) -> &str {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretValue(***redacted***)")
    }
}

impl fmt::Display for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***redacted***")
    }
}

/// Status of a secret without ever revealing its value — for display in a
/// secrets panel ("names/status only").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecretStatus {
    /// Vault has a non-empty value for this reference.
    Present,
    /// Vault knows the key but it resolves empty.
    Empty,
    /// Vault has no such key.
    Missing,
}

impl SecretStatus {
    pub fn label(self) -> &'static str {
        match self {
            SecretStatus::Present => "present",
            SecretStatus::Empty => "empty",
            SecretStatus::Missing => "missing",
        }
    }
}

/// Vault-backed secret resolver. Real deployments use an <secret-manager>/vault-backed
/// implementation; tests use [`EnvSecretManager`] over an in-memory map. No
/// secret literals ever live in config or code.
#[async_trait]
pub trait SecretManager: Send + Sync {
    /// Resolve a reference to its value, or `None` if the vault has no value.
    async fn resolve(&self, r: &SecretRef) -> Option<SecretValue>;

    /// Report presence/status WITHOUT returning the value (for display).
    async fn status(&self, r: &SecretRef) -> SecretStatus {
        match self.resolve(r).await {
            Some(v) if !v.is_empty() => SecretStatus::Present,
            Some(_) => SecretStatus::Empty,
            None => SecretStatus::Missing,
        }
    }
}

/// Vault-backed manager that reads from the process environment injected by the
/// vault agent at launch (never from literals baked into config). Used as the
/// default backend and in tests via [`with_map`].
#[derive(Default)]
pub struct EnvSecretManager {
    // Optional override map for tests; when empty, falls back to std::env.
    overrides: std::collections::HashMap<String, String>,
}

impl EnvSecretManager {
    pub fn from_env() -> Self {
        Self::default()
    }
    /// Test/inject constructor. NOT used to hold real secrets in production.
    pub fn with_map(m: std::collections::HashMap<String, String>) -> Self {
        Self { overrides: m }
    }
}

#[async_trait]
impl SecretManager for EnvSecretManager {
    async fn resolve(&self, r: &SecretRef) -> Option<SecretValue> {
        if let Some(v) = self.overrides.get(r.name()) {
            return Some(SecretValue::new(v.clone()));
        }
        std::env::var(r.name()).ok().map(SecretValue::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_value_redacts_in_debug_and_display() {
        let s = SecretValue::new("hunter2-super-secret");
        assert_eq!(format!("{s}"), "***redacted***");
        assert_eq!(format!("{s:?}"), "SecretValue(***redacted***)");
        assert!(!format!("{s:?}").contains("hunter2"));
        assert!(!format!("{s}").contains("hunter2"));
        // The only sanctioned reveal path:
        assert_eq!(s.expose(), "hunter2-super-secret");
    }

    #[tokio::test]
    async fn env_manager_status_never_returns_value() {
        let mut m = std::collections::HashMap::new();
        m.insert("TOKEN_A".to_string(), "abc".to_string());
        m.insert("TOKEN_B".to_string(), "".to_string());
        let mgr = EnvSecretManager::with_map(m);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_A")).await, SecretStatus::Present);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_B")).await, SecretStatus::Empty);
        assert_eq!(mgr.status(&SecretRef::new("TOKEN_C")).await, SecretStatus::Missing);
    }
}
