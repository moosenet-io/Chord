//! Chord control-plane client (S91 CTUI-02).
//!
//! Wraps ONLY Chord's existing, stable control endpoints:
//!   - `GET /health`                → liveness + version
//!   - `GET /api/models`            → registry: loaded/available + backend tag
//!   - `GET /api/models/:name`      → per-model detail
//!   - `GET /api/storage`           → disk usage summary
//!
//! Read-first. This client exposes NO mutating call in CTUI-02; the (gated)
//! pull/archive mutations live behind the confirm flow and are wired separately.
//! Missing/renamed fields degrade to `None` ("field unavailable") rather than
//! erroring the whole panel (API-drift tolerance). If the API exposes a
//! busy/mid-sweep flag we surface it so the UI can suppress disruptive actions.
//!
//! Auth token is resolved from the vault at call time and never logged.

use std::time::Duration;

use serde::Deserialize;

use crate::secret::SecretValue;

/// Errors talking to a Chord instance. Never carries secret material.
#[derive(Debug, thiserror::Error)]
pub enum ChordError {
    #[error("chord unreachable: {0}")]
    Unreachable(String),
    #[error("chord returned HTTP {0}")]
    Http(u16),
    #[error("malformed response: {0}")]
    Decode(String),
}

/// A model row as understood by the TUI. Every optional field degrades to
/// `None` when the API omits/renames it, so panels keep rendering.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelRow {
    pub name: String,
    /// hot/warm/cold tier, or None if the API stopped exposing it.
    pub tier: Option<String>,
    /// Whether the model is currently loaded/resident, if reported.
    pub loaded: Option<bool>,
    /// Backend tag (e.g. "llama.cpp-rocm", "ollama-rocm", "cpu",
    /// "vulkan-radv"), if reported.
    pub backend: Option<String>,
    pub size_bytes: Option<u64>,
    pub protected: Option<bool>,
}

/// The overall snapshot the Chord panels render from.
#[derive(Clone, Debug, Default)]
pub struct ChordSnapshot {
    pub version: Option<String>,
    pub models: Vec<ModelRow>,
    /// Distinct backends observed across models, for the backends panel.
    pub backends: Vec<BackendStatus>,
    pub storage: Option<StorageSummary>,
    /// True if Chord reports it is busy / mid-sweep — the UI must then suppress
    /// disruptive control actions.
    pub busy: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendStatus {
    pub name: String,
    /// Count of models currently loaded on this backend.
    pub loaded_models: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageSummary {
    pub local_used_bytes: Option<u64>,
    pub archive_used_bytes: Option<u64>,
    pub disk_pressure_percent: Option<u32>,
}

// ── Wire DTOs (lenient: everything optional so drift → None, not error) ────────

#[derive(Deserialize)]
struct HealthWire {
    #[serde(default)]
    version: Option<String>,
}

#[derive(Deserialize)]
struct ModelsWire {
    #[serde(default)]
    models: Vec<ModelWire>,
    /// Optional busy/mid-sweep flag (only present if the API exposes it).
    #[serde(default)]
    busy: Option<bool>,
    #[serde(default, rename = "sweep_active")]
    sweep_active: Option<bool>,
}

#[derive(Deserialize)]
struct ModelWire {
    name: String,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    loaded: Option<bool>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    protected: Option<bool>,
}

#[derive(Deserialize)]
struct StorageWire {
    #[serde(default)]
    local_used_bytes: Option<u64>,
    #[serde(default)]
    archive_used_bytes: Option<u64>,
    #[serde(default)]
    disk_pressure_percent: Option<u32>,
}

/// Parse the `/api/models` body into rows + busy flag. Pure + drift-tolerant,
/// factored out so it can be unit tested without a socket.
pub fn parse_models(body: &str) -> Result<(Vec<ModelRow>, bool), ChordError> {
    let wire: ModelsWire = serde_json::from_str(body).map_err(|e| ChordError::Decode(e.to_string()))?;
    let busy = wire.busy.or(wire.sweep_active).unwrap_or(false);
    let rows = wire
        .models
        .into_iter()
        .map(|m| ModelRow {
            name: m.name,
            tier: m.tier,
            loaded: m.loaded,
            backend: m.backend,
            size_bytes: m.size_bytes,
            protected: m.protected,
        })
        .collect();
    Ok((rows, busy))
}

/// Derive per-backend status from model rows (no dedicated endpoint needed;
/// backend attribution lives on each model in the stable API).
pub fn derive_backends(models: &[ModelRow]) -> Vec<BackendStatus> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for m in models {
        if let Some(b) = &m.backend {
            let entry = counts.entry(b.clone()).or_insert(0);
            if m.loaded == Some(true) {
                *entry += 1;
            } else {
                // ensure the backend appears even with zero loaded
                let _ = *entry;
            }
        }
    }
    counts
        .into_iter()
        .map(|(name, loaded_models)| BackendStatus { name, loaded_models })
        .collect()
}

/// Thin async client. Holds a reqwest client + base URL; token resolved by the
/// caller and passed per-request so rotations are picked up.
pub struct ChordClient {
    client: reqwest::Client,
    base_url: String,
    timeout: Duration,
}

impl ChordClient {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Self {
        ChordClient {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            timeout,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    async fn get(&self, path: &str, token: Option<&SecretValue>) -> Result<String, ChordError> {
        let mut req = self.client.get(self.url(path)).timeout(self.timeout);
        if let Some(t) = token {
            req = req.bearer_auth(t.expose());
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ChordError::Unreachable(if e.is_timeout() { "timeout".into() } else { "connect".into() }))?;
        let code = resp.status();
        if !code.is_success() {
            return Err(ChordError::Http(code.as_u16()));
        }
        resp.text().await.map_err(|e| ChordError::Decode(e.to_string()))
    }

    /// Fetch and assemble the full read-only snapshot. Individual sub-calls that
    /// fail with drift degrade gracefully rather than failing the whole fetch,
    /// except a hard-unreachable `/api/models` which is the primary signal.
    pub async fn fetch_snapshot(&self, token: Option<&SecretValue>) -> Result<ChordSnapshot, ChordError> {
        let mut snap = ChordSnapshot::default();

        // Health/version is best-effort.
        if let Ok(body) = self.get("/health", token).await {
            if let Ok(h) = serde_json::from_str::<HealthWire>(&body) {
                snap.version = h.version;
            }
        }

        // Models are the primary payload.
        let body = self.get("/api/models", token).await?;
        let (models, busy) = parse_models(&body)?;
        snap.busy = busy;
        snap.backends = derive_backends(&models);
        snap.models = models;

        // Storage is best-effort (drift → None).
        if let Ok(body) = self.get("/api/storage", token).await {
            if let Ok(s) = serde_json::from_str::<StorageWire>(&body) {
                snap.storage = Some(StorageSummary {
                    local_used_bytes: s.local_used_bytes,
                    archive_used_bytes: s.archive_used_bytes,
                    disk_pressure_percent: s.disk_pressure_percent,
                });
            }
        }

        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_models_payload() {
        let body = r#"{
            "models": [
                {"name":"llama3.3:70b","tier":"warm","loaded":true,"backend":"vulkan-radv","size_bytes":42000000000,"protected":false},
                {"name":"qwen3-coder:30b","tier":"hot","loaded":true,"backend":"llama.cpp-rocm"}
            ],
            "busy": true
        }"#;
        let (rows, busy) = parse_models(body).unwrap();
        assert!(busy, "mid-sweep busy flag surfaced");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].backend.as_deref(), Some("vulkan-radv"));
        assert_eq!(rows[1].tier.as_deref(), Some("hot"));
        // Missing fields on row 2 degrade to None, not error.
        assert_eq!(rows[1].size_bytes, None);
        assert_eq!(rows[1].protected, None);
    }

    /// API-drift: a renamed/removed field must degrade to "field unavailable"
    /// (None), never break parsing of the panel.
    #[test]
    fn missing_fields_degrade_gracefully() {
        let body = r#"{"models":[{"name":"solo-model"}]}"#;
        let (rows, busy) = parse_models(body).unwrap();
        assert!(!busy, "absent busy flag defaults to not-busy");
        assert_eq!(rows[0].name, "solo-model");
        assert_eq!(rows[0].tier, None);
        assert_eq!(rows[0].loaded, None);
        assert_eq!(rows[0].backend, None);
    }

    #[test]
    fn alternate_sweep_flag_is_honored() {
        let body = r#"{"models":[],"sweep_active":true}"#;
        let (_rows, busy) = parse_models(body).unwrap();
        assert!(busy, "sweep_active alias respected for mid-sweep state");
    }

    #[test]
    fn derives_backends_from_models() {
        let rows = vec![
            ModelRow { name: "a".into(), backend: Some("cpu".into()), loaded: Some(true), ..Default::default() },
            ModelRow { name: "b".into(), backend: Some("cpu".into()), loaded: Some(false), ..Default::default() },
            ModelRow { name: "c".into(), backend: Some("vulkan-radv".into()), loaded: Some(true), ..Default::default() },
        ];
        let backends = derive_backends(&rows);
        assert_eq!(backends.len(), 2);
        let cpu = backends.iter().find(|b| b.name == "cpu").unwrap();
        assert_eq!(cpu.loaded_models, 1, "only loaded models counted");
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse_models("not json {{").is_err());
    }
}
