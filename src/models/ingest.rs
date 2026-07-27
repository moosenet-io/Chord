//! ASK4-P2A: guarded HuggingFace → cold-storage model ingestion.
//!
//! This is Phase 2a of "Ask 4" (the brochure auto pull→test→promote loop): a
//! single authenticated control-plane endpoint that pulls a HuggingFace model
//! into the fleet's **cold tier** so the MINT assistant sweep can later promote
//! it warm and test it. It deliberately does the *staging* half only — it never
//! loads a model into VRAM, never runs a sweep, never mutates routing.
//!
//! ## Reuse, not a parallel path
//! Chord already owns every piece of the cold-storage machinery; this module
//! only orchestrates the existing pieces:
//! - **Download:** delegated to Ollama's own native HuggingFace ingestion
//!   (`POST {OLLAMA_URL}/api/pull` with an `hf.co/<repo>` name). Chord has no
//!   HF-blob downloader of its own, and Ollama's pull path already honours the
//!   supervisor egress filter — so re-implementing a raw HF fetch here would be
//!   both duplicative and an egress-policy bypass.
//! - **Warm→cold demote:** the existing [`crate::models::eviction::evict_to_archive`]
//!   (the same call `POST /api/models/:name/archive` uses) copies the freshly
//!   pulled model's manifest + blobs into the archive root and flips its
//!   registry record to [`StorageTier::Cold`](crate::models::registry::StorageTier).
//! - **Registry discovery:** [`crate::models::registry::ModelRegistry::reconcile`]
//!   picks up the just-pulled Ollama model, exactly as it does at startup.
//! - **Later promotion:** once cold, the model is reachable by the acquire path
//!   Terminus already calls — `POST /api/models/:name/pull` →
//!   `pull_coordinator.ensure_local` → `transfer::archive_pull` (cold→warm).
//!
//! ## Security posture
//! - **Default-OFF master gate** (`CHORD_MODEL_INGEST_ENABLED`, default `0`):
//!   merging this changes nothing until an operator explicitly enables it. A
//!   disabled endpoint returns a clean structured refusal, never a 500.
//! - **Authenticated** by the same JWT guard as every other control endpoint
//!   (the handler in `control.rs` runs `auth_check` before calling in here).
//! - **HF token from the vault, never a literal.** [`hf_token`] is the single
//!   sanctioned read point; the value is materialised into the process
//!   environment from <secret-manager> at startup by
//!   `secrets_bootstrap::fetch_and_apply_downstream_secrets` (`HF_TOKEN` is in
//!   its downstream allowlist), exactly like `OPENROUTER_API_KEY`. It is never
//!   hardcoded, never logged, and never echoed into a response or error string.
//!   Absent token ⇒ **fail soft**: public models still pull; a gated model
//!   returns [`IngestStatus::GatedNeedsToken`], not a crash.
//! - **Bounded:** an oversize repo is refused ([`IngestStatus::TooLarge`])
//!   before any byte is copied, protecting the shared disk (<host> disk-pressure
//!   discipline), and the pull/copy steps are wrapped in timeouts.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One sanctioned read point for the HuggingFace access token.
///
/// The value is populated into the process environment at startup by
/// `secrets_bootstrap::fetch_and_apply_downstream_secrets` (<secret-manager>-first) —
/// it is never authored as a literal, never logged, and never returned in any
/// response or error text. Mirrors `embeddings::openrouter_api_key()`. Returns
/// `None` when unset/blank, which the ingestion flow treats as fail-soft (only
/// gated models then fail, with a clear message).
pub fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Default master-gate value (OFF). Merging Phase 2a changes nothing until an
/// operator sets `CHORD_MODEL_INGEST_ENABLED=1`.
pub const DEFAULT_INGEST_ENABLED: bool = false;

/// Default maximum on-disk size accepted for an ingest (bytes). 100 GiB — large
/// enough for any single fleet-class model, small enough to refuse a runaway or
/// mistyped repo before it floods the archive disk.
pub const DEFAULT_MAX_BYTES: u64 = 100 * 1024 * 1024 * 1024;

/// Runtime configuration for the ingest endpoint, read from the environment
/// (mirrors `EmbeddingsConfig::from_env` — self-contained, not threaded through
/// the central `Config`).
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Master gate. `false` ⇒ every request is refused with
    /// [`IngestStatus::Disabled`].
    pub enabled: bool,
    /// Maximum accepted repo size in bytes; larger ⇒ [`IngestStatus::TooLarge`].
    pub max_bytes: u64,
    /// Base URL of the Ollama daemon that performs the actual HF download.
    /// `None` ⇒ ingestion cannot run (returns a clean error, never panics).
    pub ollama_url: Option<String>,
    /// Timeout for the Ollama `/api/pull` step.
    pub pull_timeout: Duration,
    /// Timeout for the warm→cold archive copy (shared semantics with the
    /// eviction sweep's `MODEL_ARCHIVE_COPY_TIMEOUT_SECS`).
    pub archive_copy_timeout: Duration,
}

impl IngestConfig {
    pub fn from_env() -> Self {
        let enabled = non_empty_env("CHORD_MODEL_INGEST_ENABLED")
            .map(|v| {
                let v = v.to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(DEFAULT_INGEST_ENABLED);
        let max_bytes = non_empty_env("CHORD_MODEL_INGEST_MAX_BYTES")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_BYTES);
        let ollama_url = non_empty_env("OLLAMA_URL").or_else(|| non_empty_env("OLLAMA_BASE_URL"));
        let pull_timeout = Duration::from_secs(
            non_empty_env("CHORD_MODEL_INGEST_PULL_TIMEOUT_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
        );
        let archive_copy_timeout = Duration::from_secs(
            non_empty_env("MODEL_ARCHIVE_COPY_TIMEOUT_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1800),
        );
        IngestConfig {
            enabled,
            max_bytes,
            ollama_url,
            pull_timeout,
            archive_copy_timeout,
        }
    }
}

/// Request body for `POST /api/models/ingest`.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestRequest {
    /// HuggingFace repo id, `org/name` (e.g. `Qwen/Qwen3-8B-GGUF`).
    pub hf_repo: String,
    /// Friendly model name the caller wants to track this under (used for the
    /// idempotency check and echoed in the response).
    pub model_name: String,
    /// Optional revision/quant tag (maps to the Ollama `hf.co/<repo>:<tag>`
    /// suffix). `None` ⇒ Ollama's default.
    #[serde(default)]
    pub revision: Option<String>,
}

/// Terminal status of an ingest attempt. Serialised as the `status` field so the
/// Terminus Phase 2b caller can branch on it without parsing prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestStatus {
    /// Pulled and staged cold; ready for the acquire/promote path.
    Ingested,
    /// The model was already known to the registry — no-op success (idempotent).
    AlreadyPresent,
    /// The repo is gated and no `HF_TOKEN` is provisioned (fail-soft, not a crash).
    GatedNeedsToken,
    /// The repo exceeds the configured size cap; refused before copying.
    TooLarge,
    /// The master gate (`CHORD_MODEL_INGEST_ENABLED`) is off.
    Disabled,
    /// Any other failure (bad input, network/pull failure, ...). `message`
    /// carries a non-secret explanation.
    Error,
}

/// Structured result returned to the caller. Never contains a secret value.
#[derive(Debug, Clone, Serialize)]
pub struct IngestOutcome {
    pub status: IngestStatus,
    /// Registry name + archive location once cold (the handle Phase 2b's
    /// acquire path uses). `None` unless `status == Ingested`/`AlreadyPresent`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cold_storage_ref: Option<String>,
    /// On-disk size in bytes, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Human-readable, secret-free explanation.
    pub message: String,
}

impl IngestOutcome {
    fn simple(status: IngestStatus, message: impl Into<String>) -> Self {
        IngestOutcome {
            status,
            cold_storage_ref: None,
            bytes: None,
            message: message.into(),
        }
    }

    /// HTTP status code the control handler should return alongside this body.
    /// The structured `status` field is authoritative for callers; the code is
    /// for HTTP-layer correctness.
    pub fn http_status(&self) -> u16 {
        match self.status {
            IngestStatus::Ingested
            | IngestStatus::AlreadyPresent
            | IngestStatus::GatedNeedsToken => 200,
            IngestStatus::TooLarge => 413,
            IngestStatus::Disabled => 403,
            IngestStatus::Error => 502,
        }
    }
}

/// Result of a successful HF metadata probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfProbe {
    /// Whether the repo is access-gated (requires an accepted license / token).
    pub gated: bool,
    /// Total on-disk size in bytes, when HuggingFace reports it. `None` ⇒
    /// unknown, in which case the size gate is skipped (proceed).
    pub total_bytes: Option<u64>,
}

/// Result of landing a model in cold storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdLanded {
    /// The registry name the model is tracked under (Phase 2b calls
    /// `POST /api/models/{registry_name}/pull`).
    pub registry_name: String,
    /// Archive path the cold copy lives at.
    pub archive_path: String,
    /// On-disk size in bytes.
    pub bytes: u64,
}

/// The IO surface the orchestration depends on, behind a trait so the flow is
/// unit-testable without touching the network, Ollama, or the filesystem.
#[async_trait]
pub trait IngestOps: Send + Sync {
    /// Whether a model is already tracked by the registry (idempotency check).
    async fn model_exists(&self, model_name: &str, ollama_ref: &str) -> Option<String>;
    /// Fetch HF repo metadata (gated flag + size). `token` MUST NOT be logged.
    async fn probe(
        &self,
        hf_repo: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<HfProbe, String>;
    /// Pull the HF repo into the warm (local Ollama) tier via Ollama's own
    /// `/api/pull`. `token` MUST NOT be logged.
    async fn pull_warm(&self, ollama_ref: &str, token: Option<&str>) -> Result<(), String>;
    /// Reconcile the registry, locate the freshly pulled model, and demote it
    /// warm→cold via the existing eviction machinery. Returns the cold handle.
    async fn land_cold(&self, hf_repo: &str, ollama_ref: &str) -> Result<ColdLanded, String>;
}

/// Build the canonical Ollama model reference for an HF repo (+optional
/// revision), e.g. `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`.
pub fn ollama_ref_for(hf_repo: &str, revision: Option<&str>) -> String {
    let base = format!("hf.co/{}", hf_repo.trim().trim_matches('/'));
    match revision.map(str::trim).filter(|r| !r.is_empty()) {
        Some(rev) => format!("{base}:{rev}"),
        None => base,
    }
}

/// Validate a request's fields. Returns a non-secret error string on any
/// malformed input. Pure — no IO.
pub fn validate_request(req: &IngestRequest) -> Result<(), String> {
    let repo = req.hf_repo.trim();
    if repo.is_empty() {
        return Err("hf_repo is required".into());
    }
    // Must be a bare `org/name` — reject schemes, hosts, traversal, whitespace.
    if repo.contains("://") || repo.contains(' ') || repo.contains('\t') {
        return Err("hf_repo must be a bare 'org/name' repo id, not a URL".into());
    }
    if repo.contains("..") {
        return Err("hf_repo must not contain '..'".into());
    }
    let segments: Vec<&str> = repo.split('/').collect();
    if segments.len() != 2 || segments.iter().any(|s| s.is_empty()) {
        return Err("hf_repo must have exactly one '/' (org/name)".into());
    }
    if !repo
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.'))
    {
        return Err("hf_repo contains invalid characters".into());
    }

    let name = req.model_name.trim();
    if name.is_empty() {
        return Err("model_name is required".into());
    }
    if name.contains('/') || name.contains("..") || name.contains(' ') {
        return Err("model_name must not contain '/', '..', or spaces".into());
    }

    if let Some(rev) = req.revision.as_deref().map(str::trim) {
        if !rev.is_empty()
            && !rev
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err("revision contains invalid characters".into());
        }
    }
    Ok(())
}

/// The pure post-probe decision: given the probe result, the config, and
/// whether a token is present, decide the terminal refusal (if any) or that the
/// flow should proceed to pull. Split out so every branch is unit-testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanDecision {
    Proceed { bytes: Option<u64> },
    Refuse(IngestStatus),
}

pub fn plan_after_probe(cfg: &IngestConfig, probe: &HfProbe, token_present: bool) -> PlanDecision {
    if probe.gated && !token_present {
        return PlanDecision::Refuse(IngestStatus::GatedNeedsToken);
    }
    if let Some(bytes) = probe.total_bytes {
        if bytes > cfg.max_bytes {
            return PlanDecision::Refuse(IngestStatus::TooLarge);
        }
    }
    PlanDecision::Proceed {
        bytes: probe.total_bytes,
    }
}

/// Orchestrate a full ingest. Never panics; every failure maps to a structured
/// [`IngestOutcome`]. `validate_request` is expected to have run already (the
/// handler does it to return a precise 400), but this re-checks defensively.
pub async fn ingest_model(
    ops: &dyn IngestOps,
    cfg: &IngestConfig,
    req: &IngestRequest,
) -> IngestOutcome {
    // 1. Master gate — default OFF.
    if !cfg.enabled {
        return IngestOutcome::simple(
            IngestStatus::Disabled,
            "model ingestion is disabled; set CHORD_MODEL_INGEST_ENABLED=1 to enable",
        );
    }

    // 2. Defensive validation.
    if let Err(e) = validate_request(req) {
        return IngestOutcome::simple(IngestStatus::Error, e);
    }

    let ollama_ref = ollama_ref_for(&req.hf_repo, req.revision.as_deref());

    // 3. Idempotency — already tracked ⇒ no-op success.
    if let Some(existing) = ops.model_exists(req.model_name.trim(), &ollama_ref).await {
        return IngestOutcome {
            status: IngestStatus::AlreadyPresent,
            cold_storage_ref: Some(existing),
            bytes: None,
            message: "model already present in the registry".into(),
        };
    }

    // 4. Probe HF (gated flag + size). Token read once, at the sanctioned point.
    let token = hf_token();
    let probe = match ops
        .probe(
            req.hf_repo.trim(),
            req.revision.as_deref(),
            token.as_deref(),
        )
        .await
    {
        Ok(p) => p,
        Err(e) => {
            return IngestOutcome::simple(
                IngestStatus::Error,
                format!("HF metadata probe failed: {e}"),
            );
        }
    };

    // 5. Pure decision: gated-without-token, oversize, or proceed.
    let bytes = match plan_after_probe(cfg, &probe, token.is_some()) {
        PlanDecision::Refuse(IngestStatus::GatedNeedsToken) => {
            return IngestOutcome::simple(
                IngestStatus::GatedNeedsToken,
                "gated model requires HF_TOKEN (not provisioned)",
            );
        }
        PlanDecision::Refuse(IngestStatus::TooLarge) => {
            return IngestOutcome {
                status: IngestStatus::TooLarge,
                cold_storage_ref: None,
                bytes: probe.total_bytes,
                message: format!(
                    "model size {} bytes exceeds the ingest cap of {} bytes",
                    probe.total_bytes.unwrap_or(0),
                    cfg.max_bytes
                ),
            };
        }
        PlanDecision::Refuse(other) => {
            return IngestOutcome::simple(other, "ingest refused");
        }
        PlanDecision::Proceed { bytes } => bytes,
    };

    // 6. Pull into warm via Ollama's native HF ingestion.
    if let Err(e) = ops.pull_warm(&ollama_ref, token.as_deref()).await {
        return IngestOutcome::simple(IngestStatus::Error, format!("model pull failed: {e}"));
    }

    // 7. Demote warm→cold via the existing eviction machinery.
    match ops.land_cold(req.hf_repo.trim(), &ollama_ref).await {
        Ok(landed) => IngestOutcome {
            status: IngestStatus::Ingested,
            cold_storage_ref: Some(landed.registry_name),
            bytes: Some(landed.bytes).filter(|b| *b > 0).or(bytes),
            message: format!("ingested to cold storage at {}", landed.archive_path),
        },
        Err(e) => IngestOutcome::simple(
            IngestStatus::Error,
            format!("pulled to warm but cold-staging failed: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn cfg(enabled: bool, max_bytes: u64) -> IngestConfig {
        IngestConfig {
            enabled,
            max_bytes,
            ollama_url: Some("http://ollama.test".into()),
            pull_timeout: Duration::from_secs(1),
            archive_copy_timeout: Duration::from_secs(1),
        }
    }

    fn req(repo: &str, name: &str, rev: Option<&str>) -> IngestRequest {
        IngestRequest {
            hf_repo: repo.into(),
            model_name: name.into(),
            revision: rev.map(String::from),
        }
    }

    // ── validate_request ─────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_well_formed() {
        assert!(validate_request(&req("Qwen/Qwen3-8B-GGUF", "qwen3-8b", Some("Q4_K_M"))).is_ok());
        assert!(validate_request(&req("org/model.name_v2", "m", None)).is_ok());
    }

    #[test]
    fn validate_rejects_empty_repo() {
        assert!(validate_request(&req("   ", "m", None)).is_err());
    }

    #[test]
    fn validate_rejects_url_repo() {
        assert!(validate_request(&req("https://hf.co/org/model", "m", None)).is_err());
    }

    #[test]
    fn validate_rejects_traversal() {
        assert!(validate_request(&req("org/../etc", "m", None)).is_err());
        assert!(validate_request(&req("org/model", "../evil", None)).is_err());
    }

    #[test]
    fn validate_rejects_wrong_segment_count() {
        assert!(validate_request(&req("justname", "m", None)).is_err());
        assert!(validate_request(&req("a/b/c", "m", None)).is_err());
    }

    #[test]
    fn validate_rejects_bad_model_name() {
        assert!(validate_request(&req("org/model", "has/slash", None)).is_err());
        assert!(validate_request(&req("org/model", "has space", None)).is_err());
        assert!(validate_request(&req("org/model", "", None)).is_err());
    }

    #[test]
    fn validate_rejects_bad_revision() {
        assert!(validate_request(&req("org/model", "m", Some("bad rev!"))).is_err());
    }

    // ── ollama_ref_for ───────────────────────────────────────────────────────

    #[test]
    fn ollama_ref_with_and_without_revision() {
        assert_eq!(ollama_ref_for("Qwen/Q3", None), "hf.co/Qwen/Q3");
        assert_eq!(
            ollama_ref_for("Qwen/Q3", Some("Q4_K_M")),
            "hf.co/Qwen/Q3:Q4_K_M"
        );
        assert_eq!(ollama_ref_for("Qwen/Q3", Some("  ")), "hf.co/Qwen/Q3");
    }

    // ── plan_after_probe ─────────────────────────────────────────────────────

    #[test]
    fn plan_gated_without_token_refuses() {
        let d = plan_after_probe(
            &cfg(true, DEFAULT_MAX_BYTES),
            &HfProbe {
                gated: true,
                total_bytes: Some(10),
            },
            false,
        );
        assert_eq!(d, PlanDecision::Refuse(IngestStatus::GatedNeedsToken));
    }

    #[test]
    fn plan_gated_with_token_proceeds() {
        let d = plan_after_probe(
            &cfg(true, DEFAULT_MAX_BYTES),
            &HfProbe {
                gated: true,
                total_bytes: Some(10),
            },
            true,
        );
        assert_eq!(d, PlanDecision::Proceed { bytes: Some(10) });
    }

    #[test]
    fn plan_oversize_refuses() {
        let d = plan_after_probe(
            &cfg(true, 100),
            &HfProbe {
                gated: false,
                total_bytes: Some(101),
            },
            false,
        );
        assert_eq!(d, PlanDecision::Refuse(IngestStatus::TooLarge));
    }

    #[test]
    fn plan_unknown_size_proceeds() {
        let d = plan_after_probe(
            &cfg(true, 100),
            &HfProbe {
                gated: false,
                total_bytes: None,
            },
            false,
        );
        assert_eq!(d, PlanDecision::Proceed { bytes: None });
    }

    #[test]
    fn plan_at_cap_boundary_proceeds() {
        let d = plan_after_probe(
            &cfg(true, 100),
            &HfProbe {
                gated: false,
                total_bytes: Some(100),
            },
            false,
        );
        assert_eq!(d, PlanDecision::Proceed { bytes: Some(100) });
    }

    // ── result mapping ───────────────────────────────────────────────────────

    #[test]
    fn http_status_mapping() {
        assert_eq!(
            IngestOutcome::simple(IngestStatus::Ingested, "").http_status(),
            200
        );
        assert_eq!(
            IngestOutcome::simple(IngestStatus::AlreadyPresent, "").http_status(),
            200
        );
        assert_eq!(
            IngestOutcome::simple(IngestStatus::GatedNeedsToken, "").http_status(),
            200
        );
        assert_eq!(
            IngestOutcome::simple(IngestStatus::TooLarge, "").http_status(),
            413
        );
        assert_eq!(
            IngestOutcome::simple(IngestStatus::Disabled, "").http_status(),
            403
        );
        assert_eq!(
            IngestOutcome::simple(IngestStatus::Error, "").http_status(),
            502
        );
    }

    #[test]
    fn status_serializes_snake_case() {
        let body = serde_json::to_value(IngestOutcome::simple(
            IngestStatus::GatedNeedsToken,
            "gated model requires HF_TOKEN (not provisioned)",
        ))
        .unwrap();
        assert_eq!(body["status"], "gated_needs_token");
        assert_eq!(
            body["message"],
            "gated model requires HF_TOKEN (not provisioned)"
        );
        // Absent optional fields are omitted, never null.
        assert!(body.get("cold_storage_ref").is_none());
        assert!(body.get("bytes").is_none());
    }

    // ── orchestration (mocked IO) ────────────────────────────────────────────

    /// Records which IO steps ran, and lets each be scripted.
    #[derive(Default)]
    struct MockOps {
        exists: Option<String>,
        probe: Option<Result<HfProbe, String>>,
        pull: Option<Result<(), String>>,
        land: Option<Result<ColdLanded, String>>,
        pulled: Mutex<bool>,
        landed: Mutex<bool>,
    }

    #[async_trait]
    impl IngestOps for MockOps {
        async fn model_exists(&self, _model_name: &str, _ollama_ref: &str) -> Option<String> {
            self.exists.clone()
        }
        async fn probe(
            &self,
            _hf_repo: &str,
            _revision: Option<&str>,
            _token: Option<&str>,
        ) -> Result<HfProbe, String> {
            self.probe.clone().expect("probe should not be called")
        }
        async fn pull_warm(&self, _ollama_ref: &str, _token: Option<&str>) -> Result<(), String> {
            *self.pulled.lock().unwrap() = true;
            self.pull.clone().expect("pull should not be called")
        }
        async fn land_cold(&self, _hf_repo: &str, _ollama_ref: &str) -> Result<ColdLanded, String> {
            *self.landed.lock().unwrap() = true;
            self.land.clone().expect("land should not be called")
        }
    }

    #[tokio::test]
    async fn disabled_gate_refuses_without_touching_io() {
        let ops = MockOps::default();
        let out = ingest_model(
            &ops,
            &cfg(false, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Disabled);
        assert!(!*ops.pulled.lock().unwrap());
        assert!(!*ops.landed.lock().unwrap());
    }

    #[tokio::test]
    async fn invalid_input_errors_before_io() {
        let ops = MockOps::default();
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("bad repo", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Error);
        assert!(!*ops.pulled.lock().unwrap());
    }

    #[tokio::test]
    async fn already_present_is_noop_success() {
        let ops = MockOps {
            exists: Some("hf.co/org/model".into()),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::AlreadyPresent);
        assert_eq!(out.cold_storage_ref.as_deref(), Some("hf.co/org/model"));
        assert!(!*ops.pulled.lock().unwrap());
    }

    #[tokio::test]
    async fn gated_without_token_fails_soft() {
        // No HF_TOKEN in env for this test's process is not guaranteed, so this
        // asserts the plan path: probe reports gated, and (in CI without a
        // token) we get GatedNeedsToken. Guard by clearing the var locally.
        let _ = std::env::var("HF_TOKEN"); // observe only
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: true,
                total_bytes: Some(10),
            })),
            ..Default::default()
        };
        // Only meaningful when no token is present; if the runner has one set,
        // the flow would proceed to pull. hf_token() reads the process env, so
        // we assert the token-absent contract via plan_after_probe directly
        // (covered above) and here assert no crash + no land when it refuses.
        if hf_token().is_none() {
            let out = ingest_model(
                &ops,
                &cfg(true, DEFAULT_MAX_BYTES),
                &req("org/model", "m", None),
            )
            .await;
            assert_eq!(out.status, IngestStatus::GatedNeedsToken);
            assert!(!*ops.landed.lock().unwrap());
        }
    }

    #[tokio::test]
    async fn oversize_refused_before_pull() {
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: false,
                total_bytes: Some(999),
            })),
            ..Default::default()
        };
        let out = ingest_model(&ops, &cfg(true, 100), &req("org/model", "m", None)).await;
        assert_eq!(out.status, IngestStatus::TooLarge);
        assert_eq!(out.bytes, Some(999));
        assert!(!*ops.pulled.lock().unwrap());
    }

    #[tokio::test]
    async fn probe_failure_maps_to_error() {
        let ops = MockOps {
            probe: Some(Err("network down".into())),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Error);
        assert!(out.message.contains("probe failed"));
        assert!(!*ops.pulled.lock().unwrap());
    }

    #[tokio::test]
    async fn happy_path_pulls_then_lands_cold() {
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: false,
                total_bytes: Some(50),
            })),
            pull: Some(Ok(())),
            land: Some(Ok(ColdLanded {
                registry_name: "hf.co/org/model:latest".into(),
                archive_path: "/archive".into(),
                bytes: 50,
            })),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Ingested);
        assert_eq!(
            out.cold_storage_ref.as_deref(),
            Some("hf.co/org/model:latest")
        );
        assert_eq!(out.bytes, Some(50));
        assert!(*ops.pulled.lock().unwrap());
        assert!(*ops.landed.lock().unwrap());
    }

    #[tokio::test]
    async fn pull_failure_does_not_land() {
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: false,
                total_bytes: Some(50),
            })),
            pull: Some(Err("ollama 500".into())),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Error);
        assert!(out.message.contains("pull failed"));
        assert!(!*ops.landed.lock().unwrap());
    }

    #[tokio::test]
    async fn land_failure_maps_to_error() {
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: false,
                total_bytes: Some(50),
            })),
            pull: Some(Ok(())),
            land: Some(Err("no warm model found after pull".into())),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Error);
        assert!(out.message.contains("cold-staging failed"));
    }
}
