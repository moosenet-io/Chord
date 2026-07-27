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
//!   registry record to [`crate::models::registry::StorageTier::Cold`].
//! - **Registry discovery:** [`crate::models::registry::ModelRegistry::reconcile`]
//!   picks up the just-pulled Ollama model, exactly as it does at startup.
//! - **Later promotion:** once cold, the model is reachable by the acquire path
//!   Terminus already calls — `POST /api/models/:name/pull` →
//!   `pull_coordinator.ensure_local` → `transfer::archive_pull` (cold→warm).
//!
//! ## Registry naming (why we never search for `hf.co/...`)
//! Ollama pulls an HF model under the source name `hf.co/<org>/<model>[:<tag>]`,
//! but [`crate::models::registry::ModelRegistry::reconcile`] **strips the
//! registry host** and records the model under `<org>/<model>:<tag>` (asserted
//! by the reconciler's own test in `registry.rs`). So the idempotency check and
//! the post-pull "find the warm model" step must use the host-stripped key — see
//! [`canonical_registry_names`]. Searching for the literal `hf.co/...` name
//! never matches and would leave the model stranded warm.
//!
//! ## `revision` is the Ollama tag / quant selector, NOT an HF git revision
//! The request's `revision` field maps to the Ollama `hf.co/<repo>:<tag>` suffix
//! — i.e. it selects a quantization/file variant (e.g. `Q4_K_M`), the way Ollama
//! consumes HF GGUF repos. It is deliberately **not** used as an HF git
//! revision in the metadata-probe URL (that would make a normal quant request
//! probe a nonexistent git ref). The probe reads the repo root and, when a
//! `revision`/quant is given, sizes only the sibling files matching that quant
//! so a multi-quant repo's whole-repo `usedStorage` doesn't cause a false
//! `too_large` refusal.
//!
//! ## Security posture
//! - **Default-OFF master gate** (`CHORD_MODEL_INGEST_ENABLED`, default `0`):
//!   merging this changes nothing until an operator explicitly enables it. The
//!   gate is checked **before** input validation, so a disabled endpoint never
//!   does any work or leaks validation detail.
//! - **Authenticated** by the same JWT guard as every other control endpoint
//!   (the handler in `control.rs` runs `auth_check` before calling in here).
//! - **HF token from the vault, never a literal.** [`hf_token`] is the single
//!   sanctioned read point; the value is materialised into the process
//!   environment from <secret-manager> at startup by
//!   `secrets_bootstrap::fetch_and_apply_downstream_secrets` (vault key
//!   `HF_PAT_MOOSE`, materialised under the standard env var `HF_TOKEN`), like
//!   `OPENROUTER_API_KEY`. It is never hardcoded, never logged, and never echoed
//!   into a response or error string.
//! - **Gated-model contract (explicit refusal, never a dead-end pull).** The
//!   actual download is delegated to the **separate Ollama daemon** over HTTP.
//!   Ollama's `/api/pull` has no per-request credential field, and current
//!   Ollama builds have no HF-token consumer for `hf.co/...` pulls at all
//!   (credential-free pull API) — so an *automated* gated pull cannot
//!   authenticate through this path, regardless of any token Chord holds.
//!   Rather than probe-pass a gated repo and then dead-end in an
//!   un-authenticatable pull, a gated repo is **refused up front** with
//!   [`IngestStatus::GatedNeedsToken`] and a clear message; **nothing is
//!   pulled**. Chord's `HF_TOKEN` is used ONLY for the metadata probe, so a
//!   gated repo is reliably *detected* (a `gated` flag, or a `401/403`) instead
//!   of being mistaken for a 404. A public repo pulls with no token at all.
//!   (If a future Ollama build gains real gated-pull support, this is the one
//!   place to relax the refusal — verify against the deployed daemon first.)
//! - **Bounded:** an oversize repo is refused ([`IngestStatus::TooLarge`])
//!   before any byte is copied, protecting the shared disk (<host> disk-pressure
//!   discipline), and the pull/copy steps are wrapped in timeouts.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One sanctioned read point for the HuggingFace access token.
///
/// The value is populated into the process environment at startup by
/// `secrets_bootstrap::fetch_and_apply_downstream_secrets` (<secret-manager>-first,
/// vault key `HF_PAT_MOOSE` → env `HF_TOKEN`) — it is never authored as a
/// literal, never logged, and never returned in any response or error text.
/// Mirrors `embeddings::openrouter_api_key()`. Returns
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
    /// Timeout for the Ollama `/api/pull` step (an HF download can be long).
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
    /// Friendly model name the caller wants to track this under (echoed in the
    /// response). NOTE: the registry key is derived from `hf_repo`/`revision`
    /// (see [`canonical_registry_names`]), NOT from this field — so a
    /// `model_name` collision can never misidentify an unrelated model.
    pub model_name: String,
    /// Optional Ollama tag / quantization selector (e.g. `Q4_K_M`), mapped to
    /// the `hf.co/<repo>:<tag>` suffix. NOT an HF git revision. `None` ⇒
    /// Ollama's default (`latest`).
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
    /// The model was already cold-stored — no-op success (idempotent).
    AlreadyPresent,
    /// The repo is gated and no `HF_TOKEN` is provisioned (fail-soft, not a crash).
    GatedNeedsToken,
    /// The repo exceeds the configured size cap; refused before copying.
    TooLarge,
    /// The master gate (`CHORD_MODEL_INGEST_ENABLED`) is off.
    Disabled,
    /// Any other failure (bad input, network/pull failure, model loaded hot, ...).
    /// `message` carries a non-secret explanation.
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
    /// Total on-disk size in bytes, when HuggingFace reports it (and, when a
    /// quant/`revision` was given, filtered to the matching files). `None` ⇒
    /// unknown, in which case the size gate is skipped (proceed).
    pub total_bytes: Option<u64>,
}

/// Storage tier of an already-known model (decoupled from
/// [`crate::models::registry::StorageTier`] so this module stays unit-testable
/// without the registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Cold,
    Warm,
    Hot,
}

/// An already-known registry record matched during the idempotency check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingModel {
    pub name: String,
    pub tier: ModelTier,
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
    /// Look up whether any of the host-stripped `candidates` is already in the
    /// registry, returning the match + its tier (idempotency). The candidates
    /// come from [`canonical_registry_names`], never a `hf.co/...` literal.
    async fn find_existing(&self, candidates: &[String]) -> Option<ExistingModel>;
    /// Fetch HF repo metadata (gated flag + quant-aware size). `token` MUST NOT
    /// be logged. `revision` (a quant selector) is used only to filter the
    /// sibling files summed for `total_bytes`, never as an HF git ref.
    async fn probe(
        &self,
        hf_repo: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<HfProbe, String>;
    /// Pull the HF repo into the warm (local Ollama) tier via Ollama's own
    /// `/api/pull`. No token is forwarded — a gated pull depends on the Ollama
    /// daemon's own environment (see the module-level gated-model contract).
    async fn pull_warm(&self, ollama_ref: &str) -> Result<(), String>;
    /// Reconcile the registry, locate the freshly pulled model by one of the
    /// host-stripped `candidates`, and demote it warm→cold via the existing
    /// eviction machinery (under the shared disk-op lock). Returns the cold
    /// handle.
    async fn land_cold(&self, candidates: &[String]) -> Result<ColdLanded, String>;
}

/// Build the canonical Ollama source reference for an HF repo (+optional quant),
/// e.g. `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`. This is the name Ollama pulls under;
/// the *registry* records it host-stripped — see [`canonical_registry_names`].
pub fn ollama_ref_for(hf_repo: &str, revision: Option<&str>) -> String {
    let base = format!("hf.co/{}", hf_repo.trim().trim_matches('/'));
    match revision.map(str::trim).filter(|r| !r.is_empty()) {
        Some(tag) => format!("{base}:{tag}"),
        None => base,
    }
}

/// The registry key(s) [`crate::models::registry::ModelRegistry::reconcile`]
/// records for an Ollama-pulled HF model, i.e. with the `hf.co/` host
/// **stripped** (matching the reconciler's own behaviour: `hf.co/org/model:tag`
/// → `org/model:tag`).
///
/// - With a `revision`/quant tag ⇒ exactly `org/model:<tag>`.
/// - Without one ⇒ both `org/model:latest` (Ollama's default tag) and the bare
///   `org/model`, since which of the two the reconciler records depends on how
///   the manifest leaf is named on disk.
///
/// This is the load-bearing derivation the idempotency check and the post-pull
/// "find the warm model" step rely on; it is unit-tested against the shape the
/// reconciler test asserts.
pub fn canonical_registry_names(hf_repo: &str, revision: Option<&str>) -> Vec<String> {
    let repo = hf_repo.trim().trim_matches('/').to_string();
    match revision.map(str::trim).filter(|r| !r.is_empty()) {
        Some(tag) => vec![format!("{repo}:{tag}")],
        None => vec![format!("{repo}:latest"), repo],
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

pub fn plan_after_probe(cfg: &IngestConfig, probe: &HfProbe) -> PlanDecision {
    // Gated repos are refused up front (regardless of any token Chord holds):
    // the Ollama-delegated pull has no way to authenticate them. See the
    // module-level gated-model contract.
    if probe.gated {
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

/// Given a `land_cold` result, build the success outcome (shared by the
/// already-warm and pulled-fresh paths).
fn ingested_outcome(landed: ColdLanded, probe_bytes: Option<u64>) -> IngestOutcome {
    IngestOutcome {
        status: IngestStatus::Ingested,
        cold_storage_ref: Some(landed.registry_name),
        bytes: Some(landed.bytes).filter(|b| *b > 0).or(probe_bytes),
        message: format!("ingested to cold storage at {}", landed.archive_path),
    }
}

/// Orchestrate a full ingest. Never panics; every failure maps to a structured
/// [`IngestOutcome`]. Order: gate → validate → idempotency → probe → pull →
/// land-cold.
pub async fn ingest_model(
    ops: &dyn IngestOps,
    cfg: &IngestConfig,
    req: &IngestRequest,
) -> IngestOutcome {
    // 1. Master gate — default OFF — checked BEFORE validation.
    if !cfg.enabled {
        return IngestOutcome::simple(
            IngestStatus::Disabled,
            "model ingestion is disabled; set CHORD_MODEL_INGEST_ENABLED=1 to enable",
        );
    }

    // 2. Validation.
    if let Err(e) = validate_request(req) {
        return IngestOutcome::simple(IngestStatus::Error, e);
    }

    let ollama_ref = ollama_ref_for(&req.hf_repo, req.revision.as_deref());
    let candidates = canonical_registry_names(&req.hf_repo, req.revision.as_deref());

    // 3. Idempotency — tier-aware.
    if let Some(existing) = ops.find_existing(&candidates).await {
        match existing.tier {
            ModelTier::Cold => {
                return IngestOutcome {
                    status: IngestStatus::AlreadyPresent,
                    cold_storage_ref: Some(existing.name),
                    bytes: None,
                    message: "model already cold-stored".into(),
                };
            }
            ModelTier::Hot => {
                return IngestOutcome::simple(
                    IngestStatus::Error,
                    format!(
                        "model {} is loaded (hot); unload it before ingesting to cold",
                        existing.name
                    ),
                );
            }
            ModelTier::Warm => {
                // Already local — skip the pull, just demote it to cold.
                return match ops.land_cold(&candidates).await {
                    Ok(landed) => ingested_outcome(landed, None),
                    Err(e) => IngestOutcome::simple(
                        IngestStatus::Error,
                        format!("warm model present but cold-staging failed: {e}"),
                    ),
                };
            }
        }
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

    // 5. Pure decision: gated (refused), oversize, or proceed.
    let bytes = match plan_after_probe(cfg, &probe) {
        PlanDecision::Refuse(IngestStatus::GatedNeedsToken) => {
            return IngestOutcome::simple(
                IngestStatus::GatedNeedsToken,
                "gated model: automated ingest is not supported by the Ollama-delegated \
                 pull path (no HF credential mechanism on /api/pull); acquire it out-of-band \
                 or use a public/mirror repo",
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

    // 6. Pull into warm via Ollama's native HF ingestion. Only public repos
    //    reach here (gated repos are refused at step 5), so no credential is
    //    involved — a failure is reported plainly.
    if let Err(e) = ops.pull_warm(&ollama_ref).await {
        return IngestOutcome::simple(IngestStatus::Error, format!("model pull failed: {e}"));
    }

    // 7. Demote warm→cold via the existing eviction machinery.
    match ops.land_cold(&candidates).await {
        Ok(landed) => ingested_outcome(landed, bytes),
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

    // ── canonical_registry_names (the load-bearing name derivation) ───────────

    #[test]
    fn canonical_names_are_host_stripped_matching_reconcile() {
        // reconcile records `hf.co/org/model:tag` as `org/model:tag` — NEVER the
        // `hf.co/...` literal. This is the exact class of bug the gate caught.
        assert_eq!(
            canonical_registry_names("org/model", Some("Q4_K_M")),
            vec!["org/model:Q4_K_M".to_string()]
        );
        // No revision ⇒ the reconciler may record either `:latest` or bare.
        assert_eq!(
            canonical_registry_names("org/model", None),
            vec!["org/model:latest".to_string(), "org/model".to_string()]
        );
        // Crucially: none of the candidates carry the `hf.co/` host.
        for c in canonical_registry_names("org/model", Some("Q4_K_M")) {
            assert!(
                !c.starts_with("hf.co/"),
                "candidate must be host-stripped: {c}"
            );
        }
        for c in canonical_registry_names("org/model", None) {
            assert!(
                !c.starts_with("hf.co/"),
                "candidate must be host-stripped: {c}"
            );
        }
    }

    // ── plan_after_probe ─────────────────────────────────────────────────────

    #[test]
    fn plan_gated_refuses_regardless_of_token() {
        // Gated repos are refused up front — a token Chord holds does not change
        // this (the Ollama-delegated pull can't authenticate gated repos).
        let gated = HfProbe {
            gated: true,
            total_bytes: Some(10),
        };
        assert_eq!(
            plan_after_probe(&cfg(true, DEFAULT_MAX_BYTES), &gated),
            PlanDecision::Refuse(IngestStatus::GatedNeedsToken)
        );
    }

    #[test]
    fn plan_oversize_refuses() {
        let d = plan_after_probe(
            &cfg(true, 100),
            &HfProbe {
                gated: false,
                total_bytes: Some(101),
            },
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

    /// Records which IO steps ran, and captures the `candidates` that
    /// find_existing/land_cold were called with, so tests assert the live-adapter
    /// contract (host-stripped names) — closing the test-gap the gate flagged (a
    /// mock that rubber-stamps a `hf.co/...` name would fail these assertions).
    #[derive(Default)]
    struct MockOps {
        existing: Option<ExistingModel>,
        probe: Option<Result<HfProbe, String>>,
        pull: Option<Result<(), String>>,
        land: Option<Result<ColdLanded, String>>,
        pulled: Mutex<bool>,
        land_candidates: Mutex<Option<Vec<String>>>,
        find_candidates: Mutex<Option<Vec<String>>>,
    }

    #[async_trait]
    impl IngestOps for MockOps {
        async fn find_existing(&self, candidates: &[String]) -> Option<ExistingModel> {
            *self.find_candidates.lock().unwrap() = Some(candidates.to_vec());
            self.existing.clone()
        }
        async fn probe(
            &self,
            _hf_repo: &str,
            _revision: Option<&str>,
            _token: Option<&str>,
        ) -> Result<HfProbe, String> {
            self.probe.clone().expect("probe should not be called")
        }
        async fn pull_warm(&self, _ollama_ref: &str) -> Result<(), String> {
            *self.pulled.lock().unwrap() = true;
            self.pull.clone().expect("pull should not be called")
        }
        async fn land_cold(&self, candidates: &[String]) -> Result<ColdLanded, String> {
            *self.land_candidates.lock().unwrap() = Some(candidates.to_vec());
            self.land.clone().expect("land should not be called")
        }
    }

    #[tokio::test]
    async fn disabled_gate_refuses_before_validation() {
        // Even a malformed request is refused with `disabled`, not `error`,
        // because the gate is checked first.
        let ops = MockOps::default();
        let out = ingest_model(
            &ops,
            &cfg(false, DEFAULT_MAX_BYTES),
            &req("bad repo", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Disabled);
        assert!(!*ops.pulled.lock().unwrap());
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
    async fn cold_present_is_noop_success_and_uses_host_stripped_candidates() {
        let ops = MockOps {
            existing: Some(ExistingModel {
                name: "org/model:Q4_K_M".into(),
                tier: ModelTier::Cold,
            }),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", Some("Q4_K_M")),
        )
        .await;
        assert_eq!(out.status, IngestStatus::AlreadyPresent);
        assert_eq!(out.cold_storage_ref.as_deref(), Some("org/model:Q4_K_M"));
        assert!(!*ops.pulled.lock().unwrap());
        // The idempotency lookup used the host-stripped key, never `hf.co/...`.
        let cands = ops.find_candidates.lock().unwrap().clone().unwrap();
        assert_eq!(cands, vec!["org/model:Q4_K_M".to_string()]);
    }

    #[tokio::test]
    async fn warm_present_skips_pull_and_lands_cold() {
        let ops = MockOps {
            existing: Some(ExistingModel {
                name: "org/model:latest".into(),
                tier: ModelTier::Warm,
            }),
            land: Some(Ok(ColdLanded {
                registry_name: "org/model:latest".into(),
                archive_path: "/archive".into(),
                bytes: 42,
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
        assert_eq!(out.cold_storage_ref.as_deref(), Some("org/model:latest"));
        assert!(
            !*ops.pulled.lock().unwrap(),
            "warm model must not be re-pulled"
        );
        // land_cold was handed the host-stripped candidates.
        let cands = ops.land_candidates.lock().unwrap().clone().unwrap();
        assert!(cands.iter().all(|c| !c.starts_with("hf.co/")));
    }

    #[tokio::test]
    async fn hot_present_returns_error() {
        let ops = MockOps {
            existing: Some(ExistingModel {
                name: "org/model:latest".into(),
                tier: ModelTier::Hot,
            }),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        assert_eq!(out.status, IngestStatus::Error);
        assert!(out.message.contains("hot"));
        assert!(!*ops.pulled.lock().unwrap());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn gated_is_refused_without_pulling() {
        // A gated repo is refused before any pull, regardless of whether Chord
        // holds a token — nothing is pulled, no dead-end.
        for token in ["", "some-token"] {
            if token.is_empty() {
                std::env::remove_var("HF_TOKEN");
            } else {
                std::env::set_var("HF_TOKEN", token); // pii-test-fixture
            }
            let ops = MockOps {
                probe: Some(Ok(HfProbe {
                    gated: true,
                    total_bytes: Some(10),
                })),
                ..Default::default()
            };
            let out = ingest_model(
                &ops,
                &cfg(true, DEFAULT_MAX_BYTES),
                &req("org/model", "m", None),
            )
            .await;
            std::env::remove_var("HF_TOKEN");
            assert_eq!(out.status, IngestStatus::GatedNeedsToken);
            assert!(!*ops.pulled.lock().unwrap());
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
                registry_name: "org/model:latest".into(),
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
        assert_eq!(out.cold_storage_ref.as_deref(), Some("org/model:latest"));
        assert_eq!(out.bytes, Some(50));
        assert!(*ops.pulled.lock().unwrap());
        // land_cold received host-stripped candidates (never `hf.co/...`).
        let cands = ops.land_candidates.lock().unwrap().clone().unwrap();
        assert!(cands.iter().all(|c| !c.starts_with("hf.co/")));
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
        assert!(ops.land_candidates.lock().unwrap().is_none());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn gated_with_token_still_refused_and_never_pulls() {
        // Even with a Chord-side token, a gated repo is refused up front (the
        // pull path can't authenticate it) — the pull step is never reached.
        std::env::set_var("HF_TOKEN", "x"); // pii-test-fixture
        let ops = MockOps {
            probe: Some(Ok(HfProbe {
                gated: true,
                total_bytes: Some(10),
            })),
            ..Default::default()
        };
        let out = ingest_model(
            &ops,
            &cfg(true, DEFAULT_MAX_BYTES),
            &req("org/model", "m", None),
        )
        .await;
        std::env::remove_var("HF_TOKEN");
        assert_eq!(out.status, IngestStatus::GatedNeedsToken);
        assert!(!*ops.pulled.lock().unwrap());
        assert!(out.message.contains("gated model"));
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
